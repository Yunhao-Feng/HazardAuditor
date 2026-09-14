#!/usr/bin/env python3
"""Prepare privacy-safe, trajectory-level SFT files for LLaMA-Factory.

Source data is never bundled with HazardAuditor. Callers provide explicit
training and validation JSON/JSONL files containing content, label, and reason.
The script hashes trajectories, removes train/validation overlap, deduplicates
training rows, and optionally balances the unsafe class through deterministic
oversampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from hazard_auditor.output import format_guard_target
from hazard_auditor.prompting import SYSTEM_PROMPT, build_user_prompt, canonical_json


DEFAULT_OUTPUT_DIR = Path("artifacts/sft_data")


class DataPreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceItem:
    source: str
    source_split: str
    source_index: int
    content: Any
    label: int
    reason: str
    content_sha256: str
    oversample_copy: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare standardized HazardAuditor SFT data."
    )
    parser.add_argument(
        "--train-file",
        action="append",
        required=True,
        metavar="[SOURCE=]PATH",
        help="Training JSON/JSONL file; repeat for multiple sources.",
    )
    parser.add_argument(
        "--validation-file",
        action="append",
        required=True,
        metavar="[SOURCE=]PATH",
        help="Held-out JSON/JSONL file; repeat for multiple sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unsafe-to-safe-ratio",
        type=float,
        default=0.90,
        help="Training-only unsafe:safe target ratio (default: 0.90).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def content_hash(content: Any) -> str:
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def parse_source_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        source, raw_path = value.split("=", 1)
        source = source.strip()
        path = Path(raw_path).expanduser()
    else:
        path = Path(value).expanduser()
        source = path.stem
    if not source:
        raise DataPreparationError(f"empty source name in {value!r}")
    path = path.resolve()
    if not path.is_file():
        raise DataPreparationError(f"source file not found: {path}")
    return source, path


def read_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataPreparationError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
        return records
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataPreparationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, list):
        raise DataPreparationError(f"{path}: top level must be a JSON array")
    return value


def load_items(specs: Sequence[str], *, split: str) -> list[SourceItem]:
    items: list[SourceItem] = []
    for spec in specs:
        source, path = parse_source_path(spec)
        for index, record in enumerate(read_records(path)):
            location = f"{path}[{index}]"
            if not isinstance(record, dict):
                raise DataPreparationError(f"{location}: record must be an object")
            missing = {"content", "label", "reason"} - set(record)
            unexpected = set(record) - {"content", "label", "reason", "source"}
            if missing or unexpected:
                raise DataPreparationError(
                    f"{location}: expected content/label/reason and optional source"
                )
            content = record["content"]
            if not isinstance(content, (list, str)) or content in ([], ""):
                raise DataPreparationError(f"{location}: content must be non-empty")
            label = record["label"]
            if type(label) is not int or label not in (0, 1):
                raise DataPreparationError(f"{location}: label must be 0 or 1")
            reason = record["reason"]
            if not isinstance(reason, str) or not reason.strip():
                raise DataPreparationError(f"{location}: reason must be non-empty")
            row_source = record.get("source", source)
            if not isinstance(row_source, str) or not row_source.strip():
                raise DataPreparationError(f"{location}: source must be text")
            items.append(
                SourceItem(
                    source=row_source.strip(),
                    source_split=split,
                    source_index=index,
                    content=content,
                    label=label,
                    reason=" ".join(reason.split()),
                    content_sha256=content_hash(content),
                )
            )
    return items


def deduplicate(
    items: Sequence[SourceItem], *, forbidden_hashes: set[str]
) -> tuple[list[SourceItem], dict[str, int]]:
    groups: dict[str, list[SourceItem]] = {}
    removed_overlap = 0
    for item in items:
        if item.content_sha256 in forbidden_hashes:
            removed_overlap += 1
            continue
        groups.setdefault(item.content_sha256, []).append(item)

    selected: list[SourceItem] = []
    removed_duplicates = 0
    for sha256, group in groups.items():
        labels = {item.label for item in group}
        if len(labels) != 1:
            raise DataPreparationError(
                f"identical trajectory {sha256} has conflicting labels"
            )
        selected.append(
            min(group, key=lambda item: (item.source, item.source_index))
        )
        removed_duplicates += len(group) - 1
    selected.sort(key=lambda item: (item.source, item.source_index))
    return selected, {
        "removed_validation_overlap": removed_overlap,
        "removed_exact_duplicates": removed_duplicates,
    }


def oversample_unsafe(
    items: Sequence[SourceItem], *, target_ratio: float, seed: int
) -> tuple[list[SourceItem], dict[str, Any]]:
    if not 0 < target_ratio < 1:
        raise DataPreparationError("unsafe-to-safe ratio must be between 0 and 1")
    safe_items = [item for item in items if item.label == 0]
    unsafe_items = [item for item in items if item.label == 1]
    if not safe_items or not unsafe_items:
        raise DataPreparationError("training data must contain both labels")
    target_unsafe = round(len(safe_items) * target_ratio)
    remaining = max(0, target_unsafe - len(unsafe_items))
    extras: list[SourceItem] = []
    ordered = sorted(
        unsafe_items, key=lambda item: (item.content_sha256, item.source)
    )
    copy_number = 1
    while remaining:
        candidates = list(ordered)
        random.Random(seed + copy_number * 1_000_003).shuffle(candidates)
        selected = candidates[: min(remaining, len(candidates))]
        extras.extend(
            replace(item, oversample_copy=copy_number) for item in selected
        )
        remaining -= len(selected)
        copy_number += 1
    return list(items) + extras, {
        "target_unsafe_to_safe_ratio": target_ratio,
        "before": {"safe": len(safe_items), "unsafe": len(unsafe_items)},
        "added_unsafe_copies": len(extras),
        "after": {
            "safe": len(safe_items),
            "unsafe": len(unsafe_items) + len(extras),
        },
    }


def sft_messages(item: SourceItem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item.content)},
        {"role": "assistant", "content": format_guard_target(item.reason, item.label)},
    ]


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            count += 1
    return count


def write_split(
    directory: Path,
    *,
    name: str,
    items: Sequence[SourceItem],
) -> int:
    dataset_path = directory / f"guard_{name}.jsonl"
    manifest_path = directory / f"guard_{name}_manifest.jsonl"
    count = write_jsonl(
        dataset_path,
        ({"messages": sft_messages(item)} for item in items),
    )
    write_jsonl(
        manifest_path,
        (
            {
                "sft_line": index,
                "source": item.source,
                "source_split": item.source_split,
                "source_index": item.source_index,
                "is_oversampled": item.oversample_copy > 0,
                "oversample_copy": item.oversample_copy,
                "label": item.label,
                "label_text": "unsafe" if item.label else "safe",
                "content_sha256": item.content_sha256,
            }
            for index, item in enumerate(items, 1)
        ),
    )
    return count


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise DataPreparationError(f"stale staging directory exists: {staging}")
    if output.exists() and not args.overwrite:
        raise DataPreparationError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    staging.mkdir(parents=True)
    try:
        validation = load_items(args.validation_file, split="validation")
        validation_hashes = {item.content_sha256 for item in validation}
        if len(validation_hashes) != len(validation):
            raise DataPreparationError("validation data contains duplicate trajectories")
        raw_training = load_items(args.train_file, split="train")
        unique_training, removals = deduplicate(
            raw_training, forbidden_hashes=validation_hashes
        )
        training, oversampling = oversample_unsafe(
            unique_training,
            target_ratio=args.unsafe_to_safe_ratio,
            seed=args.seed,
        )
        random.Random(args.seed).shuffle(training)

        train_count = write_split(staging, name="train", items=training)
        validation_count = write_split(
            staging, name="validation", items=validation
        )
        dataset_info = {
            "hazard_auditor_train": {
                "file_name": "guard_train.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            },
            "hazard_auditor_validation": {
                "file_name": "guard_validation.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            },
        }
        (staging / "dataset_info.json").write_text(
            json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8"
        )
        audit = {
            "seed": args.seed,
            "training_rows": train_count,
            "validation_rows": validation_count,
            "training_labels": dict(Counter(item.label for item in training)),
            "validation_labels": dict(Counter(item.label for item in validation)),
            "deduplication": removals,
            "oversampling": oversampling,
        }
        (staging / "audit_report.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

        if output.exists():
            if output.parent == output or output == Path.cwd().resolve():
                raise DataPreparationError(f"refusing to replace broad path: {output}")
            shutil.rmtree(output)
        staging.replace(output)
        print(json.dumps(audit, indent=2))
        return 0
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataPreparationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
