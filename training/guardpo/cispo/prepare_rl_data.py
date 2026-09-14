#!/usr/bin/env python3
"""Convert RLGuard SFT JSONL into leakage-safe VERL RL datasets.

The records use JSON Lines encoding with a ``.json`` suffix. This is deliberate:
VERL 0.7 dispatches RLHFDataset by suffix and accepts ``.json`` (not
``.jsonl``), while Hugging Face's JSON loader accepts one object per line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.guardpo.cispo.constants import INT_TO_LABEL, LABEL_TO_INT, VALID_LABELS
from training.guardpo.cispo.formatting import extract_untrusted_trajectory, parse_reference_response


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "artifacts" / "sft_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "guardpo_data"


class RLDataError(RuntimeError):
    """Raised when source alignment or safety invariants are violated."""


@dataclass(frozen=True)
class SourceItem:
    """An original, non-oversampled SFT record ready for VERL conversion."""

    source: str
    label: str
    reason: str
    messages: list[dict[str, str]]
    content_sha256: str
    source_index: int
    sft_line: int


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RLDataError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise RLDataError(f"{path}:{line_number}: expected a JSON object")
            yield item


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                encoded = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                payload = (encoded + "\n").encode("utf-8")
                handle.write(payload.decode("utf-8"))
                digest.update(payload)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count, digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _normalize_messages(raw_messages: Any, *, source_desc: str) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or len(raw_messages) != 3:
        raise RLDataError(f"{source_desc}: expected system/user/assistant messages")
    normalized: list[dict[str, str]] = []
    for position, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise RLDataError(f"{source_desc}: message {position} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise RLDataError(f"{source_desc}: message {position} has invalid role/content")
        normalized.append({"role": role, "content": content})
    roles = [message["role"] for message in normalized]
    if roles != ["system", "user", "assistant"]:
        raise RLDataError(f"{source_desc}: unexpected roles {roles!r}")
    return normalized


def load_aligned_sft_records(
    data_path: Path,
    manifest_path: Path,
    *,
    originals_only: bool,
) -> list[SourceItem]:
    """Read data and manifest in lockstep and verify all label invariants."""

    data_iter = _iter_jsonl(data_path)
    manifest_iter = _iter_jsonl(manifest_path)
    output: list[SourceItem] = []
    line_number = 0

    while True:
        try:
            data_item = next(data_iter)
            data_done = False
        except StopIteration:
            data_item = None
            data_done = True
        try:
            manifest = next(manifest_iter)
            manifest_done = False
        except StopIteration:
            manifest = None
            manifest_done = True

        if data_done and manifest_done:
            break
        line_number += 1
        if data_done != manifest_done:
            raise RLDataError(
                f"{data_path.name} and {manifest_path.name} have different record counts "
                f"near aligned record {line_number}"
            )
        assert data_item is not None and manifest is not None

        if originals_only and bool(manifest.get("is_oversampled")):
            continue

        source_desc = f"{data_path}:{line_number}"
        messages = _normalize_messages(data_item.get("messages"), source_desc=source_desc)
        parsed = parse_reference_response(messages[-1]["content"])
        if not parsed.valid or parsed.label is None:
            raise RLDataError(
                f"{source_desc}: reference assistant response is not canonical: {parsed.error}"
            )

        manifest_label = manifest.get("label")
        if isinstance(manifest_label, bool):
            manifest_label = int(manifest_label)
        if manifest_label not in INT_TO_LABEL:
            raise RLDataError(f"{source_desc}: unsupported manifest label {manifest_label!r}")
        label = INT_TO_LABEL[int(manifest_label)]
        if label != parsed.label:
            raise RLDataError(
                f"{source_desc}: assistant label {parsed.label!r} != manifest label {label!r}"
            )

        source = manifest.get("source")
        content_sha256 = manifest.get("content_sha256")
        source_index = manifest.get("source_index")
        sft_line = manifest.get("sft_line", line_number)
        if not isinstance(source, str) or not source:
            raise RLDataError(f"{source_desc}: missing source")
        if not isinstance(content_sha256, str) or len(content_sha256) != 64:
            raise RLDataError(f"{source_desc}: invalid content_sha256")
        if not isinstance(source_index, int):
            raise RLDataError(f"{source_desc}: invalid source_index")
        if not isinstance(sft_line, int):
            raise RLDataError(f"{source_desc}: invalid sft_line")

        output.append(
            SourceItem(
                source=source,
                label=label,
                reason=parsed.analysis,
                messages=messages[:2],
                content_sha256=content_sha256,
                source_index=source_index,
                sft_line=sft_line,
            )
        )

    if not output:
        raise RLDataError(f"no usable records loaded from {data_path}")
    return output


def _to_verl_record(
    item: SourceItem,
    *,
    split: str,
    prepared_index: int,
    sampling_draw: int = 0,
    sampling_repeat: bool = False,
) -> dict[str, Any]:
    """Create the standard RLHFDataset record without exposing the answer."""

    prompt = [dict(message) for message in item.messages]
    user_content = prompt[-1]["content"]
    trajectory = extract_untrusted_trajectory(user_content)
    if not trajectory:
        raise RLDataError(
            f"empty trajectory for {item.source}:{item.source_index} ({item.content_sha256})"
        )

    return {
        "data_source": item.source,
        "prompt": prompt,
        "ability": "trajectory_safety_classification",
        "reward_model": {
            "style": "strict_outcome_rule",
            "ground_truth": {
                "label": item.label,
                "label_int": LABEL_TO_INT[item.label],
                "reason": item.reason,
                "source": item.source,
                "content_sha256": item.content_sha256,
            },
        },
        "extra_info": {
            "index": prepared_index,
            "split": split,
            "source": item.source,
            "source_index": item.source_index,
            "sft_line": item.sft_line,
            "label": item.label,
            "label_int": LABEL_TO_INT[item.label],
            "content_sha256": item.content_sha256,
            "sampling_draw": sampling_draw,
            "sampling_repeat": sampling_repeat,
        },
    }


def _stratum_key(item: SourceItem) -> tuple[str, str]:
    return item.source, item.label


def build_balanced_schedule(
    originals: Sequence[SourceItem],
    *,
    samples_per_stratum: int,
    seed: int,
) -> list[tuple[SourceItem, int, bool]]:
    """Build an exactly balanced, round-robin source×label virtual epoch."""

    if samples_per_stratum <= 0:
        raise RLDataError("samples_per_stratum must be positive")

    pools: dict[tuple[str, str], list[SourceItem]] = defaultdict(list)
    for item in originals:
        pools[_stratum_key(item)].append(item)

    sources = sorted({item.source for item in originals})
    expected = [(source, label) for source in sources for label in VALID_LABELS]
    missing = [key for key in expected if not pools.get(key)]
    if missing:
        raise RLDataError(f"cannot balance empty source×label strata: {missing!r}")

    rng = random.Random(seed)
    shuffled_pools: dict[tuple[str, str], list[SourceItem]] = {}
    cursors: dict[tuple[str, str], int] = {}
    draws: dict[tuple[str, str], int] = defaultdict(int)
    for key in expected:
        shuffled = list(pools[key])
        rng.shuffle(shuffled)
        shuffled_pools[key] = shuffled
        cursors[key] = 0

    schedule: list[tuple[SourceItem, int, bool]] = []
    for _round_index in range(samples_per_stratum):
        # Alternating label order avoids making every ten-record cycle start
        # with the same label while retaining exact stratum balance.
        round_keys = list(expected)
        if _round_index % 2:
            round_keys.reverse()
        for key in round_keys:
            pool = shuffled_pools[key]
            cursor = cursors[key]
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            item = pool[cursor]
            cursors[key] = cursor + 1
            draw_number = draws[key]
            is_repeat = draw_number >= len(pools[key])
            draws[key] += 1
            schedule.append((item, draw_number, is_repeat))

    return schedule


def _count_items(items: Iterable[SourceItem]) -> dict[str, Any]:
    source_label = Counter((item.source, item.label) for item in items)
    labels = Counter(item.label for item in items)
    sources = Counter(item.source for item in items)
    return {
        "total": sum(labels.values()),
        "labels": dict(sorted(labels.items())),
        "sources": dict(sorted(sources.items())),
        "source_label": {
            f"{source}/{label}": count
            for (source, label), count in sorted(source_label.items())
        },
    }


def prepare_datasets(
    *,
    input_dir: Path,
    output_dir: Path,
    samples_per_stratum: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    train_path = input_dir / "guard_train.jsonl"
    train_manifest_path = input_dir / "guard_train_manifest.jsonl"
    validation_path = input_dir / "guard_validation.jsonl"
    validation_manifest_path = input_dir / "guard_validation_manifest.jsonl"
    for required in (
        train_path,
        train_manifest_path,
        validation_path,
        validation_manifest_path,
    ):
        if not required.is_file():
            raise RLDataError(f"required input does not exist: {required}")

    output_paths = {
        "train_unique": output_dir / "train_unique.json",
        "train_balanced": output_dir / "train_balanced.json",
        "validation": output_dir / "validation.json",
        "summary": output_dir / "preparation_summary.json",
    }
    legacy_output_paths = [
        output_dir / "train_unique.jsonl",
        output_dir / "train_balanced.jsonl",
        output_dir / "validation.jsonl",
    ]
    existing = [
        path
        for path in [*output_paths.values(), *legacy_output_paths]
        if path.exists()
    ]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise RLDataError(f"output already exists; pass --overwrite to replace: {rendered}")
    if overwrite:
        # Remove only the three obsolete filenames produced by the earlier
        # RLGuard schema. No recursive deletion or directory-wide globbing.
        for legacy_path in legacy_output_paths:
            legacy_path.unlink(missing_ok=True)

    originals = load_aligned_sft_records(
        train_path,
        train_manifest_path,
        originals_only=True,
    )
    validation = load_aligned_sft_records(
        validation_path,
        validation_manifest_path,
        originals_only=False,
    )

    original_hashes = [item.content_sha256 for item in originals]
    validation_hashes = [item.content_sha256 for item in validation]
    if len(set(original_hashes)) != len(original_hashes):
        raise RLDataError("duplicate content_sha256 detected among original RL training items")
    overlap = sorted(set(original_hashes).intersection(validation_hashes))
    if overlap:
        raise RLDataError(
            f"train/validation leakage detected for {len(overlap)} content hashes; "
            f"first={overlap[0]}"
        )
    system_prompt_hashes = {
        hashlib.sha256(item.messages[0]["content"].encode("utf-8")).hexdigest()
        for item in [*originals, *validation]
    }
    if len(system_prompt_hashes) != 1:
        raise RLDataError(
            "training and validation do not share exactly one system prompt; "
            f"found {len(system_prompt_hashes)} distinct prompts"
        )

    schedule = build_balanced_schedule(
        originals,
        samples_per_stratum=samples_per_stratum,
        seed=seed,
    )

    unique_count, unique_sha = _atomic_write_jsonl(
        output_paths["train_unique"],
        (
            _to_verl_record(item, split="train_unique", prepared_index=index)
            for index, item in enumerate(originals)
        ),
    )
    balanced_count, balanced_sha = _atomic_write_jsonl(
        output_paths["train_balanced"],
        (
            _to_verl_record(
                item,
                split="train_balanced",
                prepared_index=index,
                sampling_draw=draw,
                sampling_repeat=is_repeat,
            )
            for index, (item, draw, is_repeat) in enumerate(schedule)
        ),
    )
    validation_count, validation_sha = _atomic_write_jsonl(
        output_paths["validation"],
        (
            _to_verl_record(item, split="validation", prepared_index=index)
            for index, item in enumerate(validation)
        ),
    )

    scheduled_items = [item for item, _draw, _repeat in schedule]
    repeat_counts = Counter(
        f"{item.source}/{item.label}"
        for item, _draw, is_repeat in schedule
        if is_repeat
    )
    summary = {
        "schema_version": 2,
        "seed": seed,
        "samples_per_stratum": samples_per_stratum,
        "inputs": {
            "directory": str(input_dir.resolve()),
            "training_data": str(train_path.resolve()),
            "training_manifest": str(train_manifest_path.resolve()),
            "validation_data": str(validation_path.resolve()),
            "validation_manifest": str(validation_manifest_path.resolve()),
        },
        "original_training": _count_items(originals),
        "balanced_training": _count_items(scheduled_items),
        "validation": _count_items(validation),
        "sampling_repeats": dict(sorted(repeat_counts.items())),
        "train_validation_hash_overlap": 0,
        "system_prompt_sha256": next(iter(system_prompt_hashes)),
        "outputs": {
            "train_unique": {
                "path": str(output_paths["train_unique"].resolve()),
                "records": unique_count,
                "sha256": unique_sha,
            },
            "train_balanced": {
                "path": str(output_paths["train_balanced"].resolve()),
                "records": balanced_count,
                "sha256": balanced_sha,
            },
            "validation": {
                "path": str(output_paths["validation"].resolve()),
                "records": validation_count,
                "sha256": validation_sha,
            },
        },
    }
    _atomic_write_json(output_paths["summary"], summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-stratum", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_datasets(
            input_dir=args.input_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            samples_per_stratum=args.samples_per_stratum,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except RLDataError as exc:
        raise SystemExit(f"RL data preparation failed: {exc}") from exc

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
