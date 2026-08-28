#!/usr/bin/env python3
"""Evaluate an RLGuard checkpoint on all five held-out test groups.

The script reads the exact prepared validation split used during SFT:

* AgentHazard test
* ASSE test
* ATBench test
* RJudge test
* deterministic 5% Vera holdout

It supports one GPU with ``python`` and independent multi-GPU data parallelism
with ``torchrun``. Per-rank JSONL prediction shards are append-only so an
interrupted evaluation can resume without regenerating completed examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from guard_output import extract_guard_label


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "qwen3_guard_8b_full_sft" / "checkpoint-2704"
)
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "data" / "guard_sft"
DEFAULT_TOKENIZER_FALLBACK = PROJECT_ROOT / "model_cache" / "qwen3_guard_8b"
EXPECTED_SOURCES = ("AgentHazard", "ASSE", "ATBench", "RJudge", "Vera")
WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
METADATA_FILENAME = "evaluation_metadata.json"
METRICS_FILENAME = "metrics.json"
TABLE_FILENAME = "metrics.md"
CONSOLIDATED_PREDICTIONS_FILENAME = "predictions.jsonl"
SCHEMA_VERSION = 2

# Version 1 metadata already contains these prediction-affecting fields. They
# must match before old per-rank prediction shards can be resumed.
REQUIRED_RESUME_METADATA_KEYS = (
    "checkpoint",
    "checkpoint_config_sha256",
    "validation_dataset",
    "validation_dataset_sha256",
    "validation_manifest",
    "validation_manifest_sha256",
    "selected_item_count",
    "limit_per_benchmark",
    "cutoff_len",
    "max_new_tokens",
    "batch_size",
    "dtype",
    "attn_implementation",
    "label_parser",
)

# These prediction fingerprints were added after version 1. Compare them when
# present in an older metadata file, while allowing a trustworthy v1 file that
# could not have recorded them yet.
OPTIONAL_RESUME_METADATA_KEYS = (
    "checkpoint_weight_file",
    "checkpoint_weight_size",
    "checkpoint_chat_template_sha256",
    "requested_tokenizer",
)


class EvaluationError(RuntimeError):
    """Raised when an evaluation would be incomplete or incomparable."""


@dataclass(frozen=True)
class EvalItem:
    dataset_index: int
    source: str
    source_split: str
    source_index: int
    content_sha256: str
    gold_label: str
    input_messages: list[dict[str, str]]


@dataclass(frozen=True)
class RuntimeContext:
    rank: int
    local_rank: int
    world_size: int
    device: Any
    distributed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate safe/unsafe verdicts for the five RLGuard validation "
            "groups and print per-benchmark binary metrics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Consolidated Hugging Face checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help=(
            "Directory containing guard_validation.jsonl and its manifest "
            f"(default: {DEFAULT_VALIDATION_DIR})"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help=(
            "Optional tokenizer directory. By default the checkpoint is used, "
            "falling back to model_cache/qwen3_guard_8b when necessary."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Prediction/metric directory. Defaults to "
            "eval_outputs/<resolved-checkpoint-name>."
        ),
    )
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=16_000,
        help="Keep at most this many prompt tokens from the prefix (default: 16000).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum generated CoT plus label tokens (default: 384).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU generation batch size (default: 1).",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model inference dtype (default: bf16).",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
        help="Transformers attention backend (default: flash_attention_2).",
    )
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        help="Smoke testing only: evaluate at most N rows from each source.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Each rank prints progress every N newly completed rows (default: 10).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing_dir(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise EvaluationError(f"{description} directory not found: {resolved}")
    return resolved


def validate_checkpoint(path: Path) -> Path:
    checkpoint = resolve_existing_dir(path, description="checkpoint")
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise EvaluationError(f"checkpoint config.json not found: {config_path}")
    if not any((checkpoint / name).is_file() for name in WEIGHT_FILENAMES):
        raise EvaluationError(
            f"{checkpoint} has no consolidated Hugging Face model weights. "
            "Files under global_step*/ are DeepSpeed training state and cannot "
            "be passed directly to AutoModelForCausalLM."
        )
    return checkpoint


def checkpoint_weight_file(checkpoint: Path) -> Path:
    for name in WEIGHT_FILENAMES:
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise EvaluationError(f"no consolidated model weights found in {checkpoint}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"required JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise EvaluationError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def validate_message(value: Any, *, location: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise EvaluationError(f"{location}: invalid role/content message")
    role = value["role"]
    content = value["content"]
    if role not in {"system", "user", "assistant"}:
        raise EvaluationError(f"{location}: unsupported role {role!r}")
    if not isinstance(content, str) or not content:
        raise EvaluationError(f"{location}: message content must be non-empty text")
    return {"role": role, "content": content}


def load_validation_items(
    validation_dir: Path,
    *,
    limit_per_benchmark: int | None,
) -> tuple[list[EvalItem], Path, Path]:
    if limit_per_benchmark is not None and limit_per_benchmark <= 0:
        raise EvaluationError("--limit-per-benchmark must be positive")

    dataset_path = validation_dir / "guard_validation.jsonl"
    manifest_path = validation_dir / "guard_validation_manifest.jsonl"
    dataset_rows = read_jsonl(dataset_path)
    manifest_rows = read_jsonl(manifest_path)
    if len(dataset_rows) != len(manifest_rows):
        raise EvaluationError(
            "validation dataset/manifest line mismatch: "
            f"{len(dataset_rows)} != {len(manifest_rows)}"
        )

    seen_per_source: Counter[str] = Counter()
    items: list[EvalItem] = []
    for dataset_index, (dataset_row, manifest) in enumerate(
        zip(dataset_rows, manifest_rows)
    ):
        location = f"validation row {dataset_index + 1}"
        source = manifest.get("source")
        if source not in EXPECTED_SOURCES:
            raise EvaluationError(f"{location}: unexpected source {source!r}")
        if (
            limit_per_benchmark is not None
            and seen_per_source[source] >= limit_per_benchmark
        ):
            continue

        messages_value = dataset_row.get("messages")
        if not isinstance(messages_value, list) or len(messages_value) != 3:
            raise EvaluationError(
                f"{location}: expected system/user/assistant messages"
            )
        messages = [
            validate_message(message, location=f"{location}.messages[{index}]")
            for index, message in enumerate(messages_value)
        ]
        if [message["role"] for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise EvaluationError(
                f"{location}: message roles must be system/user/assistant"
            )

        label = manifest.get("label")
        if type(label) is not int or label not in (0, 1):
            raise EvaluationError(f"{location}: manifest label must be 0 or 1")
        gold_label = "unsafe" if label == 1 else "safe"
        target_label = extract_guard_label(
            messages[-1]["content"],
            allow_fallback=False,
        )
        if target_label != gold_label:
            raise EvaluationError(
                f"{location}: assistant target and manifest label disagree"
            )

        source_split = manifest.get("source_split")
        source_index = manifest.get("source_index")
        content_sha256 = manifest.get("content_sha256")
        if not isinstance(source_split, str):
            raise EvaluationError(f"{location}: invalid source_split")
        if type(source_index) is not int or source_index < 0:
            raise EvaluationError(f"{location}: invalid source_index")
        if (
            not isinstance(content_sha256, str)
            or len(content_sha256) != 64
        ):
            raise EvaluationError(f"{location}: invalid content SHA-256")

        items.append(
            EvalItem(
                dataset_index=dataset_index,
                source=source,
                source_split=source_split,
                source_index=source_index,
                content_sha256=content_sha256,
                gold_label=gold_label,
                input_messages=messages[:-1],
            )
        )
        seen_per_source[source] += 1

    missing_sources = [
        source for source in EXPECTED_SOURCES if seen_per_source[source] == 0
    ]
    if missing_sources:
        raise EvaluationError(
            f"prepared validation split is missing sources: {missing_sources}"
        )
    return items, dataset_path, manifest_path


def setup_runtime() -> tuple[Any, Any, RuntimeContext]:
    try:
        import torch
        import torch.distributed as distributed
    except ImportError as exc:
        raise EvaluationError(
            f"PyTorch is required in the evaluation environment: {exc}"
        ) from exc

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_run = world_size > 1

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise EvaluationError(
                f"LOCAL_RANK={local_rank} but only "
                f"{torch.cuda.device_count()} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        if distributed_run:
            raise EvaluationError("multi-process evaluation requires CUDA")
        device = torch.device("cpu")
        backend = "gloo"

    if distributed_run:
        if device.type == "cuda":
            try:
                distributed.init_process_group(backend=backend, device_id=device)
            except TypeError:
                # Compatibility with older PyTorch releases that predate the
                # device_id argument.
                distributed.init_process_group(backend=backend)
        else:
            distributed.init_process_group(backend=backend)

    return torch, distributed, RuntimeContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        distributed=distributed_run,
    )


def distributed_barrier(distributed: Any, runtime: RuntimeContext) -> None:
    if runtime.distributed:
        if runtime.device.type == "cuda":
            distributed.barrier(device_ids=[runtime.local_rank])
        else:
            distributed.barrier()


def resolve_output_dir(
    requested: Path | None,
    *,
    checkpoint: Path,
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    return (PROJECT_ROOT / "eval_outputs" / checkpoint.name).resolve()


def evaluation_metadata(
    *,
    checkpoint: Path,
    dataset_path: Path,
    manifest_path: Path,
    item_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    weight_path = checkpoint_weight_file(checkpoint)
    template_path = checkpoint / "chat_template.jinja"
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weight_file": weight_path.name,
        "checkpoint_weight_size": weight_path.stat().st_size,
        "checkpoint_chat_template_sha256": (
            sha256_file(template_path) if template_path.is_file() else None
        ),
        "requested_tokenizer": (
            str(args.tokenizer.expanduser().resolve())
            if args.tokenizer is not None
            else None
        ),
        "validation_dataset": str(dataset_path.resolve()),
        "validation_dataset_sha256": sha256_file(dataset_path),
        "validation_manifest": str(manifest_path.resolve()),
        "validation_manifest_sha256": sha256_file(manifest_path),
        "selected_item_count": item_count,
        "limit_per_benchmark": args.limit_per_benchmark,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "label_parser": "canonical_then_qwen_guard_then_final_bare",
        "reported_classes": ["safe", "unsafe"],
        "binary_confusion_positive_class": "unsafe",
    }


def resume_metadata_differences(
    existing: Any,
    current: dict[str, Any],
) -> list[str]:
    """Return prediction-affecting metadata conflicts.

    Reporting-only additions (for example, adding per-class metrics) are
    intentionally excluded so they cannot invalidate compatible prediction
    shards.
    """
    if not isinstance(existing, dict):
        return ["metadata root is not a JSON object"]

    differences: list[str] = []
    existing_version = existing.get("schema_version")
    if type(existing_version) is not int or existing_version < 1:
        differences.append(
            f"schema_version: existing={existing_version!r}, expected an integer >= 1"
        )
    elif existing_version > SCHEMA_VERSION:
        differences.append(
            f"schema_version: existing={existing_version!r}, "
            f"supported<={SCHEMA_VERSION}"
        )

    for key in REQUIRED_RESUME_METADATA_KEYS:
        if key not in existing:
            differences.append(f"{key}: missing from existing metadata")
        elif existing[key] != current[key]:
            differences.append(
                f"{key}: existing={existing[key]!r}, current={current[key]!r}"
            )

    for key in OPTIONAL_RESUME_METADATA_KEYS:
        if key in existing and existing[key] != current[key]:
            differences.append(
                f"{key}: existing={existing[key]!r}, current={current[key]!r}"
            )
    return differences


def write_evaluation_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def prepare_output_dir(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILENAME
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"cannot read {metadata_path}: {exc}") from exc
        differences = resume_metadata_differences(existing, metadata)
        if differences:
            detail = "\n".join(f"  - {difference}" for difference in differences)
            raise EvaluationError(
                f"{output_dir} belongs to a different evaluation configuration. "
                "Choose a new --output-dir rather than mixing predictions.\n"
                f"Prediction-affecting differences:\n{detail}"
            )
        # Upgrade compatible v1 metadata and refresh reporting-only fields.
        if existing != metadata:
            write_evaluation_metadata(metadata_path, metadata)
        return

    unexpected = list(output_dir.iterdir())
    if unexpected:
        raise EvaluationError(
            f"output directory is non-empty but has no metadata: {output_dir}"
        )
    write_evaluation_metadata(metadata_path, metadata)


def prediction_shards(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("predictions.rank*.jsonl"))


def load_completed_predictions(
    output_dir: Path,
    *,
    items_by_index: dict[int, EvalItem],
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for path in prediction_shards(output_dir):
        with path.open(encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        f"{path}:{line_number}: invalid JSON; remove only the "
                        f"truncated final line before resuming: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EvaluationError(
                        f"{path}:{line_number}: prediction row must be an object"
                    )
                index = row.get("dataset_index")
                if type(index) is not int or index not in items_by_index:
                    raise EvaluationError(
                        f"{path}:{line_number}: unexpected dataset_index {index!r}"
                    )
                item = items_by_index[index]
                if (
                    row.get("source") != item.source
                    or row.get("gold_label") != item.gold_label
                    or row.get("content_sha256") != item.content_sha256
                ):
                    raise EvaluationError(
                        f"{path}:{line_number}: prediction provenance mismatch"
                    )
                if index in completed and completed[index] != row:
                    raise EvaluationError(
                        f"conflicting resumed predictions for dataset index {index}"
                    )
                completed[index] = row
    return completed


def load_tokenizer(
    *,
    checkpoint: Path,
    requested_tokenizer: Path | None,
) -> tuple[Any, Path]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise EvaluationError(f"Transformers is required: {exc}") from exc

    candidates: list[Path] = []
    if requested_tokenizer is not None:
        candidates.append(requested_tokenizer.expanduser().resolve())
    else:
        candidates.append(checkpoint)
        if DEFAULT_TOKENIZER_FALLBACK.is_dir():
            candidates.append(DEFAULT_TOKENIZER_FALLBACK.resolve())

    errors: list[str] = []
    tokenizer = None
    tokenizer_source = None
    for candidate in candidates:
        if not candidate.is_dir():
            errors.append(f"{candidate}: directory not found")
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(candidate),
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        tokenizer_source = candidate
        break

    if tokenizer is None or tokenizer_source is None:
        raise EvaluationError(
            "could not load tokenizer; pass --tokenizer explicitly. Attempts: "
            + " | ".join(errors)
        )

    checkpoint_template = checkpoint / "chat_template.jinja"
    if checkpoint_template.is_file():
        tokenizer.chat_template = checkpoint_template.read_text(encoding="utf-8")
    if not getattr(tokenizer, "chat_template", None):
        raise EvaluationError(
            "tokenizer has no chat template and checkpoint has no "
            "chat_template.jinja"
        )
    if tokenizer.eos_token_id is None:
        raise EvaluationError("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    return tokenizer, tokenizer_source


def dtype_value(torch: Any, name: str) -> Any:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def load_model(
    *,
    torch: Any,
    checkpoint: Path,
    runtime: RuntimeContext,
    dtype_name: str,
    attn_implementation: str,
) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise EvaluationError(f"Transformers is required: {exc}") from exc

    dtype = dtype_value(torch, dtype_name)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if runtime.device.type == "cuda":
        load_kwargs["device_map"] = {"": str(runtime.device)}

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        **load_kwargs,
    )
    if runtime.device.type == "cpu":
        model.to(runtime.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = True
    if hasattr(model.generation_config, "use_cache"):
        model.generation_config.use_cache = True
    return model


def token_container_summary(value: Any) -> str:
    """Return a bounded diagnostic for an unexpected tokenizer result."""
    details = [f"type={type(value).__name__}"]
    if isinstance(value, Mapping):
        details.append(f"keys={list(value.keys())[:10]!r}")
    shape = getattr(value, "shape", None)
    if shape is not None:
        details.append(f"shape={shape!r}")
    if isinstance(value, (list, tuple)):
        details.append(f"length={len(value)}")
        if value:
            details.append(f"first_type={type(value[0]).__name__}")
    return ", ".join(details)


def normalize_chat_template_input_ids(value: Any, *, row_number: int) -> list[int]:
    """Normalize all documented apply_chat_template token containers."""
    original_summary = token_container_summary(value)
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise EvaluationError(
                f"chat template result has no input_ids for row {row_number} "
                f"({original_summary})"
            )
        value = value["input_ids"]

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)

    # Some Transformers/tokenizer combinations retain a singleton batch
    # dimension even for one conversation.
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])

    if not isinstance(value, list) or not value:
        raise EvaluationError(
            f"chat template returned empty or invalid input_ids for row "
            f"{row_number} ({original_summary}; "
            f"input_ids={token_container_summary(value)})"
        )

    normalized: list[int] = []
    for position, token_id in enumerate(value):
        if isinstance(token_id, bool):
            raise EvaluationError(
                f"chat template returned a boolean token ID at position "
                f"{position} for row {row_number} ({original_summary})"
            )
        try:
            normalized.append(operator.index(token_id))
        except TypeError as exc:
            raise EvaluationError(
                f"chat template returned a non-integer token ID at position "
                f"{position} for row {row_number}: "
                f"{type(token_id).__name__} ({original_summary}; "
                f"input_ids={token_container_summary(value)})"
            ) from exc
    return normalized


def encode_prompt(
    tokenizer: Any,
    item: EvalItem,
    *,
    cutoff_len: int,
) -> tuple[list[int], bool, int]:
    try:
        encoded_chat = tokenizer.apply_chat_template(
            item.input_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )
    except Exception as exc:
        raise EvaluationError(
            f"failed to apply chat template for validation row "
            f"{item.dataset_index + 1}: {exc}"
        ) from exc
    input_ids = normalize_chat_template_input_ids(
        encoded_chat,
        row_number=item.dataset_index + 1,
    )
    original_length = len(input_ids)
    truncated = original_length > cutoff_len
    if truncated:
        input_ids = input_ids[:cutoff_len]
    return input_ids, truncated, original_length


def eos_token_ids(tokenizer: Any) -> list[int]:
    values = [tokenizer.eos_token_id]
    extra = getattr(tokenizer, "additional_special_tokens_ids", None)
    if isinstance(extra, list):
        values.extend(token_id for token_id in extra if type(token_id) is int)
    return list(dict.fromkeys(token_id for token_id in values if token_id >= 0))


def chunks(values: Sequence[EvalItem], size: int) -> Iterable[list[EvalItem]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def strip_trailing_pad(token_ids: list[int], pad_token_id: int) -> list[int]:
    while token_ids and token_ids[-1] == pad_token_id:
        token_ids.pop()
    return token_ids


def generate_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    runtime: RuntimeContext,
    items: list[EvalItem],
    cutoff_len: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    encoded: list[tuple[list[int], bool, int]] = [
        encode_prompt(tokenizer, item, cutoff_len=cutoff_len) for item in items
    ]
    features = [
        {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }
        for token_ids, _, _ in encoded
    ]
    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )
    batch = {
        key: value.to(runtime.device, non_blocking=True)
        for key, value in batch.items()
    }
    padded_prompt_length = batch["input_ids"].shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids(tokenizer),
        )
    elapsed = time.perf_counter() - started
    generated_suffixes = generated[:, padded_prompt_length:]

    rows: list[dict[str, Any]] = []
    for item, token_info, suffix in zip(items, encoded, generated_suffixes):
        prompt_ids, truncated, original_prompt_tokens = token_info
        output_ids = strip_trailing_pad(
            suffix.detach().cpu().tolist(),
            tokenizer.pad_token_id,
        )
        prediction_text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()
        predicted_label = extract_guard_label(
            prediction_text,
            allow_fallback=True,
        )
        rows.append(
            {
                "dataset_index": item.dataset_index,
                "source": item.source,
                "source_split": item.source_split,
                "source_index": item.source_index,
                "content_sha256": item.content_sha256,
                "gold_label": item.gold_label,
                "predicted_label": predicted_label,
                "parsed": predicted_label is not None,
                "correct": predicted_label == item.gold_label,
                "prompt_tokens": len(prompt_ids),
                "original_prompt_tokens": original_prompt_tokens,
                "prompt_truncated": truncated,
                "generated_tokens": len(output_ids),
                "batch_generation_seconds": elapsed,
                "prediction": prediction_text,
            }
        )
    return rows


def write_prediction(file: Any, row: dict[str, Any]) -> None:
    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    file.flush()
    os.fsync(file.fileno())


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise EvaluationError("cannot score an empty prediction group")
    confusion: Counter[tuple[str, str | None]] = Counter()
    for row in rows:
        confusion[(row["gold_label"], row["predicted_label"])] += 1

    total = len(rows)
    parsed = sum(row["predicted_label"] is not None for row in rows)
    correct = sum(row["predicted_label"] == row["gold_label"] for row in rows)
    metrics: dict[str, Any] = {
        "count": total,
        "safe_count": sum(row["gold_label"] == "safe" for row in rows),
        "unsafe_count": sum(row["gold_label"] == "unsafe" for row in rows),
        "accuracy": correct / total,
        "parse_rate": parsed / total,
        "unparseable": total - parsed,
        "truncated_prompts": sum(bool(row["prompt_truncated"]) for row in rows),
    }

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in ("safe", "unsafe"):
        true_positive = confusion[(label, label)]
        predicted_count = sum(
            count
            for (gold, predicted), count in confusion.items()
            if predicted == label
        )
        gold_count = sum(
            count
            for (gold, _), count in confusion.items()
            if gold == label
        )
        precision = safe_divide(true_positive, predicted_count)
        recall = safe_divide(true_positive, gold_count)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_f1"] = f1
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    metrics["macro_precision"] = sum(precision_values) / len(precision_values)
    metrics["macro_recall"] = sum(recall_values) / len(recall_values)
    metrics["macro_f1"] = sum(f1_values) / len(f1_values)
    metrics["confusion"] = {
        f"gold_{gold}__pred_{predicted or 'unparseable'}": count
        for (gold, predicted), count in sorted(
            confusion.items(),
            key=lambda pair: (pair[0][0], pair[0][1] or ""),
        )
    }
    return metrics


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def build_markdown_table(metrics_by_source: dict[str, dict[str, Any]]) -> str:
    headers = (
        "Dataset",
        "N",
        "Safe",
        "Unsafe",
        "ACC",
        "P-safe",
        "R-safe",
        "F1-safe",
        "P-unsafe",
        "R-unsafe",
        "F1-unsafe",
        "Macro-P",
        "Macro-R",
        "Macro-F1",
        "Parse rate",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for source in (*EXPECTED_SOURCES, "Overall"):
        metrics = metrics_by_source[source]
        values = (
            source,
            str(metrics["count"]),
            str(metrics["safe_count"]),
            str(metrics["unsafe_count"]),
            format_metric(metrics["accuracy"]),
            format_metric(metrics["safe_precision"]),
            format_metric(metrics["safe_recall"]),
            format_metric(metrics["safe_f1"]),
            format_metric(metrics["unsafe_precision"]),
            format_metric(metrics["unsafe_recall"]),
            format_metric(metrics["unsafe_f1"]),
            format_metric(metrics["macro_precision"]),
            format_metric(metrics["macro_recall"]),
            format_metric(metrics["macro_f1"]),
            format_metric(metrics["parse_rate"]),
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def summarize_and_write(
    *,
    output_dir: Path,
    items: list[EvalItem],
) -> str:
    items_by_index = {item.dataset_index: item for item in items}
    completed = load_completed_predictions(
        output_dir,
        items_by_index=items_by_index,
    )
    missing = sorted(set(items_by_index) - set(completed))
    if missing:
        raise EvaluationError(
            f"evaluation finished with {len(missing)} missing predictions"
        )

    rows = [completed[index] for index in sorted(completed)]
    consolidated_path = output_dir / CONSOLIDATED_PREDICTIONS_FILENAME
    with consolidated_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    metrics_by_source: dict[str, dict[str, Any]] = {}
    for source in EXPECTED_SOURCES:
        source_rows = [row for row in rows if row["source"] == source]
        metrics_by_source[source] = compute_metrics(source_rows)
    metrics_by_source["Overall"] = compute_metrics(rows)

    metrics_payload = {
        "reported_classes": ["safe", "unsafe"],
        "binary_confusion_positive_class": "unsafe",
        "unparseable_policy": "count_as_incorrect",
        "sources": metrics_by_source,
    }
    (output_dir / METRICS_FILENAME).write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    table = build_markdown_table(metrics_by_source)
    (output_dir / TABLE_FILENAME).write_text(table, encoding="utf-8")
    return table


def main() -> int:
    args = parse_args()
    if args.cutoff_len <= 0:
        raise EvaluationError("--cutoff-len must be positive")
    if args.max_new_tokens <= 0:
        raise EvaluationError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise EvaluationError("--batch-size must be positive")
    if args.print_every <= 0:
        raise EvaluationError("--print-every must be positive")

    checkpoint = validate_checkpoint(args.checkpoint)
    validation_dir = resolve_existing_dir(
        args.validation_dir,
        description="validation",
    )
    items, dataset_path, manifest_path = load_validation_items(
        validation_dir,
        limit_per_benchmark=args.limit_per_benchmark,
    )
    items_by_index = {item.dataset_index: item for item in items}
    if len(items_by_index) != len(items):
        raise EvaluationError("validation dataset indices are not unique")

    output_dir = resolve_output_dir(args.output_dir, checkpoint=checkpoint)
    metadata = evaluation_metadata(
        checkpoint=checkpoint,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        item_count=len(items),
        args=args,
    )

    torch = distributed = runtime = None
    try:
        torch, distributed, runtime = setup_runtime()
        if runtime.rank == 0:
            prepare_output_dir(output_dir, metadata)
            counts = Counter(item.source for item in items)
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "weights": checkpoint_weight_file(checkpoint).name,
                        "validation_counts": dict(counts),
                        "world_size": runtime.world_size,
                        "output_dir": str(output_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        distributed_barrier(distributed, runtime)

        completed = load_completed_predictions(
            output_dir,
            items_by_index=items_by_index,
        )
        pending = [
            item
            for item in items
            if item.dataset_index not in completed
            and item.dataset_index % runtime.world_size == runtime.rank
        ]
        print(
            f"[rank {runtime.rank}] completed={len(completed)} "
            f"local_pending={len(pending)} device={runtime.device}",
            flush=True,
        )

        if pending:
            tokenizer, tokenizer_source = load_tokenizer(
                checkpoint=checkpoint,
                requested_tokenizer=args.tokenizer,
            )
            model = load_model(
                torch=torch,
                checkpoint=checkpoint,
                runtime=runtime,
                dtype_name=args.dtype,
                attn_implementation=args.attn_implementation,
            )
            maximum_context = getattr(model.config, "max_position_embeddings", None)
            if (
                type(maximum_context) is int
                and args.cutoff_len + args.max_new_tokens > maximum_context
            ):
                raise EvaluationError(
                    "cutoff_len + max_new_tokens exceeds model context: "
                    f"{args.cutoff_len} + {args.max_new_tokens} > {maximum_context}"
                )
            print(
                f"[rank {runtime.rank}] tokenizer={tokenizer_source} "
                f"model_loaded=true",
                flush=True,
            )

            shard_path = output_dir / f"predictions.rank{runtime.rank:02d}.jsonl"
            newly_completed = 0
            with shard_path.open("a", encoding="utf-8") as shard:
                for batch_items in chunks(pending, args.batch_size):
                    rows = generate_batch(
                        torch=torch,
                        model=model,
                        tokenizer=tokenizer,
                        runtime=runtime,
                        items=batch_items,
                        cutoff_len=args.cutoff_len,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for row in rows:
                        write_prediction(shard, row)
                        newly_completed += 1
                    if (
                        newly_completed == len(rows)
                        or newly_completed % args.print_every == 0
                        or newly_completed == len(pending)
                    ):
                        print(
                            f"[rank {runtime.rank}] generated "
                            f"{newly_completed}/{len(pending)}",
                            flush=True,
                        )

            del model
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        distributed_barrier(distributed, runtime)
        if runtime.rank == 0:
            table = summarize_and_write(output_dir=output_dir, items=items)
            print(
                "\nBoth safe and unsafe class metrics are reported; "
                "unparseable outputs are incorrect.\n"
            )
            print(table)
            print(f"Predictions: {output_dir / CONSOLIDATED_PREDICTIONS_FILENAME}")
            print(f"Metrics JSON: {output_dir / METRICS_FILENAME}")
            print(f"Markdown table: {output_dir / TABLE_FILENAME}")
        distributed_barrier(distributed, runtime)
        return 0
    finally:
        if (
            distributed is not None
            and runtime is not None
            and runtime.distributed
            and distributed.is_initialized()
        ):
            distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
