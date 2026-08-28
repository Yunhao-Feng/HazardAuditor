#!/usr/bin/env python3
"""Launch LLaMA-Factory full SFT with label-only generative validation.

Run this file under torchrun. It uses LLaMA-Factory's data/model/training stack,
but installs two narrowly scoped extensions:

1. validation metrics parse only the final safe/unsafe label from generated
   CoT outputs; malformed outputs count as wrong, and GPU tensors are moved to
   host memory before NumPy decoding;
2. the final label tail receives modest extra token loss weight so long
   rationales do not drown out the verdict objective.

Validation runs once before the first optimizer update and after every epoch.
Generation temporarily enables the KV cache and preserves the configured
generation limits, while training keeps the cache disabled for gradient
checkpointing. The Hugging Face checkpoint rotation then selects the best
trained epoch by generated label accuracy and keeps at most best/latest
checkpoint weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from guard_output import extract_guard_label


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "qwen3_guard_full_sft.yaml"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
ROOT_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
)


class TrainingConfigurationError(RuntimeError):
    """Raised when the config would violate the requested training contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 8-GPU full SFT with generated-label validation."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"LLaMA-Factory YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or an explicit checkpoint directory",
    )
    parser.add_argument(
        "--no-final-cleanup",
        action="store_true",
        help="Do not remove the duplicate root model weights after training.",
    )
    return parser.parse_args()


def global_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def enable_generation_cache(model: Any) -> list[tuple[Any, Any]]:
    """Enable every relevant model cache flag and return values to restore."""
    model_config = getattr(model, "config", None)
    possible_configs = (
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model, "generation_config", None),
    )
    previous_values: list[tuple[Any, Any]] = []
    seen_config_ids: set[int] = set()
    for cache_config in possible_configs:
        if (
            cache_config is None
            or id(cache_config) in seen_config_ids
            or not hasattr(cache_config, "use_cache")
        ):
            continue
        seen_config_ids.add(id(cache_config))
        previous_values.append((cache_config, cache_config.use_cache))
        cache_config.use_cache = True
    return previous_values


def restore_generation_cache(previous_values: list[tuple[Any, Any]]) -> None:
    """Restore model cache flags after evaluation, including partial failures."""
    for cache_config, previous_use_cache in reversed(previous_values):
        cache_config.use_cache = previous_use_cache


def build_evaluation_kwargs(
    stored_generation_kwargs: Any,
    current_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Preserve LLaMA-Factory generation bounds across epoch evaluations."""
    generation_kwargs = dict(stored_generation_kwargs or {})
    generation_kwargs.update(current_kwargs)
    generation_kwargs["use_cache"] = True
    return generation_kwargs


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingConfigurationError(f"config not found: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrainingConfigurationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{path}: top level must be a mapping")
    return value


def resolve_from_project(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def checkpoint_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        match = CHECKPOINT_RE.match(child.name)
        if match and child.is_dir():
            checkpoints.append((int(match.group(1)), child))
    return [path for _, path in sorted(checkpoints)]


def configure_resume(
    config: dict[str, Any], resume: str
) -> tuple[dict[str, Any], Path]:
    output_dir = resolve_from_project(config.get("output_dir"))
    checkpoints = checkpoint_dirs(output_dir)
    if resume == "auto":
        config["resume_from_checkpoint"] = (
            str(checkpoints[-1]) if checkpoints else None
        )
    elif resume.lower() == "none":
        if checkpoints:
            raise TrainingConfigurationError(
                f"{output_dir} already contains checkpoints; use --resume auto "
                "or choose a fresh output_dir"
            )
        config["resume_from_checkpoint"] = None
    else:
        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_dir() or not CHECKPOINT_RE.match(checkpoint.name):
            raise TrainingConfigurationError(
                f"invalid explicit checkpoint directory: {checkpoint}"
            )
        config["resume_from_checkpoint"] = str(checkpoint)
    return config, output_dir


def validate_training_contract(config: dict[str, Any]) -> None:
    required = {
        "model_name_or_path",
        "dataset",
        "eval_dataset",
        "dataset_dir",
        "output_dir",
        "deepspeed",
    }
    missing = sorted(key for key in required if not config.get(key))
    if missing:
        raise TrainingConfigurationError(f"missing config keys: {missing}")
    if config.get("finetuning_type") != "full":
        raise TrainingConfigurationError("finetuning_type must be 'full'")
    if config.get("eval_strategy") != "epoch":
        raise TrainingConfigurationError("eval_strategy must be 'epoch'")
    if config.get("eval_on_start") is not True:
        raise TrainingConfigurationError(
            "eval_on_start must be true to validate the base/resumed model "
            "before the first optimizer update"
        )
    if config.get("save_strategy") != "epoch":
        raise TrainingConfigurationError("save_strategy must be 'epoch'")
    if config.get("load_best_model_at_end") is not True:
        raise TrainingConfigurationError("load_best_model_at_end must be true")
    if config.get("save_total_limit") != 2:
        raise TrainingConfigurationError("save_total_limit must be exactly 2")
    if config.get("predict_with_generate") is not True:
        raise TrainingConfigurationError(
            "predict_with_generate must be true for deployment-faithful labels"
        )
    if config.get("metric_for_best_model") not in {
        "label_accuracy",
        "eval_label_accuracy",
    }:
        raise TrainingConfigurationError(
            "metric_for_best_model must be label_accuracy"
        )
    if config.get("greater_is_better") is not True:
        raise TrainingConfigurationError("greater_is_better must be true")

    dataset_dir = resolve_from_project(config["dataset_dir"])
    dataset_info_path = dataset_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise TrainingConfigurationError(
            f"prepared dataset_info.json not found: {dataset_info_path}"
        )

    deepspeed_path = resolve_from_project(config["deepspeed"])
    if not deepspeed_path.is_file():
        raise TrainingConfigurationError(
            f"DeepSpeed config not found: {deepspeed_path}"
        )
    try:
        deepspeed = json.loads(deepspeed_path.read_text(encoding="utf-8"))
        zero_stage = deepspeed["zero_optimization"]["stage"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TrainingConfigurationError(
            f"invalid DeepSpeed config {deepspeed_path}: {exc}"
        ) from exc
    if zero_stage == 3:
        raise TrainingConfigurationError(
            "generated validation is incompatible with DeepSpeed ZeRO-3 in "
            "LLaMA-Factory; use the provided ZeRO-2 config"
        )


@dataclass
class GuardLabelMetrics:
    """Streaming-compatible generated-label metrics for LLaMA-Factory."""

    tokenizer: Any

    def __post_init__(self) -> None:
        self.reset()

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert metric tensors to host NumPy arrays without assuming CPU."""
        if isinstance(value, np.ndarray):
            return value
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            return to_numpy()
        return np.asarray(value)

    def reset(self) -> None:
        self.total = 0
        self.correct = 0
        self.parsed = 0
        self.gold_counts = {"safe": 0, "unsafe": 0}
        self.predicted_counts = {"safe": 0, "unsafe": 0}
        self.true_positive = {"safe": 0, "unsafe": 0}

    def __call__(
        self, eval_preds: Any, compute_result: bool = True
    ) -> dict[str, float] | None:
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = self._to_numpy(predictions)
        labels = self._to_numpy(labels)
        predictions = np.where(
            predictions >= 0, predictions, self.tokenizer.pad_token_id
        )
        labels = np.where(labels >= 0, labels, self.tokenizer.pad_token_id)

        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        for prediction_text, label_text in zip(
            decoded_predictions, decoded_labels
        ):
            gold = extract_guard_label(label_text, allow_fallback=False)
            if gold is None:
                raise ValueError(
                    "validation target has no canonical <label> tag: "
                    f"{label_text[:300]!r}"
                )
            predicted = extract_guard_label(prediction_text, allow_fallback=True)
            self.total += 1
            self.gold_counts[gold] += 1
            if predicted is not None:
                self.parsed += 1
                self.predicted_counts[predicted] += 1
            if predicted == gold:
                self.correct += 1
                self.true_positive[gold] += 1

        if not compute_result:
            return None
        if self.total == 0:
            raise ValueError("validation metric received zero examples")

        f1_values = []
        results: dict[str, float] = {
            "label_accuracy": self.correct / self.total,
            "label_parse_rate": self.parsed / self.total,
        }
        for label in ("safe", "unsafe"):
            tp = self.true_positive[label]
            precision = (
                tp / self.predicted_counts[label]
                if self.predicted_counts[label]
                else 0.0
            )
            recall = tp / self.gold_counts[label] if self.gold_counts[label] else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            results[f"{label}_precision"] = precision
            results[f"{label}_recall"] = recall
            results[f"{label}_f1"] = f1
            f1_values.append(f1)
        results["label_macro_f1"] = sum(f1_values) / len(f1_values)
        self.reset()
        return results


def install_llamafactory_extensions(
    *, label_loss_weight: float, label_tail_tokens: int
) -> None:
    try:
        import torch
        import torch.nn.functional as functional
        from llamafactory.train.sft import workflow
    except ImportError as exc:
        raise TrainingConfigurationError(
            "LLaMA-Factory is not importable. Install it in editable mode "
            "and then install requirements/metrics.txt plus "
            "requirements/deepspeed.txt from that checkout. Underlying "
            f"ImportError: {exc!r}"
        ) from exc

    base_trainer = workflow.CustomSeq2SeqTrainer

    class GuardWeightedSeq2SeqTrainer(base_trainer):
        """Apply modest extra loss to the final label section."""

        def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            """Use bounded cached generation, including evals triggered by train()."""
            previous_padding_side = self.processing_class.padding_side
            self.processing_class.padding_side = "left"

            # LLaMA-Factory disables the model cache whenever do_train=True.
            # That is correct for gradient-checkpointed training but makes
            # autoregressive validation on 16k-token inputs prohibitively slow.
            # Explicitly passing use_cache also takes precedence in generate().
            cache_configs = enable_generation_cache(self.model)

            # LLaMA-Factory stores max_new_tokens, EOS IDs, sampling mode and
            # other generation settings in _gen_kwargs. Seq2SeqTrainer.evaluate
            # otherwise replaces them when epoch evaluation supplies no kwargs,
            # potentially falling back to max_length=cutoff_len (16k).
            generation_kwargs = build_evaluation_kwargs(
                getattr(self, "_gen_kwargs", {}),
                kwargs,
            )
            try:
                return super().evaluate(*args, **generation_kwargs)
            finally:
                restore_generation_cache(cache_configs)
                self.processing_class.padding_side = previous_padding_side

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            if (
                label_loss_weight <= 1.0
                or "labels" not in inputs
                or getattr(self.finetuning_args, "use_asft_loss", False)
            ):
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_losses = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            valid = shift_labels.ne(-100)
            weights = valid.to(dtype=token_losses.dtype)
            for row_index in range(valid.size(0)):
                positions = torch.nonzero(valid[row_index], as_tuple=False).flatten()
                if positions.numel():
                    tail = positions[-label_tail_tokens:]
                    weights[row_index, tail] = label_loss_weight
            denominator = weights.sum().clamp_min(1.0)
            loss = (token_losses * weights).sum() / denominator
            return (loss, outputs) if return_outputs else loss

    workflow.ComputeSimilarity = GuardLabelMetrics
    workflow.CustomSeq2SeqTrainer = GuardWeightedSeq2SeqTrainer


def read_trainer_state(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "trainer_state.json"
    if not state_path.is_file():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def remove_root_weight_duplicates(output_dir: Path) -> list[str]:
    """Remove only known root model shards after checkpoint copies are verified."""
    checkpoints = checkpoint_dirs(output_dir)
    if not checkpoints:
        raise TrainingConfigurationError(
            f"refusing root-weight cleanup: no checkpoints in {output_dir}"
        )
    if not any(
        list(checkpoint.glob("model*.safetensors"))
        or list(checkpoint.glob("pytorch_model*.bin"))
        for checkpoint in checkpoints
    ):
        raise TrainingConfigurationError(
            "refusing root-weight cleanup: retained checkpoints contain no "
            "recognized full-model weight shards"
        )

    removed: list[str] = []
    seen: set[Path] = set()
    for pattern in ROOT_WEIGHT_PATTERNS:
        for path in output_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            path.unlink()
            seen.add(path)
            removed.append(path.name)
    return sorted(removed)


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise TrainingConfigurationError(
            f"refusing to replace non-symlink checkpoint alias: {path}"
        )
    path.symlink_to(target.name, target_is_directory=True)


def prune_and_describe_checkpoints(
    output_dir: Path, *, remove_root_weights: bool
) -> dict[str, Any]:
    checkpoints = checkpoint_dirs(output_dir)
    if not checkpoints:
        raise TrainingConfigurationError(f"no checkpoints found in {output_dir}")
    state = read_trainer_state(output_dir)
    best_raw = state.get("best_model_checkpoint")
    best = None
    if isinstance(best_raw, str):
        candidate = output_dir / Path(best_raw).name
        if candidate in checkpoints:
            best = candidate
    latest = checkpoints[-1]

    keep: set[Path] = {latest}
    if best is not None:
        keep.add(best)
    for checkpoint in reversed(checkpoints):
        if len(keep) >= 2:
            break
        keep.add(checkpoint)
    removed_checkpoints: list[str] = []
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)
            removed_checkpoints.append(checkpoint.name)

    if best is None:
        best = latest
    replace_symlink(output_dir / "best_checkpoint", best)
    replace_symlink(output_dir / "last_checkpoint", latest)
    removed_root_weights = (
        remove_root_weight_duplicates(output_dir) if remove_root_weights else []
    )
    summary = {
        "best_checkpoint": best.name,
        "last_checkpoint": latest.name,
        "best_metric": state.get("best_metric"),
        "metric_for_best_model": "eval_label_accuracy",
        "retained_checkpoint_weight_sets": sorted(path.name for path in keep),
        "removed_checkpoint_directories": removed_checkpoints,
        "removed_duplicate_root_weight_files": removed_root_weights,
    }
    (output_dir / "checkpoint_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config.expanduser().resolve())
    label_loss_weight = float(config.pop("guard_label_loss_weight", 4.0))
    label_tail_tokens = int(config.pop("guard_label_tail_tokens", 12))
    if label_loss_weight < 1.0:
        raise TrainingConfigurationError("guard_label_loss_weight must be >= 1")
    if label_tail_tokens <= 0:
        raise TrainingConfigurationError("guard_label_tail_tokens must be positive")

    validate_training_contract(config)
    config, output_dir = configure_resume(config, args.resume)
    install_llamafactory_extensions(
        label_loss_weight=label_loss_weight,
        label_tail_tokens=label_tail_tokens,
    )

    if global_rank() == 0:
        print(
            json.dumps(
                {
                    "config": str(args.config.expanduser().resolve()),
                    "output_dir": str(output_dir),
                    "resume_from_checkpoint": config.get("resume_from_checkpoint"),
                    "label_loss_weight": label_loss_weight,
                    "label_tail_tokens": label_tail_tokens,
                    "checkpoint_metric": "generated label accuracy",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    from llamafactory.train.tuner import run_exp

    run_exp(args=config)
    if global_rank() == 0:
        summary = prune_and_describe_checkpoints(
            output_dir,
            remove_root_weights=not args.no_final_cleanup,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TrainingConfigurationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
