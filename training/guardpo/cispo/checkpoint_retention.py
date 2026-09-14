"""Validation-aware checkpoint retention for VERL 0.7 RayPPOTrainer.

VERL's native retention is recency-based. VERL 0.7 validates before saving at
each scheduled step, so RLGuard carries that validation result into the save
hook and then keeps the union of two roles:

* the checkpoint with the best label-first validation key; and
* the most recently saved checkpoint, which is required for safe resume.

Those roles may point at the same checkpoint, so the output contains at most
two ``global_step_<N>`` directories. The launcher skips step-zero validation,
but temporarily arms VERL's native validation/save conditions after the first
actor update. It then restores the configured periodic cadence (50 by default),
giving checkpoints at step 1 and steps 50/100/... without copying VERL's fit
loop. Step-zero handling remains only for safe compatibility with an explicit
direct-plugin run that enables a baseline pass.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from training.guardpo.cispo.summarize_validation import (
    _load_jsonl,
    selection_key,
    summarize_rows,
)


STATE_FILENAME = "checkpoint_retention.json"
BEST_LINK = "best_checkpoint"
LATEST_LINK = "latest_checkpoint"
SELECTION_ORDER = (
    "overall_macro_f1",
    "worst_source_macro_f1",
    "minimum_class_recall",
)
_CHECKPOINT_PATTERN = re.compile(r"global_step_(0|[1-9][0-9]*)\Z")
_PATCH_MARKER = "_rlguard_v07_best_latest_retention_installed"
_PENDING_VALIDATION = "_rlguard_pending_checkpoint_validation"
_STEP_ONE_PROBE_ARMED = "_rlguard_step_one_probe_armed"
_STEP_ONE_PROBE_DONE = "_rlguard_step_one_probe_done"
_PERIODIC_FREQUENCIES = "_rlguard_periodic_validation_save_frequencies"


class CheckpointRetentionError(RuntimeError):
    """Raised when safe best/latest retention cannot be guaranteed."""


def _boolean_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise CheckpointRetentionError(
        f"{name} must be a boolean (0/1/false/true), got {raw!r}"
    )


def _arm_step_one_probe(trainer: Any) -> bool:
    """Make VERL's native end-of-step branch validate and save step 1 once.

    VERL 0.7 expresses periodic validation with one integer frequency, so it
    cannot directly represent ``{1, 50, 100, ...}``. After the first actor
    update we temporarily change both frequencies to one. The unmodified VERL
    loop then performs validation, merges its metrics into the normal step-1
    log, and calls the normal checkpoint hook. That hook restores the original
    frequencies after the save succeeds.
    """

    if not _boolean_env("RLGUARD_STEP_ONE_PROBE", True):
        return False
    if int(getattr(trainer, "global_steps", -1)) != 1:
        return False
    if bool(getattr(trainer, _STEP_ONE_PROBE_DONE, False)) or bool(
        getattr(trainer, _STEP_ONE_PROBE_ARMED, False)
    ):
        return False
    if getattr(trainer, "val_reward_fn", None) is None:
        raise CheckpointRetentionError(
            "step-one validation was requested but trainer.val_reward_fn is unavailable"
        )

    trainer_config = trainer.config.trainer
    save_frequency = int(trainer_config.save_freq)
    test_frequency = int(trainer_config.test_freq)
    if save_frequency < 1 or test_frequency < 1:
        raise CheckpointRetentionError(
            "step-one probe requires positive periodic save/test frequencies"
        )
    if save_frequency != test_frequency:
        raise CheckpointRetentionError(
            "step-one probe requires identical periodic save/test frequencies"
        )

    setattr(
        trainer,
        _PERIODIC_FREQUENCIES,
        (save_frequency, test_frequency),
    )
    trainer_config.save_freq = 1
    trainer_config.test_freq = 1
    setattr(trainer, _STEP_ONE_PROBE_ARMED, True)
    print(
        "RLGuard step-one probe armed: VERL will validate and save after "
        f"step 1, then restore the every-{test_frequency}-step cadence",
        flush=True,
    )
    return True


def _restore_periodic_frequencies(trainer: Any) -> bool:
    """Restore periodic validation/save settings after the step-1 checkpoint."""

    frequencies = getattr(trainer, _PERIODIC_FREQUENCIES, None)
    if frequencies is None:
        return False
    save_frequency, test_frequency = frequencies
    trainer.config.trainer.save_freq = int(save_frequency)
    trainer.config.trainer.test_freq = int(test_frequency)
    delattr(trainer, _PERIODIC_FREQUENCIES)
    if hasattr(trainer, _STEP_ONE_PROBE_ARMED):
        delattr(trainer, _STEP_ONE_PROBE_ARMED)
    setattr(trainer, _STEP_ONE_PROBE_DONE, True)
    print(
        "RLGuard step-one probe scheduling reset: restored validation/save cadence "
        f"to every {test_frequency} steps",
        flush=True,
    )
    return True


def _attach_outcome_diagnostics(actor_output: Any, batch: Any) -> None:
    """Add directly interpretable reward/format rates to VERL's step log."""

    non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
    meta_info = getattr(actor_output, "meta_info", None)
    if not isinstance(meta_info, dict):
        return
    metrics = meta_info.get("metrics")
    if not isinstance(metrics, dict):
        return

    def numeric_values(key: str) -> list[float]:
        raw = non_tensor_batch.get(key)
        if raw is None:
            return []
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if not isinstance(raw, list):
            raw = [raw]
        return [float(value) for value in raw]

    parse_values = numeric_values("parse_ok")
    correct_values = numeric_values("label_correct")
    reward_values = numeric_values("outcome_reward")
    if parse_values:
        metrics["training/valid_format_rate"] = [
            sum(parse_values) / len(parse_values)
        ]
    if correct_values:
        metrics["training/correct_label_rate"] = [
            sum(correct_values) / len(correct_values)
        ]
    if reward_values:
        denominator = len(reward_values)
        metrics["training/malformed_rate"] = [
            sum(value == -1.25 for value in reward_values) / denominator
        ]
        metrics["training/wrong_label_rate"] = [
            sum(value == -1.0 for value in reward_values) / denominator
        ]
        metrics["training/correct_reward_rate"] = [
            sum(value == 1.0 for value in reward_values) / denominator
        ]


def _save_actor_on_compute_device(trainer: Any, save_checkpoint: Any) -> None:
    """Stage a VERL 0.7 new-engine actor on GPU before FSDP state export.

    In hybrid async rollout mode, ``ActorRolloutRefWorker.wake_up()`` copies
    actor weights into vLLM and then explicitly moves only the actor model to
    CPU.  Validation ends with another rollout, so the model is still on CPU
    when ``RayPPOTrainer._save_checkpoint()`` immediately asks FSDP for a
    sharded state dict.  FSDP1 rejects that manual device state with
    ``expected ... cuda, was on cpu``.

    The rollout manager has already put vLLM to sleep before validation
    returns, so restoring the actor model to its compute device is safe.  The
    optimizer and gradients were not moved to CPU by rollout wake-up and do
    not need an extra transfer.  We deliberately leave the model on the
    compute device after saving: the next normal rollout wake-up exports the
    new weights and performs VERL's usual actor-to-CPU transition.
    """

    worker_group = getattr(trainer, "actor_rollout_wg", None)
    if worker_group is None:
        raise CheckpointRetentionError(
            "VERL trainer has no actor_rollout_wg for checkpoint staging"
        )
    move_to = getattr(worker_group, "to", None)
    if not callable(move_to):
        raise CheckpointRetentionError(
            "VERL 0.7 new-engine actor worker has no device-transfer interface"
        )

    print(
        "RLGuard checkpoint staging: moving actor model to compute device",
        flush=True,
    )
    move_to("device", model=True, optimizer=False, grad=False)
    save_checkpoint(trainer)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise CheckpointRetentionError(
            f"refusing to replace non-symlink checkpoint pointer: {link}"
        )
    temporary = link.with_name(f".{link.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(target.resolve(), target_is_directory=True)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


class BestLatestCheckpointRetention:
    """Persist validation history and retain only best/latest checkpoints."""

    def __init__(
        self,
        *,
        output_dir: Path,
        base_model: Path,
        max_checkpoints: int = 2,
    ) -> None:
        if max_checkpoints != 2:
            raise CheckpointRetentionError(
                "RLGuard retention is intentionally fixed to two roles: best and latest"
            )
        self.output_dir = output_dir.expanduser().resolve()
        self.base_model = base_model.expanduser().resolve()
        self.max_checkpoints = max_checkpoints
        self.state_path = self.output_dir / STATE_FILENAME
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state_existed = self.state_path.exists()
        self.state = self._load_state()
        if not state_existed and self._checkpoint_dirs():
            raise CheckpointRetentionError(
                "found existing global_step_* directories but no "
                "checkpoint_retention.json; refusing to guess and delete checkpoints. "
                "Use a fresh output directory or migrate the existing run explicitly."
            )

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "selection_order": list(SELECTION_ORDER),
            "base_model": str(self.base_model),
            "best": None,
            "latest": None,
            "retained_rl_steps": [],
            "validations": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointRetentionError(
                f"cannot safely read retention state {self.state_path}: {exc}"
            ) from exc
        if not isinstance(state, dict) or state.get("schema_version") != 2:
            raise CheckpointRetentionError(
                f"unsupported retention state in {self.state_path}"
            )
        if state.get("selection_order") != list(SELECTION_ORDER):
            raise CheckpointRetentionError(
                "checkpoint selection rule changed; use a new output directory "
                "or explicitly migrate checkpoint_retention.json"
            )
        recorded_base = Path(str(state.get("base_model", ""))).expanduser().resolve()
        if recorded_base != self.base_model:
            raise CheckpointRetentionError(
                f"retention state was created for base model {recorded_base}, "
                f"not {self.base_model}"
            )
        if not isinstance(state.get("validations"), dict):
            raise CheckpointRetentionError("retention state has invalid validations")
        for role in ("best", "latest"):
            reference = state.get(role)
            if reference is None:
                continue
            if not isinstance(reference, dict) or not isinstance(
                reference.get("path"), str
            ):
                raise CheckpointRetentionError(
                    f"retention state has invalid {role} checkpoint reference"
                )
            if not Path(reference["path"]).is_dir():
                raise CheckpointRetentionError(
                    f"retention state {role} checkpoint is missing: "
                    f"{reference['path']}"
                )
        return state

    def _checkpoint_path(self, step: int) -> Path:
        return self.output_dir / f"global_step_{step}"

    def _checkpoint_dirs(self) -> dict[int, Path]:
        checkpoints: dict[int, Path] = {}
        for candidate in self.output_dir.iterdir():
            match = _CHECKPOINT_PATTERN.fullmatch(candidate.name)
            if match is None:
                continue
            if candidate.is_symlink():
                raise CheckpointRetentionError(
                    f"refusing to manage symlink named as a checkpoint: {candidate}"
                )
            if not candidate.is_dir():
                raise CheckpointRetentionError(
                    f"checkpoint path is not a directory: {candidate}"
                )
            checkpoints[int(match.group(1))] = candidate
        return checkpoints

    def _best_internal_step(self) -> int | None:
        best = self.state.get("best")
        if not isinstance(best, dict) or bool(best.get("external", False)):
            return None
        return int(best["step"])

    def _latest_step(self) -> int | None:
        latest = self.state.get("latest")
        return int(latest["step"]) if isinstance(latest, dict) else None

    def _checkpoint_reference(self, step: int) -> dict[str, Any]:
        if step == 0:
            if not self.base_model.is_dir():
                raise CheckpointRetentionError(
                    f"step-zero SFT checkpoint is unavailable: {self.base_model}"
                )
            return {
                "step": 0,
                "path": str(self.base_model),
                "external": True,
            }
        checkpoint = self._checkpoint_path(step)
        if not checkpoint.is_dir():
            raise CheckpointRetentionError(
                f"validation step {step} has no saved checkpoint at {checkpoint}; "
                "save_freq and test_freq must be identical"
            )
        return {
            "step": step,
            "path": str(checkpoint),
            "external": False,
        }

    def _write_state_and_links(self) -> None:
        checkpoints = self._checkpoint_dirs()
        self.state["retained_rl_steps"] = sorted(checkpoints)
        _atomic_write_json(self.state_path, self.state)

        best = self.state.get("best")
        if isinstance(best, dict):
            _atomic_symlink(self.output_dir / BEST_LINK, Path(best["path"]))
        latest = self.state.get("latest")
        if isinstance(latest, dict):
            _atomic_symlink(self.output_dir / LATEST_LINK, Path(latest["path"]))

    def _prune(self) -> None:
        keep_steps = {
            step
            for step in (self._best_internal_step(), self._latest_step())
            if step is not None
        }
        if len(keep_steps) > self.max_checkpoints:
            raise CheckpointRetentionError(
                f"internal error: retention requested {sorted(keep_steps)}"
            )

        for step, checkpoint in self._checkpoint_dirs().items():
            if step in keep_steps:
                continue
            # Only exact, direct children discovered by _checkpoint_dirs reach
            # this destructive operation.
            print(
                f"RLGuard checkpoint retention: removing non-best/non-latest {checkpoint}",
                flush=True,
            )
            shutil.rmtree(checkpoint)

        remaining = self._checkpoint_dirs()
        if len(remaining) > self.max_checkpoints:
            raise CheckpointRetentionError(
                f"retention left too many checkpoints: {sorted(remaining)}"
            )

    def record_checkpoint(self, step: int) -> None:
        """Mark a completed VERL save as latest and prune the previous latest."""

        if step <= 0:
            raise CheckpointRetentionError(f"checkpoint step must be positive, got {step}")
        checkpoint = self._checkpoint_path(step)
        if not checkpoint.is_dir():
            raise CheckpointRetentionError(
                f"VERL reported a save but checkpoint is missing: {checkpoint}"
            )
        self.state["latest"] = {
            "step": step,
            "path": str(checkpoint),
            "external": False,
        }
        # Persist the new resume target before removing the previous latest.
        # A crash can temporarily leave an extra checkpoint, but never a state
        # file that points only at the checkpoint we are about to delete.
        self._write_state_and_links()
        self._prune()
        self._write_state_and_links()

    def record_validation(self, step: int, validation_file: Path) -> None:
        """Score one validation dump, update best, and enforce best/latest."""

        validation_file = validation_file.expanduser().resolve()
        rows = _load_jsonl(validation_file)
        summary = summarize_rows(rows)
        key = tuple(float(value) for value in selection_key(summary))
        reference = self._checkpoint_reference(step)
        candidate = {
            **reference,
            "selection_key": list(key),
            "validation_file": str(validation_file),
        }
        self.state["validations"][str(step)] = {
            "selection_key": list(key),
            "validation_file": str(validation_file),
        }

        best = self.state.get("best")
        best_key = (
            tuple(float(value) for value in best["selection_key"])
            if isinstance(best, dict)
            else None
        )
        if best_key is None or key > best_key:
            self.state["best"] = candidate
            print(
                "RLGuard checkpoint retention: new best "
                f"step={step} key={tuple(round(value, 6) for value in key)}",
                flush=True,
            )

        # Persist the selected best before deleting the displaced checkpoint.
        # This makes retention restart-safe across process or machine failure.
        self._write_state_and_links()
        self._prune()
        self._write_state_and_links()


def install_verl_checkpoint_retention() -> None:
    """Install retention hooks on VERL 0.7's RayPPOTrainer."""

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer, _PATCH_MARKER, False):
        return

    original_save_checkpoint = RayPPOTrainer._save_checkpoint
    original_validate = RayPPOTrainer._validate
    original_update_actor = RayPPOTrainer._update_actor

    def retention_manager(trainer: Any) -> BestLatestCheckpointRetention:
        manager = getattr(trainer, "_rlguard_checkpoint_retention", None)
        if manager is None:
            trainer_config = trainer.config.trainer
            if str(trainer_config.get("use_legacy_worker_impl", "auto")) != "disable":
                raise CheckpointRetentionError(
                    "RLGuard on VERL 0.7 requires trainer.use_legacy_worker_impl=disable"
                )
            if int(trainer_config.save_freq) != int(trainer_config.test_freq):
                raise CheckpointRetentionError(
                    "trainer.save_freq and trainer.test_freq must be identical"
                )
            if trainer_config.get("max_actor_ckpt_to_keep", None) is not None:
                raise CheckpointRetentionError(
                    "native max_actor_ckpt_to_keep must be null; RLGuard owns retention"
                )
            checkpoint_config = trainer.config.actor_rollout_ref.actor.get(
                "checkpoint", {}
            )
            if bool(checkpoint_config.get("async_save", False)):
                raise CheckpointRetentionError(
                    "asynchronous checkpoint saving is incompatible with immediate "
                    "validation-aware retention"
                )
            base_model = os.getenv("RLGUARD_BASE_MODEL_PATH")
            if base_model is None:
                # Backward-compatible fallback for direct plugin use. The
                # RLGuard launcher always supplies the immutable source SFT
                # checkpoint because model.path is a generated tokenizer view.
                base_model = str(trainer.config.actor_rollout_ref.model.path)
            manager = BestLatestCheckpointRetention(
                output_dir=Path(str(trainer.config.trainer.default_local_dir)),
                base_model=Path(base_model),
                max_checkpoints=int(os.getenv("RLGUARD_MAX_CHECKPOINTS", "2")),
            )
            setattr(trainer, "_rlguard_checkpoint_retention", manager)
        return manager

    def save_with_retention(trainer: Any) -> None:
        try:
            _save_actor_on_compute_device(trainer, original_save_checkpoint)
            step = int(trainer.global_steps)
            manager = retention_manager(trainer)
            manager.record_checkpoint(step)
            pending = getattr(trainer, _PENDING_VALIDATION, None)
            if pending is not None:
                pending_step, validation_file = pending
                if int(pending_step) == step:
                    manager.record_validation(step, Path(validation_file))
                    delattr(trainer, _PENDING_VALIDATION)
        finally:
            # Once VERL reaches its native step-1 save branch, return to the
            # user's periodic cadence even if the save raises. A failed save
            # still aborts the run and the checkpoint preflight will reject a
            # partial directory on restart.
            if int(getattr(trainer, "global_steps", -1)) == 1:
                _restore_periodic_frequencies(trainer)

    def validate_with_retention(trainer: Any) -> dict[str, float]:
        metrics = original_validate(trainer)
        validation_dir = trainer.config.trainer.get("validation_data_dir", None)
        if not validation_dir:
            raise CheckpointRetentionError(
                "trainer.validation_data_dir is required for best-checkpoint retention"
            )
        step = int(trainer.global_steps)
        validation_file = Path(str(validation_dir)) / f"{step}.jsonl"
        manager = retention_manager(trainer)
        checkpoint = manager._checkpoint_path(step)
        if step == 0:
            manager.record_validation(step, validation_file)
        elif checkpoint.is_dir():
            # Initial validation after a VERL resume runs on already-saved
            # weights, so it can be committed immediately.
            if manager._latest_step() != step:
                manager.record_checkpoint(step)
            manager.record_validation(step, validation_file)
        else:
            # During normal VERL 0.7 training validation precedes checkpoint
            # saving. The save hook commits this candidate after all weights
            # and dataloader state have been written successfully.
            setattr(trainer, _PENDING_VALIDATION, (step, str(validation_file)))
        return metrics

    def update_actor_with_step_one_probe(
        trainer: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        actor_output = original_update_actor(trainer, *args, **kwargs)
        batch = args[0] if args else kwargs.get("batch")
        if batch is not None:
            _attach_outcome_diagnostics(actor_output, batch)
        _arm_step_one_probe(trainer)
        return actor_output

    RayPPOTrainer._save_checkpoint = save_with_retention
    RayPPOTrainer._validate = validate_with_retention
    RayPPOTrainer._update_actor = update_actor_with_step_one_probe
    setattr(RayPPOTrainer, _PATCH_MARKER, True)


def verl_checkpoint_retention_installed() -> bool:
    """Return whether the active VERL 0.7 trainer has the hooks installed."""

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    return bool(getattr(RayPPOTrainer, _PATCH_MARKER, False))
