"""Preflight, data preparation, and standard VERL command construction."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from packaging.version import InvalidVersion, Version

from training.guardpo.cispo.model_view import ModelViewError, build_runtime_model_view


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "checkpoints" / "sft" / "best_checkpoint"
DEFAULT_TOKENIZER = DEFAULT_MODEL
DEFAULT_SFT_DATA = PROJECT_ROOT / "artifacts" / "sft_data"
DEFAULT_RL_DATA = PROJECT_ROOT / "artifacts" / "guardpo_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "checkpoints" / "guardpo"
RECIPE_VERSION = "verl-0.7-outcome-r12"
VERL_AUDITED_COMMIT = "f9c855f7cf04d603c9546bc01776c74806a879c1"
NATIVE_CUDA_ALLOCATOR_CONFIG = "backend:native"


class LaunchError(RuntimeError):
    """Raised before allocating GPUs when a run is inconsistent."""


@dataclass(frozen=True)
class PreparedPaths:
    train_unique: Path
    train_balanced: Path
    validation: Path
    summary: Path


def _prepared_paths(data_dir: Path) -> PreparedPaths:
    return PreparedPaths(
        # VERL 0.7's RLHFDataset accepts JSON Lines content, but dispatches
        # the loader by filename suffix and only recognizes `.json`.
        train_unique=data_dir / "train_unique.json",
        train_balanced=data_dir / "train_balanced.json",
        validation=data_dir / "validation.json",
        summary=data_dir / "preparation_summary.json",
    )


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"expected JSON object in {path}")
    return value


def ensure_prepared_data(args: argparse.Namespace) -> PreparedPaths:
    paths = _prepared_paths(args.data_dir)
    outputs = [paths.train_unique, paths.train_balanced, paths.validation, paths.summary]
    existing = [path for path in outputs if path.exists()]

    if args.skip_prepare:
        missing = [path for path in outputs if not path.is_file()]
        if missing:
            raise LaunchError(
                "--skip-prepare was set but prepared files are missing: "
                + ", ".join(str(path) for path in missing)
            )
        return paths

    if existing and len(existing) != len(outputs) and not args.overwrite_data:
        raise LaunchError(
            "prepared data directory is partial; inspect it or pass --overwrite-data: "
            + ", ".join(str(path) for path in existing)
        )

    should_prepare = args.overwrite_data or not existing
    if not should_prepare:
        summary = _load_json(paths.summary)
        mismatches: list[str] = []
        if summary.get("schema_version") != 2:
            mismatches.append(
                f"schema_version={summary.get('schema_version')} (required 2 for VERL 0.7)"
            )
        if summary.get("seed") != args.seed:
            mismatches.append(f"seed={summary.get('seed')} (requested {args.seed})")
        if summary.get("samples_per_stratum") != args.samples_per_stratum:
            mismatches.append(
                "samples_per_stratum="
                f"{summary.get('samples_per_stratum')} "
                f"(requested {args.samples_per_stratum})"
            )
        recorded_input = summary.get("inputs", {}).get("directory")
        if recorded_input != str(args.sft_data_dir.resolve()):
            mismatches.append(
                f"input_dir={recorded_input!r} "
                f"(requested {str(args.sft_data_dir.resolve())!r})"
            )
        if mismatches:
            raise LaunchError(
                "existing prepared data was built with different settings; "
                "pass --overwrite-data to rebuild: "
                + "; ".join(mismatches)
            )
        return paths

    command = [
        sys.executable,
        "-m",
        "training.guardpo.cispo.prepare_rl_data",
        "--input-dir",
        str(args.sft_data_dir),
        "--output-dir",
        str(args.data_dir),
        "--samples-per-stratum",
        str(args.samples_per_stratum),
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    print("Preparing VERL data:", shlex.join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise LaunchError(f"data preparation exited with code {completed.returncode}")
    return paths


def _require_hf_checkpoint(model_path: Path) -> None:
    if not model_path.is_dir():
        raise LaunchError(f"model checkpoint directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise LaunchError(f"model checkpoint is missing config.json: {model_path}")
    weight_candidates = [
        *model_path.glob("*.safetensors"),
        *model_path.glob("pytorch_model*.bin"),
    ]
    if not weight_candidates:
        raise LaunchError(
            "model checkpoint has no consolidated Hugging Face weights "
            f"(*.safetensors or pytorch_model*.bin): {model_path}"
        )


def _require_tokenizer(tokenizer_path: Path) -> None:
    if not tokenizer_path.is_dir():
        raise LaunchError(f"tokenizer directory does not exist: {tokenizer_path}")
    tokenizer_candidates = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    if not any((tokenizer_path / name).is_file() for name in tokenizer_candidates):
        raise LaunchError(
            "tokenizer directory has no recognizable tokenizer files: "
            f"{tokenizer_path}"
        )


def _effective_model_path(args: argparse.Namespace) -> Path:
    """Return the generated HF view during a real run, else the source model."""

    return Path(getattr(args, "runtime_model_view", args.model))


def _require_distribution_version(
    distribution: str,
    *,
    minimum: str | None = None,
    maximum: str | None = None,
    excluded: tuple[str, ...] = (),
) -> str:
    try:
        raw_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise LaunchError(f"required package is not installed: {distribution}") from exc
    try:
        version = Version(raw_version)
    except InvalidVersion as exc:
        raise LaunchError(
            f"cannot parse installed {distribution} version {raw_version!r}"
        ) from exc
    if minimum is not None and version < Version(minimum):
        raise LaunchError(
            f"{distribution}>={minimum} is required, found {raw_version}"
        )
    if maximum is not None and version > Version(maximum):
        raise LaunchError(
            f"{distribution}<={maximum} is required, found {raw_version}"
        )
    if any(version == Version(item) for item in excluded):
        raise LaunchError(
            f"{distribution} {raw_version} is explicitly unsupported by VERL 0.7"
        )
    return raw_version


def _load_and_validate_tokenizer(model_path: Path) -> tuple[str, int]:
    """Exercise the same Transformers path VERL uses before Ray starts."""

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise LaunchError("Transformers is not importable") from exc

    try:
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        raise LaunchError(
            "the runtime Hugging Face model/tokenizer cannot be loaded before "
            f"Ray starts ({model_path}): {type(exc).__name__}: {exc}"
        ) from exc

    if tokenizer.eos_token_id is None:
        raise LaunchError("runtime tokenizer has no eos_token_id")
    if tokenizer.chat_template is None:
        raise LaunchError(
            "runtime tokenizer has no chat template; RL data cannot be rendered "
            "the same way as SFT"
        )
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "RLGuard tokenizer preflight."},
                {"role": "user", "content": "Trajectory preflight."},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:
        raise LaunchError(
            "runtime tokenizer chat template failed on the exact RL rendering "
            f"options: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(rendered, str) or not rendered.strip():
        raise LaunchError("runtime tokenizer chat template rendered an empty prompt")

    vocab = tokenizer.get_vocab()
    if not vocab:
        raise LaunchError("runtime tokenizer vocabulary is empty")
    max_token_id = max(int(token_id) for token_id in vocab.values())
    model_vocab_size = int(getattr(model_config, "vocab_size", 0) or 0)
    if model_vocab_size <= max_token_id:
        raise LaunchError(
            "tokenizer/model vocabulary mismatch: tokenizer maximum id is "
            f"{max_token_id}, but model config vocab_size is {model_vocab_size}"
        )
    return tokenizer.__class__.__name__, len(tokenizer)


def _require_verl_v07_capabilities(verl) -> None:
    """Reject same-version VERL checkouts that predate required 0.7 APIs."""

    package_root = Path(verl.__file__).resolve().parent
    ppo_config = package_root / "trainer" / "config" / "ppo_trainer.yaml"
    rollout_config = (
        package_root
        / "trainer"
        / "config"
        / "algorithm"
        / "rollout_correction.yaml"
    )
    engine_losses = package_root / "workers" / "utils" / "losses.py"
    missing: list[str] = []

    try:
        ppo_config_text = ppo_config.read_text(encoding="utf-8")
    except OSError:
        ppo_config_text = ""
    if "algorithm@algorithm.rollout_correction" not in ppo_config_text:
        missing.append("Hydra algorithm.rollout_correction config group")
    if "reward_manager@reward_manager" not in ppo_config_text:
        missing.append("top-level reward_manager config")
    if "use_legacy_worker_impl" not in ppo_config_text:
        missing.append("new engine-worker selector")
    if not rollout_config.is_file():
        missing.append("rollout_correction.yaml")

    try:
        engine_losses_text = engine_losses.read_text(encoding="utf-8")
    except OSError:
        engine_losses_text = ""
    if "global_batch_info" not in engine_losses_text:
        missing.append("engine-worker global batch loss metadata")

    try:
        rollout_helper = importlib.import_module(
            "verl.trainer.ppo.rollout_corr_helper"
        )
    except ModuleNotFoundError:
        rollout_helper = None
    if rollout_helper is None or not hasattr(
        rollout_helper,
        "compute_rollout_correction_and_add_to_batch",
    ):
        missing.append("token-level rollout importance-sampling implementation")

    try:
        engine_workers = importlib.import_module("verl.workers.engine_workers")
        actor_rollout_worker = getattr(
            engine_workers,
            "ActorRolloutRefWorker",
            None,
        )
    except ModuleNotFoundError:
        actor_rollout_worker = None
    if actor_rollout_worker is None or not callable(
        getattr(actor_rollout_worker, "to", None)
    ):
        missing.append("new-engine actor device-transfer interface for FSDP saving")

    if missing:
        rendered = "; ".join(missing)
        raise LaunchError(
            "the editable VERL source calls itself 0.7.x but predates APIs "
            f"required by RLGuard: {rendered}. Loaded source: {package_root}. "
            "Use official VERL tag v0.7.0 at audited commit "
            f"{VERL_AUDITED_COMMIT}; changing Hydra '=' to '+' would only hide "
            "the missing implementation."
        )


def _require_reward_manager_contract() -> str:
    """Load the manager exactly as VERL importlib source mode will load it."""

    from verl.utils.import_utils import load_extern_object
    from verl.workers.reward_manager.abstract import AbstractRewardManager

    reward_manager_path = (
        PROJECT_ROOT / "training" / "guardpo" / "cispo" / "reward_manager.py"
    )
    try:
        manager_class = load_extern_object(
            module_path=str(reward_manager_path),
            object_name="RLGuardRewardManager",
        )
    except Exception as exc:
        raise LaunchError(
            "VERL cannot import RLGuardRewardManager through its configured "
            f"file loader: {type(exc).__name__}: {exc}"
        ) from exc
    if not inspect.isclass(manager_class) or not issubclass(
        manager_class, AbstractRewardManager
    ):
        raise LaunchError(
            "RLGuardRewardManager must implement VERL 0.7 AbstractRewardManager"
        )
    try:
        inspect.signature(manager_class).bind(
            tokenizer=object(),
            num_examine=0,
            compute_score=None,
            reward_fn_key="data_source",
        )
    except TypeError as exc:
        raise LaunchError(
            "RLGuardRewardManager does not accept the constructor keywords used "
            f"by VERL 0.7's synchronous loader: {exc}"
        ) from exc
    return manager_class.__name__


def _require_dataset_contract() -> str:
    """Load the async-rollout dataset through VERL's configured file loader."""

    from verl.utils.dataset import RLHFDataset
    from verl.utils.import_utils import load_extern_object

    dataset_path = (
        PROJECT_ROOT / "training" / "guardpo" / "cispo" / "rl_dataset.py"
    )
    try:
        dataset_class = load_extern_object(
            module_path=str(dataset_path),
            object_name="RLGuardRLHFDataset",
        )
    except Exception as exc:
        raise LaunchError(
            "VERL cannot import RLGuardRLHFDataset through its configured "
            f"file loader: {type(exc).__name__}: {exc}"
        ) from exc
    if not inspect.isclass(dataset_class) or not issubclass(
        dataset_class, RLHFDataset
    ):
        raise LaunchError(
            "RLGuardRLHFDataset must implement VERL 0.7 RLHFDataset"
        )
    return dataset_class.__name__


def _require_checkpoint_output_consistency(args: argparse.Namespace) -> None:
    """Reject partial or conflicting checkpoint trees before Ray starts."""

    from training.guardpo.cispo.checkpoint_retention import (
        BestLatestCheckpointRetention,
        CheckpointRetentionError,
    )

    try:
        manager = BestLatestCheckpointRetention(
            output_dir=args.output_dir,
            base_model=args.model,
            max_checkpoints=args.max_checkpoints,
        )
    except CheckpointRetentionError as exc:
        raise LaunchError(
            f"checkpoint output is not safely resumable: {exc}. "
            "Use a fresh --output-dir; do not treat a directory left by a "
            "failed FSDP state_dict export as a checkpoint."
        ) from exc

    existing_steps = sorted(manager._checkpoint_dirs())
    if args.resume_mode == "disable" and existing_steps:
        raise LaunchError(
            "--resume-mode disable cannot reuse an output directory containing "
            f"checkpoints {existing_steps}; choose a fresh --output-dir or use "
            "--resume-mode auto for a valid managed run"
        )


def _preflight_runtime(args: argparse.Namespace) -> None:
    runtime_model = _effective_model_path(args)
    _require_hf_checkpoint(runtime_model)
    _require_tokenizer(args.tokenizer)
    _require_tokenizer(runtime_model)

    allocator_configs = {
        name: os.environ.get(name)
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    }
    incompatible_allocator_configs = {
        name: value
        for name, value in allocator_configs.items()
        if value != NATIVE_CUDA_ALLOCATOR_CONFIG
    }
    if incompatible_allocator_configs:
        raise LaunchError(
            "vLLM 0.11 requires the native CUDA allocator for its weight "
            "memory pool; incompatible allocator environment: "
            f"{incompatible_allocator_configs}"
        )

    # These are the dependency ranges declared by official VERL v0.7.0.  The
    # supplied server versions (Ray 2.50.1, vLLM 0.11.0, TensorDict 0.10.0,
    # NumPy 1.26.4) satisfy them.  Check metadata without importing vLLM and
    # allocating any GPU state.
    ray_version = _require_distribution_version("ray", minimum="2.41.0")
    transformers_version = _require_distribution_version(
        "transformers", minimum="4.51.0", maximum="4.999999"
    )
    vllm_version = _require_distribution_version(
        "vllm", minimum="0.8.5", maximum="0.12.0"
    )
    _require_distribution_version(
        "tensordict",
        minimum="0.8.0",
        maximum="0.10.0",
        excluded=("0.9.0",),
    )
    _require_distribution_version("numpy", maximum="1.999999")

    try:
        verl = importlib.import_module("verl")
        core_algos = importlib.import_module("verl.trainer.ppo.core_algos")
    except ModuleNotFoundError as exc:
        raise LaunchError(
            "VERL/RLGuard could not be imported in this Python environment "
            f"(missing module: {exc.name!r}). Use the server environment "
            "described in cispo/README.md and run with that environment's python."
        ) from exc
    raw_verl_version = str(
        getattr(verl, "__version__", "")
        or importlib.metadata.version("verl")
    )
    try:
        verl_version = Version(raw_verl_version)
    except InvalidVersion as exc:
        raise LaunchError(f"cannot parse installed VERL version {raw_verl_version!r}") from exc
    if not (Version("0.7.0.dev0") <= verl_version < Version("0.8")):
        raise LaunchError(
            "this implementation targets VERL 0.7.x (server has 0.7.0.dev0); "
            f"found {raw_verl_version!r}"
        )
    if not hasattr(core_algos, "register_adv_est") or not hasattr(
        core_algos,
        "register_policy_loss",
    ):
        raise LaunchError(
            f"installed VERL {getattr(verl, '__version__', 'unknown')} lacks "
            "the external algorithm registries required by this recipe"
        )
    _require_verl_v07_capabilities(verl)
    from training.guardpo.cispo.checkpoint_retention import verl_checkpoint_retention_installed

    if not verl_checkpoint_retention_installed():
        raise LaunchError(
            "RLGuard best/latest checkpoint retention was not installed in "
            "VERL 0.7 RayPPOTrainer"
        )

    reward_manager_class = _require_reward_manager_contract()
    dataset_class = _require_dataset_contract()
    tokenizer_class, tokenizer_size = _load_and_validate_tokenizer(runtime_model)

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise LaunchError("PyTorch is not importable") from exc
    if not torch.cuda.is_available():
        raise LaunchError("CUDA is not available; refusing to start a GPU RL run")
    detected_gpus = int(torch.cuda.device_count())
    if detected_gpus < args.gpus:
        raise LaunchError(
            f"requested {args.gpus} GPUs but PyTorch detects {detected_gpus}; "
            "set --gpus explicitly for an intentional smaller smoke test"
        )

    print(
        "Runtime preflight passed:",
        f"VERL={raw_verl_version}",
        f"Ray={ray_version}",
        f"Transformers={transformers_version}",
        f"vLLM={vllm_version}",
        f"CUDA_GPUs={detected_gpus}",
        f"cuda_allocator={NATIVE_CUDA_ALLOCATOR_CONFIG}",
        f"tokenizer={tokenizer_class}[{tokenizer_size}]",
        f"dataset={dataset_class}",
        f"reward_manager={reward_manager_class}",
        "reward=local_outcome_only",
        flush=True,
    )


def _hydra_runtime_env(env_vars: dict[str, str]) -> str:
    pairs = ",".join(
        f"{key}:{json.dumps(value, ensure_ascii=True)}"
        for key, value in sorted(env_vars.items())
    )
    return "{" + pairs + "}"


def _build_plugin_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        # vLLM 0.11's CuMemAllocator memory pool rejects PyTorch expandable
        # segments. Override both the current name and its legacy alias because
        # the RL command may be launched from a shell previously used for SFT.
        "PYTORCH_ALLOC_CONF": NATIVE_CUDA_ALLOCATOR_CONFIG,
        "PYTORCH_CUDA_ALLOC_CONF": NATIVE_CUDA_ALLOCATOR_CONFIG,
        "VERL_USE_EXTERNAL_MODULES": "training.guardpo.cispo.verl_plugin",
        "RLGUARD_LABEL_LOSS_WEIGHT": str(args.label_loss_weight),
        "RLGUARD_LABEL_TAIL_TOKENS": str(args.label_tail_tokens),
        "RLGUARD_STEP_ONE_PROBE": "1" if args.step_one_probe else "0",
        # Step-zero best-checkpoint selection must point at the immutable SFT
        # checkpoint, not at the generated tokenizer/model symlink view.
        "RLGUARD_BASE_MODEL_PATH": str(args.model),
        "RLGUARD_MAX_CHECKPOINTS": str(args.max_checkpoints),
    }


def _local_ppo_mini_batch_size(args: argparse.Namespace) -> int:
    """Convert prompt-group CLI semantics to per-actor-worker response rows."""

    expanded_responses = args.ppo_mini_batch_size * args.rollout_n
    if expanded_responses % args.gpus != 0:
        raise LaunchError(
            "ppo-mini-batch-size × rollout-n must be divisible by the GPU count"
        )
    return expanded_responses // args.gpus


def build_verl_command(
    args: argparse.Namespace,
    paths: PreparedPaths,
) -> tuple[list[str], dict[str, str]]:
    reward_manager_path = (
        PROJECT_ROOT / "training" / "guardpo" / "cispo" / "reward_manager.py"
    )
    dataset_path = (
        PROJECT_ROOT / "training" / "guardpo" / "cispo" / "rl_dataset.py"
    )
    train_file = (
        paths.train_unique
        if args.train_data == "unique"
        else paths.train_balanced
    )
    validation_dump = args.output_dir / "validation_generations"
    runtime_env_vars = _build_plugin_env(args)
    model_path = _effective_model_path(args)
    local_ppo_mini_batch_size = _local_ppo_mini_batch_size(args)

    overrides = [
        "algorithm.adv_estimator=rlguard_outcome",
        "algorithm.use_kl_in_reward=False",
        "algorithm.rollout_correction.rollout_is=token",
        f"algorithm.rollout_correction.rollout_is_threshold={args.rollout_is_threshold}",
        "algorithm.rollout_correction.rollout_rs=null",
        "algorithm.rollout_correction.rollout_rs_threshold=null",
        "algorithm.rollout_correction.bypass_mode=False",
        f"data.train_files={train_file}",
        f"data.val_files={paths.validation}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.validation_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=False",
        "data.truncation=right",
        f"data.custom_cls.path={dataset_path}",
        "data.custom_cls.name=RLGuardRLHFDataset",
        "data.shuffle=False",
        "data.validation_shuffle=False",
        f"data.seed={args.seed}",
        f"data.dataloader_num_workers={args.dataloader_workers}",
        "data.trust_remote_code=True",
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"actor_rollout_ref.model.path={model_path}",
        # VERL 0.7 has multiple model-path-only tokenizer call sites. Point
        # the nominal tokenizer path at the same validated view as well, so
        # every dataset, actor, rollout, and checkpoint path is identical.
        f"actor_rollout_ref.model.tokenizer_path={model_path}",
        "actor_rollout_ref.model.external_lib=training.guardpo.cispo.verl_plugin",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.policy_loss.loss_mode=rlguard_cispo",
        f"actor_rollout_ref.actor.clip_ratio_low={args.clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={args.clip_ratio_high}",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        f"actor_rollout_ref.actor.optim.weight_decay={args.weight_decay}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={args.warmup_ratio}",
        f"actor_rollout_ref.actor.grad_clip={args.max_grad_norm}",
        # CLI semantics are prompt groups. VERL 0.7 new-engine workers receive
        # the expanded response batch after actor-DP sharding, so this Hydra
        # field must be the local response count: prompts * rollout.n / GPUs.
        "actor_rollout_ref.actor.ppo_mini_batch_size="
        f"{local_ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={args.ppo_epochs}",
        "actor_rollout_ref.actor.shuffle=True",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={args.ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(args.optimizer_offload)}",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        # hf_model is an export artifact, not a resumable FSDP state shard.
        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
        "actor_rollout_ref.actor.checkpoint.async_save=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        f"actor_rollout_ref.rollout.top_p={args.top_p}",
        "actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.rollout.max_model_len={args.max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.max_num_seqs}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={args.ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "critic.enable=False",
        # VERL 0.7 keeps reward_model and reward_manager at the config root.
        # The deterministic manager is synchronous and returns one terminal
        # outcome reward per response; no remote reward model is launched.
        "reward_model.enable=False",
        "reward_model.use_reward_loop=False",
        "reward_model.launch_reward_fn_async=False",
        "reward_manager.source=importlib",
        "reward_manager.name=RLGuardRewardManager",
        f"reward_manager.module.path={reward_manager_path}",
        "reward_manager.module.name=rlguard_reward_manager",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        "trainer.logger=[console]",
        "trainer.balance_batch=True",
        f"trainer.n_gpus_per_node={args.gpus}",
        "trainer.nnodes=1",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        # Start rollout/training immediately. The plugin uses VERL's native
        # end-of-step branch for a one-time step-1 probe, then restores
        # test_freq/save_freq for steps 50/100/...; there is no step-zero pass.
        "trainer.val_before_train=False",
        "trainer.val_only=False",
        f"trainer.default_local_dir={args.output_dir}",
        "trainer.default_hdfs_dir=null",
        # Native VERL retention is recency-only. Disable it so the RLGuard
        # hook can retain validation-best + latest under VERL 0.7's
        # validate-then-save ordering.
        "trainer.max_actor_ckpt_to_keep=null",
        "trainer.max_critic_ckpt_to_keep=null",
        "trainer.del_local_ckpt_after_load=False",
        f"trainer.log_val_generations={args.log_validation_generations}",
        f"trainer.validation_data_dir={validation_dump}",
        "trainer.rollout_data_dir=null",
        # The engine worker supplies global batch token counts to our
        # partitioned CISPO loss; 0.7's legacy dp_actor does not.
        "trainer.use_legacy_worker_impl=disable",
        f"trainer.resume_mode={args.resume_mode}",
        "+ray_kwargs.ray_init.runtime_env.env_vars="
        + _hydra_runtime_env(runtime_env_vars),
    ]
    if args.total_training_steps is not None:
        overrides.append(f"trainer.total_training_steps={args.total_training_steps}")
    if args.resume_mode == "resume_path":
        if args.resume_from is None:
            raise LaunchError("--resume-mode resume_path requires --resume-from")
        overrides.append(f"trainer.resume_from_path={args.resume_from}")
    elif args.resume_from is not None:
        raise LaunchError("--resume-from is only valid with --resume-mode resume_path")

    overrides.extend(args.extra_override)
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    environment = os.environ.copy()
    environment.update(runtime_env_vars)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("VLLM_USE_V1", "1")
    environment.setdefault("NCCL_DEBUG", "WARN")
    return command, environment


def _training_log_path(args: argparse.Namespace) -> Path:
    """Return the stable append-only log used by tmux and tail -f."""

    if args.log_file is not None:
        return args.log_file
    return args.output_dir / "logs" / "training.log"


def _run_with_live_log(
    command: Sequence[str],
    environment: dict[str, str],
    log_path: Path,
) -> int:
    """Run VERL in the foreground while byte-for-byte teeing stdout/stderr.

    A Python-level binary tee preserves tqdm carriage returns and Ray worker
    output, while keeping the child attached to the tmux foreground process
    group. The stable append-only path makes `tail -f` work across resumes.
    """

    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    header = (
        f"\n===== RLGuard run started {started_at} =====\n"
        f"cwd={PROJECT_ROOT}\n"
        f"command={shlex.join(command)}\n"
    ).encode("utf-8")
    print(f"Live training log: {log_path}", flush=True)

    with log_path.open("ab", buffering=0) as log_handle:
        log_handle.write(header)
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if process.stdout is None:
            raise LaunchError("failed to capture VERL stdout for live logging")
        try:
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                log_handle.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        except KeyboardInterrupt:
            # The child shares the foreground process group and normally
            # receives Ctrl-C too. Forward explicitly for nonstandard shells.
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        finally:
            process.stdout.close()
        return_code = process.wait()
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        footer = (
            f"\n===== RLGuard run finished {finished_at} "
            f"exit_code={return_code} =====\n"
        ).encode("utf-8")
        log_handle.write(footer)
        print(
            f"VERL exited with code {return_code}; complete log: {log_path}",
            flush=True,
        )
        return return_code


def _audit_built_command(command: Sequence[str]) -> None:
    """Catch a partially synchronized launcher before handing it to Hydra."""

    overrides = tuple(command[3:])
    forbidden_prefixes = (
        "reward.",
        "trainer.v1.",
        "trainer.use_v1=",
    )
    forbidden = [
        override
        for override in overrides
        if override.startswith(forbidden_prefixes)
    ]
    jsonl_inputs = [
        override
        for override in overrides
        if override.startswith(("data.train_files=", "data.val_files="))
        and override.endswith(".jsonl")
    ]
    required = {
        "algorithm.adv_estimator=rlguard_outcome",
        "actor_rollout_ref.actor.policy_loss.loss_mode=rlguard_cispo",
        "trainer.use_legacy_worker_impl=disable",
        "trainer.val_before_train=False",
        "reward_model.use_reward_loop=False",
        "reward_manager.source=importlib",
    }
    missing = sorted(required.difference(overrides))
    model_paths = [
        item.split("=", 1)[1]
        for item in overrides
        if item.startswith("actor_rollout_ref.model.path=")
    ]
    tokenizer_paths = [
        item.split("=", 1)[1]
        for item in overrides
        if item.startswith("actor_rollout_ref.model.tokenizer_path=")
    ]
    split_tokenizer_paths = (
        len(model_paths) != 1
        or len(tokenizer_paths) != 1
        or model_paths[0] != tokenizer_paths[0]
    )
    plugin_env_override = next(
        (
            item
            for item in overrides
            if item.startswith("+ray_kwargs.ray_init.runtime_env.env_vars=")
        ),
        "",
    )
    base_model_env_missing = "RLGUARD_BASE_MODEL_PATH" not in plugin_env_override
    step_one_probe_env_missing = "RLGUARD_STEP_ONE_PROBE" not in plugin_env_override
    if (
        forbidden
        or jsonl_inputs
        or missing
        or split_tokenizer_paths
        or base_model_env_missing
        or step_one_probe_env_missing
    ):
        raise LaunchError(
            "internally inconsistent VERL 0.7 command; launcher files appear "
            f"partially synchronized (forbidden={forbidden}, "
            f"jsonl_inputs={jsonl_inputs}, missing={missing}, "
            f"split_tokenizer_paths={split_tokenizer_paths}, "
            f"base_model_env_missing={base_model_env_missing}, "
            f"step_one_probe_env_missing={step_one_probe_env_missing})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and launch RLGuard outcome-conditioned CISPO with VERL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--sft-data-dir", type=Path, default=DEFAULT_SFT_DATA)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "append combined VERL/Ray stdout and stderr here; defaults to "
            "OUTPUT_DIR/logs/training.log"
        ),
    )
    parser.add_argument("--train-data", choices=("balanced", "unique"), default="balanced")
    parser.add_argument("--samples-per-stratum", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="build and load-check the runtime model/tokenizer, then exit before Ray",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--total-training-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "CPU optimizer offload; unsupported by this VERL 0.7 new-engine "
            "recipe unless parameter offload is also enabled"
        ),
    )

    parser.add_argument("--max-prompt-length", type=int, default=16000)
    parser.add_argument("--max-response-length", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=16896)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--ppo-max-token-len-per-gpu", type=int, default=20000)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--rollout-tp", type=int, default=2)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dataloader-workers", type=int, default=4)

    parser.add_argument("--clip-ratio-low", type=float, default=0.1)
    parser.add_argument("--clip-ratio-high", type=float, default=0.1)
    parser.add_argument("--rollout-is-threshold", type=float, default=2.0)
    parser.add_argument("--label-loss-weight", type=float, default=2.0)
    parser.add_argument("--label-tail-tokens", type=int, default=12)

    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--test-freq", type=int, default=50)
    parser.add_argument(
        "--step-one-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "validate and save once after the first actor update, then restore "
            "the periodic save/test frequency"
        ),
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=2,
        help="fixed best+latest checkpoint budget; values other than 2 are rejected",
    )
    parser.add_argument("--log-validation-generations", type=int, default=8)
    parser.add_argument("--project-name", default="hazard-auditor")
    parser.add_argument("--experiment-name", default="hazard-auditor-guardpo")
    parser.add_argument(
        "--resume-mode",
        choices=("auto", "disable", "resume_path"),
        default="auto",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="additional final Hydra override; may be repeated",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    selected_stop_modes = sum(
        bool(value)
        for value in (args.prepare_only, args.preflight_only, args.dry_run)
    )
    if selected_stop_modes > 1:
        raise LaunchError(
            "--prepare-only, --preflight-only, and --dry-run are mutually exclusive"
        )
    if args.optimizer_offload:
        raise LaunchError(
            "--optimizer-offload cannot be enabled with RLGuard's VERL 0.7 "
            "new-engine configuration: optimizer movement requires parameter "
            "movement too. The 8×A100 8B recipe keeps both on GPU."
        )
    positive_int_fields = (
        "samples_per_stratum",
        "gpus",
        "train_batch_size",
        "validation_batch_size",
        "ppo_mini_batch_size",
        "ppo_epochs",
        "total_epochs",
        "max_prompt_length",
        "max_response_length",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "ppo_max_token_len_per_gpu",
        "rollout_n",
        "rollout_tp",
        "label_tail_tokens",
        "max_checkpoints",
    )
    for name in positive_int_fields:
        value = getattr(args, name)
        if value < 1:
            raise LaunchError(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_model_len < args.max_prompt_length + args.max_response_length:
        raise LaunchError(
            "--max-model-len must be at least max-prompt-length + max-response-length"
        )
    if args.max_num_batched_tokens < args.max_model_len:
        raise LaunchError("--max-num-batched-tokens must be >= --max-model-len")
    if args.ppo_max_token_len_per_gpu < (
        args.max_prompt_length + args.max_response_length
    ):
        raise LaunchError(
            "--ppo-max-token-len-per-gpu must fit at least one maximum-length sequence"
        )
    if args.max_response_length < args.label_tail_tokens:
        raise LaunchError(
            "--max-response-length must be >= --label-tail-tokens"
        )
    if args.gpus % args.rollout_tp != 0:
        raise LaunchError("--gpus must be divisible by --rollout-tp")
    if args.train_batch_size % args.ppo_mini_batch_size != 0:
        raise LaunchError(
            "--train-batch-size must be divisible by --ppo-mini-batch-size"
        )
    if (args.ppo_mini_batch_size * args.rollout_n) % args.gpus != 0:
        raise LaunchError(
            "ppo-mini-batch-size × rollout-n must be divisible by the GPU count"
        )
    if (args.train_batch_size * args.rollout_n) % args.gpus != 0:
        raise LaunchError(
            "train-batch-size × rollout-n must be divisible by the GPU count"
        )
    # The default five sources and two labels form ten balanced strata.
    if (args.samples_per_stratum * 10) % args.train_batch_size != 0:
        raise LaunchError(
            "10 × samples-per-stratum must be divisible by train-batch-size "
            "to avoid a partial final balanced batch"
        )
    for name in ("top_p", "rollout_gpu_memory_utilization"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            raise LaunchError(f"--{name.replace('_', '-')} must be in (0,1]")
    for name in (
        "learning_rate",
        "max_grad_norm",
        "rollout_is_threshold",
    ):
        value = getattr(args, name)
        if value <= 0.0:
            raise LaunchError(f"--{name.replace('_', '-')} must be > 0")
    for name in (
        "weight_decay",
        "warmup_ratio",
        "temperature",
        "label_loss_weight",
    ):
        value = getattr(args, name)
        if value < 0.0:
            raise LaunchError(f"--{name.replace('_', '-')} must be >= 0")
    if args.clip_ratio_low >= 1.0:
        raise LaunchError("--clip-ratio-low must be < 1 so the lower ratio stays positive")
    if args.clip_ratio_low < 0.0 or args.clip_ratio_high < 0.0:
        raise LaunchError("CISPO clip ratios must be >= 0")
    if args.total_training_steps is not None and args.total_training_steps < 1:
        raise LaunchError("--total-training-steps must be >= 1")
    if args.save_freq == 0 or args.test_freq == 0:
        raise LaunchError("--save-freq and --test-freq must be positive or -1")
    if args.save_freq < 1 or args.test_freq < 1:
        raise LaunchError(
            "best-checkpoint retention requires positive --save-freq and --test-freq"
        )
    if args.save_freq != args.test_freq:
        raise LaunchError(
            "best-checkpoint retention requires --save-freq == --test-freq so "
            "every scored RL step has saved weights"
        )
    if args.max_checkpoints != 2:
        raise LaunchError(
            "--max-checkpoints is fixed at 2: one validation-best and one latest"
        )
    protected_overrides = {
        "algorithm.adv_estimator",
        "actor_rollout_ref.actor.policy_loss.loss_mode",
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.tokenizer_path",
        "actor_rollout_ref.actor.checkpoint.async_save",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
        "actor_rollout_ref.actor.fsdp_config.param_offload",
        "data.custom_cls.name",
        "data.custom_cls.path",
        "ray_kwargs.ray_init.runtime_env.env_vars",
        "trainer.default_local_dir",
        "trainer.del_local_ckpt_after_load",
        "trainer.max_actor_ckpt_to_keep",
        "trainer.max_critic_ckpt_to_keep",
        "trainer.use_legacy_worker_impl",
        "trainer.save_freq",
        "trainer.test_freq",
        "trainer.val_before_train",
        "trainer.validation_data_dir",
        "reward_model.use_reward_loop",
    }
    for override in args.extra_override:
        key = override.split("=", 1)[0].strip().lstrip("+")
        if key in protected_overrides:
            raise LaunchError(
                f"--extra-override cannot replace objective/retention setting {key}; "
                "use the corresponding named RLGuard argument where available"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"RLGuard recipe: {RECIPE_VERSION}", flush=True)
    for name in (
        "model",
        "tokenizer",
        "sft_data_dir",
        "data_dir",
        "output_dir",
        "log_file",
        "resume_from",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())

    try:
        _validate_arguments(args)
        paths = ensure_prepared_data(args)
        if args.prepare_only:
            print(f"Prepared data is ready in {args.data_dir}")
            return 0

        if not args.dry_run:
            _require_hf_checkpoint(args.model)
            _require_tokenizer(args.tokenizer)
            try:
                runtime_model_view, repaired_legacy_schema = build_runtime_model_view(
                    args.model,
                    args.tokenizer,
                    args.data_dir / "runtime_models",
                )
            except ModelViewError as exc:
                raise LaunchError(f"cannot construct runtime model view: {exc}") from exc
            args.runtime_model_view = runtime_model_view
            print(
                "Runtime Hugging Face model view:",
                runtime_model_view,
                "(legacy tokenizer schema repaired)"
                if repaired_legacy_schema
                else "(tokenizer schema already compatible)",
                flush=True,
            )

        command, environment = build_verl_command(args, paths)
        _audit_built_command(command)
        print("VERL command:\n" + shlex.join(command), flush=True)
        if args.dry_run:
            print(
                "Dry run only: CUDA, model files, and VERL imports "
                "were not required. A real run replaces model.path with a "
                "generated weight-symlink/tokenizer runtime view.",
                flush=True,
            )
            return 0

        os.environ.update(_build_plugin_env(args))
        _require_checkpoint_output_consistency(args)
        _preflight_runtime(args)
        if args.preflight_only:
            print("Preflight-only check passed; Ray and training were not started.", flush=True)
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Starting VERL in the foreground with live file logging; "
            "use tmux as documented.",
            flush=True,
        )
        return _run_with_live_log(
            command,
            environment,
            _training_log_path(args),
        )
    except LaunchError as exc:
        raise SystemExit(f"RL launch refused: {exc}") from exc
    return 0
