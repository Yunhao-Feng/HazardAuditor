"""VERL 0.7 extension for RLGuard outcome-conditioned CISPO.

Load with ``VERL_USE_EXTERNAL_MODULES=training.guardpo.cispo.verl_plugin``. No VERL source file
is patched. The module registers a CRL-style batch-centered outcome advantage
and a token-level CISPO policy loss through VERL's public registries. It also
installs validation-aware best/latest checkpoint retention hooks.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from training.guardpo.cispo.checkpoint_retention import install_verl_checkpoint_retention
from training.guardpo.cispo.outcome import batch_centered_outcome_advantages
from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss


ADVANTAGE_NAME = "rlguard_outcome"
POLICY_LOSS_NAME = "rlguard_cispo"


# VERL imports this external module before constructing the trainer. Install
# the hook once in every process; only the Ray trainer driver invokes it.
install_verl_checkpoint_retention()


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}")
    return value


def _nonnegative_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0.0:
        raise RuntimeError(f"{name} must be >= 0, got {value}")
    return value


def _build_region_masks(
    response_mask: torch.Tensor,
    label_tail_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split right-padded valid response tokens into analysis and label tail."""

    mask = response_mask.to(dtype=torch.bool)
    sequence_length = mask.shape[-1]
    positions = torch.arange(sequence_length, device=mask.device).unsqueeze(0)
    valid_lengths = mask.sum(dim=-1, keepdim=True)
    label_starts = torch.clamp(valid_lengths - label_tail_tokens, min=0)
    label_mask = mask & (positions >= label_starts)
    analysis_mask = mask & ~label_mask
    return analysis_mask, label_mask


@register_adv_est(ADVANTAGE_NAME)
def compute_rlguard_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | list[Any] | None = None,
    config: Any = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast one batch-centered outcome advantage over every response token.

    Prompt IDs are deliberately ignored. Unlike GRPO's per-prompt relative
    estimator, this follows the CRL Extractor objective: all generated responses
    in the current rollout batch share one reward baseline, without standard
    deviation normalization.
    """

    del index, config
    try:
        advantages, _sequence_rewards, _batch_mean = (
            batch_centered_outcome_advantages(
                token_level_rewards,
                response_mask,
            )
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    returns = advantages.clone()
    return advantages, returns


def _global_sequence_region_mean(
    loss_matrix: torch.Tensor,
    region_mask: torch.Tensor,
    *,
    dp_size: int,
    global_batch_size: int,
) -> torch.Tensor:
    """FSDP-correct mean of per-sequence region means.

    The same sequence outcome advantage is still present on every valid token,
    but a response with 300 analysis tokens must not contribute ten times the
    total gradient of one with 30. We first average within each response region
    and then average responses over the complete PPO mini-batch. Empty regions
    contribute zero. VERL calls this function once per dynamic micro-batch but
    provides the complete global batch size, so the fixed denominator preserves
    correct FSDP scaling across both DP ranks and micro-batches.
    """

    if global_batch_size < 1:
        raise RuntimeError(f"invalid global batch size: {global_batch_size}")
    mask = region_mask.to(dtype=loss_matrix.dtype)
    region_lengths = mask.sum(dim=-1)
    per_sequence_numerators = (loss_matrix * mask).sum(dim=-1)
    per_sequence_means = per_sequence_numerators / region_lengths.clamp_min(1.0)
    per_sequence_means = per_sequence_means * (region_lengths > 0).to(
        dtype=loss_matrix.dtype
    )
    local_numerator = per_sequence_means.sum()
    # FSDP averages gradients across DP ranks. Multiplying each local loss by
    # dp_size makes the averaged gradient equal the true global sequence mean.
    return local_numerator / global_batch_size * dp_size


def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Avoid NaN metrics for a rare empty analysis/response region."""

    denominator = mask.float().sum()
    if float(denominator.item()) == 0.0:
        return values.sum() * 0.0
    return (values * mask.float()).sum() / denominator


@register_policy_loss(POLICY_LOSS_NAME)
def compute_rlguard_cispo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Token-level CISPO with sequence-balanced analysis/label aggregation."""

    if config is None:
        raise RuntimeError("RLGuard CISPO requires VERL ActorConfig")
    if loss_agg_mode != "token-mean":
        raise RuntimeError(
            "RLGuard sequence-balanced CISPO requires VERL's token-mean "
            "dispatch mode before applying its custom aggregation; "
            f"received {loss_agg_mode!r}"
        )

    clip_ratio_low = (
        config.clip_ratio_low
        if config.clip_ratio_low is not None
        else config.clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high
        if config.clip_ratio_high is not None
        else config.clip_ratio
    )
    label_loss_weight = _nonnegative_float_env(
        "RLGUARD_LABEL_LOSS_WEIGHT",
        2.0,
    )
    label_tail_tokens = _positive_int_env("RLGUARD_LABEL_TAIL_TOKENS", 12)

    # This is the CISPO proximal ratio pi_theta / pi_old. VERL may additionally
    # supply token-level rollout correction weights for pi_old / pi_rollout;
    # the two ratios solve different distribution-shift problems.
    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_ratio_low,
        max=1.0 + clip_ratio_high,
    )
    clipped_ratio_sg = clipped_ratio.detach()
    loss_matrix = -clipped_ratio_sg * advantages * log_prob
    if rollout_is_weights is not None:
        loss_matrix = loss_matrix * rollout_is_weights

    analysis_mask, label_mask = _build_region_masks(
        response_mask,
        label_tail_tokens,
    )
    global_batch_info = getattr(config, "global_batch_info", {}) or {}
    dp_size = int(global_batch_info.get("dp_size", 1))
    global_batch_size = global_batch_info.get("global_batch_size")
    if global_batch_size is None:
        raise RuntimeError(
            "VERL did not provide the global batch size required by "
            "RLGuard sequence-balanced CISPO"
        )
    total_global_sequences = int(global_batch_size)
    analysis_loss = _global_sequence_region_mean(
        loss_matrix,
        analysis_mask,
        dp_size=dp_size,
        global_batch_size=total_global_sequences,
    )
    label_loss = _global_sequence_region_mean(
        loss_matrix,
        label_mask,
        dp_size=dp_size,
        global_batch_size=total_global_sequences,
    )
    total_loss = analysis_loss + label_loss_weight * label_loss

    response_mask_float = response_mask.float()
    ppo_kl = _safe_masked_mean(-log_ratio, response_mask_float)
    clip_fraction = _safe_masked_mean(
        (ratio != clipped_ratio).float(),
        response_mask_float,
    )
    active_advantage = advantages.abs() > 0
    active_fraction = _safe_masked_mean(
        active_advantage.float(),
        response_mask_float,
    )
    analysis_advantage = _safe_masked_mean(
        advantages.abs(),
        analysis_mask.float(),
    )
    label_advantage = _safe_masked_mean(
        advantages.abs(),
        label_mask.float(),
    )

    metrics = {
        "actor/pg_clipfrac": clip_fraction.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
        "actor/rlguard_analysis_loss": analysis_loss.detach().item(),
        "actor/rlguard_label_loss": label_loss.detach().item(),
        "actor/rlguard_active_token_fraction": active_fraction.detach().item(),
        "actor/rlguard_analysis_adv_abs": analysis_advantage.detach().item(),
        "actor/rlguard_label_adv_abs": label_advantage.detach().item(),
    }
    return total_loss, metrics
