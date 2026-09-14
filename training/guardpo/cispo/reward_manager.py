"""Deterministic sequence-outcome reward manager for VERL 0.7.

The generated analysis and final label form one policy response. A single
strict label/format reward is placed on the last valid response token, matching
VERL's standard outcome-reward representation. The advantage estimator later
broadcasts the resulting batch-centered sequence advantage over every response
token; no external judge or network access participates in training.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from training.guardpo.cispo.constants import INT_TO_LABEL, VALID_LABELS
from training.guardpo.cispo.outcome import terminal_outcome_tensor
from training.guardpo.cispo.rewards import score_strict_label
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int) and value in INT_TO_LABEL:
        return INT_TO_LABEL[value]
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in VALID_LABELS:
            return lowered
        if lowered in {"0", "1"}:
            return INT_TO_LABEL[int(lowered)]
    raise RuntimeError(f"unsupported ground-truth label: {value!r}")


class RLGuardRewardManager(AbstractRewardManager):
    """Strict, symmetric safe/unsafe outcome reward with no remote calls."""

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        reward_router_address=None,
        reward_model_tokenizer=None,
        **_kwargs,
    ):
        # Keep the complete constructor contract used by VERL 0.7's importlib
        # loader, even though this deterministic manager needs only tokenizer
        # and reward_fn_key.
        del num_examine, reward_router_address, reward_model_tokenizer
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def run_single(self, data: DataProto) -> dict[str, Any]:
        """Decode and score one generated response."""

        data_item = data[-1]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(
            data_item.batch["attention_mask"][-response_length:].sum().item()
        )
        valid_response_ids = response_ids[:valid_response_length]
        response_text = self.tokenizer.decode(
            valid_response_ids,
            skip_special_tokens=True,
        )

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        if not isinstance(ground_truth, dict):
            raise RuntimeError("reward_model.ground_truth must be a dictionary")
        target_label = _normalize_label(ground_truth.get("label"))
        label_score = score_strict_label(response_text, target_label)
        parsed = label_score.parsed

        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        data_source = str(
            data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
        )
        content_sha256 = str(
            ground_truth.get("content_sha256")
            or extra_info.get("content_sha256")
            or ""
        )
        reward_extra_info = {
            "outcome_reward": float(label_score.reward),
            # Compatibility/logging alias only; this is not a second reward.
            "label_reward": float(label_score.reward),
            "label_correct": float(label_score.correct),
            "parse_ok": float(parsed.valid),
            "pred": label_score.predicted_label,
            "target_label": target_label,
            "source": data_source,
            "content_sha256": content_sha256,
            "analysis_chars": float(len(parsed.analysis)),
        }
        return {
            "reward_score": float(label_score.reward),
            "reward_extra_info": reward_extra_info,
        }

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Provide VERL 0.7's synchronous rule-reward interface."""

        if "rm_scores" in data.batch.keys():
            reward_tensor = data.batch["rm_scores"]
            if not return_dict:
                return reward_tensor
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {
                key: data.non_tensor_batch[key]
                for key in reward_extra_keys
                if key in data.non_tensor_batch
            }
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }

        results = [
            self.run_single(data[index : index + 1])
            for index in range(len(data))
        ]
        reward_tensor = self.assemble_rm_scores(
            data,
            [float(result["reward_score"]) for result in results],
        )
        reward_extra_info: dict[str, list[Any]] = defaultdict(list)
        for result in results:
            for key, value in result.get("reward_extra_info", {}).items():
                reward_extra_info[key].append(value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor

    @classmethod
    def assemble_rm_scores(
        cls,
        data: DataProto,
        scores: list[float],
    ) -> torch.Tensor:
        """Place one sequence outcome on each response's final valid token."""

        try:
            return terminal_outcome_tensor(
                data.batch["responses"],
                data.batch["attention_mask"],
                scores,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
