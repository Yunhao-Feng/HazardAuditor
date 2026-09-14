"""RLGuard dataset adapter for VERL 0.7's asynchronous AgentLoop."""

from __future__ import annotations

from training.guardpo.cispo.prompt_truncation import right_truncate_last_user_message
from verl.utils.dataset import RLHFDataset


class RLGuardRLHFDataset(RLHFDataset):
    """Apply real right truncation to the raw chat consumed by AgentLoop."""

    def __getitem__(self, item):
        row = super().__getitem__(item)
        result = right_truncate_last_user_message(
            self.tokenizer,
            row["raw_prompt"],
            max_prompt_length=self.max_prompt_length,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        row["raw_prompt"] = result.messages

        extra_info = dict(row.get("extra_info") or {})
        extra_info["prompt_original_tokens"] = result.original_tokens
        extra_info["prompt_visible_tokens"] = result.final_tokens
        extra_info["prompt_right_truncated"] = result.truncated
        row["extra_info"] = extra_info
        return row
