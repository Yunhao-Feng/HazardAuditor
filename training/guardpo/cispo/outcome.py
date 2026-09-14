"""Pure batch-centered outcome advantage used by RLGuard CISPO."""

from __future__ import annotations

import torch


def terminal_outcome_tensor(
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    scores: list[float],
) -> torch.Tensor:
    """Encode one scalar reward at each response's final valid token."""

    if responses.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("responses and attention mask must be rank 2")
    if attention_mask.shape[0] != responses.shape[0]:
        raise ValueError("responses and attention mask batch sizes differ")
    if attention_mask.shape[1] < responses.shape[1]:
        raise ValueError("attention mask is shorter than responses")
    if responses.shape[1] < 1:
        raise ValueError("response tensor must reserve at least one token position")
    if len(scores) != responses.shape[0]:
        raise ValueError(
            f"reward count {len(scores)} does not match batch {responses.shape[0]}"
        )

    reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
    response_length = responses.shape[-1]
    response_mask = attention_mask[:, -response_length:].bool()
    valid_lengths = response_mask.sum(dim=-1)
    for row, score in enumerate(scores):
        # An empty generation has no policy token to reinforce. Position 0
        # preserves the scalar in VERL's unmasked reward logs, while the
        # response mask prevents it from creating a policy gradient.
        final_position = max(int(valid_lengths[row].item()) - 1, 0)
        reward_tensor[row, final_position] = float(score)
    return reward_tensor


def batch_centered_outcome_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center one sequence reward over the batch and broadcast to all tokens.

    This is the direct analogue of CRL's Experience Extractor estimator
    ``A_i = r_i - mean(r)``. Rewards are summed over each response because VERL
    stores a terminal sequence outcome in a token-shaped tensor. The same
    centered scalar is then assigned to every valid token in that response.

    Returns ``(token_advantages, sequence_rewards, batch_mean_reward)``.
    """

    if token_level_rewards.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("reward and response mask tensors must be rank 2")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            "reward and response mask shapes differ: "
            f"{token_level_rewards.shape} vs {response_mask.shape}"
        )
    if token_level_rewards.shape[0] == 0:
        raise ValueError("cannot center rewards over an empty batch")

    with torch.no_grad():
        sequence_rewards = token_level_rewards.float().sum(dim=-1)
        batch_mean_reward = sequence_rewards.mean()
        sequence_advantages = sequence_rewards - batch_mean_reward
        token_advantages = (
            sequence_advantages.unsqueeze(-1)
            * response_mask.to(dtype=sequence_advantages.dtype)
        )
    return token_advantages, sequence_rewards, batch_mean_reward
