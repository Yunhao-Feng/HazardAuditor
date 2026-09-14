"""Pure deterministic reward logic, intentionally independent of VERL."""

from __future__ import annotations

from dataclasses import dataclass

from training.guardpo.cispo.constants import (
    DEFAULT_LABEL_REWARD_CORRECT,
    DEFAULT_LABEL_REWARD_MALFORMED,
    DEFAULT_LABEL_REWARD_WRONG,
    VALID_LABELS,
)
from training.guardpo.cispo.formatting import ParsedGuardResponse, parse_guard_response


@dataclass(frozen=True)
class StrictLabelScore:
    reward: float
    correct: bool
    predicted_label: str
    parsed: ParsedGuardResponse


def score_strict_label(response_text: str, target_label: str) -> StrictLabelScore:
    """Score safe and unsafe symmetrically under the canonical protocol."""

    if target_label not in VALID_LABELS:
        raise ValueError(f"invalid target label: {target_label!r}")
    parsed = parse_guard_response(response_text)
    if not parsed.valid or parsed.label is None:
        return StrictLabelScore(
            reward=DEFAULT_LABEL_REWARD_MALFORMED,
            correct=False,
            predicted_label="__invalid__",
            parsed=parsed,
        )
    if parsed.label == target_label:
        return StrictLabelScore(
            reward=DEFAULT_LABEL_REWARD_CORRECT,
            correct=True,
            predicted_label=parsed.label,
            parsed=parsed,
        )
    return StrictLabelScore(
        reward=DEFAULT_LABEL_REWARD_WRONG,
        correct=False,
        predicted_label=parsed.label,
        parsed=parsed,
    )

