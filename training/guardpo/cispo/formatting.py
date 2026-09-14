"""Strict parsing helpers for RLGuard responses and SFT references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from training.guardpo.cispo.constants import VALID_LABELS


_CANONICAL_RESPONSE_RE = re.compile(
    r"\A\s*"
    r"<analysis>\s*(?P<analysis>.*?)\s*</analysis>\s*"
    r"<label>(?P<label>safe|unsafe)</label>"
    r"\s*\Z",
    flags=re.DOTALL,
)
@dataclass(frozen=True)
class ParsedGuardResponse:
    """Result of strict parsing without permissive fallback behavior."""

    valid: bool
    analysis: str
    label: str | None
    error: str | None = None


def parse_guard_response(text: str) -> ParsedGuardResponse:
    """Parse only the canonical RLGuard response shape.

    This intentionally rejects prose fallbacks, duplicate sections, case
    variants, empty analyses, and trailing text. Training should reward the
    format that deployment will parse, not a more permissive surrogate.
    """

    if not isinstance(text, str):
        return ParsedGuardResponse(False, "", None, "response_not_string")

    match = _CANONICAL_RESPONSE_RE.fullmatch(text)
    if match is None:
        return ParsedGuardResponse(False, "", None, "noncanonical_sections")

    analysis = match.group("analysis").strip()
    label = match.group("label")
    if not analysis:
        return ParsedGuardResponse(False, "", None, "empty_analysis")
    if label not in VALID_LABELS:
        return ParsedGuardResponse(False, analysis, None, "invalid_label")

    # A nested output section inside the analysis is a duplicate/malformed
    # protocol even if the outer regular expression could otherwise consume it.
    lowered_analysis = analysis.lower()
    forbidden = ("<analysis>", "</analysis>", "<label>", "</label>")
    if any(tag in lowered_analysis for tag in forbidden):
        return ParsedGuardResponse(False, analysis, None, "nested_output_section")

    return ParsedGuardResponse(True, analysis, label)


def parse_reference_response(text: str) -> ParsedGuardResponse:
    """Parse an existing labeled reference, permitting a missing rationale.

    Some legacy validation examples have a correct binary label but no reason.
    They remain useful for label evaluation. Generated actor responses still
    go through :func:`parse_guard_response` and must contain a non-empty
    analysis.
    """

    if not isinstance(text, str):
        return ParsedGuardResponse(False, "", None, "response_not_string")
    match = _CANONICAL_RESPONSE_RE.fullmatch(text)
    if match is None:
        return ParsedGuardResponse(False, "", None, "noncanonical_sections")
    analysis = match.group("analysis").strip()
    label = match.group("label")
    lowered_analysis = analysis.lower()
    forbidden = ("<analysis>", "</analysis>", "<label>", "</label>")
    if any(tag in lowered_analysis for tag in forbidden):
        return ParsedGuardResponse(False, analysis, None, "nested_output_section")
    return ParsedGuardResponse(True, analysis, label)


def extract_untrusted_trajectory(user_prompt: str) -> str:
    """Extract the trajectory body while retaining a safe fallback.

    Right-truncated actor prompts may legitimately lose the closing marker.
    In that case the visible suffix after the opening marker is still the
    correct evidence boundary for any offline audit or diagnostic.
    """

    if not isinstance(user_prompt, str):
        return ""
    opening = "<untrusted_trajectory>"
    closing = "</untrusted_trajectory>"
    start = user_prompt.find(opening)
    if start >= 0:
        content_start = start + len(opening)
        # Use the last closing marker so a marker embedded in attacker-controlled
        # trajectory text cannot hide the rest of the actor-visible evidence.
        end = user_prompt.rfind(closing)
        if end >= content_start:
            return user_prompt[content_start:end].strip()
        return user_prompt[content_start:].strip()
    return user_prompt.strip()


def canonical_response(analysis: str, label: str) -> str:
    """Build the exact response format used in SFT and RL."""

    analysis = analysis.strip()
    if not analysis:
        raise ValueError("analysis must be non-empty")
    if label not in VALID_LABELS:
        raise ValueError(f"unsupported label: {label!r}")
    return f"<analysis>\n{analysis}\n</analysis>\n<label>{label}</label>"
