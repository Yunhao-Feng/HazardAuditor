"""Canonical output formatting and label extraction for RLGuard."""

from __future__ import annotations

import re
from typing import Literal


GuardLabel = Literal["safe", "unsafe"]

_TAGGED_LABEL_RE = re.compile(
    r"<label>\s*(safe|unsafe|0|1)\s*</label>",
    flags=re.IGNORECASE,
)
_QWEN_GUARD_RE = re.compile(
    r"\bSafety\s*:\s*(Safe|Unsafe|Controversial)\b",
    flags=re.IGNORECASE,
)
_FINAL_BARE_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(unsafe|safe)\s*$",
    flags=re.IGNORECASE,
)
_STRUCTURAL_TAG_RE = re.compile(
    r"</?(?:analysis|label)>",
    flags=re.IGNORECASE,
)


def normalize_label(value: str) -> GuardLabel:
    normalized = value.strip().lower()
    if normalized in {"unsafe", "1", "controversial"}:
        return "unsafe"
    if normalized in {"safe", "0"}:
        return "safe"
    raise ValueError(f"unsupported guard label: {value!r}")


def extract_guard_label(text: str, *, allow_fallback: bool = True) -> GuardLabel | None:
    """Extract the final verdict, preferring the explicit RLGuard label tag.

    Fallbacks keep evaluation robust to a partially fine-tuned checkpoint that
    emits Qwen3Guard's native ``Safety: ...`` schema or a final bare label.
    Malformed/unparseable output must remain incorrect rather than defaulting
    to safe.
    """
    if not isinstance(text, str):
        return None
    tagged = _TAGGED_LABEL_RE.findall(text)
    if tagged:
        return normalize_label(tagged[-1])
    if not allow_fallback:
        return None

    native = _QWEN_GUARD_RE.findall(text)
    if native:
        return normalize_label(native[-1])
    bare = _FINAL_BARE_LABEL_RE.findall(text)
    if bare:
        return normalize_label(bare[-1])
    return None


def neutralize_target_tags(text: str) -> str:
    """Keep source rationales from injecting our assistant-output delimiters."""
    return _STRUCTURAL_TAG_RE.sub(
        lambda match: "\\u003c" + match.group(0)[1:],
        text,
    )


def format_guard_target(reason: str, label: int | str) -> str:
    if isinstance(label, int):
        if label not in (0, 1):
            raise ValueError(f"integer label must be 0 or 1, got {label!r}")
        label_text: GuardLabel = "unsafe" if label == 1 else "safe"
    else:
        label_text = normalize_label(label)
    reason = neutralize_target_tags(" ".join(reason.split()))
    return (
        f"<analysis>\n{reason}\n</analysis>\n"
        f"<label>{label_text}</label>"
    )
