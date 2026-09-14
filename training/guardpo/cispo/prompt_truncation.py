"""Token-budgeted right truncation for VERL async raw chat prompts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class PromptTruncationError(RuntimeError):
    """Raised when a prompt cannot fit without damaging fixed instructions."""


@dataclass(frozen=True)
class PromptTruncationResult:
    messages: list[dict[str, Any]]
    original_tokens: int
    final_tokens: int
    truncated: bool


def _chat_token_count(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    apply_chat_template_kwargs: Mapping[str, Any],
) -> int:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=True,
        **dict(apply_chat_template_kwargs),
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if encoded is None:
        raise PromptTruncationError("chat template returned no input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise PromptTruncationError("chat template returned a batched prompt")
        encoded = encoded[0]
    try:
        return len(encoded)
    except TypeError as exc:
        raise PromptTruncationError(
            f"chat template returned unsupported input_ids type {type(encoded).__name__}"
        ) from exc


def right_truncate_last_user_message(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_prompt_length: int,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> PromptTruncationResult:
    """Keep fixed instructions and remove only the trajectory's trailing text.

    VERL 0.7's async AgentLoop re-applies the chat template to ``raw_prompt``
    and bypasses ``data.truncation``. Returning a shortened, still-valid chat
    message makes the AgentLoop and actor observe the same bounded prefix.
    """

    if max_prompt_length < 1:
        raise PromptTruncationError("max_prompt_length must be positive")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise PromptTruncationError("messages must be a sequence of mappings")

    template_kwargs = dict(apply_chat_template_kwargs or {})
    copied = deepcopy(list(messages))
    original_tokens = _chat_token_count(tokenizer, copied, template_kwargs)
    if original_tokens <= max_prompt_length:
        return PromptTruncationResult(
            messages=copied,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            truncated=False,
        )

    user_index = None
    for index in range(len(copied) - 1, -1, -1):
        message = copied[index]
        if isinstance(message, Mapping) and message.get("role") == "user":
            user_index = index
            break
    if user_index is None:
        raise PromptTruncationError("overlong prompt has no user message to truncate")

    user_message = copied[user_index]
    content = user_message.get("content")
    if not isinstance(content, str):
        raise PromptTruncationError(
            "overlong prompt's final user content is not plain text"
        )

    # Search by Unicode character prefix. Every accepted candidate is measured
    # through the exact tokenizer chat template, so the returned prompt is
    # guaranteed to respect the token budget even though BPE token counts are
    # not perfectly linear in character count.
    lower = 0
    upper = len(content)
    best_content: str | None = None
    best_tokens: int | None = None
    while lower <= upper:
        midpoint = (lower + upper) // 2
        user_message["content"] = content[:midpoint]
        candidate_tokens = _chat_token_count(tokenizer, copied, template_kwargs)
        if candidate_tokens <= max_prompt_length:
            best_content = content[:midpoint]
            best_tokens = candidate_tokens
            lower = midpoint + 1
        else:
            upper = midpoint - 1

    if best_content is None or best_tokens is None:
        user_message["content"] = ""
        minimum_tokens = _chat_token_count(tokenizer, copied, template_kwargs)
        raise PromptTruncationError(
            "fixed system/chat-template tokens exceed max_prompt_length: "
            f"minimum={minimum_tokens}, limit={max_prompt_length}"
        )

    user_message["content"] = best_content
    final_tokens = _chat_token_count(tokenizer, copied, template_kwargs)
    if final_tokens > max_prompt_length:
        raise PromptTruncationError(
            f"internal truncation error: final={final_tokens}, limit={max_prompt_length}"
        )
    return PromptTruncationResult(
        messages=copied,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        truncated=True,
    )
