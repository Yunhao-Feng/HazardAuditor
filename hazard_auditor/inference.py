"""Deterministic local inference for the generative HazardAuditor model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .output import GuardLabel, extract_guard_analysis, extract_guard_label
from .prompting import build_messages


DEFAULT_MODEL = "Yunhao-Feng/HazardAuditor"


@dataclass(frozen=True)
class AuditResult:
    analysis: str | None
    label: GuardLabel | None
    raw_output: str
    prompt_tokens: int
    generated_tokens: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HazardAuditor:
    """Trajectory-level safety auditor backed by a generative checkpoint."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_MODEL,
        *,
        dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
    ) -> "HazardAuditor":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Inference requires torch, transformers, and accelerate."
            ) from exc

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                resolved_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                resolved_dtype = torch.float16
            else:
                resolved_dtype = torch.float32
        else:
            resolved_dtype = dtype_map[dtype]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has no EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=resolved_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        ).eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        return cls(model=model, tokenizer=tokenizer)

    def audit(
        self,
        trajectory: Any,
        *,
        cutoff_len: int = 16_000,
        max_new_tokens: int = 384,
    ) -> AuditResult:
        if cutoff_len <= 0 or max_new_tokens <= 0:
            raise ValueError("cutoff_len and max_new_tokens must be positive")

        import torch

        encoded = self.tokenizer.apply_chat_template(
            build_messages(trajectory),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        original_length = int(input_ids.shape[-1])
        truncated = original_length > cutoff_len
        if truncated:
            input_ids = input_ids[:, :cutoff_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :cutoff_len]

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        generation_inputs = {"input_ids": input_ids.to(device)}
        if attention_mask is not None:
            generation_inputs["attention_mask"] = attention_mask.to(device)

        eos_ids = [self.tokenizer.eos_token_id]
        extra_eos = getattr(self.tokenizer, "additional_special_tokens_ids", None)
        if isinstance(extra_eos, list):
            eos_ids.extend(value for value in extra_eos if type(value) is int)
        eos_ids = list(dict.fromkeys(value for value in eos_ids if value >= 0))

        prompt_length = generation_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **generation_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
        output_ids = generated[0, prompt_length:].detach().cpu().tolist()
        while output_ids and output_ids[-1] == self.tokenizer.pad_token_id:
            output_ids.pop()
        raw_output = self.tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip()
        return AuditResult(
            analysis=extract_guard_analysis(raw_output),
            label=extract_guard_label(raw_output, allow_fallback=True),
            raw_output=raw_output,
            prompt_tokens=min(original_length, cutoff_len),
            generated_tokens=len(output_ids),
            truncated=truncated,
        )
