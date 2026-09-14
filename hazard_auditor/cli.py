"""Command-line interface for HazardAuditor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inference import DEFAULT_MODEL, HazardAuditor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one computer-use agent trajectory with HazardAuditor."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cutoff-len", type=int, default=16_000)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser.parse_args()


def load_trajectory(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "content" not in payload:
        raise ValueError("input JSON must be an object with a content field")
    return payload["content"]


def main() -> int:
    args = parse_args()
    try:
        trajectory = load_trajectory(args.input)
        auditor = HazardAuditor.from_pretrained(
            args.model,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        result = auditor.audit(
            trajectory,
            cutoff_len=args.cutoff_len,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.label is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
