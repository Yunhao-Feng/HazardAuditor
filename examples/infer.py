#!/usr/bin/env python3
"""Minimal HazardAuditor inference example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hazard_auditor import HazardAuditor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Yunhao-Feng/HazardAuditor")
    parser.add_argument(
        "--input", type=Path, default=Path(__file__).with_name("trajectory.json")
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    trajectory = json.loads(args.input.read_text(encoding="utf-8"))["content"]
    auditor = HazardAuditor.from_pretrained(
        args.model, attn_implementation=args.attn_implementation
    )
    print(json.dumps(auditor.audit(trajectory).to_dict(), indent=2))


if __name__ == "__main__":
    main()
