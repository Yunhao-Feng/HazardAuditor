#!/usr/bin/env python3
"""Summarize VERL validation-generation dumps as binary safety metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from training.guardpo.cispo.constants import VALID_LABELS


class ValidationSummaryError(RuntimeError):
    """Raised when validation dumps are missing or inconsistent."""


@dataclass(frozen=True)
class Metrics:
    n: int
    safe: int
    unsafe: int
    accuracy: float
    safe_precision: float
    safe_recall: float
    safe_f1: float
    unsafe_precision: float
    unsafe_recall: float
    unsafe_f1: float
    macro_f1: float
    parse_rate: float


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def compute_metrics(rows: Iterable[dict[str, Any]]) -> Metrics:
    materialized = list(rows)
    if not materialized:
        raise ValidationSummaryError("cannot compute metrics for an empty row set")

    target_counts = {label: 0 for label in VALID_LABELS}
    true_positive = {label: 0 for label in VALID_LABELS}
    false_positive = {label: 0 for label in VALID_LABELS}
    false_negative = {label: 0 for label in VALID_LABELS}
    correct = 0
    parsed = 0
    for row in materialized:
        target = str(row.get("target_label", "")).strip().lower()
        prediction = str(row.get("pred", "")).strip().lower()
        if target not in VALID_LABELS:
            ground_truth = row.get("gts")
            if isinstance(ground_truth, dict):
                target = str(ground_truth.get("label", "")).strip().lower()
        if target not in VALID_LABELS:
            raise ValidationSummaryError(f"row has invalid target label: {target!r}")
        target_counts[target] += 1

        parse_ok = prediction in VALID_LABELS and float(row.get("parse_ok", 1.0)) > 0.5
        if parse_ok:
            parsed += 1
        if prediction == target and parse_ok:
            correct += 1

        for label in VALID_LABELS:
            if prediction == label and target == label and parse_ok:
                true_positive[label] += 1
            elif prediction == label and target != label and parse_ok:
                false_positive[label] += 1
            elif target == label and (prediction != label or not parse_ok):
                false_negative[label] += 1

    class_values: dict[str, tuple[float, float, float]] = {}
    for label in VALID_LABELS:
        precision = _safe_div(
            true_positive[label],
            true_positive[label] + false_positive[label],
        )
        recall = _safe_div(
            true_positive[label],
            true_positive[label] + false_negative[label],
        )
        class_values[label] = (precision, recall, _f1(precision, recall))

    n = len(materialized)
    safe_values = class_values["safe"]
    unsafe_values = class_values["unsafe"]
    return Metrics(
        n=n,
        safe=target_counts["safe"],
        unsafe=target_counts["unsafe"],
        accuracy=correct / n,
        safe_precision=safe_values[0],
        safe_recall=safe_values[1],
        safe_f1=safe_values[2],
        unsafe_precision=unsafe_values[0],
        unsafe_recall=unsafe_values[1],
        unsafe_f1=unsafe_values[2],
        macro_f1=(safe_values[2] + unsafe_values[2]) / 2.0,
        parse_rate=parsed / n,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValidationSummaryError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationSummaryError(f"{path}:{line_number}: expected object")
            rows.append(row)
    if not rows:
        raise ValidationSummaryError(f"validation dump is empty: {path}")
    return rows


def discover_step_files(path: Path) -> dict[int, Path]:
    if path.is_file():
        try:
            step = int(path.stem)
        except ValueError as exc:
            raise ValidationSummaryError(
                f"validation filename must be a numeric step: {path.name}"
            ) from exc
        return {step: path}
    if not path.is_dir():
        raise ValidationSummaryError(f"validation path does not exist: {path}")
    discovered: dict[int, Path] = {}
    for candidate in path.glob("*.jsonl"):
        try:
            step = int(candidate.stem)
        except ValueError:
            continue
        discovered[step] = candidate
    if not discovered:
        raise ValidationSummaryError(f"no numeric *.jsonl validation dumps in {path}")
    return dict(sorted(discovered.items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Metrics]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source", "unknown"))
        by_source[source].append(row)
    output = {
        source: compute_metrics(source_rows)
        for source, source_rows in sorted(by_source.items())
    }
    output["Overall"] = compute_metrics(rows)
    return output


def _format_table(summary: dict[str, Metrics]) -> str:
    header = (
        "| Dataset | N | Safe | Unsafe | ACC | P-safe | R-safe | F1-safe | "
        "P-unsafe | R-unsafe | F1-unsafe | Macro-F1 | Parse |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for source, metrics in summary.items():
        lines.append(
            f"| {source} | {metrics.n} | {metrics.safe} | {metrics.unsafe} | "
            f"{metrics.accuracy:.4f} | {metrics.safe_precision:.4f} | "
            f"{metrics.safe_recall:.4f} | {metrics.safe_f1:.4f} | "
            f"{metrics.unsafe_precision:.4f} | {metrics.unsafe_recall:.4f} | "
            f"{metrics.unsafe_f1:.4f} | {metrics.macro_f1:.4f} | "
            f"{metrics.parse_rate:.4f} |"
        )
    return "\n".join(lines)


def selection_key(summary: dict[str, Metrics]) -> tuple[float, float, float]:
    """Return the label-first lexicographic checkpoint-selection key."""

    overall = summary["Overall"]
    source_metrics = [
        metrics
        for source, metrics in summary.items()
        if source != "Overall"
    ]
    worst_source_macro_f1 = min(metrics.macro_f1 for metrics in source_metrics)
    min_class_recall = min(overall.safe_recall, overall.unsafe_recall)
    return (
        overall.macro_f1,
        worst_source_macro_f1,
        min_class_recall,
    )


# Backward-compatible private alias for callers written before retention was
# automated. New code should import selection_key.
_selection_key = selection_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "validation_path",
        type=Path,
        nargs="?",
        default=Path("artifacts/checkpoints/guardpo/validation_generations"),
    )
    parser.add_argument(
        "--step",
        default="latest",
        help="numeric step, 'latest', or 'best'",
    )
    parser.add_argument("--all-steps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        step_files = discover_step_files(args.validation_path.expanduser().resolve())
        summaries = {
            step: summarize_rows(_load_jsonl(path))
            for step, path in step_files.items()
        }
        if args.all_steps:
            print("| Step | Overall Macro-F1 | Worst-source Macro-F1 | Min class recall |")
            print("|---:|---:|---:|---:|")
            for step, summary in summaries.items():
                key = selection_key(summary)
                print(
                    f"| {step} | {key[0]:.4f} | {key[1]:.4f} | "
                    f"{key[2]:.4f} |"
                )
            return 0

        if args.step == "latest":
            selected_step = max(summaries)
        elif args.step == "best":
            selected_step = max(summaries, key=lambda step: selection_key(summaries[step]))
        else:
            selected_step = int(args.step)
            if selected_step not in summaries:
                raise ValidationSummaryError(
                    f"step {selected_step} is unavailable; have {sorted(summaries)}"
                )
        print(f"Validation step: {selected_step}")
        print(_format_table(summaries[selected_step]))
        print(
            "Checkpoint key:",
            tuple(round(value, 6) for value in selection_key(summaries[selected_step])),
        )
    except (ValidationSummaryError, ValueError) as exc:
        raise SystemExit(f"validation summary failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
