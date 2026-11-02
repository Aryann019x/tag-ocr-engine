from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from .metrics import CRITICAL_FIELDS, FieldScore


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def write_report(
    out_dir: Path,
    meta: dict,
    summaries: dict[str, dict],
    scores: list[FieldScore],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "details.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "model", "file", "field", "expected", "actual", "status",
            "correct", "scored", "false_read", "unexpected", "failure",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for score in scores:
            writer.writerow(asdict(score))

    lines = [
        "# OCR benchmark",
        "",
        f"- Run: {meta['timestamp']}",
        f"- Images: {meta['image_count']} (`{meta['images_path']}`)",
        f"- Labels: `{meta['labels_path']}`",
        f"- Mode: {meta['mode']}",
        f"- Critical fields: {', '.join(CRITICAL_FIELDS)}",
        "",
        "| Model | Full row | Critical | Field | False READ | READ precision | REVIEW recall | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, summary in summaries.items():
        lines.append(
            f"| {model} | {_pct(summary['full_row_rate'])} | "
            f"{_pct(summary['critical_accuracy'])} | {_pct(summary['field_accuracy'])} | "
            f"{summary['false_reads']} | {_pct(summary['read_precision'])} | "
            f"{_pct(summary['review_recall'])} | {_seconds(summary['p50_s'])} | "
            f"{_seconds(summary['p95_s'])} |"
        )

    for model, summary in summaries.items():
        lines.extend([
            "",
            f"## {model}",
            "",
            f"- Images: {summary['images']}",
            f"- Field accuracy: {summary['fields_correct']}/{summary['fields_scored']} "
            f"({_pct(summary['field_accuracy'])})",
            f"- Critical accuracy: {summary['critical_correct']}/{summary['critical_scored']} "
            f"({_pct(summary['critical_accuracy'])})",
            f"- Full rows: {summary['full_rows']}/{summary['images']} "
            f"({_pct(summary['full_row_rate'])})",
            f"- False READ: {summary['false_reads']}",
            f"- Unexpected values on blank labels: {summary['unexpected']}",
            f"- Failures: {summary['ocr_misses']} OCR misses, "
            f"{summary['mapper_misses']} mapper misses",
            "",
            "| Field | Correct | Scored | False READ |",
            "|---|---:|---:|---:|",
        ])
        for field, stats in summary["per_field"].items():
            lines.append(
                f"| {field} | {stats['correct']} | {stats['scored']} | {stats['false_read']} |"
            )

    lines.extend([
        "",
        "## How to read this",
        "",
        "- **Full row** means every scored field on that image was correct.",
        "- **Critical** covers serial, complaint number, part code, and purchase date.",
        "- **False READ** means the field was marked `READ` but the value was wrong.",
        "- **OCR miss** means the expected value was not in the transcript at all.",
        "- **Mapper miss** means the value was in the transcript but the mapper missed it.",
        "- Small sample sizes make percentages noisy. Use reviewed tags, not the",
        "  three synthetic demo images, before drawing conclusions about a model.",
        "",
    ])

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
