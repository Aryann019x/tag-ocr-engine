from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import config
import pipeline

from .adapters import OllamaAdapter
from .dataset import collect_images, load_labels
from .metrics import score_scan, summarize
from .report import write_report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Score OCR models against reviewed tag labels.",
    )
    parser.add_argument("--images", type=Path, default=Path("samples"), help="image file or folder")
    parser.add_argument("--labels", type=Path, required=True, help="labels CSV")
    parser.add_argument(
        "--models",
        default=config.DEFAULT_MODEL,
        help=f"comma-separated Ollama model tags (default: {config.DEFAULT_MODEL})",
    )
    parser.add_argument("--out", type=Path, help="output folder (default: reports/<UTC timestamp>)")
    parser.add_argument("--skip-warmup", action="store_true", help="do not preload each model")
    parser.add_argument("--save-raw", action="store_true", help="save raw OCR transcripts")
    parser.add_argument("--limit", type=int, help="score only the first N images")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)

    try:
        images = collect_images(args.images)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.limit is not None:
        if args.limit < 1:
            print("error: --limit must be at least 1", file=sys.stderr)
            return 1
        images = images[:args.limit]
    if not images:
        print(f"error: no images found in {args.images}", file=sys.stderr)
        return 1

    try:
        labels = load_labels(args.labels)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    missing = [image.name for image in images if image.name not in labels]
    if missing:
        print(f"error: no label row for: {', '.join(missing)}", file=sys.stderr)
        return 1

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        print("error: --models is empty", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    out_dir = args.out or (Path("reports") / now.strftime("%Y%m%dT%H%M%SZ"))
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")

    all_scores = []
    summaries: dict[str, dict] = {}

    for model in models:
        adapter = OllamaAdapter(model)
        if not args.skip_warmup:
            print(f"warming {model} ...", flush=True)
            state = pipeline.start_warmup(model, blocking=True)
            if state.get("state") != "ready":
                print(f"  warning: {state.get('detail') or state.get('state')}", flush=True)

        model_scores = []
        latencies: list[float] = []
        for index, image in enumerate(images, start=1):
            try:
                scan = adapter.scan(image)
            except Exception as exc:
                print(f"\n{model}: {image.name} failed: {exc}", file=sys.stderr)
                return 1

            latencies.append(scan.elapsed)
            model_scores.extend(score_scan(model, image.name, scan, labels[image.name]))
            print(f"[{model}] {index}/{len(images)} {image.name} {scan.elapsed:.2f}s", flush=True)

            if args.save_raw:
                raw_dir = out_dir / "raw" / _model_slug(model)
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{image.stem}.txt").write_text(scan.raw_ocr, encoding="utf-8")

        summaries[model] = summarize(model_scores, len(images), latencies)
        all_scores.extend(model_scores)

    summary_path = write_report(
        out_dir,
        {
            "timestamp": timestamp,
            "image_count": len(images),
            "labels_path": args.labels.as_posix(),
            "images_path": args.images.as_posix(),
            "mode": "end-to-end",
        },
        summaries,
        all_scores,
    )

    print()
    print(f"{'model':30} {'full':>7} {'critical':>9} {'field':>7} {'falseREAD':>10} {'p50':>7}")
    for model, summary in summaries.items():
        print(
            f"{model:30} {_pct(summary['full_row_rate']):>7} "
            f"{_pct(summary['critical_accuracy']):>9} {_pct(summary['field_accuracy']):>7} "
            f"{summary['false_reads']:>10} {_seconds(summary['p50_s']):>7}"
        )
    print()
    print(f"report: {summary_path}")
    return 0
