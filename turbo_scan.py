#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast & Furious CLI Batch & Watch-Folder Scanner for TAG Spare Part Tags.

Modes:
1. Default (All Samples):
     python turbo_scan.py
2. Batch Folder / Files -> CSV + JSONL:
     python turbo_scan.py --input path/to/folder --csv results.csv --jsonl results.jsonl
3. Watch Folder ("Scan & Go" Live Daemon):
     python turbo_scan.py --watch path/to/drop_folder --csv live_scans.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV_COLUMNS = [
    "file",
    "elapsed_s",
    "passes",
    "angle",
    "read_count",
    "filled_count",
    "Branch",
    "BCCD Name",
    "Spare Part Code",
    "Product Description",
    "Product Sr. No.",
    "Date of Purchase",
    "Complaint No.",
    "Nature Of Defect",
    "Technician Name",
]


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [
        p
        for p in sorted(target.glob("*.*"))
        if p.is_file() and p.suffix.lower() in exts and "Blank_Test" not in p.name
    ]


def append_csv(csv_path: Path, row_dict: dict) -> None:
    exists = csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in CSV_COLUMNS})


def append_jsonl(jsonl_path: Path, payload: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_row_summary(img_path: Path, res: dict) -> dict:
    flat = res["result"]
    return {
        "file": img_path.name,
        "elapsed_s": res["elapsed"],
        "passes": res["passes"],
        "angle": res["angle"],
        "read_count": res.get("read_count", 0),
        "filled_count": res.get("filled_count", 0),
        **flat,
    }


def print_scan_card(idx: int, total: int, img_path: Path, res: dict) -> None:
    t = res.get("timings", {})
    print(
        f"\n[{idx}/{total}] ⚡ {img_path.name} — {res['elapsed']:.2f}s "
        f"(vision {t.get('prefill_s', 0):.2f}s + decode {t.get('decode_s', 0):.2f}s | "
        f"{res['passes']}-pass @ {res['angle']}° | READ {res.get('read_count', 0)}/9)"
    )
    for k, v in res["fields"].items():
        val = v.get("value") or "—"
        st = v.get("status", "READ")
        conf = v.get("confidence", "")
        conf_tag = f" ({conf})" if conf and conf not in ("EXACT", "HIGH") else ""
        cands = [c.get("canonical") or c.get("code") for c in v.get("candidates", []) if (c.get("canonical") or c.get("code")) != v.get("value")]
        cand_str = f"  alt={cands[:2]}" if cands else ""
        print(f"   {k:20s} : {val:36s} [{st:10s}]{conf_tag}{cand_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", type=Path, default=Path(__file__).parent / "samples", help="Image file or folder to scan")
    ap.add_argument("--watch", "-w", type=Path, help="Watch a folder continuously and auto-scan new images as they arrive")
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "scan_results.csv", help="Output CSV path")
    ap.add_argument("--jsonl", type=Path, help="Optional output JSONL path")
    ap.add_argument("--model", default=pipeline.DEFAULT_MODEL, help="Ollama OCR model (default: glm-ocr:q8_0)")
    ap.add_argument("--dual", action="store_true", help="Force 2-rotation pass on every image")
    args = ap.parse_args()

    print(f"🚀 Preloading {args.model} onto GPU (num_ctx={pipeline.NUM_CTX}, 896x560 patch grid)...")
    warm = pipeline.start_warmup(args.model, blocking=True)
    print(f"✅ {warm.get('detail')} in {warm.get('seconds')}s")

    if args.watch:
        watch_dir = args.watch
        watch_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        print(f"👀 Watching {watch_dir.resolve()} for new tag images... (Ctrl+C to stop)")
        try:
            while True:
                for img in collect_images(watch_dir):
                    key = f"{img.name}:{img.stat().st_mtime_ns}"
                    if key in seen:
                        continue
                    seen.add(key)
                    res = pipeline.scan_and_resolve(img, model=args.model, dual_rotation=args.dual)
                    print_scan_card(len(seen), len(seen), img, res)
                    row = format_row_summary(img, res)
                    if args.csv:
                        append_csv(args.csv, row)
                    if args.jsonl:
                        append_jsonl(args.jsonl, {"file": img.name, **res})
                time.sleep(0.4)
        except KeyboardInterrupt:
            print("\nStopped watch mode.")
            return 0

    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return 1

    if args.csv and args.csv.is_file():
        args.csv.unlink()
    if args.jsonl and args.jsonl.is_file():
        args.jsonl.unlink()

    t_batch = time.perf_counter()
    total_read = 0
    total_filled = 0
    elapsed_list = []

    for idx, img in enumerate(images, 1):
        res = pipeline.scan_and_resolve(img, model=args.model, dual_rotation=args.dual)
        elapsed_list.append(res["elapsed"])
        total_read += res.get("read_count", 0)
        total_filled += res.get("filled_count", 0)
        print_scan_card(idx, len(images), img, res)
        row = format_row_summary(img, res)
        if args.csv:
            append_csv(args.csv, row)
        if args.jsonl:
            append_jsonl(args.jsonl, {"file": img.name, **res})

    wall = time.perf_counter() - t_batch
    avg_s = sum(elapsed_list) / len(elapsed_list)
    warm_list = elapsed_list[1:] if len(elapsed_list) > 1 else elapsed_list
    warm_avg = sum(warm_list) / len(warm_list)

    print("\n" + "=" * 78)
    print(
        f"🏁 TURBO BATCH COMPLETE: {len(images)} tags in {wall:.2f}s | "
        f"Avg: {avg_s:.2f}s/tag (Warm Avg: {warm_avg:.2f}s/tag)"
    )
    print(
        f"📊 EXTRACTION SCORE   : {total_read}/{total_filled} filled fields verified READ "
        f"({100.0 * total_read / max(total_filled, 1):.1f}% READ confidence across {len(images) * 9} total slots)"
    )
    if args.csv:
        print(f"💾 Saved CSV          : {args.csv}")
    if args.jsonl:
        print(f"💾 Saved JSONL        : {args.jsonl}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
