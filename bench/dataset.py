from __future__ import annotations

import csv
from pathlib import Path

from pipeline import KEYS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"not an image file: {target}")
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"image path not found: {target}")
    return [
        path
        for path in sorted(target.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    """Read a labels CSV keyed by image filename.

    Required columns: file, then the nine field names used by pipeline.KEYS.
    Values are expected canonical output, not the raw print on the tag.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        missing = [name for name in ("file", *KEYS) if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

        labels: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            filename = (row.get("file") or "").strip()
            if not filename:
                continue
            if filename in labels:
                raise ValueError(f"{path}:{line_number}: duplicate row for {filename}")
            labels[filename] = {
                field: (row.get(field) or "").strip()
                for field in KEYS
            }

    if not labels:
        raise ValueError(f"{path} has no data rows")
    return labels
