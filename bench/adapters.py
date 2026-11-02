from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pipeline


@dataclass(frozen=True)
class Scan:
    values: dict[str, str]
    statuses: dict[str, str]
    raw_ocr: str
    elapsed: float


class OllamaAdapter:
    """Run one Ollama vision model through the normal scan pipeline."""

    def __init__(self, model: str):
        self.model = model

    def scan(self, image: Path) -> Scan:
        result = pipeline.scan_and_resolve(image, model=self.model)
        fields = result.get("fields") or {}
        values = {key: str((fields.get(key) or {}).get("value") or "") for key in pipeline.KEYS}
        statuses = {
            key: str((fields.get(key) or {}).get("status") or ("READ" if values[key] else "UNREADABLE"))
            for key in pipeline.KEYS
        }
        return Scan(
            values=values,
            statuses=statuses,
            raw_ocr=str(result.get("raw_ocr") or ""),
            elapsed=float(result.get("elapsed") or 0.0),
        )
