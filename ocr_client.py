"""Minimal Ollama client and OCR helpers used by pipeline.py.

Only the Ollama HTTP API is used, so there is no dependency on a model
implementation or on any vendor SDK.  Set OLLAMA_URL if your server lives
somewhere other than the default local port.
"""
from __future__ import annotations

import os
import re

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")

OCR_PROMPT = (
    "Text Recognition: Transcribe ALL printed labels and handwritten values "
    "line by line in reading order, including serial and complaint numbers. "
    "Plain lines only, no JSON, no markdown, stop after one pass."
)

# Words that reliably appear on a correctly oriented form.  The score picks
# the winning rotation when two candidate orientations are OCR'd.
KEYWORDS = [
    "branch", "bccd", "product", "serial", "purchase", "complaint",
    "spare", "defect", "technician", "n30",
]


def orientation_score(text: str) -> int:
    """Higher is better: keyword hits plus a small digit-density bonus."""
    t = (text or "").lower()
    return sum(1 for k in KEYWORDS if k in t) + len(re.findall(r"\d", t)) // 5
