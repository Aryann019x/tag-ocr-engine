from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re

from pipeline import KEYS

CRITICAL_FIELDS = (
    "Product Sr. No.",
    "Complaint No.",
    "Spare Part Code",
    "Date of Purchase",
)
ID_FIELDS = frozenset({"Product Sr. No.", "Complaint No.", "Spare Part Code"})
DATE_FIELD = "Date of Purchase"

_DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%Y-%m-%d",
)


@dataclass(frozen=True)
class FieldScore:
    model: str
    file: str
    field: str
    expected: str
    actual: str
    status: str
    correct: bool
    scored: bool
    false_read: bool
    unexpected: bool
    failure: str


def _text(value: str) -> str:
    spaced = re.sub(r"[^\w]+", " ", (value or "").strip(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", spaced).strip().casefold()


def _compact(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value or "").upper()


def _parse_date(value: str) -> str | None:
    text = (value or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def field_matches(field: str, expected: str, actual: str) -> bool:
    expected = (expected or "").strip()
    actual = (actual or "").strip()
    if not expected:
        return not actual
    if not actual:
        return False
    if field == DATE_FIELD:
        expected_date = _parse_date(expected)
        actual_date = _parse_date(actual)
        if expected_date and actual_date:
            return expected_date == actual_date
    if field in ID_FIELDS:
        token = _compact(expected)
        return bool(token) and token == _compact(actual)
    return _text(expected) == _text(actual)


def appears_in_text(field: str, expected: str, raw_text: str) -> bool:
    """True when the expected value is in the transcript, even if mapping missed it."""
    expected = (expected or "").strip()
    raw_text = raw_text or ""
    if not expected or not raw_text:
        return False
    if field == DATE_FIELD:
        parsed = _parse_date(expected)
        if parsed:
            year, month, day = parsed.split("-")
            raw_compact = _compact(raw_text)
            variants = {
                f"{day}{month}{year}",
                f"{day}{month}{year[2:]}",
                f"{year}{month}{day}",
            }
            return any(variant in raw_compact for variant in variants)
    if field in ID_FIELDS:
        token = _compact(expected)
        return bool(token) and token in _compact(raw_text)
    return _text(expected) in _text(raw_text)


def score_scan(model: str, file: str, scan, expected: dict[str, str]) -> list[FieldScore]:
    scores: list[FieldScore] = []
    for field in KEYS:
        want = (expected.get(field) or "").strip()
        got = (scan.values.get(field) or "").strip()
        status = (scan.statuses.get(field) or "").strip().upper()
        if not status:
            status = "READ" if got else "UNREADABLE"

        if not want:
            scores.append(FieldScore(
                model=model,
                file=file,
                field=field,
                expected="",
                actual=got,
                status=status,
                correct=False,
                scored=False,
                false_read=False,
                unexpected=bool(got),
                failure="",
            ))
            continue

        correct = field_matches(field, want, got)
        false_read = status == "READ" and not correct
        if correct:
            failure = ""
        elif appears_in_text(field, want, scan.raw_ocr):
            failure = "mapper_miss"
        else:
            failure = "ocr_miss"

        scores.append(FieldScore(
            model=model,
            file=file,
            field=field,
            expected=want,
            actual=got,
            status=status,
            correct=correct,
            scored=True,
            false_read=false_read,
            unexpected=False,
            failure=failure,
        ))
    return scores


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[index]


def summarize(scores: list[FieldScore], image_count: int, latencies: list[float]) -> dict:
    scored = [score for score in scores if score.scored]
    correct = [score for score in scored if score.correct]
    critical = [score for score in scored if score.field in CRITICAL_FIELDS]
    critical_correct = [score for score in critical if score.correct]
    read = [score for score in scored if score.status == "READ"]
    read_correct = [score for score in read if score.correct]
    wrong = [score for score in scored if not score.correct]
    caught = [score for score in wrong if score.status in {"REVIEW", "UNREADABLE"}]

    files = sorted({score.file for score in scores})
    full_rows = 0
    for file in files:
        file_scores = [score for score in scores if score.file == file]
        file_scored = [score for score in file_scores if score.scored]
        if (
            file_scored
            and all(score.correct for score in file_scored)
            and not any(score.unexpected for score in file_scores)
        ):
            full_rows += 1

    per_field = {}
    for field in KEYS:
        field_scores = [score for score in scored if score.field == field]
        per_field[field] = {
            "scored": len(field_scores),
            "correct": sum(1 for score in field_scores if score.correct),
            "false_read": sum(1 for score in field_scores if score.false_read),
        }

    return {
        "images": image_count,
        "files": len(files),
        "fields_scored": len(scored),
        "fields_correct": len(correct),
        "field_accuracy": len(correct) / len(scored) if scored else None,
        "critical_scored": len(critical),
        "critical_correct": len(critical_correct),
        "critical_accuracy": len(critical_correct) / len(critical) if critical else None,
        "full_rows": full_rows,
        "full_row_rate": full_rows / image_count if image_count else None,
        "false_reads": sum(1 for score in scored if score.false_read),
        "read_count": len(read),
        "read_precision": len(read_correct) / len(read) if read else None,
        "wrong_count": len(wrong),
        "wrong_caught": len(caught),
        "review_recall": len(caught) / len(wrong) if wrong else None,
        "unexpected": sum(1 for score in scores if score.unexpected),
        "ocr_misses": sum(1 for score in scored if score.failure == "ocr_miss"),
        "mapper_misses": sum(1 for score in scored if score.failure == "mapper_miss"),
        "per_field": per_field,
        "p50_s": _percentile(latencies, 50),
        "p95_s": _percentile(latencies, 95),
    }
