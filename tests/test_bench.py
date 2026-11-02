from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from bench.adapters import Scan
from bench.dataset import collect_images, load_labels
from bench.metrics import (
    appears_in_text,
    field_matches,
    score_scan,
    summarize,
)
from pipeline import KEYS


def expected_row() -> dict[str, str]:
    return {
        "Branch": "Testnagar",
        "BCCD Name": "Demo Electronics",
        "Product Description": "MAIN PCB ASSY DEMO 100",
        "Product Sr. No.": "N304123456789",
        "Date of Purchase": "12/03/2024",
        "Complaint No.": "TST12057605796344-1",
        "Spare Part Code": "100001",
        "Nature Of Defect": "Dead",
        "Technician Name": "Alex Demo",
    }


def scan_from(values: dict[str, str], statuses: dict[str, str] | None = None, raw: str = "") -> Scan:
    return Scan(
        values=values,
        statuses=statuses or {field: "READ" for field in KEYS},
        raw_ocr=raw,
        elapsed=1.5,
    )


class MatchTests(unittest.TestCase):
    def test_date_accepts_two_digit_year(self):
        self.assertTrue(field_matches("Date of Purchase", "12/03/2024", "12/03/24"))

    def test_serial_ignores_separators(self):
        self.assertTrue(field_matches("Product Sr. No.", "N304123456789", "N304-123-456-789"))

    def test_text_is_case_and_space_insensitive(self):
        self.assertTrue(field_matches("Branch", "Testnagar", "  testnagar "))

    def test_wrong_value_does_not_match(self):
        self.assertFalse(field_matches("Branch", "Testnagar", "Demopur"))

    def test_value_in_transcript(self):
        raw = "Product Sr.No. : N304 123 456 789\nBranch : Testnagar"
        self.assertTrue(appears_in_text("Product Sr. No.", "N304123456789", raw))
        self.assertTrue(appears_in_text("Branch", "Testnagar", raw))

    def test_date_in_transcript(self):
        raw = "Date Of Purchase : 12/03/24"
        self.assertTrue(appears_in_text("Date of Purchase", "12/03/2024", raw))


class ScoreTests(unittest.TestCase):
    def test_false_read_is_counted(self):
        expected = expected_row()
        values = dict(expected)
        values["Complaint No."] = "TST12057605796344-2"
        scan = scan_from(values, raw="Branch : Testnagar")
        scores = score_scan("demo", "tag.jpg", scan, expected)
        summary = summarize(scores, 1, [scan.elapsed])

        self.assertEqual(summary["false_reads"], 1)
        self.assertEqual(summary["fields_correct"], 8)
        self.assertEqual(summary["critical_correct"], 3)
        self.assertEqual(summary["ocr_misses"], 1)
        self.assertEqual(summary["full_rows"], 0)

    def test_mapper_miss_when_value_is_in_transcript(self):
        expected = expected_row()
        values = dict(expected)
        values["Product Sr. No."] = ""
        scan = scan_from(values, raw="Product Sr.No. : N304 123 456 789")
        scores = score_scan("demo", "tag.jpg", scan, expected)
        summary = summarize(scores, 1, [scan.elapsed])

        self.assertEqual(summary["mapper_misses"], 1)
        self.assertEqual(summary["ocr_misses"], 0)

    def test_review_is_not_a_false_read(self):
        expected = expected_row()
        values = dict(expected)
        values["Branch"] = "Demopur"
        statuses = {field: "READ" for field in KEYS}
        statuses["Branch"] = "REVIEW"
        scan = scan_from(values, statuses=statuses, raw="Branch : Demopur")
        scores = score_scan("demo", "tag.jpg", scan, expected)
        summary = summarize(scores, 1, [scan.elapsed])

        self.assertEqual(summary["false_reads"], 0)
        self.assertEqual(summary["review_recall"], 1.0)

    def test_unexpected_value_on_blank_label(self):
        expected = expected_row()
        expected["Product Description"] = ""
        values = dict(expected)
        values["Product Description"] = "MAIN PCB ASSY DEMO 100"
        scan = scan_from(values)
        scores = score_scan("demo", "tag.jpg", scan, expected)
        summary = summarize(scores, 1, [scan.elapsed])

        self.assertEqual(summary["unexpected"], 1)
        self.assertEqual(summary["fields_scored"], 8)
        self.assertEqual(summary["full_rows"], 0)


class DatasetTests(unittest.TestCase):
    def test_load_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file", *KEYS])
                writer.writerow(["tag.jpg", *[expected_row()[field] for field in KEYS]])

            labels = load_labels(path)
            self.assertEqual(labels["tag.jpg"]["Branch"], "Testnagar")
            self.assertEqual(labels["tag.jpg"]["Spare Part Code"], "100001")

    def test_missing_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("file,Branch\n tag.jpg,Testnagar\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_collect_images_skips_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"x")
            (root / "b.txt").write_text("x", encoding="utf-8")
            images = collect_images(root)
            self.assertEqual([path.name for path in images], ["a.jpg"])

    def test_single_non_image_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                collect_images(path)


if __name__ == "__main__":
    unittest.main()
