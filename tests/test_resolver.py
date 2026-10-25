"""GPU-free regression tests for the entity resolver.

These tests cover the deterministic part of the engine: normalization,
validators, closed-set part codes and the resolver guards. They do not load
OpenCV, Ollama or any model, so they run in CI in a couple of seconds.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

from lookup import EntityResolver


class ResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.r = EntityResolver()

    def assertRead(self, info: dict) -> None:
        self.assertEqual(info.get("status"), "READ", info)

    # ---------------------------------------------------------------- Branch
    def test_branch_case_and_spacing(self) -> None:
        for raw in ("Testnagar", "TESTNAGAR", "test nagar"):
            info = self.r.resolve_branch(raw)
            self.assertRead(info)
            self.assertEqual(info["value"], "Testnagar")

    def test_dealer_is_never_a_branch(self) -> None:
        info = self.r.resolve_branch("Demo Electronics")
        self.assertNotEqual(info.get("status"), "READ", info)

    # ------------------------------------------------------------------ BCCD
    def test_bccd_case_and_typo(self) -> None:
        for raw in ("Demo Electronics", "DEMO ELECTRONICS", "Demo Electronic"):
            info = self.r.resolve_bccd(raw, branch="Testnagar")
            self.assertRead(info)
            self.assertEqual(info["value"], "Demo Electronics")

    def test_placeholder_bccd_is_never_output(self) -> None:
        info = self.r.resolve_bccd("BCCD-001", branch="Testnagar")
        self.assertNotEqual(info.get("status"), "READ", info)
        self.assertEqual(info.get("value"), "")

    def test_unknown_script_is_not_read_as_a_dealer(self) -> None:
        info = self.r.resolve_bccd("শিবম ইলেকট্রনিক্স", branch="Testnagar")
        self.assertNotEqual(info.get("status"), "READ", info)

    # ------------------------------------------------------------- Part code
    def test_part_code_closed_set(self) -> None:
        info = self.r.resolve_part_code("100001")
        self.assertRead(info)
        self.assertEqual(info["value"], "100001")
        self.assertIn("DEMO", info.get("derived_description", "").upper())

    def test_unknown_part_code_is_not_invented(self) -> None:
        info = self.r.resolve_part_code("999999")
        self.assertNotEqual(info.get("status"), "READ", info)

    # ----------------------------------------------------------------- Serial
    def test_serial_normalization(self) -> None:
        info = self.r.validate_serial("N304123456789")
        self.assertRead(info)
        self.assertEqual(info["value"], "N304123456789")

    def test_serial_prefix_repair(self) -> None:
        info = self.r.validate_serial("N804123456789")
        self.assertRead(info)
        self.assertEqual(info["value"], "N304123456789")

    def test_serial_regroups_only_when_separators_exist(self) -> None:
        info = self.r.validate_serial("N304 0285 90742")
        self.assertRead(info)
        self.assertEqual(info["value"], "N304-0285-90742")

        info = self.r.validate_serial("N304028590742")
        self.assertRead(info)
        self.assertEqual(info["value"], "N304028590742")

    # -------------------------------------------------------------- Complaint
    def test_complaint(self) -> None:
        info = self.r.validate_complaint("TST12057605796344-1", "Testnagar")
        self.assertRead(info)
        self.assertEqual(info["value"], "TST12057605796344-1")

    def test_short_complaint_needs_review(self) -> None:
        info = self.r.validate_complaint("TST12", "Testnagar")
        self.assertNotEqual(info.get("status"), "READ", info)

    # ------------------------------------------------------------------- Date
    def test_date_formats_normalize_to_dd_mm_yyyy(self) -> None:
        cases = {
            "12/03/24": "12/03/2024",
            "12-03-2024": "12/03/2024",
            "2024-03-12": "12/03/2024",
            "3/12/24": "03/12/2024",
        }
        for raw, expected in cases.items():
            info = self.r.validate_date(raw)
            self.assertRead(info)
            self.assertEqual(info["value"], expected, raw)

    def test_date_rejects_impossible_calendar_days(self) -> None:
        for raw in ("31/02/2024", "29/02/2023"):
            info = self.r.validate_date(raw)
            self.assertNotEqual(info.get("status"), "READ", raw)

    def test_leap_day_is_valid(self) -> None:
        info = self.r.validate_date("29/02/2024")
        self.assertRead(info)
        self.assertEqual(info["value"], "29/02/2024")

    # -------------------------------------------------------------- Technician
    def test_technician_exact(self) -> None:
        info = self.r.resolve_technician(
            "Alex Demo", bccd_name="Demo Electronics", branch="Testnagar"
        )
        self.assertRead(info)
        self.assertEqual(info["value"], "Alex Demo")

    def test_unknown_technician_is_flagged(self) -> None:
        info = self.r.resolve_technician(
            "Zed Unknown", bccd_name="Demo Electronics", branch="Testnagar"
        )
        self.assertNotEqual(info.get("status"), "READ", info)

    # ------------------------------------------------------------- Field swap
    def test_swapped_branch_and_dealer_are_corrected(self) -> None:
        raw = {
            "Branch": "Demo Electronics",
            "BCCD Name": "Testnagar",
            "Product Description": "",
            "Product Sr. No.": "",
            "Date of Purchase": "",
            "Complaint No.": "",
            "Spare Part Code": "",
            "Nature Of Defect": "",
            "Technician Name": "",
        }
        out = self.r.normalize_branch_bccd(raw)
        self.assertEqual(out["Branch"], "Testnagar")
        self.assertEqual(out["BCCD Name"], "Demo Electronics")


if __name__ == "__main__":
    unittest.main()
