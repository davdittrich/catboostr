#!/usr/bin/env python3
"""Tests for build_matrix.py (ticket catboost-8z4.34 / P2.1).

Run: python3 tools/parity/test_build_matrix.py
"""
import json
import unittest
from pathlib import Path

import build_matrix as bm


class TestMergeAgainstRealFixture(unittest.TestCase):
    """Runs the actual merge against the real capability_diff.json fixture."""

    @classmethod
    def setUpClass(cls):
        cls.diff = json.loads(bm.INPUT_PATH.read_text())
        cls.rows, cls.merged_count, cls.passthrough_count = bm.build_matrix(cls.diff)

    def test_produces_770_rows_from_364_merged_and_406_passthrough(self):
        self.assertEqual(len(self.rows), 770)
        self.assertEqual(self.merged_count, 364)
        self.assertEqual(self.passthrough_count, 406)

    def test_every_passthrough_row_has_single_element_members(self):
        merged_ids = set()
        seen = set()
        for row in self.rows:
            if row["inventory_row_id"] not in seen:
                seen.add(row["inventory_row_id"])
        # Passthrough rows are those whose inventory_row_id equals a raw
        # gap's or covered entry's own capability string (i.e. not a
        # dedup_key format). build_matrix.py merges diff["gaps"] and
        # diff["covered"] into one set (spec Sec 4.5: covered/name-matched
        # capabilities still need a matrix row), so both sources' dedup_keys
        # count here.
        dedup_keys = {
            g["dedup_key"] for g in self.diff["gaps"] if g.get("dedup_key")
        } | {
            c["dedup_key"] for c in self.diff["covered"] if c.get("dedup_key")
        }
        for row in self.rows:
            if row["inventory_row_id"] not in dedup_keys:
                self.assertEqual(
                    len(row["members"]), 1,
                    f"passthrough row {row['inventory_row_id']!r} must have "
                    "exactly one member",
                )

    def test_merged_row_members_preserve_every_raw_member_verbatim(self):
        by_id = {row["inventory_row_id"]: row for row in self.rows}
        groups: dict = {}
        for entry in self.diff["gaps"] + self.diff["covered"]:
            key = entry.get("dedup_key")
            if key:
                groups.setdefault(key, []).append(entry)
        for dedup_key, raw_members in groups.items():
            row = by_id[dedup_key]
            expected = []
            for g in raw_members:
                member = {"capability": g["capability"], "owner": g["owner"]}
                if "matched_r_symbol" in g:
                    member["matched_r_symbol"] = g["matched_r_symbol"]
                expected.append(member)
            self.assertEqual(row["members"], expected)

    def test_universal_flags_marked_non_capability(self):
        universal_rows = [r for r in self.rows if r["universal"]]
        self.assertEqual(len(universal_rows), len(self.diff["universal_flags"]))
        for row in universal_rows:
            self.assertEqual(row["method"], "non_capability")
            self.assertFalse(row["needs_disposition"])

    def test_no_row_outside_known_categories_gets_a_guessed_method(self):
        for row in self.rows:
            if row["kind"] in ("parameter", "flag") and not row["universal"]:
                self.assertEqual(row["method"], "parameter_or_flag")
            elif row["kind"] == "enum_member":
                self.assertEqual(row["method"], "enum_member")
            elif row["universal"]:
                self.assertEqual(row["method"], "non_capability")
            else:
                cap = row["members"][0]["capability"].lower()
                if "plot_tree" in cap or "calc_feature_statistics" in cap:
                    self.assertEqual(row["method"], "structural")
                    self.assertFalse(row["needs_disposition"])
                else:
                    self.assertIsNone(row["method"])
                    self.assertTrue(row["needs_disposition"])

    def test_cross_check_counts_match_capability_diff_aggregates(self):
        summary = bm.summarize(
            self.rows, self.merged_count, self.passthrough_count, self.diff
        )
        self.assertEqual(summary["discrepancies"], [])


class TestPrecisionCeilingGuard(unittest.TestCase):
    """Spec Sec 4.3: CLI oracle rows may never declare tolerance tighter
    than the CLI's measured ~10-significant-digit precision (1e-9 floor)."""

    def test_rejects_cli_tolerance_tighter_than_floor(self):
        bad_row = {
            "inventory_row_id": "bad-row",
            "oracle": "cli",
            "tolerance": 1e-10,
        }
        with self.assertRaises(ValueError):
            bm.validate_precision_ceiling(bad_row)

    def test_accepts_cli_tolerance_at_or_above_floor(self):
        ok_row = {"inventory_row_id": "ok-row", "oracle": "cli", "tolerance": 1e-9}
        bm.validate_precision_ceiling(ok_row)  # must not raise

    def test_python_oracle_is_not_subject_to_the_cli_floor(self):
        python_row = {
            "inventory_row_id": "py-row",
            "oracle": "python",
            "tolerance": 1e-12,
        }
        bm.validate_precision_ceiling(python_row)  # must not raise


class TestToleranceJustificationGuard(unittest.TestCase):
    """Spec Sec 4.3: a non-default tolerance with an empty justification is
    a schema violation, not a valid row."""

    def test_rejects_nondefault_tolerance_with_empty_justification(self):
        bad_row = {
            "inventory_row_id": "bad-row",
            "oracle": "python",
            "tolerance": 1e-6,
            "tolerance_justification": "",
        }
        with self.assertRaises(ValueError):
            bm.validate_tolerance_justification(bad_row)

    def test_accepts_nondefault_tolerance_with_justification(self):
        ok_row = {
            "inventory_row_id": "ok-row",
            "oracle": "python",
            "tolerance": 1e-6,
            "tolerance_justification": "documented output-format precision limit",
        }
        bm.validate_tolerance_justification(ok_row)  # must not raise

    def test_accepts_default_tolerance_with_empty_justification(self):
        ok_row = {
            "inventory_row_id": "ok-row",
            "oracle": "python",
            "tolerance": 1e-12,
            "tolerance_justification": "",
        }
        bm.validate_tolerance_justification(ok_row)  # must not raise

    def test_accepts_null_tolerance_with_empty_justification(self):
        ok_row = {
            "inventory_row_id": "ok-row",
            "oracle": "cli",
            "tolerance": None,
            "tolerance_justification": "",
        }
        bm.validate_tolerance_justification(ok_row)  # must not raise


if __name__ == "__main__":
    unittest.main()
