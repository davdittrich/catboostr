#!/usr/bin/env python3
"""Tests for apply_disposition.py (ticket catboost-8z4.36 / P2.3).

Run: python3 tools/parity/test_apply_disposition.py
"""
import json
import unittest

import apply_disposition as ad
import build_matrix as bm

# Fields build_matrix.py's build_row() puts on every row; apply_disposition.py
# must never change any of these on any row (it only adds disposition fields,
# and may rewrite `method`/`tolerance`/`needs_disposition` for method/mode-
# shaped rows per the ticket's own boundary).
BASE_FIELDS_ALWAYS_PRESERVED = (
    "inventory_row_id", "oracle", "kind", "test_id", "state",
    "tolerance_justification", "members", "universal",
)


class TestFullPipeline(unittest.TestCase):
    """capability_diff.json -> matrix.json -> matrix.dispositioned.json,
    end to end, against the real committed fixtures."""

    @classmethod
    def setUpClass(cls):
        diff = json.loads(bm.INPUT_PATH.read_text())
        cls.built_rows, _, _ = bm.build_matrix(diff)
        cls.committed_matrix = json.loads(bm.OUTPUT_PATH.read_text())
        cls.judgments = json.loads(ad.JUDGMENTS_PATH.read_text())
        cls.dispositioned = ad.apply_disposition(cls.built_rows, cls.judgments)
        cls.committed_dispositioned = json.loads(ad.OUTPUT_PATH.read_text())

    def test_rebuilt_matrix_matches_committed_matrix_json(self):
        self.assertEqual(self.built_rows, self.committed_matrix)

    def test_dispositioned_row_count_and_category_counts(self):
        self.assertEqual(len(self.dispositioned), 725)
        counts = ad.summarize(self.dispositioned)
        self.assertEqual(counts["parameter_flag_family"], 361)
        self.assertEqual(counts["universal_flag"], 2)
        self.assertEqual(counts["enum_member_attachment"], 24)
        self.assertEqual(counts["method_mode_shaped"], 338)

    def test_dispositioned_rows_match_matrix_rows_except_documented_fields(self):
        by_id_matrix = {r["inventory_row_id"]: r for r in self.built_rows}
        for row in self.dispositioned:
            base = by_id_matrix[row["inventory_row_id"]]
            for field in BASE_FIELDS_ALWAYS_PRESERVED:
                self.assertEqual(row[field], base[field], field)
            # method/tolerance may only change for method_mode_shaped rows
            # that build_matrix.py left as method: null.
            if base["method"] is not None:
                self.assertEqual(row["method"], base["method"])
                self.assertEqual(row["tolerance"], base["tolerance"])

    def test_needs_disposition_matches_method_nullness(self):
        # Important #1's contract: needs_disposition and method may never
        # disagree once a row has been through apply_disposition.
        for row in self.dispositioned:
            self.assertEqual(row["needs_disposition"], row["method"] is None)

    def test_elementwise_rows_carry_default_tolerance(self):
        for row in self.dispositioned:
            if row.get("final_method") == "elementwise":
                self.assertEqual(row["tolerance"], bm.DEFAULT_TOLERANCE[row["oracle"]])

    def test_regenerated_output_matches_committed_matrix_dispositioned_json(self):
        # Pins the committed artifact: it must be exactly reproducible from
        # matrix.json + disposition_judgments.json, not a hand-edited file.
        self.assertEqual(self.dispositioned, self.committed_dispositioned)


if __name__ == "__main__":
    unittest.main()
