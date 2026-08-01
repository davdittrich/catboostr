#!/usr/bin/env python3
"""Tests structural_mapping.py (ticket catboost-8z4.35 / P2.2).

Run: python3 tools/parity/test_structural_mapping.py
"""
import unittest

import structural_mapping as sm


class TestCanonicalForm(unittest.TestCase):
    def test_keys_sorted_recursively(self):
        doc = {"b": 1, "a": {"z": 1, "y": 2}}
        self.assertEqual(list(sm.to_canonical(doc).keys()), ["a", "b"])
        self.assertEqual(list(sm.to_canonical(doc)["a"].keys()), ["y", "z"])

    def test_list_order_preserved_not_sorted(self):
        doc = {"xs": [3, 1, 2]}
        self.assertEqual(sm.to_canonical(doc)["xs"], [3, 1, 2])

    def test_nan_and_inf_tagged_not_dropped(self):
        doc = {"x": float("nan"), "y": float("inf"), "z": float("-inf")}
        c = sm.to_canonical(doc)
        self.assertEqual(c["x"], {"__float__": "NaN"})
        self.assertEqual(c["y"], {"__float__": "Infinity"})
        self.assertEqual(c["z"], {"__float__": "-Infinity"})

    def test_canonical_dumps_is_stable_regardless_of_input_key_order(self):
        a = sm.canonical_dumps({"b": 1, "a": 2})
        b = sm.canonical_dumps({"a": 2, "b": 1})
        self.assertEqual(a, b)


class TestResolvePath(unittest.TestCase):
    def test_plain_dotted_path(self):
        doc = {"a": {"b": 5}}
        self.assertEqual(sm.resolve_path(doc, "a.b"), [5])

    def test_broadcast_over_list_addresses_nested_structure(self):
        doc = {"children": [{"v": 1}, {"v": 2}, {"v": 3}]}
        self.assertEqual(sm.resolve_path(doc, "children[].v"), [1, 2, 3])

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            sm.resolve_path({"a": 1}, "b")


class TestValidateMappingSchema(unittest.TestCase):
    """Ticket III Logic guard: (a) both sides declared, (b) comparison
    type recognized, (c) excluded leaves carry a reason."""

    def test_rejects_field_missing_comparison_type(self):
        mapping = {"fields": [{"r_field": "x", "oracle_field": "x"}]}
        with self.assertRaises(ValueError):
            sm.validate_mapping(mapping)

    def test_rejects_exclude_without_reason(self):
        mapping = {"fields": [{"r_field": "x", "exclude": True}]}
        with self.assertRaises(ValueError):
            sm.validate_mapping(mapping)

    def test_rejects_numeric_without_tolerance(self):
        mapping = {
            "fields": [{"r_field": "x", "oracle_field": "x", "compare": "numeric"}]
        }
        with self.assertRaises(ValueError):
            sm.validate_mapping(mapping)

    def test_rejects_nondefault_tolerance_without_justification(self):
        mapping = {
            "fields": [
                {
                    "r_field": "x",
                    "oracle_field": "x",
                    "compare": "numeric",
                    "tolerance": {"oracle": "python", "value": 1e-6},
                }
            ]
        }
        with self.assertRaises(ValueError):
            sm.validate_mapping(mapping)

    def test_rejects_cli_tolerance_tighter_than_precision_floor(self):
        mapping = {
            "fields": [
                {
                    "r_field": "x",
                    "oracle_field": "x",
                    "compare": "numeric",
                    "tolerance": {"oracle": "cli", "value": 1e-10},
                }
            ]
        }
        with self.assertRaises(ValueError):
            sm.validate_mapping(mapping)

    def test_accepts_well_formed_mapping(self):
        mapping = {
            "fields": [
                {"r_field": "x", "oracle_field": "x", "compare": "exact"},
                {"r_field": "t", "exclude": True, "exclude_reason": "timestamp"},
            ]
        }
        sm.validate_mapping(mapping)  # must not raise


class TestValidateAgainstDocumentsPerturbation(unittest.TestCase):
    """Ticket step 3: validator must catch a deliberately broken mapping
    against real documents, not just a malformed schema."""

    def setUp(self):
        self.mapping = {
            "fields": [
                {"r_field": "name", "oracle_field": "name", "compare": "exact"},
                {
                    "r_field": "value",
                    "oracle_field": "value",
                    "compare": "numeric",
                    "tolerance": {"oracle": "python", "value": 1e-12},
                },
            ]
        }
        self.r_doc = {"name": "age", "value": 1.5}
        self.oracle_doc = {"name": "age", "value": 1.5}

    def test_passes_on_matching_documents(self):
        sm.validate_against_documents(self.mapping, self.r_doc, self.oracle_doc)  # no raise

    def test_fails_when_field_missing_on_oracle_side(self):
        broken = {"fields": [dict(f) for f in self.mapping["fields"]]}
        broken["fields"][0]["oracle_field"] = "does_not_exist"
        with self.assertRaises(ValueError):
            sm.validate_against_documents(broken, self.r_doc, self.oracle_doc)

    def test_fails_when_numeric_leaf_declared_exact(self):
        broken = {"fields": [dict(f) for f in self.mapping["fields"]]}
        broken["fields"][1]["compare"] = "exact"
        broken["fields"][1].pop("tolerance", None)
        with self.assertRaises(ValueError):
            sm.validate_against_documents(broken, self.r_doc, self.oracle_doc)


class TestWorkedExample(unittest.TestCase):
    """Ticket step 4: passes on matching data, fails on perturbed data --
    both outcomes, not just the happy path."""

    @classmethod
    def setUpClass(cls):
        cls.mapping = sm.load_mapping(sm.MAPPINGS_DIR / "example_worked.json")
        cls.r_doc = {
            "feature": "age",
            "bins": [
                {"lower": 0.0, "upper": 10.0, "mean_target": 0.31},
                {"lower": 10.0, "upper": 20.0, "mean_target": 0.42},
            ],
            "generated_at": "2026-08-02T00:00:00Z",
        }
        cls.oracle_doc = {
            "feature_name": "age",
            "buckets": [
                {"lo": 0.0, "hi": 10.0, "target_mean": 0.31},
                {"lo": 10.0, "hi": 20.0, "target_mean": 0.42},
            ],
            "generated_at": "2026-08-02T00:00:01Z",
        }

    def test_passes_on_matching_data(self):
        ok, mismatches = sm.compare(self.mapping, self.r_doc, self.oracle_doc)
        self.assertTrue(ok)
        self.assertEqual(mismatches, [])

    def test_fails_on_perturbed_numeric_leaf(self):
        import copy

        perturbed = copy.deepcopy(self.oracle_doc)
        perturbed["buckets"][1]["target_mean"] = 0.99
        ok, mismatches = sm.compare(self.mapping, self.r_doc, perturbed)
        self.assertFalse(ok)
        self.assertTrue(mismatches)
        self.assertIn("mean_target", mismatches[0])

    def test_fails_on_perturbed_exact_leaf(self):
        import copy

        perturbed = copy.deepcopy(self.oracle_doc)
        perturbed["feature_name"] = "size"
        ok, mismatches = sm.compare(self.mapping, self.r_doc, perturbed)
        self.assertFalse(ok)
        self.assertTrue(mismatches)

    def test_both_sides_nan_is_a_match_not_a_mismatch(self):
        import copy
        import math

        r_doc = copy.deepcopy(self.r_doc)
        oracle_doc = copy.deepcopy(self.oracle_doc)
        r_doc["bins"][1]["mean_target"] = math.nan
        oracle_doc["buckets"][1]["target_mean"] = math.nan
        ok, mismatches = sm.compare(self.mapping, r_doc, oracle_doc)
        self.assertTrue(ok)
        self.assertEqual(mismatches, [])

    def test_nan_vs_number_is_a_mismatch(self):
        import copy
        import math

        r_doc = copy.deepcopy(self.r_doc)
        oracle_doc = copy.deepcopy(self.oracle_doc)
        r_doc["bins"][1]["mean_target"] = math.nan
        # oracle_doc["buckets"][1]["target_mean"] stays a real number (0.42)
        ok, mismatches = sm.compare(self.mapping, r_doc, oracle_doc)
        self.assertFalse(ok)
        self.assertTrue(mismatches)

    def test_excluded_field_never_compared(self):
        # generated_at differs between r_doc and oracle_doc by design; must
        # not surface as a mismatch because it is declared excluded.
        ok, mismatches = sm.compare(self.mapping, self.r_doc, self.oracle_doc)
        self.assertTrue(ok)
        self.assertFalse(any("generated_at" in m for m in mismatches))


if __name__ == "__main__":
    unittest.main()
