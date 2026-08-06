#!/usr/bin/env python3
"""Tests for parse_native_copyoption_names()'s coverage guard (catboost-8z4.67
fix round 1).

Self-contained: writes synthetic plain_options_helper.cpp-shaped snippets to
a temp file rather than depending on a vendor/catboost checkout, so this runs
without tools/vendor/acquire.sh.

Run: python3 tools/parity/test_gen_param_reference.py
"""
import tempfile
import unittest
from pathlib import Path

import gen_param_reference as gpr


class TestParseNativeCopyoptionNames(unittest.TestCase):
    def _run(self, cpp_src):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".cpp", delete=False
        ) as f:
            f.write(cpp_src)
            path = Path(f.name)
        try:
            return gpr.parse_native_copyoption_names(path)
        finally:
            path.unlink()

    def test_copyoption_and_copyoptionwithnewkey_both_captured(self):
        names = self._run(
            'CopyOption(plainOptions, "thread_count", &systemOptions, &seenKeys);\n'
            'CopyOptionWithNewKey(plainOptions, "od_pval", "stop_pvalue", '
            '&odConfig, &seenKeys);\n'
        )
        self.assertEqual(names, {"thread_count", "od_pval"})

    def test_reverse_direction_copyoptionwithnewkey_not_captured(self):
        # Opposite-direction call (options struct -> plainOptionsJson, used
        # for serialization): first arg is not the literal "plainOptions",
        # so its second-arg string names a *source* key in an internal
        # struct, not an accepted external input name.
        names = self._run(
            'CopyOptionWithNewKey(odConfig, "type", "od_type", '
            '&plainOptionsJson, &seenKeys);\n'
        )
        self.assertEqual(names, set())

    def test_unscanned_copyoption_family_helper_fails_loudly(self):
        # Regression guard: if a future vendor pin adds a third
        # CopyOption-family helper (or the existing two regexes are
        # accidentally narrowed), the broader family scan inside
        # parse_native_copyoption_names() must catch the gap instead of
        # silently under-scanning -- this is what would have caught the
        # original CopyOptionWithNewKey gap (catboost-8z4.67) before it was
        # fixed, had CopyOptionWithNewKey not already been a known pattern.
        with self.assertRaises(SystemExit):
            self._run(
                'CopyOptionRenamed(plainOptions, "some_future_name", '
                '&someOptions, &seenKeys);\n'
            )

    def test_differently_shaped_copy_helpers_do_not_false_positive(self):
        # CopyCtrDescription / CopyPerFeatureCtrDescription /
        # CopyPerFloatFeatureQuantization are differently-shaped helpers
        # (not CopyOption*) already covered via the 139-name Python-surface
        # inventory; the coverage guard must not flag them.
        names = self._run(
            'CopyCtrDescription(plainOptions, "ctr_description", '
            '"simple_ctrs", &ctrOptions, &seenKeys);\n'
        )
        self.assertEqual(names, set())


if __name__ == "__main__":
    unittest.main()
