#!/usr/bin/env python3
"""Generate the calc_feature_statistics oracle fixture (catboost-8z4.50 /
P4.1): deterministic train data + the pinned Python catboost==1.2.10
CatBoostClassifier.calc_feature_statistics()'s observable output, for the
structural differential test against catboost.calc_feature_statistics()
(R/catboost.R).

calc_feature_statistics() dispatches to _object._get_binarized_statistics
(catboost/private/libs/quantized_pool_analysis/quantized_pool_analysis.h
NCB::GetBinarizedStatistics) -- an identical vendor C++ entry point on both
sides, so R and Python are expected to match within float32 rounding, not
merely "close" (relative tolerance 1e-6, looser than the 1e-12 default
because Borders/MeanTarget/MeanPrediction/PredictionsOnVaryingFeature are
computed in float32 in the vendor function's internal accumulators before
being surfaced as float64 -- see quantized_pool_analysis.cpp:226-413).

Two float features (num1, num2) and one one-hot categorical feature (cat1)
exercise both statistics branches (see docstring's 'borders'-vs-'cat_values'
split). plot=False throughout: rendering is explicitly out of scope (spec
Sec 4.3's Structural row -- plot/plot_file exist only for the interactive
widget, never differential-tested).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_calc_feature_statistics_fixture.py
"""
import json
import os

import numpy as np

from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "calc_feature_statistics.json")

N_ROWS = 24

NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25, -10.0, 11.5,
        1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0, -9.5, 10.25, -11.0, 12.5]
NUM2 = [float(i) * 0.3 - 3.0 for i in range(N_ROWS)]
CAT1 = ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c", "b", "a",
        "b", "a", "c", "b", "a", "c", "b", "a", "c", "b", "a", "c"]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]

FEATURE_NAMES = ["num1", "num2", "cat1"]


def _np_to_list(value):
    if isinstance(value, np.ndarray):
        return [_np_to_list(v) for v in value.tolist()] if value.dtype == object else value.tolist()
    return value


def _stat_to_jsonable(stat):
    out = {}
    for key, value in stat.items():
        out[key] = _np_to_list(value)
    return out


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    X = [[NUM1[i], NUM2[i], CAT1[i]] for i in range(N_ROWS)]
    pool = Pool(X, LABEL, cat_features=[2], feature_names=FEATURE_NAMES)

    model = CatBoostClassifier(
        iterations=10, depth=3, loss_function="Logloss",
        random_seed=42, thread_count=1, verbose=False,
        # one_hot_max_size >= 3 (unique count of cat1): GetBinarizedStatistics
        # (quantized_pool_analysis.cpp:570) only supports one-hot-encoded
        # categorical features, never CTR-encoded ones.
        one_hot_max_size=4,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train"),
    )
    model.fit(pool)

    # 1. feature=None (all features), default prediction_type (Probability,
    #    derived from Logloss) -- the primary, all-features-at-once case.
    all_features_default = model.calc_feature_statistics(pool, feature=None, plot=False)

    # 2. Single named float feature, is_for_one_feature branch, explicit
    #    prediction_type="RawFormulaVal".
    single_float = model.calc_feature_statistics(
        pool, feature="num1", prediction_type="RawFormulaVal", plot=False
    )

    # 3. Single named categorical feature, is_for_one_feature branch --
    #    exercises the 'cat_values' branch (borders popped, cat_values set).
    single_cat = model.calc_feature_statistics(pool, feature="cat1", plot=False)

    expected = {
        "all_features_default": {
            name: _stat_to_jsonable(all_features_default[name]) for name in FEATURE_NAMES
        },
        "single_float": _stat_to_jsonable(single_float),
        "single_cat": _stat_to_jsonable(single_cat),
    }

    fixture = {
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "cat1": CAT1,
            "label": LABEL,
            "feature_names": FEATURE_NAMES,
        },
        "expected": expected,
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=True)
    print("wrote", FIXTURE_PATH)


if __name__ == "__main__":
    main()
