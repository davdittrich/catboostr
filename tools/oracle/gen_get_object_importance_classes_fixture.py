#!/usr/bin/env python3
"""Generate the get_object_importance-method oracle fixture
(catboost-8z4.80): pins Python catboost==1.2.10's
model.get_object_importance(pool, train_pool) output (the LeafInfluence
algorithm, catboost/python-package/catboost/core.py:3603) for one model
trained via each of the four estimator classes -- CatBoost,
CatBoostClassifier, CatBoostRegressor, CatBoostRanker (core.py).

get_object_importance() is defined once on the CatBoost base class
(core.py:3603) and is not overridden by any of the three subclasses, so R's
single catboost.get_object_importance() function (R/catboost.R:4110) is the
parity target for all four matrix rows. It routes to the LeafInfluence
algorithm's derivative calculator (catboost/private/libs/documents_
importance/ders_helpers.cpp:76-95), which only implements a fixed allow-list
of loss functions (Logloss/CrossEntropy/RMSE/MAE/Quantile/Expectile/
LogLinQuantile/MAPE/Poisson) and raises CB_ENSURE("... is not supported yet
in ostr mode") for every other loss -- including MultiClass and every
ranking loss. So only 2 of the 4 estimator classes' natural per-class loss
families (see gen_eval_metrics_classes_fixture.py) can exercise the real
numeric LeafInfluence output; this fixture covers those two:
  - CatBoostClassifier:     Logloss (binary classification) -- allow-listed
  - CatBoostRegressor:      RMSE -- allow-listed

CatBoost.get_object_importance (MultiClass) and CatBoostRanker.
get_object_importance (YetiRank) are NOT covered by real numeric values --
both loss functions hit the ders_helpers.cpp:95 CB_ENSURE and are closed
instead via an error-match differential test (same C++ raise site in both
languages): CatBoost.MultiClass already has one
(tests/testthat/test_object_importance_multiclass.R, catboost-8z4.54);
CatBoostRanker.YetiRank gets one added alongside this fixture's R test file.
This fixture also pins the exact Python-side exception message for both,
so the R test compares against an empirically-captured string (not one
inferred from reading vendor C++ source).

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_get_object_importance_classes_fixture.py
"""
import json
import os

from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

from _classes_common import (
    BINARY_LABEL, FEATURE_NAMES, GROUP_ID, MULTICLASS_LABEL, NUM1, NUM2, REGRESSION_LABEL, x,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "get_object_importance_classes.json")

# Held-out pool to score object importance for (first 5 rows of x()); the
# remaining 15 rows are the training pool.
EVAL_N = 5

COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_ostr"))


def split(features, label, group_id=None):
    eval_x, train_x = features[:EVAL_N], features[EVAL_N:]
    eval_y, train_y = label[:EVAL_N], label[EVAL_N:]
    if group_id is None:
        return Pool(eval_x, eval_y, feature_names=FEATURE_NAMES), Pool(train_x, train_y, feature_names=FEATURE_NAMES)
    eval_g, train_g = group_id[:EVAL_N], group_id[EVAL_N:]
    return (Pool(eval_x, eval_y, feature_names=FEATURE_NAMES, group_id=eval_g),
            Pool(train_x, train_y, feature_names=FEATURE_NAMES, group_id=train_g))


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}}

    # CatBoostClassifier: Logloss (binary).
    eval_pool, train_pool = split(x(), BINARY_LABEL)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(train_pool)
    indices, scores = model.get_object_importance(eval_pool, train_pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostClassifier"] = {"indices": list(indices), "scores": list(scores)}

    # CatBoostRegressor: RMSE.
    eval_pool, train_pool = split(x(), REGRESSION_LABEL)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(train_pool)
    indices, scores = model.get_object_importance(eval_pool, train_pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1, "num2": NUM2, "label": REGRESSION_LABEL, "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressor"] = {"indices": list(indices), "scores": list(scores)}

    # CatBoost base class: MultiClass -- not allow-listed, so this raises;
    # pin the exact Python-side error message rather than infer it.
    eval_pool, train_pool = split(x(), MULTICLASS_LABEL)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(train_pool)
    try:
        model.get_object_importance(eval_pool, train_pool)
        raise AssertionError("expected CatBoostError for MultiClass, got no exception")
    except Exception as e:  # noqa: BLE001 -- deliberately capturing CatBoostError's message
        fixture["inputs"]["CatBoost"] = {
            "num1": NUM1, "num2": NUM2, "label": MULTICLASS_LABEL, "feature_names": FEATURE_NAMES,
        }
        fixture["expected"]["CatBoost"] = {"error": str(e)}

    # CatBoostRanker: YetiRank -- not allow-listed either, same as above.
    eval_pool, train_pool = split(x(), BINARY_LABEL, GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(train_pool)
    try:
        model.get_object_importance(eval_pool, train_pool)
        raise AssertionError("expected CatBoostError for YetiRank, got no exception")
    except Exception as e:  # noqa: BLE001 -- deliberately capturing CatBoostError's message
        fixture["inputs"]["CatBoostRanker"] = {
            "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "group_id": GROUP_ID, "feature_names": FEATURE_NAMES,
        }
        fixture["expected"]["CatBoostRanker"] = {"error": str(e)}

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
