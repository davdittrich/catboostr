#!/usr/bin/env python3
"""Generate the shrink-method oracle fixture (catboost-8z4.81): pins Python
catboost==1.2.10's model.shrink(ntree_end, ntree_start) post-shrink
predict(prediction_type="RawFormulaVal") output for one model trained via
each of the four estimator classes -- CatBoost, CatBoostClassifier,
CatBoostRegressor, CatBoostRanker (catboost/python-package/catboost/core.py).

shrink() is defined once on the CatBoost base class (core.py, wraps the
native TFullModel::Truncate) and is not overridden by any of the three
subclasses; both languages route to the same native truncation entry point
(src/catboostr.cpp's CatBoostShrinkModel_R vs. the Python _object.Truncate
binding), so R's single catboost.shrink() function (R/catboost.R:4160) is
the parity target for all four matrix rows (CatBoost.shrink /
CatBoostClassifier.shrink / CatBoostRegressor.shrink /
CatBoostRanker.shrink). shrink() itself has no output of its own (Python
returns None, R returns the native call's status) -- like
drop_unused_features, its effect is observable only via predictions taken
before/after the call, so this fixture pins both the pre-shrink and
post-shrink RawFormulaVal predictions on the same 10-tree model truncated
to trees [2, 8), reusing the same per-class loss families as the
eval_metrics / get_feature_importance / predict / staged_predict fixtures:
  - CatBoost (base):        MultiClass (3-class)
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_shrink_classes_fixture.py
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
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "shrink_classes.json")

NTREE_START = 2
NTREE_END = 8
COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_shrink"))


def shrink_before_after(model, pool, has_prediction_type=True):
    # CatBoostRanker.predict() overrides the base class and hard-codes
    # prediction_type="RawFormulaVal" internally, so it does not accept a
    # prediction_type kwarg at all (see gen_predict_classes_fixture.py) --
    # values are still RawFormulaVal on both sides.
    kwargs = {"prediction_type": "RawFormulaVal"} if has_prediction_type else {}
    before = model.predict(pool, **kwargs).tolist()
    model.shrink(ntree_end=NTREE_END, ntree_start=NTREE_START)
    after = model.predict(pool, **kwargs).tolist()
    return before, after


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}, "ntree_start": NTREE_START, "ntree_end": NTREE_END}

    # CatBoost base class: MultiClass.
    pool = Pool(x(), MULTICLASS_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(pool)
    before, after = shrink_before_after(model, pool)
    fixture["inputs"]["CatBoost"] = {
        "num1": NUM1, "num2": NUM2, "label": MULTICLASS_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoost"] = {"before": before, "after": after}

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    before, after = shrink_before_after(model, pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostClassifier"] = {"before": before, "after": after}

    # CatBoostRegressor: RMSE.
    pool = Pool(x(), REGRESSION_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    before, after = shrink_before_after(model, pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1, "num2": NUM2, "label": REGRESSION_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressor"] = {"before": before, "after": after}

    # CatBoostRanker: YetiRank (requires group_id).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES, group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    before, after = shrink_before_after(model, pool, has_prediction_type=False)
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "group_id": GROUP_ID,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRanker"] = {"before": before, "after": after}

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
