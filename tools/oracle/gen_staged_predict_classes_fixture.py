#!/usr/bin/env python3
"""Generate the staged_predict-method oracle fixture (catboost-8z4.81): pins
Python catboost==1.2.10's model.staged_predict(pool,
prediction_type="RawFormulaVal", ntree_start=0, ntree_end=0, eval_period=3)
per-stage output for one model trained via each of the four estimator
classes -- CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker
(catboost/python-package/catboost/core.py).

staged_predict() IS overridden per-subclass with different DEFAULT
prediction_type values -- same override pattern as predict()
(CatBoostClassifier.staged_predict defaults to 'Class', core.py:5699;
CatBoostRegressor resolves via _get_default_prediction_type(),
core.py:6320-6329; CatBoostRanker.staged_predict hard-codes 'RawFormulaVal'
and drops the kwarg entirely) -- but all four produce identical
RawFormulaVal output when that prediction_type is explicitly requested on
both sides, which is exactly what this fixture and its paired test do.
Default-prediction_type parity itself is NOT covered here and is out of
scope for this ticket (catboost-8z4.81); see the follow-up ticket filed for
that gap. R's single catboost.staged_predict() function (R/catboost.R:3851)
is the parity target for all four matrix rows
(CatBoost.staged_predict / CatBoostClassifier.staged_predict /
CatBoostRegressor.staged_predict / CatBoostRanker.staged_predict); what
varies per row is only the loss/task shape the model was fit with, so this
fixture reuses the same per-class loss families as the eval_metrics /
get_feature_importance / predict fixtures:
  - CatBoost (base):        MultiClass (3-class)
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

10 trees, eval_period=3 -> 4 stages (trees [0:3], [0:6], [0:9], [0:10]),
covering both full eval_period increments and the final partial stage.

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_staged_predict_classes_fixture.py
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
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "staged_predict_classes.json")

EVAL_PERIOD = 3
COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_staged"))


def staged_predict_for(model, pool, has_prediction_type=True):
    # CatBoostRanker.staged_predict() overrides the base class and hard-codes
    # prediction_type="RawFormulaVal" internally, so it does not accept a
    # prediction_type kwarg at all (same override as predict(), see
    # gen_predict_classes_fixture.py) -- values are still RawFormulaVal on
    # both sides.
    kwargs = dict(ntree_start=0, ntree_end=0, eval_period=EVAL_PERIOD)
    if has_prediction_type:
        kwargs["prediction_type"] = "RawFormulaVal"
    stages = model.staged_predict(pool, **kwargs)
    return [stage.tolist() for stage in stages]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}, "eval_period": EVAL_PERIOD}

    # CatBoost base class: MultiClass.
    pool = Pool(x(), MULTICLASS_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(pool)
    fixture["inputs"]["CatBoost"] = {
        "num1": NUM1, "num2": NUM2, "label": MULTICLASS_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoost"] = staged_predict_for(model, pool)

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostClassifier"] = staged_predict_for(model, pool)

    # CatBoostRegressor: RMSE.
    pool = Pool(x(), REGRESSION_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1, "num2": NUM2, "label": REGRESSION_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressor"] = staged_predict_for(model, pool)

    # CatBoostRanker: YetiRank (requires group_id).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES, group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "group_id": GROUP_ID,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRanker"] = staged_predict_for(model, pool, has_prediction_type=False)

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
