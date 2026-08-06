#!/usr/bin/env python3
"""Generate the predict-method oracle fixture (catboost-8z4.81): pins Python
catboost==1.2.10's model.predict(pool, prediction_type="RawFormulaVal")
output for one model trained via each of the four estimator classes --
CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker
(catboost/python-package/catboost/core.py).

predict() IS overridden per-subclass with different DEFAULT prediction_type
values -- CatBoostClassifier.predict() defaults to prediction_type='Class'
(core.py:5552), CatBoostRegressor.predict() resolves its default via
_get_default_prediction_type() (core.py:6183, 6320-6329) to
'Exponent'/'RMSEWithUncertainty' for some losses, and CatBoostRanker.predict()
hard-codes 'RawFormulaVal' and drops the kwarg entirely -- but all four
produce identical RawFormulaVal output when that prediction_type is
explicitly requested on both sides, which is exactly what this fixture and
its paired test do. Default-prediction_type parity itself is NOT covered
here and is out of scope for this ticket (catboost-8z4.81); see the
follow-up ticket filed for that gap. R's single catboost.predict() function
(R/catboost.R:3772, which delegates to predict.catboost.Model) is the
parity target for all four matrix rows
(CatBoost.predict / CatBoostClassifier.predict / CatBoostRegressor.predict /
CatBoostRanker.predict); what varies per row is only the loss/task shape the
model was fit with, so this fixture reuses the same per-class loss families
as the eval_metrics/get_feature_importance fixtures
(gen_eval_metrics_classes_fixture.py):
  - CatBoost (base):        MultiClass (3-class)
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_predict_classes_fixture.py
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
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "predict_classes.json")

COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_predict"))


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}}

    # CatBoost base class: MultiClass.
    pool = Pool(x(), MULTICLASS_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(pool)
    fixture["inputs"]["CatBoost"] = {
        "num1": NUM1, "num2": NUM2, "label": MULTICLASS_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoost"] = model.predict(pool, prediction_type="RawFormulaVal").tolist()

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostClassifier"] = model.predict(pool, prediction_type="RawFormulaVal").tolist()

    # CatBoostRegressor: RMSE.
    pool = Pool(x(), REGRESSION_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1, "num2": NUM2, "label": REGRESSION_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressor"] = model.predict(pool, prediction_type="RawFormulaVal").tolist()

    # CatBoostRanker: YetiRank (requires group_id). CatBoostRanker.predict()
    # overrides the base class and hard-codes prediction_type="RawFormulaVal"
    # internally (core.py: `self._predict(X, 'RawFormulaVal', ...)`), so it
    # does not accept a prediction_type kwarg at all -- unlike R's single
    # catboost.predict(), which always exposes prediction_type regardless of
    # which class the model was conceptually trained under. Values are still
    # RawFormulaVal on both sides, so parity is unaffected; just omit the
    # kwarg here.
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES, group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "group_id": GROUP_ID,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRanker"] = model.predict(pool).tolist()

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
