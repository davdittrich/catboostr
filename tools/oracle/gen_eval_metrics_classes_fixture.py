#!/usr/bin/env python3
"""Generate the eval_metrics-method oracle fixture (catboost-8z4.79): pins
Python catboost==1.2.10's model.eval_metrics() output for one model trained
via each of the four estimator classes -- CatBoost, CatBoostClassifier,
CatBoostRegressor, CatBoostRanker (catboost/python-package/catboost/core.py).

eval_metrics() is defined once on the CatBoost base class (core.py:3234) and
is not overridden by any of the three subclasses, so R's single
catboost.eval_metrics() function (R/catboost.R) is the parity target for all
four matrix rows (CatBoost.eval_metrics / CatBoostClassifier.eval_metrics /
CatBoostRegressor.eval_metrics / CatBoostRanker.eval_metrics); what varies
per row is only the loss/task shape the model was fit with, so this fixture
pins one fit per class covering a distinct loss family:
  - CatBoost (base):        MultiClass (3-class), exercised via the base
                             class directly rather than CatBoostClassifier
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_eval_metrics_classes_fixture.py
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
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "eval_metrics_classes.json")

COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_evalm"))


def eval_metrics_for(model, pool, metrics):
    return model.eval_metrics(pool, metrics, ntree_start=0, ntree_end=0, eval_period=1)


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}}

    # CatBoost base class: MultiClass.
    pool = Pool(x(), MULTICLASS_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(pool)
    metrics = ["MultiClass", "Accuracy"]
    fixture["inputs"]["CatBoost"] = {
        "num1": NUM1, "num2": NUM2, "label": MULTICLASS_LABEL,
        "feature_names": FEATURE_NAMES, "metrics": metrics,
    }
    fixture["expected"]["CatBoost"] = eval_metrics_for(model, pool, metrics)

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    metrics = ["Logloss", "AUC"]
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL,
        "feature_names": FEATURE_NAMES, "metrics": metrics,
    }
    fixture["expected"]["CatBoostClassifier"] = eval_metrics_for(model, pool, metrics)

    # CatBoostRegressor: RMSE.
    pool = Pool(x(), REGRESSION_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    metrics = ["RMSE", "MAE"]
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1, "num2": NUM2, "label": REGRESSION_LABEL,
        "feature_names": FEATURE_NAMES, "metrics": metrics,
    }
    fixture["expected"]["CatBoostRegressor"] = eval_metrics_for(model, pool, metrics)

    # CatBoostRanker: YetiRank (requires group_id).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES, group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    metrics = ["PFound", "NDCG"]
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "group_id": GROUP_ID,
        "feature_names": FEATURE_NAMES, "metrics": metrics,
    }
    fixture["expected"]["CatBoostRanker"] = eval_metrics_for(model, pool, metrics)

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
