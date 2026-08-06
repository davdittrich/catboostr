#!/usr/bin/env python3
"""Generate the get_feature_importance-method oracle fixture
(catboost-8z4.80): pins Python catboost==1.2.10's
model.get_feature_importance(pool, type="FeatureImportance") output (the
Sec 4.3 default type -- PredictionValuesChange for non-ranking losses,
LossFunctionChange for ranking losses, catboost/python-package/catboost/
core.py:3404-3417) for one model trained via each of the four estimator
classes -- CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker
(core.py).

get_feature_importance() is defined once on the CatBoost base class
(core.py:3385) and is not overridden by any of the three subclasses, so R's
single catboost.get_feature_importance() function (R/catboost.R:3984) is the
parity target for all four matrix rows (CatBoost.get_feature_importance /
CatBoostClassifier.get_feature_importance /
CatBoostRegressor.get_feature_importance /
CatBoostRanker.get_feature_importance); what varies per row is only the
loss/task shape the model was fit with, so this fixture reuses the same
per-class loss families as the eval_metrics fixture
(gen_eval_metrics_classes_fixture.py):
  - CatBoost (base):        MultiClass (3-class)
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_get_feature_importance_classes_fixture.py
"""
import json
import os

from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "get_feature_importance_classes.json")

N_ROWS = 20
NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
        -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0]
NUM2 = [0.3 * i for i in range(N_ROWS)]
FEATURE_NAMES = ["num1", "num2"]

BINARY_LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
MULTICLASS_LABEL = [i % 3 for i in range(N_ROWS)]
REGRESSION_LABEL = [0.1 * i - 1.0 for i in range(N_ROWS)]
GROUP_ID = [i // 4 for i in range(N_ROWS)]  # 5 groups of 4

COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_fstr"))


def x():
    return [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]


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
    fixture["expected"]["CatBoost"] = model.get_feature_importance(pool, type="FeatureImportance").tolist()

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostClassifier"] = model.get_feature_importance(pool, type="FeatureImportance").tolist()

    # CatBoostRegressor: RMSE.
    pool = Pool(x(), REGRESSION_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1, "num2": NUM2, "label": REGRESSION_LABEL,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressor"] = model.get_feature_importance(pool, type="FeatureImportance").tolist()

    # CatBoostRanker: YetiRank (requires group_id). FeatureImportance
    # resolves to LossFunctionChange for ranking losses, which requires a
    # pool -- pass the same training pool used for eval_metrics.
    pool = Pool(x(), BINARY_LABEL, feature_names=FEATURE_NAMES, group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1, "num2": NUM2, "label": BINARY_LABEL, "group_id": GROUP_ID,
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRanker"] = model.get_feature_importance(pool, type="FeatureImportance").tolist()

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
