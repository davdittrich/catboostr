#!/usr/bin/env python3
"""Generate the training-history-introspection oracle fixture
(catboost-8z4.118, P10.D): pins Python catboost==1.2.10's
best_iteration_/best_score_/evals_result_/classes_ properties and their
get_best_iteration()/get_best_score()/get_evals_result() method equivalents
(catboost/python-package/catboost/core.py:1851-2098) for one model trained
via each of the four estimator classes -- CatBoost, CatBoostClassifier,
CatBoostRegressor, CatBoostRanker.

All three get_*/*_ pairs are defined once on the CatBoost base class and are
not overridden by any of the three subclasses, so R's single
catboost.get_best_iteration()/catboost.get_best_score()/
catboost.get_evals_result() functions (and the classes_/best_iteration_/
best_score_/evals_result_ fields create.model.base() attaches to every
model object) are the parity target for all 28 matrix rows (7 families x 4
classes). Each model is fit with an eval_set and early_stopping_rounds so
best_iteration_ is a real, non-trivial value (not just "last iteration"),
exercising the actual early-stopping bookkeeping rather than a degenerate
case.

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_training_history_classes_fixture.py
"""
import json
import os

from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from catboost.core import _NumpyAwareEncoder

from _classes_common import (
    BINARY_LABEL, FEATURE_NAMES, GROUP_ID, MULTICLASS_LABEL, NUM1, NUM2, REGRESSION_LABEL, x,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "training_history_classes.json")

# Whole-group split (5 groups of 4 rows, GROUP_ID = i // 4): first 4 groups
# (16 rows) train, last group (4 rows) eval -- keeps every group intact so
# the CatBoostRanker/YetiRank case (which needs group_id) stays valid.
TRAIN_N = 16

COMMON = dict(iterations=30, depth=2, random_seed=42, thread_count=1, verbose=False,
              early_stopping_rounds=5,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_history"))


def split(label, group_id=None):
    features = x()
    train_x, eval_x = features[:TRAIN_N], features[TRAIN_N:]
    train_y, eval_y = label[:TRAIN_N], label[TRAIN_N:]
    if group_id is None:
        return (Pool(train_x, train_y, feature_names=FEATURE_NAMES),
                Pool(eval_x, eval_y, feature_names=FEATURE_NAMES))
    train_g, eval_g = group_id[:TRAIN_N], group_id[TRAIN_N:]
    return (Pool(train_x, train_y, feature_names=FEATURE_NAMES, group_id=train_g),
            Pool(eval_x, eval_y, feature_names=FEATURE_NAMES, group_id=eval_g))


def history_for(model):
    return {
        "classes_": model.classes_,
        "best_iteration_": model.best_iteration_,
        "best_score_": model.best_score_,
        "evals_result_": model.evals_result_,
        "get_best_iteration": model.get_best_iteration(),
        "get_best_score": model.get_best_score(),
        "get_evals_result": model.get_evals_result(),
    }


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}}

    # CatBoost base class: MultiClass (3-class) -- exercises classes_ with a
    # non-trivial multi-value label set.
    train_pool, eval_pool = split(MULTICLASS_LABEL)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoost"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": MULTICLASS_LABEL[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": MULTICLASS_LABEL[TRAIN_N:],
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoost"] = history_for(model)

    # CatBoostClassifier: Logloss (binary).
    train_pool, eval_pool = split(BINARY_LABEL)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": BINARY_LABEL[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": BINARY_LABEL[TRAIN_N:],
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostClassifier"] = history_for(model)

    # CatBoostRegressor: RMSE -- exercises classes_ being empty (non-classification).
    train_pool, eval_pool = split(REGRESSION_LABEL)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": REGRESSION_LABEL[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": REGRESSION_LABEL[TRAIN_N:],
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressor"] = history_for(model)

    # CatBoostRanker: YetiRank (requires group_id).
    train_pool, eval_pool = split(BINARY_LABEL, GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": BINARY_LABEL[:TRAIN_N], "group_id": GROUP_ID[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": BINARY_LABEL[TRAIN_N:],
        "eval_group_id": GROUP_ID[TRAIN_N:],
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRanker"] = history_for(model)

    # No eval_set: exercises the case where LearnMetricsHistory/
    # LearnBestError are still populated every iteration (unconditional on
    # having a test set) but BestIteration/TestMetricsHistory/TestBestError
    # stay empty/undefined (only ever touched by the eval-set-only code
    # path, catboost/libs/train_lib/train_model.cpp CalcErrors/AddTestError).
    train_pool, _ = split(REGRESSION_LABEL)
    model = CatBoostRegressor(loss_function="RMSE", **{k: v for k, v in COMMON.items() if k != "early_stopping_rounds"})
    model.fit(train_pool)
    fixture["inputs"]["CatBoostRegressorNoEvalSet"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": REGRESSION_LABEL[:TRAIN_N],
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostRegressorNoEvalSet"] = history_for(model)

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2, cls=_NumpyAwareEncoder)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
