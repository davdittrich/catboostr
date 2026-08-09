#!/usr/bin/env python3
"""Generate the sklearn-accessor oracle fixture (catboost-8z4.116 / P10.B):
pins Python catboost==1.2.10's post-fit accessor output (tree_count_,
learning_rate_, random_seed_, n_features_in_/get_n_features_in(),
get_cat_feature_indices()/get_text_feature_indices()/
get_embedding_feature_indices(), get_all_params()/get_param(), and the
observable effect of set_feature_names()) for one model trained via each of
the four estimator classes -- CatBoost, CatBoostClassifier, CatBoostRegressor,
CatBoostRanker (catboost/python-package/catboost/core.py) -- plus
CatBoostClassifier.predict_proba()/staged_predict_proba(), which have no
CatBoost/Regressor/Ranker analogue in the sklearn API.

Reuses the same per-class loss families as the predict/staged_predict/
get_feature_importance fixtures (gen_predict_classes_fixture.py etc):
  - CatBoost (base):        MultiClass (3-class)
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

Unlike those fixtures, this one adds a third, categorical feature (cat1) on
top of _classes_common's num1/num2 -- required to make
get_cat_feature_indices() non-vacuous (num1/num2-only pools trivially return
an empty list for every accessor, which would not actually distinguish "R
returns the right indices" from "R always returns empty"). This is a fixture
local addition (not a change to _classes_common.py, which every other
gen_*_classes_fixture.py script depends on verbatim).

get_all_params()/get_param() comparison is restricted to the keys that are
mechanically comparable between Python's flat get_all_params() dict and R's
catboost.get_plain_params()'s flat list (iterations/depth/learning_rate/
random_seed/loss_function) -- both are literally the same underlying
resolved-training-options structure (catboost.get_model_params() is a
*nested* structure and is not a faithful get_all_params() match; see
catboost-8z4.116 report for why get_model_params was rejected as the parity
target).

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_sklearn_accessor_classes_fixture.py
"""
import json
import os

from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

from _classes_common import (
    BINARY_LABEL, GROUP_ID, MULTICLASS_LABEL, NUM1, NUM2, REGRESSION_LABEL,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "sklearn_accessor_classes.json")

CAT1 = ["a", "b"] * (len(NUM1) // 2)
FEATURE_NAMES = ["num1", "num2", "cat1"]
NEW_FEATURE_NAMES = ["renamed_num1", "renamed_num2", "renamed_cat1"]
CAT_FEATURES = ["cat1"]

COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_sklearn_accessor"))

PARAM_KEYS = ("iterations", "depth", "learning_rate", "random_seed", "loss_function")
# get_param() only echoes back a value the caller explicitly passed in (it is
# not the resolved-training-options view get_all_params()/R's
# catboost.get_plain_params() are); learning_rate is left to auto-resolve in
# COMMON below, so Python's get_param("learning_rate") is None -- excluded
# here so this fixture only asserts keys where get_param's raw-echo semantics
# and get_all_params/get_plain_params's resolved-value semantics coincide.
GET_PARAM_KEYS = ("iterations", "depth", "random_seed", "loss_function")

EVAL_PERIOD = 3


def features():
    return [[NUM1[i], NUM2[i], CAT1[i]] for i in range(len(NUM1))]


def accessors_for(model, pool, is_classifier):
    result = {
        "tree_count_": model.tree_count_,
        "learning_rate_": model.learning_rate_,
        "random_seed_": model.random_seed_,
        "n_features_in_": model.n_features_in_,
        "get_n_features_in": model.get_n_features_in(),
        "get_cat_feature_indices": model.get_cat_feature_indices(),
        "get_text_feature_indices": model.get_text_feature_indices(),
        "get_embedding_feature_indices": model.get_embedding_feature_indices(),
        "get_all_params": {k: model.get_all_params()[k] for k in PARAM_KEYS},
        "get_param": {k: model.get_param(k) for k in GET_PARAM_KEYS},
    }
    if is_classifier:
        result["predict_proba"] = model.predict_proba(pool).tolist()
        staged = model.staged_predict_proba(pool, ntree_start=0, ntree_end=0, eval_period=EVAL_PERIOD)
        result["staged_predict_proba"] = [stage.tolist() for stage in staged]
    # set_feature_names() renames the model's own feature metadata in place;
    # done last since a renamed model is no longer prediction-compatible with
    # a pool built under the original names (model_dataset_compatibility.cpp).
    model.set_feature_names(NEW_FEATURE_NAMES)
    result["feature_names_after_set"] = list(model.feature_names_)
    return result


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}, "eval_period": EVAL_PERIOD}

    common_inputs = {"num1": NUM1, "num2": NUM2, "cat1": CAT1, "feature_names": FEATURE_NAMES,
                      "new_feature_names": NEW_FEATURE_NAMES, "cat_features": CAT_FEATURES}

    # CatBoost base class: MultiClass.
    pool = Pool(features(), MULTICLASS_LABEL, feature_names=FEATURE_NAMES, cat_features=CAT_FEATURES)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(pool)
    fixture["inputs"]["CatBoost"] = dict(common_inputs, label=MULTICLASS_LABEL)
    fixture["expected"]["CatBoost"] = accessors_for(model, pool, is_classifier=False)

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(features(), BINARY_LABEL, feature_names=FEATURE_NAMES, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostClassifier"] = dict(common_inputs, label=BINARY_LABEL)
    fixture["expected"]["CatBoostClassifier"] = accessors_for(model, pool, is_classifier=True)

    # CatBoostRegressor: RMSE.
    pool = Pool(features(), REGRESSION_LABEL, feature_names=FEATURE_NAMES, cat_features=CAT_FEATURES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRegressor"] = dict(common_inputs, label=REGRESSION_LABEL)
    fixture["expected"]["CatBoostRegressor"] = accessors_for(model, pool, is_classifier=False)

    # CatBoostRanker: YetiRank (requires group_id).
    pool = Pool(features(), BINARY_LABEL, feature_names=FEATURE_NAMES, cat_features=CAT_FEATURES,
                group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    fixture["inputs"]["CatBoostRanker"] = dict(common_inputs, label=BINARY_LABEL, group_id=GROUP_ID)
    fixture["expected"]["CatBoostRanker"] = accessors_for(model, pool, is_classifier=False)

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
