#!/usr/bin/env python3
"""Generate the drop_unused_features-method oracle fixture
(catboost-8z4.80): pins Python catboost==1.2.10's model.drop_unused_features()
before/after effect (catboost/python-package/catboost/core.py:3696, which
just calls the same native TModelTrees::DropUnusedFeatures() as R's
catboost.drop_unused_features(), src/catboostr.cpp:2400) for one model
trained via each of the four estimator classes -- CatBoost,
CatBoostClassifier, CatBoostRegressor, CatBoostRanker.

drop_unused_features() has no return value of its own -- both languages'
wrappers just mutate the model in place (Python returns None; R returns a
constant TRUE regardless of outcome, src/catboostr.cpp:2405) -- so this is a
state-mutator row: the real observable is the model's state before/after the
call, verified via the paired feature-names getter (model.feature_names_ /
catboost.get_model_feature_names) and via predictions staying identical
(dropping unused features must not change tree evaluation). Each fit uses a
strong single informative feature (num1) plus two pure-noise features
(noise1, noise2) with depth=1 so that, given the fixed seed and single
thread, both C++ builds deterministically never split on the noise features
-- they are the ones DropUnusedFeatures() removes. Reuses the same per-class
loss families as the eval_metrics fixture
(gen_eval_metrics_classes_fixture.py):
  - CatBoost (base):        MultiClass (3-class)
  - CatBoostClassifier:     Logloss (binary classification)
  - CatBoostRegressor:      RMSE
  - CatBoostRanker:         YetiRank (requires group_id)

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_drop_unused_features_classes_fixture.py
"""
import json
import os

from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

from _classes_common import BINARY_LABEL, GROUP_ID, MULTICLASS_LABEL, N_ROWS, REGRESSION_LABEL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "drop_unused_features_classes.json")

# Pure noise, uncorrelated with any label below.
NOISE1 = [0.91, 0.12, 0.44, 0.77, 0.03, 0.65, 0.28, 0.99, 0.51, 0.08,
          0.36, 0.72, 0.19, 0.83, 0.47, 0.60, 0.05, 0.94, 0.22, 0.68]
NOISE2 = [-0.4, 0.8, -0.2, 0.1, 0.9, -0.6, 0.3, -0.9, 0.5, -0.1,
          0.7, -0.3, 0.6, -0.7, 0.2, -0.8, 0.4, -0.5, 0.0, 1.0]
FEATURE_NAMES = ["num1", "noise1", "noise2"]

# num1 is built directly from each class's own label (scaled +/- a small
# deterministic jitter), so it is by construction a stronger split
# candidate than the two pure-noise features for every one of the four
# label shapes below -- not just one shared correlation pattern that
# happens to favor some classes over others.
NUM1_FOR_BINARY = [10.0 if lbl == 1 else -10.0 for lbl in BINARY_LABEL]
NUM1_FOR_MULTICLASS = [(lbl - 1) * 50.0 for lbl in MULTICLASS_LABEL]
NUM1_FOR_REGRESSION = [lbl * 100.0 for lbl in REGRESSION_LABEL]

# depth=1, iterations=3: few, shallow splits so the strongly-informative
# num1 wins every split given the fixed seed/single thread -- noise1/noise2
# are never used and are exactly what DropUnusedFeatures() removes.
COMMON = dict(iterations=3, depth=1, random_seed=42, thread_count=1, verbose=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_drop_unused"))


def x(num1):
    return [[num1[i], NOISE1[i], NOISE2[i]] for i in range(N_ROWS)]


def record(fixture, class_name, model, pool, num1, label, group_id=None, predict_kwargs=None):
    # R's catboost.predict() has no per-class dispatch and always defaults to
    # prediction_type="RawFormulaVal" (R/catboost.R:3751); CatBoostRanker's
    # predict() has no prediction_type kwarg at all and returns raw formula
    # values unconditionally, so it needs no override.
    predict_kwargs = predict_kwargs or {}

    inputs = {"num1": num1, "noise1": NOISE1, "noise2": NOISE2, "label": label,
              "feature_names": FEATURE_NAMES}
    if group_id is not None:
        inputs["group_id"] = group_id
    fixture["inputs"][class_name] = inputs

    before_names = model.feature_names_
    before_pred = model.predict(pool, **predict_kwargs).tolist()
    model.drop_unused_features()
    after_names = model.feature_names_
    after_pred = model.predict(pool, **predict_kwargs).tolist()

    fixture["expected"][class_name] = {
        "before_feature_names": list(before_names),
        "after_feature_names": list(after_names),
        "before_prediction": before_pred,
        "after_prediction": after_pred,
    }


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}}

    # CatBoost base class: MultiClass.
    pool = Pool(x(NUM1_FOR_MULTICLASS), MULTICLASS_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(pool)
    record(fixture, "CatBoost", model, pool, NUM1_FOR_MULTICLASS, MULTICLASS_LABEL,
           predict_kwargs={"prediction_type": "RawFormulaVal"})

    # CatBoostClassifier: Logloss (binary).
    pool = Pool(x(NUM1_FOR_BINARY), BINARY_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(pool)
    record(fixture, "CatBoostClassifier", model, pool, NUM1_FOR_BINARY, BINARY_LABEL,
           predict_kwargs={"prediction_type": "RawFormulaVal"})

    # CatBoostRegressor: RMSE.
    pool = Pool(x(NUM1_FOR_REGRESSION), REGRESSION_LABEL, feature_names=FEATURE_NAMES)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(pool)
    record(fixture, "CatBoostRegressor", model, pool, NUM1_FOR_REGRESSION, REGRESSION_LABEL,
           predict_kwargs={"prediction_type": "RawFormulaVal"})

    # CatBoostRanker: YetiRank (requires group_id).
    pool = Pool(x(NUM1_FOR_BINARY), BINARY_LABEL, feature_names=FEATURE_NAMES, group_id=GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(pool)
    record(fixture, "CatBoostRanker", model, pool, NUM1_FOR_BINARY, BINARY_LABEL, group_id=GROUP_ID)

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
