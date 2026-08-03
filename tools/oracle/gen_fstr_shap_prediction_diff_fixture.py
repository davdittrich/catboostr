#!/usr/bin/env python3
"""Generate ShapInteractionValues / PredictionDiff oracle fixture
(catboost-8z4.53 / P4.4): deterministic train data + a pinned Python
catboost==1.2.10 model's observable get_feature_importance() output for
type='ShapInteractionValues' and type='PredictionDiff' -- the differential
test data for catboost.get_feature_importance()'s argument-value parity
(R/catboost.R:3130).

Both types route through CatBoostClassifier._calc_fstr() (core.py:3555),
which for ShapInteractionValues calls the native CalcShapFeatureInteractionMulti
entry point (catboost/libs/fstr/calc_fstr.h) and for PredictionDiff calls
GetFeatureImportances(EFstrType::PredictionDiff, ...) -> GetPredictionDiff
(catboost/libs/fstr/compare_documents.h), both wrapping the same C++ engine
the R package's src/catboostr.cpp binds. This fixture pins get_feature_importance()'s
observable numpy output for both types so the R port can be checked byte-for-byte
(within float tolerance) against the same C++ computation.

Also pins a MultiClass model's ShapInteractionValues output (fix round 1,
review finding: src/catboostr.cpp's CatBoostCalcRegularFeatureEffect_R has a
separate 4-nested-loop reordering branch for `multiClass` -- model->GetDimensionsCount() > 1
-- with an extra `dim` axis that the binary-classification fixture above never
exercises).

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_fstr_shap_prediction_diff_fixture.py
"""
import json
import os

from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "fstr_shap_prediction_diff.json")

N_ROWS = 20
NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
        -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0]
NUM2 = [0.3 * i for i in range(N_ROWS)]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
# 3-class label for the MultiClass ShapInteractionValues fixture (fix round 1).
MULTICLASS_LABEL = [i % 3 for i in range(N_ROWS)]
FEATURE_NAMES = ["num1", "num2"]

# ShapInteractionValues evaluated on the first 5 training rows (small enough
# to keep the fixture readable; enough rows to exercise the doc axis).
SHAP_INTERACTION_ROWS = 5

# PredictionDiff requires data.num_row() == 2 exactly.
PREDICTION_DIFF_ROWS = [0, 1]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    X = [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]
    pool = Pool(X, LABEL, feature_names=FEATURE_NAMES)

    model = CatBoostClassifier(
        iterations=10, depth=2, loss_function="Logloss",
        random_seed=42, thread_count=1, verbose=False,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_fstr"),
    )
    model.fit(pool)

    shap_pool = Pool(
        X[:SHAP_INTERACTION_ROWS], LABEL[:SHAP_INTERACTION_ROWS],
        feature_names=FEATURE_NAMES,
    )
    shap_interaction_values = model.get_feature_importance(
        data=shap_pool, type="ShapInteractionValues",
    )

    diff_pool = Pool(
        [X[i] for i in PREDICTION_DIFF_ROWS],
        feature_names=FEATURE_NAMES,
    )
    prediction_diff = model.get_feature_importance(
        data=diff_pool, type="PredictionDiff",
    )

    # MultiClass model: exercises the `multiClass` 4-nested-loop reordering
    # branch (extra `dim` axis) in CatBoostCalcRegularFeatureEffect_R that the
    # binary Logloss model above never reaches.
    multiclass_pool = Pool(X, MULTICLASS_LABEL, feature_names=FEATURE_NAMES)
    multiclass_model = CatBoostClassifier(
        iterations=10, depth=2, loss_function="MultiClass",
        random_seed=42, thread_count=1, verbose=False,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_fstr_multiclass"),
    )
    multiclass_model.fit(multiclass_pool)
    multiclass_shap_interaction_values = multiclass_model.get_feature_importance(
        data=shap_pool, type="ShapInteractionValues",
    )

    fixture = {
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "label": LABEL,
            "multiclass_label": MULTICLASS_LABEL,
            "feature_names": FEATURE_NAMES,
            "shap_interaction_rows": SHAP_INTERACTION_ROWS,
            "prediction_diff_rows": PREDICTION_DIFF_ROWS,
        },
        "expected": {
            # shape: (n_objects, n_features + 1, n_features + 1)
            "shap_interaction_values": shap_interaction_values.tolist(),
            # shape: (n_features,)
            "prediction_diff": prediction_diff.tolist(),
            # shape: (n_objects, n_classes, n_features + 1, n_features + 1)
            "multiclass_shap_interaction_values": multiclass_shap_interaction_values.tolist(),
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
