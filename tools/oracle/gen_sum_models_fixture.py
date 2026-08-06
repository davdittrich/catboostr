#!/usr/bin/env python3
"""Generate the oracle fixture proving R's catboost.sum_models matches
Python's sum_models() (catboost-8z4.78 -- closing the catboost.sum_models row
surfaced by catboost-8z4.66's pipeline fix).

Python's top-level sum_models() (core.py) constructs a CatBoost() and calls
its _sum_models() (_catboost.pyx:5884), which calls SumModels() (catboost/
libs/model/model_export/... model.h) directly on the input models' native
TFullModel objects. R's catboost.sum_models (R/catboost.R) does the same:
CatBoostSumModels_R (src/catboostr.cpp) calls the identical SumModels() entry
point on its own loaded TFullModel handles.

Because SumModels() operates purely on already-trained model artifacts (not
on training data), this fixture trains two models in Python, serializes them
to portable .cbm files, and sums/predicts in Python once as ground truth. The
R-side test loads the *same* .cbm files (not R-retrained models) via
catboost.load_model and calls catboost.sum_models on them, isolating the
comparison to SumModels() itself rather than conflating it with training
parity (already covered separately by test_params_precision.R).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_sum_models_fixture.py
"""
import json
import os
import sys

import catboost
from catboost import CatBoostRegressor, Pool, sum_models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/oracle -> tools -> repo root
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")

N_ROWS = 50
FEATURES = [
    [round(0.4 * i - 3.0, 4), round(1.2 * ((i * 5) % 9) - 5.0, 4)]
    for i in range(N_ROWS)
]
LABEL_A = [round(0.3 * i - 1.0 + 0.7 * ((i * 3) % 5), 4) for i in range(N_ROWS)]
LABEL_B = [round(-0.2 * i + 2.0 + 0.5 * ((i * 2) % 7), 4) for i in range(N_ROWS)]

PARAMS_A = {"iterations": 12, "depth": 3, "learning_rate": 0.2, "loss_function": "RMSE",
            "random_seed": 1, "thread_count": 1, "verbose": False, "allow_writing_files": False}
PARAMS_B = {"iterations": 8, "depth": 4, "learning_rate": 0.1, "loss_function": "RMSE",
            "random_seed": 2, "thread_count": 1, "verbose": False, "allow_writing_files": False}

WEIGHTS = [0.6, 0.4]
CTR_MERGE_POLICY = "IntersectingCountersAverage"


def train(params, label):
    pool = Pool(FEATURES, label)
    model = CatBoostRegressor(**params)
    model.fit(pool)
    return model


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    model_a = train(PARAMS_A, LABEL_A)
    model_b = train(PARAMS_B, LABEL_B)

    model_a_path = os.path.join(FIXTURE_DIR, "sum_models_a.cbm")
    model_b_path = os.path.join(FIXTURE_DIR, "sum_models_b.cbm")
    model_a.save_model(model_a_path)
    model_b.save_model(model_b_path)

    pred_pool = Pool(FEATURES)
    summed = sum_models([model_a, model_b], weights=WEIGHTS, ctr_merge_policy=CTR_MERGE_POLICY)
    predictions = [float(v) for v in summed.predict(pred_pool)]

    fixture = {
        "catboost_version": catboost.__version__,
        "weights": WEIGHTS,
        "ctr_merge_policy": CTR_MERGE_POLICY,
        "model_a_file": os.path.basename(model_a_path),
        "model_b_file": os.path.basename(model_b_path),
        "inputs": {
            "features": FEATURES,
        },
        "expected": {
            "predictions": predictions,
        },
    }

    fixture_path = os.path.join(FIXTURE_DIR, "sum_models.json")
    with open(fixture_path, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"n_predictions={len(predictions)}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
