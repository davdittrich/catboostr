#!/usr/bin/env python3
"""Generate the oracle fixture proving catboost.train's params JSON
round-trips learning_rate/l2_leaf_reg at >10 significant digits without
truncation (catboost-8z4.65: prepare_train_export_parameters used
jsonlite::toJSON(..., digits = 10), silently truncating hyperparameters
before they ever reached the native training call -- same defect class as
prepare_grid_json's earlier fix, catboost-8z4.60).

Python never truncates these values (passed as native doubles, no JSON
round trip involved), so its predictions are the ground truth: if R's
inbound JSON still truncated learning_rate/l2_leaf_reg to 10 significant
digits, the trained model would differ measurably from this fixture's
predictions.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_train_high_precision_fixture.py
"""
import csv
import json
import os
import random
import sys

import catboost
import numpy as np
from catboost import CatBoostRegressor, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/oracle -> tools -> repo root
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
TRAIN_DIR = os.path.join(SCRIPT_DIR, ".catboost_train")
SEED = 42
N_ROWS = 40
# 15 and 16 significant digits respectively -- both well past jsonlite's
# digits = 10 default truncation point.
PARAMS = {
    "loss_function": "RMSE",
    "iterations": 10,
    "depth": 4,
    "learning_rate": 0.123456789012345,
    "l2_leaf_reg": 3.141592653589793,
    "random_seed": SEED,
    "thread_count": 1,
    "verbose": False,
}
TRAIN_KWARGS = {"train_dir": TRAIN_DIR}


def make_dataset(seed: int, n_rows: int):
    rng = random.Random(seed)
    rows = []
    for _ in range(n_rows):
        num1 = rng.uniform(-10.0, 10.0)
        num2 = rng.gauss(0.0, 1.0)
        target = num1 * 0.3 + num2 * 0.7 + rng.uniform(-1.0, 1.0)
        rows.append({"num1": num1, "num2": num2, "target": target})
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: repr(v) for k, v in r.items()})


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    rows = make_dataset(SEED, N_ROWS)
    feature_cols = ["num1", "num2"]
    write_csv(f"{FIXTURE_DIR}/train_high_precision_data.csv", rows, feature_cols + ["target"])

    X = [[r["num1"], r["num2"]] for r in rows]
    y = [r["target"] for r in rows]
    pool = Pool(X, y, feature_names=feature_cols)

    model = CatBoostRegressor(**PARAMS, **TRAIN_KWARGS)
    model.fit(pool)

    raw_preds = model.predict(pool)
    raw_preds = [float(v) for v in np.asarray(raw_preds).ravel()]

    with open(f"{FIXTURE_DIR}/train_high_precision.json", "w") as f:
        json.dump(
            {
                "catboost_version": catboost.__version__,
                "params": PARAMS,
                "predictions": raw_preds,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"n_predictions={len(raw_preds)}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
