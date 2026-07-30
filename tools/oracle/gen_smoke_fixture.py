#!/usr/bin/env python3
"""Generate the oracle smoke fixture: deterministic dataset + params + CatBoost
predictions + model file, all written full-precision (repr round-trip).

Run via: uv run --project tools/oracle python3 tools/oracle/gen_smoke_fixture.py
(cwd-independent: fixture output dir and CatBoost's train_dir are both
resolved from this file's own location, not the invocation cwd, so running
from elsewhere cannot litter the repo root with catboost_info/.)

Determinism check (brief step 7): run this script twice in two fresh
processes and diff the two predictions JSON files. They must be byte-identical.
"""
import csv
import json
import os
import random
import sys

import catboost
import numpy as np
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/oracle -> tools -> repo root
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
TRAIN_DIR = os.path.join(SCRIPT_DIR, ".catboost_train")  # CatBoost scratch dir, stays in tools/oracle/
SEED = 42
N_ROWS = 40
PARAMS = {
    "loss_function": "Logloss",
    "iterations": 20,
    "depth": 4,
    "learning_rate": 0.1,
    "random_seed": SEED,
    "verbose": False,
}
# train_dir is a CatBoost runtime scratch dir (learn/, tmp/, *.tsv), not a fixture
# output — kept separate from PARAMS so it never lands in smoke_params.json.
TRAIN_KWARGS = {"train_dir": TRAIN_DIR}


def make_dataset(seed: int, n_rows: int):
    rng = random.Random(seed)
    cat_levels = ["a", "b", "c"]
    rows = []
    for _ in range(n_rows):
        num1 = rng.uniform(-10.0, 10.0)
        num2 = rng.gauss(0.0, 1.0)
        cat1 = rng.choice(cat_levels)
        # deterministic-but-nontrivial label rule + noise
        score = num1 * 0.3 + num2 * 0.7 + (1.0 if cat1 == "a" else -0.5)
        label = 1 if score + rng.uniform(-1.0, 1.0) > 0 else 0
        rows.append({"num1": num1, "num2": num2, "cat1": cat1, "target": label})
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # repr() round-trips Python floats exactly (float_repr_style=short since 3.1)
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    rows = make_dataset(SEED, N_ROWS)
    feature_cols = ["num1", "num2", "cat1"]
    write_csv(f"{FIXTURE_DIR}/smoke_data.csv", rows, feature_cols + ["target"])

    X = [[r["num1"], r["num2"], r["cat1"]] for r in rows]
    y = [r["target"] for r in rows]
    pool = Pool(X, y, cat_features=[2], feature_names=feature_cols)

    model = CatBoostClassifier(**PARAMS, **TRAIN_KWARGS)
    model.fit(pool)

    raw_preds = model.predict(pool, prediction_type="RawFormulaVal")
    raw_preds = [float(v) for v in np.asarray(raw_preds).ravel()]

    with open(f"{FIXTURE_DIR}/smoke_params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)
        f.write("\n")

    with open(f"{FIXTURE_DIR}/smoke_predictions.json", "w") as f:
        json.dump(
            {
                "catboost_version": catboost.__version__,
                "prediction_type": "RawFormulaVal",
                "predictions": raw_preds,
            },
            f,
            indent=2,
        )
        f.write("\n")

    model.save_model(f"{FIXTURE_DIR}/smoke_model.cbm")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"n_predictions={len(raw_preds)}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
