#!/usr/bin/env python3
"""Generate the multi-target oracle fixture (catboost-8z4.44 / P3.7, part a).

Pins Python catboost==1.2.10's fit/predict output for the two multi-target
losses named in the Phase 3 gate text -- MultiRMSE (float label matrix) and
MultiLogloss (binary label matrix) -- so R's existing
catboost.load_pool/catboost.train/catboost.predict path can be compared
against a real oracle instead of only against its own self-consistency test
(tests/testthat/test_pool.R:39,:53).

No new implementation is exercised here: P2.4's report ("Finding 2:
multi-target losses") already root-caused that only get_object_importance is
broken for multi-target models, while fit/predict work. This fixture turns
that claim into a checked assertion.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_multitarget_fixture.py
(cwd-independent, same convention as gen_smoke_fixture.py.)
"""
import json
import os
import sys

from catboost import CatBoost, Pool
import catboost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "multitarget.json")

N_ROWS = 16

# Three float features, deterministic and free of ties (ties make split
# selection order-dependent and therefore a poor differential probe).
FEATURES = [
    [round(0.5 * i - 4.0, 4), round(1.5 * ((i * 7) % 11) - 8.0, 4), round(((i * 5) % 13) / 3.0, 4)]
    for i in range(N_ROWS)
]

# MultiRMSE: two continuous, correlated-but-distinct targets.
MULTIRMSE_LABEL = [
    [round(0.3 * i + 1.0, 4), round(-0.7 * i + 5.0, 4)]
    for i in range(N_ROWS)
]

# MultiLogloss: two independent binary targets.
MULTILOGLOSS_LABEL = [[i % 2, (i // 2) % 2] for i in range(N_ROWS)]

PARAMS = {
    "iterations": 10,
    "depth": 3,
    "learning_rate": 0.3,
    "random_seed": 42,
    "thread_count": 1,
    "verbose": False,
}


def train_and_predict(label, loss_function, train_dir):
    pool = Pool(FEATURES, label)
    model = CatBoost(dict(PARAMS, loss_function=loss_function,
                          train_dir=os.path.join(SCRIPT_DIR, train_dir)))
    model.fit(pool)
    prediction = model.predict(pool, prediction_type="RawFormulaVal")
    return pool, [[float(v) for v in row] for row in prediction]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    rmse_pool, multirmse_predict = train_and_predict(
        MULTIRMSE_LABEL, "MultiRMSE", ".catboost_train_multirmse")
    logloss_pool, multilogloss_predict = train_and_predict(
        MULTILOGLOSS_LABEL, "MultiLogloss", ".catboost_train_multilogloss")

    fixture = {
        "catboost_version": catboost.__version__,
        "params": PARAMS,
        "inputs": {
            "features": FEATURES,
            "multirmse_label": MULTIRMSE_LABEL,
            "multilogloss_label": MULTILOGLOSS_LABEL,
        },
        "expected": {
            "num_row": int(rmse_pool.num_row()),
            "num_col": int(rmse_pool.num_col()),
            # get_label() of a multi-target Pool is (num_row x target_count).
            "multirmse_get_label": [[float(v) for v in row] for row in rmse_pool.get_label()],
            "multilogloss_get_label": [
                [float(v) for v in row] for row in logloss_pool.get_label()
            ],
            "multirmse_predict": multirmse_predict,
            "multilogloss_predict": multilogloss_predict,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"multirmse_predict[0]={multirmse_predict[0]}", file=sys.stderr)
    print(f"multilogloss_predict[0]={multilogloss_predict[0]}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
