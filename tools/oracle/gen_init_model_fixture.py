#!/usr/bin/env python3
"""Generate init_model (continue training) oracle fixture (catboost-8z4.58 / P5.1).

Pins Python catboost==1.2.10's CatBoost.fit(..., init_model=...) behavior --
train a base model on one chunk of data, then continue training it on a
second chunk via init_model -- so R's new catboost.train(init_model=...)
argument can be checked against a real oracle instead of only against
R-internal self-consistency.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_init_model_fixture.py
(cwd-independent, same convention as gen_multitarget_fixture.py.)
"""
import json
import os
import sys

from catboost import CatBoost, Pool
import catboost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "init_model.json")

N_ROWS = 16

FEATURES = [
    [round(0.5 * i - 4.0, 4), round(1.5 * ((i * 7) % 11) - 8.0, 4), round(((i * 5) % 13) / 3.0, 4)]
    for i in range(N_ROWS)
]
LABEL = [round(0.3 * i - 1.0, 4) for i in range(N_ROWS)]

# Base model trained on the first half of the rows; continuation trained on
# the second half via init_model. Splitting the data (not just re-running on
# the same rows) makes the continuation numerically distinguishable from a
# from-scratch fit on the same params -- a real differential check on
# init_model, not just on iterations count.
SPLIT = N_ROWS // 2

BASE_PARAMS = {
    "iterations": 10,
    "depth": 3,
    "learning_rate": 0.3,
    "random_seed": 42,
    "thread_count": 1,
    "loss_function": "RMSE",
    "verbose": False,
}
CONTINUE_PARAMS = dict(BASE_PARAMS, iterations=5)


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    base_features = FEATURES[:SPLIT]
    base_label = LABEL[:SPLIT]
    continue_features = FEATURES[SPLIT:]
    continue_label = LABEL[SPLIT:]

    base_pool = Pool(base_features, base_label)
    base_model = CatBoost(dict(BASE_PARAMS, train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_init_model_base")))
    base_model.fit(base_pool)
    base_predict = [float(v) for v in base_model.predict(base_pool, prediction_type="RawFormulaVal")]

    continue_pool = Pool(continue_features, continue_label)
    continued_model = CatBoost(dict(CONTINUE_PARAMS, train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_init_model_continue")))
    continued_model.fit(continue_pool, init_model=base_model)
    continued_predict = [float(v) for v in continued_model.predict(continue_pool, prediction_type="RawFormulaVal")]

    all_pool = Pool(FEATURES, LABEL)
    continued_predict_all = [float(v) for v in continued_model.predict(all_pool, prediction_type="RawFormulaVal")]

    fixture = {
        "catboost_version": catboost.__version__,
        "base_params": BASE_PARAMS,
        "continue_params": CONTINUE_PARAMS,
        "inputs": {
            "features": FEATURES,
            "label": LABEL,
            "split": SPLIT,
        },
        "expected": {
            "base_predict": base_predict,
            "continued_predict": continued_predict,
            "continued_predict_all": continued_predict_all,
            "base_tree_count": int(base_model.tree_count_),
            "continued_tree_count": int(continued_model.tree_count_),
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"base_tree_count={base_model.tree_count_}", file=sys.stderr)
    print(f"continued_tree_count={continued_model.tree_count_}", file=sys.stderr)
    print(f"continued_predict[0]={continued_predict[0]}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
