#!/usr/bin/env python3
"""Generate grid_search / randomized_search oracle fixture (catboost-8z4.59 / P5.2).

Pins Python catboost==1.2.10's CatBoost.grid_search / CatBoost.randomized_search
behavior. Both call the exact same native NCB::GridSearch / NCB::RandomizedSearch
entry points (catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h)
that R's new catboost.grid_search / catboost.randomized_search now call directly
(_catboost.pyx:4403 -> self._object._tune_hyperparams, cpdef at _catboost.pyx:5933,
n_iter == -1 branch -> GridSearch, else -> RandomizedSearch), so best-params and
cv_results are expected to match bit-for-bit, not just approximately.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_grid_search_fixture.py
"""
import json
import os
import sys

from catboost import CatBoost, Pool
import catboost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "grid_search.json")

N_ROWS = 60

FEATURES = [
    [round(0.5 * i - 4.0, 4), round(1.5 * ((i * 7) % 11) - 8.0, 4), round(((i * 5) % 13) / 3.0, 4)]
    for i in range(N_ROWS)
]
LABEL = [round(0.3 * i - 1.0 + 0.7 * ((i * 3) % 5), 4) for i in range(N_ROWS)]

BASE_PARAMS = {
    "iterations": 20,
    "loss_function": "RMSE",
    "thread_count": 1,
    "verbose": False,
}

PARAM_GRID = {
    "depth": [3, 5],
    "learning_rate": [0.05, 0.3],
}

PARAM_DISTRIBUTIONS = {
    "depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1, 0.2, 0.3],
}

# catboost-8z4 Phase 5 whole-branch fix-wave Finding 1: grid_search/
# randomized_search used to truncate hyperparameter values to 10 significant
# digits both outbound (BestOptionValuesToRList's ToString(TJsonValue),
# default DefaultDoubleNDigits=10) and inbound (prepare_grid_json's
# jsonlite::toJSON(..., digits=10)). learning_rate below has 15 significant
# digits so a >10-sig-digit round trip is actually exercised.
HIGH_PRECISION_PARAM_GRID = {
    "depth": [3],
    "learning_rate": [0.123456789012345, 0.05],
}


def run_search(kind, extra, param_grid=None):
    pool = Pool(FEATURES, LABEL)
    model = CatBoost(dict(BASE_PARAMS, train_dir=os.path.join(SCRIPT_DIR, f".catboost_train_{kind}")))
    if kind in ("grid_search", "grid_search_high_precision"):
        result = model.grid_search(
            param_grid if param_grid is not None else PARAM_GRID, pool, cv=3, partition_random_seed=0,
            calc_cv_statistics=True, search_by_train_test_split=True,
            refit=True, shuffle=True, stratified=False, train_size=0.8,
            verbose=False,
        )
    else:
        result = model.randomized_search(
            PARAM_DISTRIBUTIONS, pool, cv=3, n_iter=6, partition_random_seed=0,
            calc_cv_statistics=True, search_by_train_test_split=True,
            refit=True, shuffle=True, stratified=False, train_size=0.8,
            verbose=False,
        )
    refit_predict = [float(v) for v in model.predict(pool, prediction_type="RawFormulaVal")]
    return {
        "params": result["params"],
        "cv_results": {k: [float(x) for x in v] for k, v in result["cv_results"].items()},
        "refit_predict": refit_predict,
        "refit_tree_count": int(model.tree_count_),
    }


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    grid_result = run_search("grid_search", {})
    randomized_result = run_search("randomized_search", {})
    high_precision_result = run_search(
        "grid_search_high_precision", {}, param_grid=HIGH_PRECISION_PARAM_GRID
    )

    fixture = {
        "catboost_version": catboost.__version__,
        "base_params": BASE_PARAMS,
        "param_grid": PARAM_GRID,
        "param_distributions": PARAM_DISTRIBUTIONS,
        "high_precision_param_grid": HIGH_PRECISION_PARAM_GRID,
        "inputs": {
            "features": FEATURES,
            "label": LABEL,
        },
        "expected": {
            "grid_search": grid_result,
            "randomized_search": randomized_result,
            "grid_search_high_precision": high_precision_result,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"grid_search.params={grid_result['params']}", file=sys.stderr)
    print(f"randomized_search.params={randomized_result['params']}", file=sys.stderr)
    print(f"grid_search_high_precision.params={high_precision_result['params']}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
