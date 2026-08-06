#!/usr/bin/env python3
"""Generate the oracle fixture proving R's catboost.cv matches Python's cv()
(catboost-8z4.78 -- closing the catboost.cv row surfaced by catboost-8z4.66's
pipeline fix).

Python's top-level cv() (core.py) calls _cv() (_catboost.pyx:6287), which
builds a TCrossValidationParams and calls CrossValidate()
(catboost/libs/train_lib/cross_validation.cpp) directly. R's catboost.cv
(R/catboost.R) does the same: CatBoostCV_R (src/catboostr.cpp) builds its own
TCrossValidationParams and calls the identical CrossValidate() entry point.
Both wrappers are thin field-for-field constructors around the same native
call, so the returned per-iteration test/train mean/std columns are expected
to match bit-for-bit, not just approximately (same precedent as
catboost-8z4.59's grid_search/randomized_search oracle, gen_grid_search_fixture.py).

This is a *different* capability from the CLI's `--cv` flag (flag:--cv row,
catboost-8z4.76 / closure_overlay.json): that flag parses into a distinct
TCvDataPartitionParams struct picking a single train/test split, a different
native code path entirely. This fixture is Python-cv() vs R-catboost.cv only.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_cv_fixture.py
"""
import json
import os
import sys

import catboost
from catboost import Pool, cv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/oracle -> tools -> repo root
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "cv.json")

N_ROWS = 80
FEATURES = [
    [round(0.5 * i - 4.0, 4), round(1.5 * ((i * 7) % 11) - 8.0, 4), round(((i * 5) % 13) / 3.0, 4)]
    for i in range(N_ROWS)
]
LABEL = [(i * 3) % 2 for i in range(N_ROWS)]

BASE_PARAMS = {
    "iterations": 15,
    "depth": 3,
    "loss_function": "Logloss",
    "learning_rate": 0.1,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}


def run_cv(type_, fold_count=3, partition_random_seed=0, shuffle=True, stratified=False):
    pool = Pool(FEATURES, LABEL)
    result = cv(
        pool,
        params=BASE_PARAMS,
        fold_count=fold_count,
        type=type_,
        partition_random_seed=partition_random_seed,
        shuffle=shuffle,
        stratified=stratified,
        as_pandas=True,
    )
    return {col: [float(v) for v in result[col]] for col in result.columns}


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    classical_result = run_cv("Classical", fold_count=4, partition_random_seed=17, shuffle=True, stratified=True)
    inverted_result = run_cv("Inverted", fold_count=4, partition_random_seed=17, shuffle=True, stratified=True)

    fixture = {
        "catboost_version": catboost.__version__,
        "base_params": BASE_PARAMS,
        "fold_count": 4,
        "partition_random_seed": 17,
        "shuffle": True,
        "stratified": True,
        "inputs": {
            "features": FEATURES,
            "label": LABEL,
        },
        "expected": {
            "classical": classical_result,
            "inverted": inverted_result,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"classical columns={sorted(classical_result.keys())}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
