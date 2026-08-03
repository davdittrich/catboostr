#!/usr/bin/env python3
"""Generate the sparse-input oracle fixture (catboost-8z4.44 / P3.7, part c).

Pins Python catboost==1.2.10's Pool construction and fit/predict output for a
Pool built from a scipy.sparse.csr_matrix, so R's new dgCMatrix input path can
be compared against the oracle's scipy-sparse path.

The stored values are baked into the fixture as a dense row list; both sides
rebuild their own sparse container from it (scipy CSR here, Matrix::dgCMatrix
in R), which keeps the fixture free of format-specific index/pointer arrays.

Note on the dataset: a degenerate mostly-zero matrix makes several split
candidates score-tied, and CatBoost's tie-breaking differs between its sparse
and dense column layouts (observed: max |dense - sparse| prediction delta
0.0298 on a 12x5 all-but-three-zeros matrix). The fixture therefore uses a
non-degenerate ~40%-filled matrix, on which the sparse and dense training
paths agree bit-for-bit; both are recorded below so the R test can assert
against the sparse path and the equality is auditable.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_sparse_fixture.py
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from catboost import CatBoost, Pool
import catboost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pool_sparse.json")

N_ROWS = 40
N_COLS = 6

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 10,
    "depth": 3,
    "learning_rate": 0.3,
    "random_seed": 42,
    "thread_count": 1,
    "verbose": False,
}


def build_dense():
    rng = np.random.RandomState(7)
    dense = np.zeros((N_ROWS, N_COLS))
    mask = rng.rand(N_ROWS, N_COLS) < 0.4
    dense[mask] = np.round(rng.randn(mask.sum()) * 2, 3)
    return dense


def train_and_predict(data, train_dir, label):
    pool = Pool(data, label)
    model = CatBoost(dict(PARAMS, train_dir=os.path.join(SCRIPT_DIR, train_dir)))
    model.fit(pool)
    return pool, [float(v) for v in model.predict(pool, prediction_type="RawFormulaVal")]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    dense = build_dense()
    label = [int(dense[i, 0] + dense[i, 1] > 0) for i in range(N_ROWS)]
    csr = sp.csr_matrix(dense)

    sparse_pool, sparse_predict = train_and_predict(csr, ".catboost_train_sparse", label)
    _, dense_predict = train_and_predict(dense, ".catboost_train_sparse_dense", label)

    max_sparse_dense_delta = float(np.max(np.abs(np.array(sparse_predict) - np.array(dense_predict))))

    fixture = {
        "catboost_version": catboost.__version__,
        "params": PARAMS,
        "inputs": {
            "dense": [[float(v) for v in row] for row in dense],
            "label": label,
            "nnz": int(csr.nnz),
        },
        "expected": {
            "num_row": int(sparse_pool.num_row()),
            "num_col": int(sparse_pool.num_col()),
            "features": [[float(v) for v in row] for row in np.asarray(sparse_pool.get_features())],
            "feature_names": list(sparse_pool.get_feature_names()),
            "label": [float(v) for v in sparse_pool.get_label()],
            "predict": sparse_predict,
            "dense_predict": dense_predict,
            "max_sparse_dense_delta": max_sparse_dense_delta,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"nnz={csr.nnz} of {N_ROWS * N_COLS}", file=sys.stderr)
    print(f"max_sparse_dense_delta={max_sparse_dense_delta}", file=sys.stderr)
    print(f"predict[:3]={sparse_predict[:3]}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
