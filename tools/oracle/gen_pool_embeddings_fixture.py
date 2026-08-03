#!/usr/bin/env python3
"""Generate the embedding-features oracle fixture (catboost-8z4.44 / P3.7, part b).

Pins Python catboost==1.2.10's Pool construction and fit/predict output for a
dataset with one embedding feature, so R's new
catboost.load_pool(embedding_features = ...) /
catboost.from_matrix(embedding_features_data = ...) path can be compared
against a real oracle.

Python holds embedding features inside the data frame (a column whose cells
are arrays) and names their positions via Pool(embedding_features = [...]).
An R matrix cell cannot hold a vector, so the R side passes the embedding
features beside the matrix as a named list of (object x dimension) numeric
matrices; the two must produce the same Pool. The feature order is therefore
pinned here as [f0, f1, emb] with emb at flat feature index 2.

Embedding processing is pinned to KNN. The default ("LDA", "KNN") is NOT
usable as a differential probe: this fork's compiled binary and the pinned
Python wheel disagree on the LDA calcer's output even when both load the very
same dsv file through the very same file loader (control run recorded in
docs/phase-3/catboost-8z4.44-report.md: R -0.218181826852 vs Python
-0.27272728 on iterations=1/depth=1), so the disagreement predates and is
independent of any Pool-construction path. The default-processing predictions
are still recorded below as predict_default_lda for auditability, but the R
test asserts against the KNN run, which agrees exactly.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_embeddings_fixture.py
"""
import json
import os
import sys

import pandas as pd

from catboost import CatBoost, Pool
import catboost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pool_embeddings.json")

N_ROWS = 24
EMBEDDING_DIM = 4

F0 = [round(0.25 * i - 3.0, 4) for i in range(N_ROWS)]
F1 = [round(((i * 7) % 13) / 2.0 - 3.0, 4) for i in range(N_ROWS)]

# Two separable clouds in embedding space so the embedding feature actually
# carries signal (an ignored feature would make the differential test vacuous).
EMBEDDING = [
    [round(float((i % 2) * 2 + (i * (d + 1) % 5) / 4.0), 4) for d in range(EMBEDDING_DIM)]
    for i in range(N_ROWS)
]

LABEL = [i % 2 for i in range(N_ROWS)]

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 10,
    "depth": 3,
    "learning_rate": 0.3,
    "random_seed": 42,
    "thread_count": 1,
    "verbose": False,
    "embedding_processing": {"embedding_processing": {"default": ["KNN"]}},
}


def fit_predict(pool, params, train_dir):
    model = CatBoost(dict(params, train_dir=os.path.join(SCRIPT_DIR, train_dir)))
    model.fit(pool)
    return [float(v) for v in model.predict(pool, prediction_type="RawFormulaVal")]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    data = pd.DataFrame({"f0": F0, "f1": F1, "emb": EMBEDDING})
    pool = Pool(data=data, label=LABEL, embedding_features=[2])

    predict = fit_predict(pool, PARAMS, ".catboost_train_embeddings")

    default_params = {k: v for k, v in PARAMS.items() if k != "embedding_processing"}
    predict_default_lda = fit_predict(pool, default_params, ".catboost_train_embeddings_lda")

    fixture = {
        "catboost_version": catboost.__version__,
        "params": PARAMS,
        "inputs": {
            "f0": F0,
            "f1": F1,
            "embedding": EMBEDDING,
            "label": LABEL,
        },
        "expected": {
            "num_row": int(pool.num_row()),
            "num_col": int(pool.num_col()),
            "feature_names": list(pool.get_feature_names()),
            "embedding_feature_indices": [int(i) for i in pool.get_embedding_feature_indices()],
            "cat_feature_indices": [int(i) for i in pool.get_cat_feature_indices()],
            "text_feature_indices": [int(i) for i in pool.get_text_feature_indices()],
            "label": [float(v) for v in pool.get_label()],
            "predict": predict,
            # Recorded for auditability only; see the module docstring -- the
            # R test does not assert against this one.
            "predict_default_lda": predict_default_lda,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"embedding_feature_indices={fixture['expected']['embedding_feature_indices']}", file=sys.stderr)
    print(f"feature_names={fixture['expected']['feature_names']}", file=sys.stderr)
    print(f"predict[:3]={predict[:3]}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
