#!/usr/bin/env python3
"""Generate the pool-introspection oracle fixture (catboost-8z4.39 / P3.2):
deterministic Pool inputs + the pinned Python catboost==1.2.10 Pool's
observable outputs for get_cat_feature_indices, get_text_feature_indices,
get_embedding_feature_indices, get_feature_names, set_feature_names,
get_features, shape, num_col, num_row, is_empty_.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_introspection_fixture.py

Text and embedding features are exercised only as "always empty" probes:
catboost.load_pool()'s public matrix/data.frame path (R/catboost.R) has no
way to mark a column as text or embedding -- catboost.from_matrix() takes
text_features_data/text_features_indices internally, but catboost.load_pool()
never forwards user input into them, and always passes NULL/0. So any Pool
reachable from R public API always has 0 text/embedding features, and this
fixture's oracle Pool (built the same way get_cat_feature_indices'
docstring in _catboost.pyx documents, via cat_features=) matches that.

get_features() only supports Pools with exclusively numeric (float) columns
(see _catboost.pyx get_features(): "Pool has non-numeric features" if
GetExternalFeatureCount() != GetFloatFeatureCount()), so it is probed on a
separate all-numeric pool.
"""
import json
import os
import sys

import numpy as np

import catboost
from catboost import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pool_introspection.json")

N_ROWS = 8

NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0]
NUM2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
CAT1 = ["a", "b", "a", "c", "b", "a", "c", "b"]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0]

NUMERIC_ONLY = [[NUM1[i], NUM2[i], float(i)] for i in range(N_ROWS)]
NEW_FEATURE_NAMES = ["renamed_num1", "renamed_cat1"]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    X = [[n, c] for n, c in zip(NUM1, CAT1)]
    mixed_pool = Pool(X, LABEL, cat_features=[1], feature_names=["num1", "cat1"])

    before = {
        "cat_feature_indices": [int(i) for i in mixed_pool.get_cat_feature_indices()],
        "text_feature_indices": [int(i) for i in mixed_pool.get_text_feature_indices()],
        "embedding_feature_indices": [int(i) for i in mixed_pool.get_embedding_feature_indices()],
        "feature_names": list(mixed_pool.get_feature_names()),
        "num_row": int(mixed_pool.num_row()),
        "num_col": int(mixed_pool.num_col()),
        "shape": list(mixed_pool.shape),
        "is_empty_": bool(mixed_pool.is_empty_),
    }

    mixed_pool.set_feature_names(NEW_FEATURE_NAMES)
    after_rename = {
        "feature_names": list(mixed_pool.get_feature_names()),
    }

    numeric_pool = Pool(NUMERIC_ONLY, LABEL, feature_names=["n1", "n2", "n3"])
    features = numeric_pool.get_features()

    # 0-row Pool, built the same way R's catboost.load_pool(matrix(nrow=0,
    # ncol=1)) constructs one -- an all-numeric matrix with 0 rows.
    empty_pool = Pool(np.empty((0, 1), dtype=np.float32))

    expected = {
        "before": before,
        "after_rename": after_rename,
        "numeric_features": [[float(v) for v in row] for row in features],
        "numeric_shape": list(numeric_pool.shape),
        "empty_is_empty_": bool(empty_pool.is_empty_),
        "empty_num_row": int(empty_pool.num_row()),
        "nonempty_is_empty_": bool(mixed_pool.is_empty_),
    }

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "cat1": CAT1,
            "label": LABEL,
            "numeric_only": NUMERIC_ONLY,
            "new_feature_names": NEW_FEATURE_NAMES,
        },
        "expected": expected,
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"cat_feature_indices={before['cat_feature_indices']}", file=sys.stderr)
    print(f"shape={before['shape']}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
