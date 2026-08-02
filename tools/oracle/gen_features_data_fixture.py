#!/usr/bin/env python3
"""Generate the FeaturesData oracle fixture (catboost-8z4.42 / P3.5):
deterministic FeaturesData inputs + the pinned Python catboost==1.2.10
FeaturesData's observable outputs for get_object_count, get_num_feature_count,
get_cat_feature_count, get_feature_count, get_feature_names -- plus the Pool
built from FeaturesData, to verify wiring into Pool construction (round-trip
cat_feature_indices/feature_names/label after going through
catboost.load_pool(data=FeaturesData(...))).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_features_data_fixture.py
"""
import json
import os
import sys

import numpy as np

import catboost
from catboost import FeaturesData, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "features_data.json")

N_ROWS = 8

NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0]
NUM2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
CAT1 = ["a", "b", "a", "c", "b", "a", "c", "b"]
CAT2 = ["x", "y", "x", "x", "y", "y", "x", "y"]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0]

NUM_FEATURE_NAMES = ["num1", "num2"]
CAT_FEATURE_NAMES = ["cat1", "cat2"]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    num_feature_data = np.array(list(zip(NUM1, NUM2)), dtype=np.float32)
    cat_feature_data = np.array(
        list(zip(CAT1, CAT2)), dtype=object
    )
    # elements must have 'bytes' type per FeaturesData docstring
    cat_feature_data_bytes = np.array(
        [[c.encode("utf-8") for c in row] for row in cat_feature_data], dtype=object
    )

    fd_mixed = FeaturesData(
        num_feature_data=num_feature_data,
        cat_feature_data=cat_feature_data_bytes,
        num_feature_names=NUM_FEATURE_NAMES,
        cat_feature_names=CAT_FEATURE_NAMES,
    )

    mixed = {
        "get_object_count": int(fd_mixed.get_object_count()),
        "get_num_feature_count": int(fd_mixed.get_num_feature_count()),
        "get_cat_feature_count": int(fd_mixed.get_cat_feature_count()),
        "get_feature_count": int(fd_mixed.get_feature_count()),
        "get_feature_names": list(fd_mixed.get_feature_names()),
    }

    # num-only FeaturesData, default (unspecified) feature names
    fd_num_only = FeaturesData(num_feature_data=num_feature_data)
    num_only = {
        "get_object_count": int(fd_num_only.get_object_count()),
        "get_num_feature_count": int(fd_num_only.get_num_feature_count()),
        "get_cat_feature_count": int(fd_num_only.get_cat_feature_count()),
        "get_feature_count": int(fd_num_only.get_feature_count()),
        "get_feature_names": list(fd_num_only.get_feature_names()),
    }

    # cat-only FeaturesData, default (unspecified) feature names
    fd_cat_only = FeaturesData(cat_feature_data=cat_feature_data_bytes)
    cat_only = {
        "get_object_count": int(fd_cat_only.get_object_count()),
        "get_num_feature_count": int(fd_cat_only.get_num_feature_count()),
        "get_cat_feature_count": int(fd_cat_only.get_cat_feature_count()),
        "get_feature_count": int(fd_cat_only.get_feature_count()),
        "get_feature_names": list(fd_cat_only.get_feature_names()),
    }

    # Round-trip through Pool construction: data=FeaturesData(...)
    pool = Pool(data=fd_mixed, label=LABEL)
    pool_from_features_data = {
        "cat_feature_indices": [int(i) for i in pool.get_cat_feature_indices()],
        "feature_names": list(pool.get_feature_names()),
        "num_row": int(pool.num_row()),
        "num_col": int(pool.num_col()),
        "get_label": [float(v) for v in pool.get_label()],
    }

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "cat1": CAT1,
            "cat2": CAT2,
            "label": LABEL,
            "num_feature_names": NUM_FEATURE_NAMES,
            "cat_feature_names": CAT_FEATURE_NAMES,
        },
        "expected": {
            "mixed": mixed,
            "num_only": num_only,
            "cat_only": cat_only,
            "pool_from_features_data": pool_from_features_data,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"mixed={mixed}", file=sys.stderr)
    print(f"pool_from_features_data={pool_from_features_data}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
