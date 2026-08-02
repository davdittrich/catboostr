#!/usr/bin/env python3
"""Generate the pool-metadata oracle fixture (catboost-8z4.38 / P3.1):
deterministic Pool inputs + the pinned Python catboost==1.2.10 Pool's
observable outputs for has_label/get_label/get_weight/get_baseline/
get_group_id_hash/num_pairs, after exercising every one of the 14
metadata accessor/mutator methods this ticket adds R equivalents for
(the remaining 8 -- set_weight, set_baseline, set_group_id,
set_group_weight, set_subgroup_id, set_pairs, set_pairs_weight,
set_timestamp -- are mutators with no matching Python getter, so their
correctness is only observable indirectly: through get_weight/
get_baseline/get_group_id_hash/num_pairs above, or (set_group_weight/
set_subgroup_id/set_timestamp) by not raising).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_metadata_fixture.py
(cwd-independent, same convention as gen_smoke_fixture.py.)

get_group_id_hash values are ui64 and are written as decimal STRINGS, not
JSON numbers: JSON numbers get parsed back into R's double (53-bit mantissa)
by both Python's own json module round-trips through float for some parsers
and, decisively, by R's jsonlite -- either path would silently truncate
values above 2^53. Strings round-trip exactly on both ends.
"""
import json
import os
import sys

import catboost
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pool_metadata.json")

N_ROWS = 12

NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25, -10.0, 11.5]
CAT1 = ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c", "b", "a"]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
WEIGHT = [1.0, 2.0, 0.5, 1.5, 3.0, 0.25, 2.5, 1.0, 0.75, 1.25, 2.25, 0.1]
BASELINE = [[0.1 * i] for i in range(N_ROWS)]  # (N x 1)

# 3 contiguous groups of 4 rows each -- SetGroupIds requires objects with
# the same group id to already be contiguous (see docs/phase-3 report for
# CheckPairs/CheckGroupWeights preconditions this shape satisfies).
GROUP_ID = [111, 111, 111, 111, 222, 222, 222, 222, 333, 333, 333, 333]
GROUP_WEIGHT = [1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0]
SUBGROUP_ID = list(range(1, N_ROWS + 1))

# Pairs must stay within a single group (CheckPairs, target.cpp:162-171) once
# a non-trivial grouping is set -- keep them inside the first group (rows 0-3).
PAIRS = [[0, 1], [1, 2], [2, 3]]
PAIRS_WEIGHT = [0.1, 0.2, 0.3]

TIMESTAMP = [1000 + i for i in range(N_ROWS)]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    X = [[n, c] for n, c in zip(NUM1, CAT1)]
    pool = Pool(X, LABEL, cat_features=[1], feature_names=["num1", "cat1"])

    pool.set_weight(WEIGHT)
    pool.set_baseline(BASELINE)
    pool.set_group_id(GROUP_ID)
    pool.set_group_weight(GROUP_WEIGHT)
    pool.set_subgroup_id(SUBGROUP_ID)
    pool.set_pairs(PAIRS)
    pool.set_pairs_weight(PAIRS_WEIGHT)
    pool.set_timestamp(TIMESTAMP)

    # Exercise training on the fully-mutated pool as an end-to-end sanity
    # check that these mutations produced an internally consistent pool
    # (mirrors the ticket's own note that P1.6-report style CLI training is
    # the ultimate arbiter -- if the pool were corrupt, fit() would raise).
    model = CatBoostClassifier(
        iterations=5, depth=2, loss_function="Logloss",
        random_seed=42, thread_count=1, verbose=False,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train"),
    )
    model.fit(pool)

    expected = {
        "has_label": bool(pool.has_label()),
        "label": [float(v) for v in pool.get_label()],
        "weight": [float(v) for v in pool.get_weight()],
        "baseline": [[float(v) for v in row] for row in pool.get_baseline()],
        "group_id_hash": [str(int(v)) for v in pool.get_group_id_hash()],
        "num_pairs": int(pool.num_pairs()),
    }

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "num1": NUM1,
            "cat1": CAT1,
            "label": LABEL,
            "weight": WEIGHT,
            "baseline": BASELINE,
            "group_id": GROUP_ID,
            "group_weight": GROUP_WEIGHT,
            "subgroup_id": SUBGROUP_ID,
            "pairs": PAIRS,
            "pairs_weight": PAIRS_WEIGHT,
            "timestamp": TIMESTAMP,
        },
        "expected": expected,
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"num_pairs={expected['num_pairs']}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
