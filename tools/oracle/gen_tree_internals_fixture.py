#!/usr/bin/env python3
"""Generate the tree-internals-accessor oracle fixture (P10.E,
catboost-8z4.119): pins Python catboost==1.2.10's get_borders()/
save_borders()/get_leaf_values()/get_leaf_weights()/get_tree_leaf_counts()/
set_leaf_values()/calc_leaf_indexes()/iterate_leaf_indexes() for one fitted
model.

All 8 methods are defined exactly once -- get_leaf_values/get_leaf_weights/
get_tree_leaf_counts/set_leaf_values on _CatBoostBase (core.py:2112-2152);
get_borders/save_borders/calc_leaf_indexes/iterate_leaf_indexes on the
CatBoost class itself (core.py:3132-3195, 3809-3827) -- and none of them is
overridden by CatBoostClassifier/CatBoostRegressor/CatBoostRanker, so a
single fixture model exercises all 32 matrix rows (8 families x 4 classes),
same precedent as gen_scale_and_bias_fixture.py/
gen_training_history_classes_fixture.py.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_tree_internals_fixture.py
"""
import json
import os

import numpy as np
from catboost import CatBoostRegressor, Pool
from catboost.core import _NumpyAwareEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "tree_internals.json")

N_ROWS = 20
NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
        -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0]
NUM2 = [0.3 * i for i in range(N_ROWS)]
LABEL = [0.1, 1.2, 0.3, 1.4, 0.5, 1.6, 1.7, 0.8, 1.9, 0.1,
         1.2, 0.3, 0.4, 1.5, 0.6, 1.7, 0.8, 1.9, 1.1, 0.2]
FEATURE_NAMES = ["num1", "num2"]

os.makedirs(FIXTURE_DIR, exist_ok=True)

X = [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]
pool = Pool(X, LABEL, feature_names=FEATURE_NAMES)

model = CatBoostRegressor(
    iterations=10, depth=2, loss_function="RMSE",
    random_seed=42, thread_count=1, logging_level="Silent",
    train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_tree_internals"),
)
model.fit(pool)

borders = model.get_borders()

borders_path = os.path.join(SCRIPT_DIR, ".tree_internals_borders.tmp")
model.save_borders(borders_path)
with open(borders_path, "r") as f:
    borders_text = f.read()
os.remove(borders_path)

leaf_values = model.get_leaf_values()
leaf_weights = model.get_leaf_weights()
tree_leaf_counts = model.get_tree_leaf_counts()

# calc_leaf_indexes/iterate_leaf_indexes: full range and a restricted
# ntree_start/ntree_end range, both over the same pool used to fit (no
# separate eval pool needed -- these methods only apply an already-fitted
# model, they don't touch fit()'s own train/test split).
leaf_indexes_full = model.calc_leaf_indexes(pool)
leaf_indexes_range = model.calc_leaf_indexes(pool, ntree_start=1, ntree_end=4)
leaf_indexes_iter = list(model.iterate_leaf_indexes(pool))

# set_leaf_values: perturb every leaf value by a deterministic offset (no
# re-fit needed) so the earlier reads above stay valid for the
# pre-mutation fixture entries.
new_leaf_values = np.array([v + 0.5 for v in leaf_values], dtype=np.float64)
model.set_leaf_values(new_leaf_values)
leaf_values_after_set = model.get_leaf_values()

# CatBoost's TFullModel caches a compiled formula evaluator that is NOT
# invalidated by a raw leaf-value mutation (confirmed empirically: predict()
# on the same in-memory handle keeps returning pre-mutation values after
# set_leaf_values() -- only SetScaleAndBias() resets that cache). The real,
# observable effect of set_leaf_values() only shows up after a save/load
# round trip, so that is what gets recorded here (and what the R
# differential test reproduces via save_model()/load_model()).
reload_path = os.path.join(SCRIPT_DIR, ".tree_internals_model.tmp.cbm")
model.save_model(reload_path)
reloaded_model = CatBoostRegressor()
reloaded_model.load_model(reload_path)
os.remove(reload_path)
predictions_after_set = reloaded_model.predict(pool)

fixture = {
    "inputs": {
        "num1": NUM1,
        "num2": NUM2,
        "label": LABEL,
        "feature_names": FEATURE_NAMES,
    },
    "expected": {
        "borders": {str(k): list(v) for k, v in borders.items()},
        "borders_text": borders_text,
        "leaf_values": leaf_values,
        "leaf_weights": leaf_weights,
        "tree_leaf_counts": tree_leaf_counts,
        "leaf_indexes_full": leaf_indexes_full,
        "leaf_indexes_range": leaf_indexes_range,
        "leaf_indexes_iter": leaf_indexes_iter,
        "new_leaf_values": new_leaf_values,
        "leaf_values_after_set": leaf_values_after_set,
        "predictions_after_set": predictions_after_set,
    },
}

with open(FIXTURE_PATH, "w") as f:
    json.dump(fixture, f, indent=2, cls=_NumpyAwareEncoder)
print(FIXTURE_PATH)
