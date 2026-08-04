#!/usr/bin/env python3
"""Generate the model-metadata oracle fixture (catboost-8z4.63 / P5.6):
deterministic train data + pinned Python catboost==1.2.10
CatBoost.get_metadata() / feature_names_ output -- differential test data
for catboost.get_metadata()/catboost.set_metadata()/catboost.get_model_feature_names()
(R/catboost.R), which read/write the same underlying
THashMap<TString, TString> TFullModel::ModelInfo (catboost/libs/model/model.h)
that Python's _MetadataHashProxy (_catboost.pyx) wraps.

Same 20-row synthetic dataset/params as gen_roc_curve_fixture.py, so this
fixture's `params` metadata value stays consistent with the already-proven
bit-exact cross-language determinism (Phase 0).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_metadata_fixture.py
"""
import json
import os

from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "metadata.json")

N_ROWS = 20
NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
        -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0]
NUM2 = [0.3 * i for i in range(N_ROWS)]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
FEATURE_NAMES = ["num1", "num2"]

os.makedirs(FIXTURE_DIR, exist_ok=True)

X = [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]
pool = Pool(X, LABEL, feature_names=FEATURE_NAMES)

model = CatBoostClassifier(
    iterations=10, depth=2, loss_function="Logloss",
    random_seed=42, thread_count=1, logging_level="Silent",
    train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_metadata"),
)
model.fit(pool)

metadata_before = dict(model.get_metadata())

# Mirror the calling convention this fixture is exercising: __setitem__ on
# the dict-like proxy returned by get_metadata(), matching CLI `metadata
# set --key custom_key --value custom_value`.
model.get_metadata()["custom_key"] = "custom_value"
metadata_after_set = dict(model.get_metadata())

# Missing-key lookup: Python's proxy raises KeyError (_catboost.pyx
# _MetadataHashProxy.__getitem__); .get() returns the default instead.
missing_key_get_default = model.get_metadata().get("no_such_key", "__default__")

feature_names = model.feature_names_

fixture = {
    "inputs": {
        "num1": NUM1,
        "num2": NUM2,
        "label": LABEL,
        "feature_names": FEATURE_NAMES,
    },
    "expected": {
        "metadata_before": metadata_before,
        "metadata_after_set": metadata_after_set,
        "missing_key_get_default": missing_key_get_default,
        "feature_names": feature_names,
    },
}

with open(FIXTURE_PATH, "w") as f:
    json.dump(fixture, f, indent=2)
print(FIXTURE_PATH)
