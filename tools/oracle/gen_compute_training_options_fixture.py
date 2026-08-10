#!/usr/bin/env python3
"""Generate the compute_training_options oracle fixture (catboost-azg):
pinned Python catboost==1.2.10 catboost.utils.compute_training_options()
output for a representative params dict + train/test DataMetaInfo --
differential test data for catboost.compute_training_options() (R/catboost.R),
which invokes the same native GetTrainingOptions resolution path
(catboost/private/libs/options) without training a model.

DataMetaInfo/TargetStats construction mirrors the vendored
catboost/python-package/ut/medium/test.py::test_compute_options (the only
place upstream exercises this API), not a real Pool -- compute_training_options
never touches the pool's actual rows, only the shape summary a DataMetaInfo
carries.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_compute_training_options_fixture.py
"""
import json
import os

from catboost.utils import DataMetaInfo, TargetStats, compute_training_options

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "compute_training_options.json")

os.makedirs(FIXTURE_DIR, exist_ok=True)

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 100,
    "learning_rate": 0.05,
    "thread_count": 1,
    "random_seed": 42,
}

TRAIN_META_INFO_INPUT = {
    "object_count": 100000,
    "feature_count": 10,
    "max_cat_features_uniq_values_on_learn": 0,
    "target_min_value": 0.0,
    "target_max_value": 1.0,
    "has_pairs": False,
}

# No target stats / no has_pairs override: exercises the "no target_stats"
# branch (Python's DataMetaInfo(target_stats=None)) that the train case above
# does not.
TEST_META_INFO_INPUT = {
    "object_count": 20000,
    "feature_count": 10,
    "max_cat_features_uniq_values_on_learn": 0,
    "has_pairs": False,
}


def to_data_meta_info(meta_info_input):
    target_stats = None
    if "target_min_value" in meta_info_input and "target_max_value" in meta_info_input:
        target_stats = TargetStats(
            min_value=meta_info_input["target_min_value"],
            max_value=meta_info_input["target_max_value"],
        )
    return DataMetaInfo(
        object_count=meta_info_input["object_count"],
        feature_count=meta_info_input["feature_count"],
        max_cat_features_uniq_values_on_learn=meta_info_input["max_cat_features_uniq_values_on_learn"],
        target_stats=target_stats,
        has_pairs=meta_info_input["has_pairs"],
    )


train_meta_info = to_data_meta_info(TRAIN_META_INFO_INPUT)
test_meta_info = to_data_meta_info(TEST_META_INFO_INPUT)

options_with_test = compute_training_options(
    options=PARAMS,
    train_meta_info=train_meta_info,
    test_meta_info=test_meta_info,
)

options_train_only = compute_training_options(
    options=PARAMS,
    train_meta_info=train_meta_info,
)

fixture = {
    "inputs": {
        "params": PARAMS,
        "train_meta_info": TRAIN_META_INFO_INPUT,
        "test_meta_info": TEST_META_INFO_INPUT,
    },
    "expected": {
        "options_with_test": options_with_test,
        "options_train_only": options_train_only,
    },
}

with open(FIXTURE_PATH, "w") as f:
    json.dump(fixture, f, indent=2)
print(FIXTURE_PATH)
