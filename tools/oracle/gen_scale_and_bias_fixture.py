#!/usr/bin/env python3
import json
import os

from catboost import CatBoostRegressor, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "scale_and_bias.json")

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
    train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_scale_and_bias"),
)
model.fit(pool)

# Default (post-fit, pre-normalize) scale/bias -- R equivalent of the
# CatBoost._CatBoostBase.get_scale_and_bias() defined once at core.py:2422-2423
# and inherited unchanged by CatBoost/CatBoostClassifier/CatBoostRegressor/
# CatBoostRanker (all four share the single _CatBoostBase implementation, so
# this one fixture covers all four matrix rows per class).
scale_before, bias_before = model.get_scale_and_bias()

# Mirror the CLI's `normalize-model --set-scale 0.8 --set-bias 0.8` calling
# convention (mode_normalize_model.cpp): scalar bias, matching
# _CatBoostBase.set_scale_and_bias's isinstance(bias, FLOAT_TYPES) branch
# (core.py:2426-2427) which wraps a scalar into [bias].
model.set_scale_and_bias(0.8, 0.8)
scale_after_scalar, bias_after_scalar = model.get_scale_and_bias()

fixture = {
    "inputs": {
        "num1": NUM1,
        "num2": NUM2,
        "label": LABEL,
        "feature_names": FEATURE_NAMES,
    },
    "expected": {
        "scale_before": scale_before,
        "bias_before": bias_before,
        "scale_after_scalar": scale_after_scalar,
        "bias_after_scalar": bias_after_scalar,
    },
}

with open(FIXTURE_PATH, "w") as f:
    json.dump(fixture, f, indent=2)
print(FIXTURE_PATH)
