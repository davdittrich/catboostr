#!/usr/bin/env python3
"""Generate the ROC-curve oracle fixture (catboost-8z4.55 / P4.6):
deterministic train data + pinned Python catboost==1.2.10
catboost.utils.get_roc_curve() output -- differential test data for
catboost.get_roc_curve() (R/catboost.R), which ports the same C++ engine
(catboost/private/libs/algo/roc_curve.cpp TRocCurve, wrapped by
_get_roc_curve in _catboost.pyx).

get_roc_curve(model, data, thread_count=-1) returns (fpr, tpr, thresholds):
raw model approxes are converted to probabilities (PrepareEval Probability),
sorted descending, and swept to build the (FPR, TPR, threshold) points, with
a synthetic intersection point inserted wherever the FNR/FPR curves cross.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_roc_curve_fixture.py
"""
import json
import os

from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "roc_curve.json")

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
    random_seed=42, thread_count=1, verbose=False,
    train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_roc"),
)
model.fit(pool)

fpr, tpr, thresholds = get_roc_curve(model, pool, thread_count=1)

fixture = {
    "inputs": {
        "num1": NUM1,
        "num2": NUM2,
        "label": LABEL,
        "feature_names": FEATURE_NAMES,
    },
    "expected": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    },
}

with open(FIXTURE_PATH, "w") as f:
    json.dump(fixture, f, indent=2)
print(FIXTURE_PATH)
