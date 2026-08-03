#!/usr/bin/env python3
"""Generate compare oracle fixture (catboost-8z4.51 / P4.2): deterministic
train data + two pinned Python catboost==1.2.10 models' observable
eval_metrics() output, the structural differential test for
catboost.compare() (R/catboost.R).

CatBoost.compare(model, data, metrics, ...) itself only draws an interactive
Jupyter widget (returns None); per spec Sec 4.3's "Structural" row and the
task brief, the comparable/testable surface is the underlying metrics-diff
data each side feeds into that widget -- which compare() builds by calling
self._eval_metrics(...) and model._eval_metrics(...) on the same pool/metrics
(catboost/python-package/catboost/core.py:3345-3349). Both calls funnel into
the same _base_eval_metrics -> TMetricsPlotCalcer C++ path
(catboost/libs/metrics), so this fixture pins that shared entry point by
calling the public model.eval_metrics() wrapper directly on each of two
models, once each -- byte-identical to what compare() computes internally.

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_compare_fixture.py
"""
import json
import os

from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "compare.json")

N_ROWS = 20
NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
        -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0]
NUM2 = [0.3 * i for i in range(N_ROWS)]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
FEATURE_NAMES = ["num1", "num2"]
METRICS = ["Logloss", "AUC"]


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    X = [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]
    pool = Pool(X, LABEL, feature_names=FEATURE_NAMES)

    model_a = CatBoostClassifier(
        iterations=10, depth=2, loss_function="Logloss",
        random_seed=42, thread_count=1, verbose=False,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_a"),
    )
    model_a.fit(pool)

    model_b = CatBoostClassifier(
        iterations=10, depth=4, loss_function="Logloss",
        random_seed=7, thread_count=1, verbose=False,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_b"),
    )
    model_b.fit(pool)

    # Same shape of calls CatBoost.compare() makes internally for each side
    # (ntree_start=0, ntree_end=0 (-> tree_count_), eval_period=1).
    model_a_metrics = model_a.eval_metrics(pool, METRICS, ntree_start=0, ntree_end=0, eval_period=1)
    model_b_metrics = model_b.eval_metrics(pool, METRICS, ntree_start=0, ntree_end=0, eval_period=1)

    fixture = {
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "label": LABEL,
            "feature_names": FEATURE_NAMES,
            "metrics": METRICS,
        },
        "expected": {
            "model": model_a_metrics,
            "other": model_b_metrics,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
