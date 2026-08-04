#!/usr/bin/env python3
"""Generate virtual_ensembles_predict oracle fixture (catboost-8z4.61 / P5.4).

Pins Python catboost==1.2.10's CatBoost.virtual_ensembles_predict behavior.
R's catboost.virtual_ensembles_predict (R/catboost.R) calls the native
CatBoostPredictVirtualEnsembles_R entry point, which wraps the same
ApplyUncertaintyPredictions (catboost/private/libs/algo/apply.cpp) that
Python's _base_virtual_ensembles_predict calls via _catboost.pyx. Both
R and Python then reshape the flat (objects x virtual_ensembles*dim)
buffer -- R via aperm(array(..., dim=c(D,V,N)), perm=c(2,1,3)), Python via
predictions.reshape(N, V, D) -- so this fixture checks that reshape is
consistent between the two languages, not just that the native call works.

RMSEWithUncertainty is used for the 'VirtEnsembles' case because it is the
only loss where D (per-ensemble-member dimension) > 1, which is required to
actually exercise the D-axis of the reshape (a D=1 loss would pass even if
the V and D axes were transposed).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_virtual_ensembles_fixture.py
"""

import json
import os
import sys

import catboost
from catboost import CatBoost, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "virtual_ensembles.json")

N_ROWS = 80
N_FEATURES = 5

FEATURES = [
    [
        round(0.5 * i - 4.0, 4),
        round(1.5 * ((i * 7) % 11) - 8.0, 4),
        round(((i * 5) % 13) / 3.0, 4),
        round(0.25 * ((i * 3) % 17) - 2.0, 4),
        round(((i * 11) % 7) - 3.0, 4),
    ]
    for i in range(N_ROWS)
]
LABEL = [
    round(0.3 * FEATURES[i][0] + 0.8 * FEATURES[i][3] + 0.05 * ((i * 3) % 5), 4)
    for i in range(N_ROWS)
]

VIRTUAL_ENSEMBLES_COUNT = 4

BASE_PARAMS = {
    "iterations": 40,
    "loss_function": "RMSEWithUncertainty",
    "thread_count": 1,
    "random_seed": 0,
    "verbose": False,
}


def train_model():
    train_pool = Pool(FEATURES[:60], LABEL[:60])
    model = CatBoost(
        dict(BASE_PARAMS, train_dir=os.path.join(SCRIPT_DIR, ".catboost_virtual_ensembles"))
    )
    model.fit(train_pool)
    return model


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    model = train_model()
    test_pool = Pool(FEATURES[60:], LABEL[60:])

    virt = model.virtual_ensembles_predict(
        test_pool,
        prediction_type="VirtEnsembles",
        virtual_ensembles_count=VIRTUAL_ENSEMBLES_COUNT,
        thread_count=1,
    )
    total_unc = model.virtual_ensembles_predict(
        test_pool,
        prediction_type="TotalUncertainty",
        virtual_ensembles_count=VIRTUAL_ENSEMBLES_COUNT,
        thread_count=1,
    )

    fixture = {
        "catboost_version": catboost.__version__,
        "base_params": BASE_PARAMS,
        "virtual_ensembles_count": VIRTUAL_ENSEMBLES_COUNT,
        "inputs": {
            "learn_features": FEATURES[:60],
            "learn_label": LABEL[:60],
            "test_features": FEATURES[60:],
        },
        "expected": {
            "virt_ensembles_shape": list(virt.shape),
            "virt_ensembles": virt.tolist(),
            "total_uncertainty_shape": list(total_unc.shape),
            "total_uncertainty": total_unc.tolist(),
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"virt_ensembles_shape={list(virt.shape)}", file=sys.stderr)
    print(f"total_uncertainty_shape={list(total_unc.shape)}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
