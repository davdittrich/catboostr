#!/usr/bin/env python3
"""Generate select_features oracle fixture (catboost-8z4.60 / P5.3).

Pins Python catboost==1.2.10's CatBoost.select_features behavior. It calls the
exact same native NCB::SelectFeatures entry point
(catboost/libs/features_selection/select_features.h, declared at
_catboost.pyx:1264 and called at _catboost.pyx:6045 from _select_features) that
R's new catboost.select_features now calls directly, so the returned summary
and the final model's predictions are expected to match bit-for-bit, not just
approximately.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_select_features_fixture.py
"""

import json
import os
import sys

import catboost
from catboost import CatBoost, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "select_features.json")

N_ROWS = 80
N_FEATURES = 6

# Deterministic, no RNG: features 0 and 3 carry the signal, the rest are noise
# of decreasing relevance, so the elimination order is a real signal, not a tie.
FEATURES = [
    [
        round(0.5 * i - 4.0, 4),
        round(1.5 * ((i * 7) % 11) - 8.0, 4),
        round(((i * 5) % 13) / 3.0, 4),
        round(0.25 * ((i * 3) % 17) - 2.0, 4),
        round(((i * 11) % 7) - 3.0, 4),
        round(((i * 2) % 5) / 2.0, 4),
    ]
    for i in range(N_ROWS)
]
LABEL = [
    round(0.3 * FEATURES[i][0] + 0.8 * FEATURES[i][3] + 0.05 * ((i * 3) % 5), 4)
    for i in range(N_ROWS)
]

BASE_PARAMS = {
    "iterations": 20,
    "loss_function": "RMSE",
    "thread_count": 1,
    "random_seed": 0,
    "verbose": False,
}

FEATURES_FOR_SELECT = list(range(N_FEATURES))
NUM_FEATURES_TO_SELECT = 3
STEPS = 2


def run_case(name, algorithm, train_final_model):
    train_pool = Pool(FEATURES[:60], LABEL[:60])
    test_pool = Pool(FEATURES[60:], LABEL[60:])
    model = CatBoost(
        dict(BASE_PARAMS, train_dir=os.path.join(SCRIPT_DIR, f".catboost_select_{name}"))
    )
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=FEATURES_FOR_SELECT,
        num_features_to_select=NUM_FEATURES_TO_SELECT,
        algorithm=algorithm,
        steps=STEPS,
        train_final_model=train_final_model,
        verbose=False,
    )
    case = {
        "selected_features": [int(v) for v in summary["selected_features"]],
        "selected_features_names": list(summary["selected_features_names"]),
        "eliminated_features": [int(v) for v in summary["eliminated_features"]],
        "eliminated_features_names": list(summary["eliminated_features_names"]),
        "loss_graph": {
            "removed_features_count": [
                int(v) for v in summary["loss_graph"]["removed_features_count"]
            ],
            "loss_values": [float(v) for v in summary["loss_graph"]["loss_values"]],
            "main_indices": [int(v) for v in summary["loss_graph"]["main_indices"]],
        },
    }
    if train_final_model:
        case["final_tree_count"] = int(model.tree_count_)
        case["final_predict"] = [
            float(v) for v in model.predict(test_pool, prediction_type="RawFormulaVal")
        ]
    return case


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    shap = run_case("shap", "RecursiveByShapValues", True)
    loss_change = run_case("loss_change", "RecursiveByLossFunctionChange", False)

    fixture = {
        "catboost_version": catboost.__version__,
        "base_params": BASE_PARAMS,
        "features_for_select": FEATURES_FOR_SELECT,
        "num_features_to_select": NUM_FEATURES_TO_SELECT,
        "steps": STEPS,
        "inputs": {
            "learn_features": FEATURES[:60],
            "learn_label": LABEL[:60],
            "test_features": FEATURES[60:],
            "test_label": LABEL[60:],
        },
        "expected": {
            "recursive_by_shap_values": shap,
            "recursive_by_loss_function_change": loss_change,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"shap.selected_features={shap['selected_features']}", file=sys.stderr)
    print(
        f"loss_change.selected_features={loss_change['selected_features']}",
        file=sys.stderr,
    )
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
