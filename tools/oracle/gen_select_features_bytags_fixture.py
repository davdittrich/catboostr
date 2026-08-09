#!/usr/bin/env python3
"""Generate select_features grouping="ByTags" oracle fixture (catboost-8z4.121).

Pins Python catboost==1.2.10's CatBoost.select_features(grouping="ByTags")
behavior: Pool(feature_tags=...) attaches tags to the pool's TFeaturesLayout
(_catboost.pyx:2341-2388), then select_features(grouping="ByTags",
features_tags_for_select=..., num_features_tags_to_select=...) calls the same
native NCB::SelectFeatures entry point as grouping="Individual" (already
covered by gen_select_features_fixture.py), just with
TFeaturesSelectOptions.Grouping == ByTags. Reuses that script's deterministic
dataset (no RNG) so this fixture differs only in the dimension under test.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_select_features_bytags_fixture.py
"""

import json
import os
import sys

import catboost
from catboost import CatBoost, Pool

from gen_select_features_fixture import BASE_PARAMS, FEATURES, LABEL, N_FEATURES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "select_features_bytags.json")

# Same 6 features as gen_select_features_fixture.py, grouped into 3 tags of 2
# features each. Tag "signal" carries both features the label actually
# depends on (0 and 3, see that script's LABEL formula), so a correct
# ByTags elimination must keep it, not treat tags as interchangeable.
FEATURE_TAGS = {
    "signal": {"features": [0, 3], "cost": 1},
    "noise_a": {"features": [1, 4], "cost": 1},
    "noise_b": {"features": [2, 5], "cost": 2},
}
FEATURES_TAGS_FOR_SELECT = ["signal", "noise_a", "noise_b"]
NUM_FEATURES_TAGS_TO_SELECT = 2
STEPS = 2


def run_case(name, algorithm, train_final_model):
    train_pool = Pool(FEATURES[:60], LABEL[:60], feature_tags=FEATURE_TAGS)
    test_pool = Pool(FEATURES[60:], LABEL[60:], feature_tags=FEATURE_TAGS)
    model = CatBoost(
        dict(BASE_PARAMS, train_dir=os.path.join(SCRIPT_DIR, f".catboost_select_bytags_{name}"))
    )
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        grouping="ByTags",
        features_tags_for_select=FEATURES_TAGS_FOR_SELECT,
        num_features_tags_to_select=NUM_FEATURES_TAGS_TO_SELECT,
        algorithm=algorithm,
        steps=STEPS,
        train_final_model=train_final_model,
        verbose=False,
    )
    case = {
        "selected_features": [int(v) for v in summary["selected_features"]],
        "eliminated_features": [int(v) for v in summary["eliminated_features"]],
        "selected_features_tags": list(summary["selected_features_tags"]),
        "eliminated_features_tags": list(summary["eliminated_features_tags"]),
        "features_tags_loss_graph": {
            "removed_features_tags_count": [
                int(v) for v in summary["features_tags_loss_graph"]["removed_features_tags_count"]
            ],
            "loss_values": [float(v) for v in summary["features_tags_loss_graph"]["loss_values"]],
            "main_indices": [int(v) for v in summary["features_tags_loss_graph"]["main_indices"]],
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
        "feature_tags": FEATURE_TAGS,
        "features_tags_for_select": FEATURES_TAGS_FOR_SELECT,
        "num_features_tags_to_select": NUM_FEATURES_TAGS_TO_SELECT,
        "steps": STEPS,
        "n_features": N_FEATURES,
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
    print(f"shap.selected_features_tags={shap['selected_features_tags']}", file=sys.stderr)
    print(
        f"loss_change.selected_features_tags={loss_change['selected_features_tags']}",
        file=sys.stderr,
    )
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
