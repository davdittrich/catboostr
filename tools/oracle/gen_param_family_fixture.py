#!/usr/bin/env python3
"""Generate the Python-oracle fixture for P5.5 (catboost-8z4.62) family-level
closure of the 139 kind:"parameter" matrix rows.

Trains several small, deliberately-batched CatBoost models -- each batch is a
family grouping of hyperparameters that are mutually compatible (do not
interact adversarially), covering the native training hyperparameters that
had no existing differential-test coverage (see task-5 audit). Writes
predictions (RawFormulaVal) for each batch at full float64 precision so the
R side can compare at tolerance 1e-12 (spec Sec 4.3 default).

Run via:
  uv run --frozen --project tools/oracle python3 tools/oracle/gen_param_family_fixture.py
"""
import json
import os
import random
import sys

import catboost
import numpy as np
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
TRAIN_DIR = os.path.join(SCRIPT_DIR, ".catboost_train_param_family")
SEED = 42
N_ROWS = 60


def make_dataset(seed, n_rows):
    rng = random.Random(seed)
    cat_levels = ["a", "b", "c"]
    rows = []
    for _ in range(n_rows):
        num1 = rng.uniform(-10.0, 10.0)
        num2 = rng.gauss(0.0, 1.0)
        cat1 = rng.choice(cat_levels)
        score = 0.3 * num1 + 0.7 * num2 + (1.0 if cat1 == "a" else -0.5)
        label = 1 if score + rng.uniform(-1.0, 1.0) > 0 else 0
        rows.append((num1, num2, cat1, label))
    return rows


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    rows = make_dataset(SEED, N_ROWS)
    X = [[r[0], r[1], r[2]] for r in rows]
    y = [r[3] for r in rows]
    feature_names = ["num1", "num2", "cat1"]
    pool = Pool(X, y, cat_features=[2], feature_names=feature_names)

    snapshot_path = os.path.join(TRAIN_DIR, "family_a.snapshot")
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    batches = {
        # G1: core training control, regularization, leaf estimation,
        # overfitting detector, output/logging settings.
        "core_training_control": dict(
            loss_function="Logloss", eval_metric="AUC", custom_metric=["Accuracy"],
            iterations=25, learning_rate=0.08, depth=4,
            l2_leaf_reg=4.0, model_size_reg=0.4, rsm=0.9,
            random_seed=SEED, random_strength=1.5, nan_mode="Min",
            leaf_estimation_method="Newton", leaf_estimation_iterations=2,
            leaf_estimation_backtracking="AnyImprovement",
            bootstrap_type="Bayesian", bagging_temperature=0.6,
            boosting_type="Plain", boost_from_average=True,
            min_data_in_leaf=2, fold_permutation_block=1, fold_len_multiplier=2.0,
            sampling_frequency="PerTreeLevel",
            model_shrink_mode="Constant", model_shrink_rate=0.01,
            od_type="IncToDec", od_pval=0.05, od_wait=3,
            use_best_model=False, best_model_min_trees=1,
            metric_period=3, logging_level="Silent",
            allow_const_label=False, allow_writing_files=False,
            approx_on_full_history=False, thread_count=1,
            name="phase5_family_core", metadata={"family": "phase5_core"},
            train_dir=os.path.join(TRAIN_DIR, "core"),
            task_type="CPU",
            data_partition="FeatureParallel",
            dev_score_calc_obj_block_size=1000, dev_efb_max_buckets=128,
            sparse_features_conflict_fraction=0.0,
        ),
        # G2: CTR + binarization settings.
        "ctr_and_binarization": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            feature_border_type="GreedyLogSum",
            per_float_feature_quantization=["0:border_count=32"],
            simple_ctr=["Borders:CtrBorderCount=8"],
            combinations_ctr=["Borders:CtrBorderCount=8"],
            ctr_target_border_count=1, counter_calc_method="Full",
            max_ctr_complexity=2, ctr_leaf_count_limit=100,
            store_all_simple_ctr=False, final_ctr_computation_mode="Default",
            target_border=0.5,
            train_dir=os.path.join(TRAIN_DIR, "ctr"),
        ),
        # G3: grow_policy / max_leaves / score_function (Lossguide-only combo).
        "lossguide_grow_policy": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            grow_policy="Lossguide", max_leaves=16, score_function="L2",
            train_dir=os.path.join(TRAIN_DIR, "lossguide"),
        ),
        # G4: sampling / subsample family (MVS bootstrap).
        "mvs_sampling": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            bootstrap_type="MVS", subsample=0.8, mvs_reg=0.5,
            sampling_unit="Object",
            train_dir=os.path.join(TRAIN_DIR, "mvs"),
        ),
        # G5: class-weighting family -- mutually exclusive, so 3 tiny models.
        "class_weights": dict(
            loss_function="Logloss", iterations=10, verbose=False,
            random_seed=SEED, thread_count=1, class_weights=[1.0, 1.2],
            train_dir=os.path.join(TRAIN_DIR, "cw"),
        ),
        "auto_class_weights": dict(
            loss_function="Logloss", iterations=10, verbose=False,
            random_seed=SEED, thread_count=1, auto_class_weights="Balanced",
            train_dir=os.path.join(TRAIN_DIR, "acw"),
        ),
        # G7: feature-penalty family (all no-op weights/penalties => must not
        # change predictions vs G1's baseline structurally, but we only need
        # bit-exact match to the *Python* run of the identical config).
        "feature_penalties": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            monotone_constraints=[0, 0, 0], feature_weights=[1.0, 1.0, 1.0],
            first_feature_use_penalties=[0.0, 0.0, 0.0],
            per_object_feature_penalties=[0.0] * N_ROWS,
            penalties_coefficient=1.0,
            train_dir=os.path.join(TRAIN_DIR, "penalties"),
        ),
        # G9: Langevin boosting family (CPU-supported).
        "langevin": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            langevin=True, diffusion_temperature=1000.0, posterior_sampling=False,
            train_dir=os.path.join(TRAIN_DIR, "langevin"),
        ),
        # G10: snapshot family.
        "snapshot": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            save_snapshot=True, snapshot_file=snapshot_path, snapshot_interval=1,
            allow_writing_files=True,
            train_dir=os.path.join(TRAIN_DIR, "snapshot"),
        ),
    }

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "num1": [r[0] for r in rows], "num2": [r[1] for r in rows],
            "cat1": [r[2] for r in rows], "label": y,
            "feature_names": feature_names, "cat_features": [2],
        },
        "batches": {},
    }
    for batch_name, params in batches.items():
        model = CatBoostClassifier(**params)
        model.fit(pool)
        preds = model.predict(pool, prediction_type="RawFormulaVal")
        preds = [float(v) for v in np.asarray(preds).ravel().tolist()]
        # Strip filesystem-path / non-serializable-for-R params before saving
        # (R reconstructs equivalent tmp paths itself); keep everything else
        # so the R side can literally replay the same `params` list.
        # "verbose" is excluded: it is purely a console-output cosmetic (does
        # not affect predictions) and, unlike Python's fit(verbose=), R's
        # `params` list has no client-side bool->period translation for it
        # (native's flat "verbose" option is an int print-period, not a
        # bool -- see output_file_options.cpp VerbosePeriod); the R side
        # sets logging_level="Silent" directly instead (see
        # test_param_family_coverage.R).
        json_params = {
            k: v for k, v in params.items()
            if k not in ("train_dir", "snapshot_file", "input_borders", "verbose")
        }
        fixture["batches"][batch_name] = {
            "params": json_params,
            "predictions": preds,
        }

    with open(os.path.join(FIXTURE_DIR, "param_family_coverage.json"), "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"batches={list(batches.keys())}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
