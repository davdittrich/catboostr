#!/usr/bin/env python3
"""Generate the Python-oracle fixture for P5.5 (catboost-8z4.62) family-level
closure of the 139 kind:"parameter" matrix rows.

Trains several small, deliberately-batched CatBoost models -- each batch is a
family grouping of hyperparameters that are mutually compatible (do not
interact adversarially), covering the native training hyperparameters that
had no existing differential-test coverage (see task-5 audit). Writes
predictions (RawFormulaVal) for each batch at full float64 precision so the
R side can compare at tolerance 1e-12 (spec Sec 4.3 default) unless a batch's
own comment says otherwise (RNG-stochastic bootstrap/Langevin batches, and
the used_ram_limit isolation probe, are deliberately kept separate so any
tolerance widening attributes to exactly one variable, not a 30+-parameter
bundle -- confound-isolation, see review round fix).

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
    output_borders_path = os.path.join(TRAIN_DIR, "output_borders.tsv")

    batches = {
        # G1: core training control, regularization, leaf estimation,
        # overfitting detector, output/logging settings. Bit-exact at the
        # real 1e-12 default -- confirmed by bisection (see
        # task-5-report.md fix-round notes): the review round's original
        # attribution of a ~1e-7 divergence to bagging_temperature (Bayesian
        # bootstrap RNG) was WRONG (bagging_temperature alone reproduces
        # bit-exact, see "bayesian_bootstrap" below); the real, isolated,
        # confirmed cause was model_shrink_rate (moved to its own batch,
        # "model_shrink_rate_isolated", below) -- model_shrink_mode alone
        # (without model_shrink_rate) is also bit-exact and stays here.
        "core_training_control": dict(
            loss_function="Logloss", eval_metric="AUC", custom_metric=["Accuracy"],
            iterations=25, learning_rate=0.08, depth=4,
            l2_leaf_reg=4.0, model_size_reg=0.4, rsm=0.9,
            random_seed=SEED, random_strength=1.5, nan_mode="Min",
            leaf_estimation_method="Newton", leaf_estimation_iterations=2,
            leaf_estimation_backtracking="AnyImprovement",
            boosting_type="Plain", boost_from_average=True,
            min_data_in_leaf=2, fold_permutation_block=1, fold_len_multiplier=2.0,
            sampling_frequency="PerTreeLevel",
            model_shrink_mode="Constant",
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
            ignored_features=[1],
        ),
        # Isolated (single-variable-vs-core-defaults) Bayesian-bootstrap
        # batch: verified bit-exact in isolation (bisection, fix round) --
        # kept as its own small batch for clarity, not because it needs a
        # widened tolerance (it doesn't).
        "bayesian_bootstrap": dict(
            loss_function="Logloss", iterations=25, verbose=False,
            random_seed=SEED, thread_count=1,
            bootstrap_type="Bayesian", bagging_temperature=0.6,
        ),
        # Isolated (single-variable-vs-core-defaults) model_shrink_rate
        # batch: bisection (fix round, see task-5-report.md) confirmed this
        # is the real, sole cause of a ~1.7e-7-magnitude divergence in the
        # original bundled batch (re-measured ~3.9e-7 in this isolated
        # batch -- different exact value, same order of magnitude, not a
        # discrepancy), originally (and wrongly) attributed to bagging_temperature.
        # Plausibly floating-point evaluation-order sensitivity in the
        # repeated multiplicative shrinkage applied across iterations
        # (Constant mode), not an RNG stream difference -- deterministic on
        # both sides, but order-of-operations-sensitive.
        "model_shrink_rate_isolated": dict(
            loss_function="Logloss", iterations=25, verbose=False,
            random_seed=SEED, thread_count=1,
            model_shrink_mode="Constant", model_shrink_rate=0.01,
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
        # G2b: legacy/alternate CTR-description spellings + per-feature CTR
        # override, verified individually against the real native library
        # first (see task-5-report.md fix-round notes) -- per_feature_ctr
        # needs an explicit "index:Type:Prior=N/D:..." spec (an *unindexed*
        # prior, or targeting a non-categorical feature index, both throw a
        # native error), and ctr_history_unit's only CPU-supported value is
        # its own default ("Sample"; "Group" throws
        # "unimplemented for task type CPU").
        "ctr_extra": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            custom_loss=["Logloss"],
            ctr_description=["Borders:CtrBorderCount=8"],
            ctr_history_unit="Sample",
            per_feature_ctr=["2:Borders:Prior=0.5/1:CtrBorderCount=8"],
            train_dir=os.path.join(TRAIN_DIR, "ctr_extra"),
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
        # G5: class-weighting family -- mutually exclusive, so 2 tiny models.
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
        # G7: feature-penalty family (all no-op weights/penalties).
        "feature_penalties": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            monotone_constraints=[0, 0, 0], feature_weights=[1.0, 1.0, 1.0],
            first_feature_use_penalties=[0.0, 0.0, 0.0],
            per_object_feature_penalties=[0.0] * N_ROWS,
            penalties_coefficient=1.0,
            train_dir=os.path.join(TRAIN_DIR, "penalties"),
        ),
        # G9: Langevin boosting family (CPU-supported, itself an RNG-driven
        # stochastic method -- already isolated to just these 3 params, no
        # further splitting needed).
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
        # G11: output_borders (verified as a real flat top-level native
        # option; input_borders is NOT -- see task-5-report.md fix-round
        # notes -- so only output_borders gets a differential batch).
        "output_borders": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            output_borders=output_borders_path,
            train_dir=os.path.join(TRAIN_DIR, "output_borders"),
        ),
        # G12: used_ram_limit, isolated (single variable vs. core defaults)
        # to measure its real R-vs-Python divergence in isolation, after it
        # was found to diverge by ~1e-7 when bundled with 34 other
        # core_training_control parameters and the attribution to
        # used_ram_limit itself was never confirmed in isolation.
        "used_ram_limit_isolated": dict(
            loss_function="Logloss", iterations=25, verbose=False,
            random_seed=SEED, thread_count=1,
            used_ram_limit="512mb",
            train_dir=os.path.join(TRAIN_DIR, "ram_limit"),
        ),
        # G13: same run as G12 but *without* used_ram_limit, all other
        # params identical -- the direct A/B baseline for the isolation
        # comparison (single variable differs: used_ram_limit itself).
        "used_ram_limit_baseline": dict(
            loss_function="Logloss", iterations=25, verbose=False,
            random_seed=SEED, thread_count=1,
            train_dir=os.path.join(TRAIN_DIR, "ram_baseline"),
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
        # "verbose" is excluded: it is purely a console-output cosmetic (does
        # not affect predictions) and, unlike Python's fit(verbose=), R's
        # `params` list has no client-side bool->period translation for it
        # (native's flat "verbose" option is an int print-period, not a
        # bool -- see output_file_options.cpp VerbosePeriod); the R side
        # sets logging_level="Silent" directly instead (see
        # test_param_family_coverage.R).
        json_params = {
            k: v for k, v in params.items()
            if k not in ("train_dir", "snapshot_file", "output_borders", "verbose")
        }
        fixture["batches"][batch_name] = {
            "params": json_params,
            "predictions": preds,
        }

    # NB: graph (Pool's `graph=` argument) is NOT generated here. Fix round 2
    # tried it (a fresh Pool with graph=+group_id=) and found catboostr's own
    # R-side Pool-construction glue throws "Internal CatBoost Error ...
    # Unimplemented" for that combination -- a real R-glue limitation, not a
    # Python-parity gap (see test_params_validation.R's reproducing
    # assertion and task-5-report.md). Since no R test can consume it, no
    # graph fixture data is generated (fix round 3: removed dead
    # graph_pairs/graph_pool/fixture["graph_inputs"] code that no test
    # read).

    # G15: text_features + dictionaries/tokenizers/text_processing/
    # feature_calcers (review-round fix -- these 4 training params were
    # closed on test_params_validation.R with the params validation gate
    # accepting the key syntactically but no real end-to-end text-feature
    # differential test; text_features itself was closed on
    # test_text_processing.R, which only exercises the standalone
    # Tokenizer/Dictionary API, not catboost.train with a text-bearing
    # Pool). A dedicated small text-bearing dataset + Pool, auto-detected
    # text feature (character column -> text_features_indices, matching
    # R/catboost.R:401-404's auto-detection), trained with an explicit
    # tokenizers/dictionaries/text_processing/feature_calcers config.
    text_words = [["quick", "brown", "fox", "lazy", "dog"],
                  ["red", "fast", "car", "loud", "engine"]]
    text_rng = random.Random(SEED + 1)
    text_rows = []
    for i in range(N_ROWS):
        label = i % 2
        words = text_rng.sample(text_words[label], k=3)
        text_rows.append((rows[i][0], rows[i][1], rows[i][2], " ".join(words), label))
    text_X = [[r[0], r[1], r[2], r[3]] for r in text_rows]
    text_y = [r[4] for r in text_rows]
    text_feature_names = feature_names + ["text1"]
    text_pool = Pool(text_X, text_y, cat_features=[2], text_features=[3], feature_names=text_feature_names)
    fixture["text_inputs"] = {
        "num1": [r[0] for r in text_rows], "num2": [r[1] for r in text_rows],
        "cat1": [r[2] for r in text_rows], "text1": [r[3] for r in text_rows],
        "label": text_y,
    }
    # Native forbids combining `text_processing` with the separate
    # `tokenizers`/`dictionaries`/`feature_calcers` trio in the same call
    # ("You should provide either `text_processing` option or `tokenizers`,
    # `dictionaries`, `feature_calcers` options" -- text_processing_options.cpp:394),
    # so this needs two batches, not one.
    text_family_params = {
        "text_processing_only": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            text_processing={"feature_processing": {"default": [
                {"dictionaries_names": ["Word"], "feature_calcers": ["BoW"], "tokenizers_names": ["Space"]}
            ]}},
            train_dir=os.path.join(TRAIN_DIR, "text_a"),
        ),
        "tokenizers_dictionaries_calcers": dict(
            loss_function="Logloss", iterations=15, verbose=False,
            random_seed=SEED, thread_count=1,
            dictionaries=[{"dictionary_id": "Word"}],
            tokenizers=[{"tokenizer_id": "Space"}],
            feature_calcers=["BoW"],
            train_dir=os.path.join(TRAIN_DIR, "text_b"),
        ),
    }
    for name, params in text_family_params.items():
        model = CatBoostClassifier(**params)
        model.fit(text_pool)
        preds = model.predict(text_pool, prediction_type="RawFormulaVal")
        fixture["batches"][name] = {
            "params": {k: v for k, v in params.items() if k not in ("train_dir", "verbose")},
            "predictions": [float(v) for v in np.asarray(preds).ravel().tolist()],
        }

    with open(os.path.join(FIXTURE_DIR, "param_family_coverage.json"), "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"batches={list(batches.keys())}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
