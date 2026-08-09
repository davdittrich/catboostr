#!/usr/bin/env python3
"""Generate the oracle fixture for catboost-8z4.120 (P10.F): pins Python
catboost==1.2.10's output for the "advanced eval/plotting + probability-
threshold + log-proba" family --
create_metric_calcer/get_test_eval(s)/plot_partial_dependence/plot_predictions
(each x4 estimator classes) plus the CatBoostClassifier-only
get_probability_threshold/set_probability_threshold/predict_log_proba/
staged_predict_log_proba (core.py:1830-1850, 3354-3374, 3847-4020,
5655-5914).

Follows the same per-class loss convention as
gen_training_history_classes_fixture.py/gen_eval_metrics_classes_fixture.py
(CatBoost=MultiClass, CatBoostClassifier=Logloss, CatBoostRegressor=RMSE,
CatBoostRanker=YetiRank) and the same 16-train/4-eval whole-group split as
gen_training_history_classes_fixture.py, reusing ONE fitted model per class
across all of that class's sub-families (get_test_eval(s), create_metric_calcer,
plot_predictions) EXCEPT plot_partial_dependence for the "CatBoost" row:
GetPartialDependence() (catboost/libs/fstr/partial_dependence.cpp:187-196)
CB_ENSUREs `GetDimensionsCount() == 1`, i.e. explicitly refuses multiclass
models, so that row needs its own binary-loss model
(CatBoostPartialDependence, Logloss) instead of the shared MultiClass one.

Regenerate fixture with:
uv run --frozen --project tools/oracle python3 tools/oracle/gen_advanced_eval_plotting_probability_fixture.py
"""
import json
import os

from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from catboost.core import _NumpyAwareEncoder

from _classes_common import (
    BINARY_LABEL, FEATURE_NAMES, GROUP_ID, MULTICLASS_LABEL, NUM1, NUM2, REGRESSION_LABEL, x,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "advanced_eval_plotting_probability.json")

TRAIN_N = 16

# use_best_model=False: CatBoost defaults use_best_model=True whenever an
# eval_set is given, which truncates the returned model to best_iteration_+1
# trees -- on this tiny dataset that collapses tree_count_ to 1, degenerating
# staged_predict_log_proba's eval_period=3 fixture to a single batch. Forcing
# it off keeps all `iterations` trees so the staged/create_metric_calcer
# fixtures exercise a non-trivial multi-iteration sequence.
COMMON = dict(iterations=10, depth=2, random_seed=42, thread_count=1, verbose=False,
              use_best_model=False,
              train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_advanced"))


def split(label, group_id=None):
    features = x()
    train_x, eval_x = features[:TRAIN_N], features[TRAIN_N:]
    train_y, eval_y = label[:TRAIN_N], label[TRAIN_N:]
    if group_id is None:
        return (Pool(train_x, train_y, feature_names=FEATURE_NAMES),
                Pool(eval_x, eval_y, feature_names=FEATURE_NAMES))
    train_g, eval_g = group_id[:TRAIN_N], group_id[TRAIN_N:]
    return (Pool(train_x, train_y, feature_names=FEATURE_NAMES, group_id=train_g),
            Pool(eval_x, eval_y, feature_names=FEATURE_NAMES, group_id=eval_g))


def metric_calcer_result(model, metrics, train_pool, n_chunks):
    calcer = model.create_metric_calcer(metrics)
    if n_chunks == 1:
        calcer.add(train_pool)
    else:
        half = train_pool.num_row() // 2
        calcer.add(train_pool.slice(list(range(half))))
        calcer.add(train_pool.slice(list(range(half, train_pool.num_row()))))
    result = calcer.eval_metrics()
    # EvalMetricsResult isn't JSON-serializable directly -- get_result(name)
    # is the same per-iteration values list model.eval_metrics() returns
    # keyed by the metric's *canonicalized* description (e.g. "NDCG" becomes
    # "NDCG:type=Base"), same as model.eval_metrics()'s own return keys (see
    # eval_metrics_classes.json).
    return {key: result.get_result(key) for key in result._metric_descriptions}


def plot_predictions_per_document(model, pool, features_to_change):
    # model.plot_predictions() (core.py:3904) seeds its output list with
    # `[{}] * data.num_row()` -- Python's list-repeat of a mutable dict
    # aliases all N slots to the SAME dict object, so every document's entry
    # ends up sharing state and the final call's writes silently clobber all
    # earlier documents' predictions. Work around the upstream aliasing bug
    # by calling plot_predictions() once per single-row Pool (no aliasing
    # possible when num_row() == 1), so each document keeps its own real
    # perturbation predictions instead of all rows converging on the last one.
    # Build each row as its own ungrouped Pool rather than pool.slice([i]):
    # CatBoostRanker's train_pool carries group_id, and slicing a single
    # object out of a multi-object group violates Pool's "whole group or
    # nothing" subset invariant. Grouping is irrelevant to plot_predictions()
    # (it perturbs a single object and calls self.predict(doc) row-by-row),
    # so a plain feature-only Pool is equivalent and side-steps the invariant.
    features = pool.get_features()
    rows = []
    for i in range(pool.num_row()):
        row_pool = Pool([features[i]], feature_names=pool.get_feature_names())
        row_predictions, _ = model.plot_predictions(row_pool, features_to_change, plot=False)
        doc = row_predictions[0]
        rows.append({str(k): (v.tolist() if hasattr(v, "tolist") else v) for k, v in doc.items()})
    return rows


def class_block(name, model, train_pool, eval_pool, metrics, predict_kind,
                 predict_type, n_metric_chunks=2):
    block = {
        "get_test_eval": model.get_test_eval(),
        "get_test_evals": model.get_test_evals(),
        "create_metric_calcer": metric_calcer_result(model, metrics, train_pool, n_metric_chunks),
    }
    # plot_predictions() perturbs doc[feature] and calls self.predict(doc)
    # unchanged -- for CatBoostClassifier that resolves to its 'Class'
    # default, for every other class here to 'RawFormulaVal' (see the
    # per-input "predict_type" fixture field, threaded through to R's
    # explicit prediction_type argument -- R has no per-subclass default).
    block["plot_predictions"] = plot_predictions_per_document(model, train_pool, [0])
    return block


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {"inputs": {}, "expected": {}}

    # ---- CatBoost base class: MultiClass ----
    train_pool, eval_pool = split(MULTICLASS_LABEL)
    model = CatBoost(dict(loss_function="MultiClass", **COMMON))
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoost"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": MULTICLASS_LABEL[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": MULTICLASS_LABEL[TRAIN_N:],
        "feature_names": FEATURE_NAMES, "metrics": ["MultiClass", "Accuracy"],
        "predict_type": "RawFormulaVal",
    }
    fixture["expected"]["CatBoost"] = class_block(
        "CatBoost", model, train_pool, eval_pool, ["MultiClass", "Accuracy"], "raw", "RawFormulaVal")

    # plot_partial_dependence needs a non-multiclass model (see module docstring).
    pd_train_pool, _ = split(BINARY_LABEL)
    pd_model = CatBoost(dict(loss_function="Logloss", **COMMON))
    pd_model.fit(pd_train_pool)
    fixture["inputs"]["CatBoostPartialDependence"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": BINARY_LABEL[:TRAIN_N],
        "feature_names": FEATURE_NAMES,
    }
    fixture["expected"]["CatBoostPartialDependence"] = {
        "plot_partial_dependence": pd_model.plot_partial_dependence(pd_train_pool, 0, plot=False)[0].tolist(),
    }

    # ---- CatBoostClassifier: Logloss ----
    train_pool, eval_pool = split(BINARY_LABEL)
    model = CatBoostClassifier(loss_function="Logloss", **COMMON)
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoostClassifier"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": BINARY_LABEL[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": BINARY_LABEL[TRAIN_N:],
        "feature_names": FEATURE_NAMES, "metrics": ["Logloss", "AUC"],
        "predict_type": "Class",
    }
    fixture["expected"]["CatBoostClassifier"] = class_block(
        "CatBoostClassifier", model, train_pool, eval_pool, ["Logloss", "AUC"], "class", "Class")
    fixture["expected"]["CatBoostClassifier"]["plot_partial_dependence"] = (
        model.plot_partial_dependence(train_pool, 0, plot=False)[0].tolist()
    )
    # CatBoostClassifier-only rows: predict_log_proba/staged_predict_log_proba/
    # get_probability_threshold/set_probability_threshold.
    fixture["expected"]["CatBoostClassifier"]["predict_log_proba"] = (
        model.predict_log_proba(train_pool).tolist()
    )
    staged_log_proba = [batch.tolist() for batch in model.staged_predict_log_proba(train_pool, eval_period=3)]
    fixture["expected"]["CatBoostClassifier"]["staged_predict_log_proba"] = staged_log_proba
    fixture["expected"]["CatBoostClassifier"]["get_probability_threshold_default"] = model.get_probability_threshold()
    model.set_probability_threshold(0.3)
    fixture["expected"]["CatBoostClassifier"]["get_probability_threshold_after_set"] = model.get_probability_threshold()

    # ---- CatBoostRegressor: RMSE ----
    train_pool, eval_pool = split(REGRESSION_LABEL)
    model = CatBoostRegressor(loss_function="RMSE", **COMMON)
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoostRegressor"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": REGRESSION_LABEL[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": REGRESSION_LABEL[TRAIN_N:],
        "feature_names": FEATURE_NAMES, "metrics": ["RMSE", "MAE"],
        "predict_type": "RawFormulaVal",
    }
    fixture["expected"]["CatBoostRegressor"] = class_block(
        "CatBoostRegressor", model, train_pool, eval_pool, ["RMSE", "MAE"], "raw", "RawFormulaVal")
    fixture["expected"]["CatBoostRegressor"]["plot_partial_dependence"] = (
        model.plot_partial_dependence(train_pool, 0, plot=False)[0].tolist()
    )

    # ---- CatBoostRanker: YetiRank (requires group_id) ----
    train_pool, eval_pool = split(BINARY_LABEL, GROUP_ID)
    model = CatBoostRanker(loss_function="YetiRank", **COMMON)
    model.fit(train_pool, eval_set=eval_pool)
    fixture["inputs"]["CatBoostRanker"] = {
        "num1": NUM1[:TRAIN_N], "num2": NUM2[:TRAIN_N], "label": BINARY_LABEL[:TRAIN_N], "group_id": GROUP_ID[:TRAIN_N],
        "eval_num1": NUM1[TRAIN_N:], "eval_num2": NUM2[TRAIN_N:], "eval_label": BINARY_LABEL[TRAIN_N:],
        "eval_group_id": GROUP_ID[TRAIN_N:],
        "feature_names": FEATURE_NAMES, "metrics": ["PFound", "NDCG"],
        "predict_type": "RawFormulaVal",
    }
    fixture["expected"]["CatBoostRanker"] = class_block(
        "CatBoostRanker", model, train_pool, eval_pool, ["PFound", "NDCG"], "raw", "RawFormulaVal", n_metric_chunks=1)
    fixture["expected"]["CatBoostRanker"]["plot_partial_dependence"] = (
        model.plot_partial_dependence(train_pool, 0, plot=False)[0].tolist()
    )

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2, cls=_NumpyAwareEncoder)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
