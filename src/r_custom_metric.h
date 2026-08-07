// Fork-owned custom-R-metric trampolines (catboost-8z4.90, P6.4).
//
// Wires an R closure list (evaluate / is_max_optimal / get_final_error /
// is_additive) into TCustomMetricDescriptor's six C-linkage function-pointer
// members (vendor/catboost/catboost/libs/metrics/metric.h:105-140), using
// catboost-8z4.93's TRCallbackBridge to marshal every call -- exactly the
// same mechanism r_custom_objective.cpp (catboost-8z4.89) uses for
// TCustomObjectiveDescriptor -- back to R's main thread.
//
// Mirrors the four-method contract of Python's CustomMetric /
// MultiTargetCustomMetric (_catboost.pyx:198-241, _BuildCustomMetricDescriptor
// at _catboost.pyx:1907): evaluate(approx, target, weight) -> (error,
// weight), is_max_optimal(), get_final_error(error, weight), is_additive().
// R has no class hierarchy to dispatch single- vs multi-target eval on (the
// way Python dispatches on `issubclass(metricObject, MultiTargetCustomMetric)`),
// so the R list carries an explicit `multi_target` flag instead; exactly one
// of EvalFunc/EvalMultiTargetFunc may be set (TCustomMetricDescriptor::
// IsMultiTargetMetric(), metric.h:149, CB_ENSUREs on both being defined).
#pragma once

#include <catboost/libs/metrics/metric.h>

// STL/vendor headers (pulled in transitively by r_callback_bridge.h and
// metric.h) must come before any R header: R.h's `#define length(x)` (and
// friends) corrupt libstdc++/libc++ internals if processed first
// (catboost-8z4.88 finding; same ordering r_custom_objective.h follows).
#include "r_callback_bridge.h"

#include <Rinternals.h>

namespace NCatboostR {

// Built by BuildCustomMetricDescriptor and stored for the lifetime of one
// CatBoostFit_R call; TCustomMetricDescriptor::CustomData points at it.
struct TRCustomMetricContext {
    TRCallbackBridge* Bridge = nullptr;
    // R closures, all function(...) -> ... . R_NilValue for the two optional
    // ones (GetFinalErrorFun, IsAdditiveFun) if the user's
    // custom_eval_metric_object list did not define them -- RGetFinalError /
    // RIsAdditive then fall back to TMetric's own CPU defaults
    // (metric.cpp:67-70's error/weight; false) without touching R at all.
    SEXP EvaluateFun = R_NilValue;
    SEXP IsMaxOptimalFun = R_NilValue;
    SEXP GetFinalErrorFun = R_NilValue;
    SEXP IsAdditiveFun = R_NilValue;
};

// Returns Nothing() for R_NilValue (no custom eval metric supplied -- built-in
// eval_metric/custom_metric params-list behaviour is unaffected). Otherwise
// validates customEvalMetricParam is a named list defining at least
// evaluate/is_max_optimal, and fills *context (whose address becomes the
// descriptor's CustomData).
TMaybe<TCustomMetricDescriptor> BuildCustomMetricDescriptor(
    SEXP customEvalMetricParam,
    TRCallbackBridge* bridge,
    TRCustomMetricContext* context
);

}  // namespace NCatboostR
