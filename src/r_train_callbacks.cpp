// See r_train_callbacks.h for the design rationale.
#include "r_train_callbacks.h"

#include <catboost/libs/helpers/json_helpers.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>

#include <stdexcept>

namespace NCatboostR {

namespace {

// Package-namespace lookup for a non-exported R helper (.catboost_run_train_
// callbacks lives in R/catboost.R, never exported -- it is package-internal
// plumbing, not part of the public API). R_FindNamespace + Rf_eval (which
// forces the lazy-loaded binding, unlike Rf_findVarInFrame) is the standard
// way to reach a package's own namespace environment from its compiled code.
SEXP FindPackageFunction(const char* name) {
    SEXP pkgName = PROTECT(Rf_ScalarString(Rf_mkChar("catboostr")));
    SEXP ns = PROTECT(R_FindNamespace(pkgName));
    SEXP fun = PROTECT(Rf_eval(Rf_install(name), ns));
    UNPROTECT(3);
    return fun;
}

// --- main-thread-only marshaling (run inside TRCallbackBridge::Call) -------
//
// history.SaveMetrics() is the exact JSON shape R/catboost.R's
// .catboost_get_training_metrics already knows how to parse from a finished
// model's "training" metadata (catboost_logger_helpers.cpp:16-33,
// full_model_saver.cpp:575) -- reusing it here means this glue needs zero
// bespoke C++ marshaling of the nested per-iteration/per-metric history.
//
// `dispatcher` must already be resolved (see TRTrainCallbacksContext::
// Dispatcher) -- this function only ever reaches R via R_tryEval, never via
// a fresh namespace/symbol lookup, so any R-level error unwinds through the
// normal R_tryEval-caught path instead of a raw longjmp.
bool RunTrainCallbacksOnMainThread(SEXP dispatcher, SEXP callbacksParam, const TMetricsAndTimeLeftHistory& history) {
    const TString metricsJson = WriteTJsonValue(history.SaveMetrics());

    SEXP metricsJsonParam = PROTECT(Rf_ScalarString(Rf_mkChar(metricsJson.c_str())));
    SEXP iterationParam = PROTECT(Rf_ScalarInteger(static_cast<int>(history.LearnMetricsHistory.size())));
    SEXP call = PROTECT(Rf_lang4(dispatcher, callbacksParam, iterationParam, metricsJsonParam));

    int errorOccurred = 0;
    SEXP result = PROTECT(R_tryEval(call, R_GlobalEnv, &errorOccurred));
    if (errorOccurred) {
        UNPROTECT(4);
        throw std::runtime_error("catboost: error in R 'callbacks'");
    }
    const bool shouldContinue = Rf_asLogical(result) == TRUE;
    UNPROTECT(4);
    return shouldContinue;
}

bool RTrainAfterIteration(const TMetricsAndTimeLeftHistory& history, void* customData) {
    auto* context = static_cast<TRTrainCallbacksContext*>(customData);

    bool shouldContinue = true;
    TActiveWorkScope workScope(*context->Bridge);
    context->Bridge->Call([&] {
        shouldContinue = RunTrainCallbacksOnMainThread(context->Dispatcher, context->CallbacksParam, history);
    });
    return shouldContinue;
}

}  // namespace

TMaybe<TCustomCallbackDescriptor> BuildTrainCallbacksDescriptor(
    SEXP callbacksParam,
    TRCallbackBridge* bridge,
    TRTrainCallbacksContext* context
) {
    if (callbacksParam == R_NilValue) {
        return Nothing();
    }

    context->Bridge = bridge;
    context->CallbacksParam = callbacksParam;

    // Resolved here, synchronously, while this is still R's one and only
    // thread -- TRCallbackBridge::Run() (which spawns the background
    // training thread that will later invoke RTrainAfterIteration) has not
    // been called yet. A raw R error surfacing from this lookup therefore
    // longjmps through the ordinary CatBoostFit_R .Call() frame, exactly
    // like any other argument-parsing error, with no background thread to
    // orphan.
    SEXP dispatcher = FindPackageFunction(".catboost_run_train_callbacks");
    R_PreserveObject(dispatcher);
    context->Dispatcher = dispatcher;

    TCustomCallbackDescriptor descriptor;
    descriptor.CustomData = context;
    descriptor.AfterIterationFunc = RTrainAfterIteration;
    return descriptor;
}

}  // namespace NCatboostR
