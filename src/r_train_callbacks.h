// Fork-owned per-iteration training-callback trampoline (catboost-8z4.122).
//
// Wires params$callbacks (a list of R functions, see
// apply_train_callbacks_params()/catboost.train's roxygen doc in
// R/catboost.R) into ITrainingCallbacks' companion ICustomCallbacks /
// TCustomCallbackDescriptor::AfterIterationFunc hook
// (vendor/catboost/catboost/libs/train_lib/train_model.h) -- the exact same
// native mechanism Python's `callbacks` kwarg uses
// (_catboost.pyx:1921 _BuildCustomCallbackDescritor /
// _catboost.pyx:1340 _CallbackAfterIteration), previously never wired up on
// the R side (CatBoostFit_R always passed Nothing() for it).
//
// Marshaling follows r_custom_objective.h/r_custom_metric.h's established
// pattern: TRCallbackBridge moves the call from whatever thread CatBoost's
// boosting loop is running on back to R's main thread. Unlike
// calc_ders_range/calc_ders_multi (which call a user closure directly with
// numeric args this glue builds by hand), AfterIterationFunc's payload is
// TMetricsAndTimeLeftHistory -- an entire nested metrics history, not a few
// scalars -- so this reuses the *native* SaveMetrics() JSON serialization
// (catboost/libs/loggers/catboost_logger_helpers.h/.cpp, the exact same JSON
// shape stored in a finished model's "training" metadata and already parsed
// by R/catboost.R's .catboost_get_training_metrics) and hands the JSON
// string to an R-side dispatcher, .catboost_run_train_callbacks(), which
// does the JSON parsing/shaping/dispatch in R -- reusing
// catboost.get_evals_result()'s existing transpose helper instead of a
// second, C++-side reimplementation of that shape.
#pragma once

#include <catboost/libs/train_lib/train_model.h>

// STL headers (pulled in transitively by r_callback_bridge.h) must come
// before any R header: R.h's `#define length(x)` (and friends) corrupt
// libstdc++/libc++'s own headers if processed first (catboost-8z4.88
// finding; same ordering r_custom_objective.h/r_callback_bridge.cpp follow).
#include "r_callback_bridge.h"

#include <Rinternals.h>

namespace NCatboostR {

// Built by BuildTrainCallbacksDescriptor and stored for the lifetime of one
// CatBoostFit_R call; TCustomCallbackDescriptor::CustomData points at it.
struct TRTrainCallbacksContext {
    TRCallbackBridge* Bridge = nullptr;
    // The R list of callback functions itself (params$callbacks, already
    // stripped of the JSON-exported params by
    // apply_train_callbacks_params()) -- R_NilValue if none was supplied.
    // Protected for the descriptor's lifetime simply by being a live
    // argument of the enclosing CatBoostFit_R .Call() frame.
    SEXP CallbacksParam = R_NilValue;

    // .catboost_run_train_callbacks, resolved ONCE by
    // BuildTrainCallbacksDescriptor while still on R's real main thread --
    // i.e. before TRCallbackBridge::Run() ever spawns the background
    // training thread. Per-iteration dispatch (RTrainAfterIteration) then
    // only ever calls this already-resolved function via R_tryEval, never
    // repeating the R_FindNamespace/Rf_eval lookup itself. Doing that lookup
    // per iteration would run it from inside TRCallbackBridge::Call()'s
    // action, which DOES execute on the main thread but with no setjmp
    // frame of its own around it (see r_train_callbacks.cpp) -- a raw
    // Rf_eval() error there would longjmp straight out of
    // TRCallbackBridge::DrainLoop(), skipping TRCallbackBridge::Run()'s
    // `background.join()` and leaving the still-running background training
    // thread orphaned instead of unwinding cleanly the way R_tryEval-guarded
    // errors already do.
    //
    // R_NilValue until resolved; protected via R_PreserveObject for the
    // context's lifetime (released by the destructor below), since a plain
    // PROTECT()'s stack-discipline protection does not survive returning
    // from BuildTrainCallbacksDescriptor.
    SEXP Dispatcher = R_NilValue;

    ~TRTrainCallbacksContext() {
        if (Dispatcher != R_NilValue) {
            R_ReleaseObject(Dispatcher);
        }
    }
};

// Returns Nothing() for R_NilValue (no callbacks supplied -- behaviour is
// then byte-for-byte unchanged from before this ticket). Otherwise fills
// *context (whose address becomes the descriptor's CustomData).
TMaybe<TCustomCallbackDescriptor> BuildTrainCallbacksDescriptor(
    SEXP callbacksParam,
    TRCallbackBridge* bridge,
    TRTrainCallbacksContext* context
);

}  // namespace NCatboostR
