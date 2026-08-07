// Fork-owned custom-R-objective trampolines (catboost-8z4.89, P6.3).
//
// Wires an R closure pair (calc_ders_range / calc_ders_multi) into
// TCustomObjectiveDescriptor's three C-linkage function-pointer members
// (vendor/catboost/catboost/private/libs/algo_helpers/custom_objective_descriptor.h),
// using catboost-8z4.93's TRCallbackBridge to marshal every call from
// whatever thread CatBoost invokes it on (a TBB worker, in general) back to
// R's main thread, where alone it is safe to touch R's C API.
//
// Mirrors the argument shape of Python's _ObjectiveCalcDersRange /
// _ObjectiveCalcDersMultiClass / _ObjectiveCalcDersMultiTarget
// (_catboost.pyx:1669-1783): calc_ders_range(approx, target, weight) and
// calc_ders_multi(approx, target, weight), called for both the MultiClass
// (scalar target) and MultiTarget (vector target) variants -- R has no
// scalar/length-1-vector distinction, so both pass `target` as a numeric
// vector.
#pragma once

#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

// STL headers (pulled in transitively by r_callback_bridge.h) must come
// before any R header: R.h's `#define length(x) Rf_length(x)` (and friends)
// corrupt libstdc++/libc++'s own headers if processed first (catboost-8z4.88
// finding; same ordering r_callback_bridge.cpp already follows).
#include "r_callback_bridge.h"

#include <Rinternals.h>

namespace NCatboostR {

// Built by BuildCustomObjectiveDescriptor and stored for the lifetime of one
// CatBoostFit_R call; TCustomObjectiveDescriptor::CustomData points at it.
struct TRCustomObjectiveContext {
    TRCallbackBridge* Bridge = nullptr;
    // R closures: function(approx, target, weight) -> ... . R_NilValue if the
    // user's custom_objective list did not define that method.
    SEXP CalcDersRangeFun = R_NilValue;
    SEXP CalcDersMultiFun = R_NilValue;
};

// Returns Nothing() for R_NilValue (no custom objective supplied -- built-in
// losses are unaffected). Otherwise validates customObjectiveParam is a
// named list defining at least one of calc_ders_range/calc_ders_multi and
// fills *context (whose address becomes the descriptor's CustomData).
TMaybe<TCustomObjectiveDescriptor> BuildCustomObjectiveDescriptor(
    SEXP customObjectiveParam,
    TRCallbackBridge* bridge,
    TRCustomObjectiveContext* context
);

}  // namespace NCatboostR
