// Implementation of the fork-owned custom-R-metric trampolines
// (catboost-8z4.90, P6.4). See r_custom_metric.h for the design.

// Standard/vendor headers before any R header -- R.h's `#define length(x)`
// (etc.) corrupts libstdc++/libc++'s own headers (<locale>, pulled in
// transitively by exception.h's TString machinery) if processed first
// (catboost-8z4.88 finding). r_custom_metric.h itself is R-clean until its
// own last line, but is still included last here so nothing above it can
// accidentally reorder that.
#include <catboost/libs/helpers/exception.h>

#include <algorithm>

#include "r_custom_metric.h"
// catboost-8z4.90: shared with r_custom_objective.cpp -- byte-for-byte
// identical named-list lookup in both TUs.
#include "r_list_utils.h"

namespace NCatboostR {

namespace {

// --- main-thread-only marshaling (run inside TRCallbackBridge::Call) -------

// approx (approxDim per-dim arrays, each covering the whole fold) -> an
// R `count` x `approxDim` numeric matrix holding only the [begin, begin +
// count) slice -- shared by the single- and multi-target eval trampolines.
SEXP BuildApproxBatchMatrix(TConstArrayRef<TConstArrayRef<double>> approx, int begin, int count) {
    const int approxDim = static_cast<int>(approx.size());
    SEXP matrix = PROTECT(Rf_allocMatrix(REALSXP, count, approxDim));
    double* data = REAL(matrix);
    for (int dim = 0; dim < approxDim; ++dim) {
        const double* col = approx[dim].data() + begin;
        std::copy(col, col + count, data + static_cast<size_t>(dim) * count);
    }
    return matrix;  // leaves 1 PROTECT on the stack
}

// weight (whole-fold array, or empty if weights are not in use) -> an R
// numeric vector holding the [begin, begin + count) slice, or R_NilValue.
SEXP BuildWeightBatchVector(TConstArrayRef<float> weight, int begin, int count) {
    if (weight.empty()) {
        return R_NilValue;  // caller still PROTECTs its own R_NilValue, harmlessly
    }
    SEXP vec = PROTECT(Rf_allocVector(REALSXP, count));
    const float* src = weight.data() + begin;
    double* dst = REAL(vec);
    std::copy(src, src + count, dst);
    return vec;
}

// evaluate()'s return value: a named list(error = <scalar>, weight =
// <scalar>), mirroring TMetricHolder's 2-element Stats layout (sum of
// per-object error, sum of weight) that metric.cpp's TCustomMetric::Eval /
// TMultiTargetCustomMetric::Eval both CB_ENSURE on (metric.cpp:4865-4869,
// 4931-4935).
TMetricHolder ParseEvaluateResult(SEXP result) {
    CB_ENSURE(
        TYPEOF(result) == VECSXP,
        "catboost: custom_eval_metric_object$evaluate must return list(error, weight)"
    );
    SEXP errorSEXP = GetListElementByName(result, "error");
    SEXP weightSEXP = GetListElementByName(result, "weight");
    CB_ENSURE(
        errorSEXP != R_NilValue && weightSEXP != R_NilValue,
        "catboost: custom_eval_metric_object$evaluate must return a named list with "
            "'error' and 'weight' elements"
    );
    TMetricHolder holder(2);
    holder.Stats[0] = Rf_asReal(errorSEXP);
    holder.Stats[1] = Rf_asReal(weightSEXP);
    return holder;
}

// evaluate(approx, target, weight) -> list(error, weight), single-target
// case: target is one flat array for the whole fold.
TMetricHolder RunSingleTargetEvaluateOnMainThread(
    SEXP evaluateFun,
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    int begin,
    int end
) {
    const int count = end - begin;
    SEXP approxSEXP = PROTECT(BuildApproxBatchMatrix(approx, begin, count));

    SEXP targetSEXP = PROTECT(Rf_allocVector(REALSXP, count));
    std::copy(target.data() + begin, target.data() + begin + count, REAL(targetSEXP));

    SEXP weightSEXP = PROTECT(BuildWeightBatchVector(weight, begin, count));

    SEXP call = PROTECT(Rf_lang4(evaluateFun, approxSEXP, targetSEXP, weightSEXP));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(4);
        throw std::runtime_error("catboost: error in R custom eval metric's evaluate");
    }
    PROTECT(result);
    TMetricHolder holder = ParseEvaluateResult(result);
    UNPROTECT(5);
    return holder;
}

// evaluate(approx, target, weight) -> list(error, weight), multi-target
// case: target is targetDim per-dim arrays, marshaled the same way as approx.
TMetricHolder RunMultiTargetEvaluateOnMainThread(
    SEXP evaluateFun,
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end
) {
    const int count = end - begin;
    SEXP approxSEXP = PROTECT(BuildApproxBatchMatrix(approx, begin, count));

    const int targetDim = static_cast<int>(target.size());
    SEXP targetSEXP = PROTECT(Rf_allocMatrix(REALSXP, count, targetDim));
    double* targetData = REAL(targetSEXP);
    for (int dim = 0; dim < targetDim; ++dim) {
        const float* col = target[dim].data() + begin;
        double* out = targetData + static_cast<size_t>(dim) * count;
        std::copy(col, col + count, out);
    }

    SEXP weightSEXP = PROTECT(BuildWeightBatchVector(weight, begin, count));

    SEXP call = PROTECT(Rf_lang4(evaluateFun, approxSEXP, targetSEXP, weightSEXP));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(4);
        throw std::runtime_error("catboost: error in R custom eval metric's evaluate");
    }
    PROTECT(result);
    TMetricHolder holder = ParseEvaluateResult(result);
    UNPROTECT(5);
    return holder;
}

bool RunIsMaxOptimalOnMainThread(SEXP isMaxOptimalFun) {
    SEXP call = PROTECT(Rf_lang1(isMaxOptimalFun));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(1);
        throw std::runtime_error("catboost: error in R custom eval metric's is_max_optimal");
    }
    bool isMaxOptimal = Rf_asLogical(result) == TRUE;
    UNPROTECT(1);
    return isMaxOptimal;
}

bool RunIsAdditiveOnMainThread(SEXP isAdditiveFun) {
    SEXP call = PROTECT(Rf_lang1(isAdditiveFun));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(1);
        throw std::runtime_error("catboost: error in R custom eval metric's is_additive");
    }
    bool isAdditive = Rf_asLogical(result) == TRUE;
    UNPROTECT(1);
    return isAdditive;
}

// get_final_error(error): `error` is a length-2 numeric vector, the same
// (sum error, sum weight) pair evaluate() reports via Stats.
double RunGetFinalErrorOnMainThread(SEXP getFinalErrorFun, const TMetricHolder& error) {
    SEXP errorSEXP = PROTECT(Rf_allocVector(REALSXP, 2));
    REAL(errorSEXP)[0] = error.Stats[0];
    REAL(errorSEXP)[1] = error.Stats[1];

    SEXP call = PROTECT(Rf_lang2(getFinalErrorFun, errorSEXP));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(2);
        throw std::runtime_error("catboost: error in R custom eval metric's get_final_error");
    }
    double finalError = Rf_asReal(result);
    UNPROTECT(2);
    return finalError;
}

// --- TCustomMetricDescriptor trampolines (called from any thread) ----------

TMetricHolder RSingleTargetEval(
    TConstArrayRef<TConstArrayRef<double>>& approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    void* customData
) {
    auto* context = static_cast<TRCustomMetricContext*>(customData);
    TActiveWorkScope workScope(*context->Bridge);
    TMetricHolder holder(2);
    context->Bridge->Call([&] {
        holder = RunSingleTargetEvaluateOnMainThread(context->EvaluateFun, approx, target, weight, begin, end);
    });
    return holder;
}

TMetricHolder RMultiTargetEval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    void* customData
) {
    auto* context = static_cast<TRCustomMetricContext*>(customData);
    TActiveWorkScope workScope(*context->Bridge);
    TMetricHolder holder(2);
    context->Bridge->Call([&] {
        holder = RunMultiTargetEvaluateOnMainThread(context->EvaluateFun, approx, target, weight, begin, end);
    });
    return holder;
}

// No R closure to ask -- catboost.train()'s custom_eval_metric_object list
// has no get_description slot (not part of the Input Specification), so
// every R custom metric reports the same fixed name. GetDescriptionFunc is a
// required (non-TMaybe) descriptor member, hence still needs a concrete
// implementation.
TString RGetDescription(void* /* customData */) {
    return TString("CustomMetric");
}

bool RIsMaxOptimal(void* customData) {
    auto* context = static_cast<TRCustomMetricContext*>(customData);
    TActiveWorkScope workScope(*context->Bridge);
    bool result = false;
    context->Bridge->Call([&] {
        result = RunIsMaxOptimalOnMainThread(context->IsMaxOptimalFun);
    });
    return result;
}

bool RIsAdditive(void* customData) {
    auto* context = static_cast<TRCustomMetricContext*>(customData);
    if (context->IsAdditiveFun == R_NilValue) {
        // Matches Python's own default: _MetricIsAdditive returns
        // hasattr(metricObject, 'is_additive') and ... (_catboost.pyx:1326-1328).
        return false;
    }
    TActiveWorkScope workScope(*context->Bridge);
    bool result = false;
    context->Bridge->Call([&] {
        result = RunIsAdditiveOnMainThread(context->IsAdditiveFun);
    });
    return result;
}

double RGetFinalError(const TMetricHolder& error, void* customData) {
    auto* context = static_cast<TRCustomMetricContext*>(customData);
    if (context->GetFinalErrorFun == R_NilValue) {
        // TMetric::GetFinalError's own CPU default (metric.cpp:67-70): no R
        // call needed, so this is safe to run on whatever thread called us.
        return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
    }
    TActiveWorkScope workScope(*context->Bridge);
    double result = 0;
    context->Bridge->Call([&] {
        result = RunGetFinalErrorOnMainThread(context->GetFinalErrorFun, error);
    });
    return result;
}

}  // namespace

TMaybe<TCustomMetricDescriptor> BuildCustomMetricDescriptor(
    SEXP customEvalMetricParam,
    TRCallbackBridge* bridge,
    TRCustomMetricContext* context
) {
    if (customEvalMetricParam == R_NilValue) {
        return Nothing();
    }
    CB_ENSURE(TYPEOF(customEvalMetricParam) == VECSXP, "catboost: custom_eval_metric_object must be a named list");

    context->Bridge = bridge;
    context->EvaluateFun = GetListElementByName(customEvalMetricParam, "evaluate");
    context->IsMaxOptimalFun = GetListElementByName(customEvalMetricParam, "is_max_optimal");
    context->GetFinalErrorFun = GetListElementByName(customEvalMetricParam, "get_final_error");
    context->IsAdditiveFun = GetListElementByName(customEvalMetricParam, "is_additive");
    CB_ENSURE(
        context->EvaluateFun != R_NilValue,
        "catboost: custom_eval_metric_object must define an 'evaluate' function"
    );
    CB_ENSURE(
        context->IsMaxOptimalFun != R_NilValue,
        "catboost: custom_eval_metric_object must define an 'is_max_optimal' function"
    );

    // R has no evaluate()-overload-by-subclass dispatch the way Python's
    // MultiTargetCustomMetric provides, so multi-target eval is opted into
    // explicitly. TCustomMetricDescriptor::IsMultiTargetMetric()
    // (metric.h:149) CB_ENSUREs that EvalFunc/EvalMultiTargetFunc are never
    // both defined at once, so exactly one is wired below.
    SEXP multiTargetSEXP = GetListElementByName(customEvalMetricParam, "multi_target");
    const bool isMultiTarget = multiTargetSEXP != R_NilValue && Rf_asLogical(multiTargetSEXP) == TRUE;

    TCustomMetricDescriptor descriptor;
    descriptor.CustomData = context;
    descriptor.GetDescriptionFunc = RGetDescription;
    descriptor.IsMaxOptimalFunc = RIsMaxOptimal;
    descriptor.IsAdditiveFunc = RIsAdditive;
    descriptor.GetFinalErrorFunc = RGetFinalError;
    if (isMultiTarget) {
        descriptor.EvalMultiTargetFunc = RMultiTargetEval;
    } else {
        descriptor.EvalFunc = RSingleTargetEval;
    }
    return descriptor;
}

}  // namespace NCatboostR
