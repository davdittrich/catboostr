// Implementation of the fork-owned custom-R-objective trampolines
// (catboost-8z4.89, P6.3). See r_custom_objective.h for the design.

// Standard/vendor headers before any R header -- R.h's `#define length(x)`
// (etc.) corrupts libstdc++/libc++'s own headers (<locale>, pulled in
// transitively by exception.h's TString machinery) if processed first
// (catboost-8z4.88 finding). r_custom_objective.h itself is R-clean until
// its own last line, but is still included last here so nothing above it
// can accidentally reorder that.
#include <catboost/libs/helpers/exception.h>

#include <algorithm>
#include <cstring>

#include "r_custom_objective.h"

namespace NCatboostR {

namespace {

// R has no public Rf_getListElement helper; this is the standard
// "Writing R Extensions" idiom for looking up a named list element.
SEXP GetListElementByName(SEXP list, const char* name) {
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if (Rf_isNull(names)) {
        return R_NilValue;
    }
    for (R_xlen_t i = 0; i < Rf_xlength(list); ++i) {
        if (std::strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
            return VECTOR_ELT(list, i);
        }
    }
    return R_NilValue;
}

// --- main-thread-only marshaling (run inside TRCallbackBridge::Call) -------

// calc_ders_range(approx, target, weight) -> numeric matrix, `count` rows,
// 2 columns (der1, der2).
void RunCalcDersRangeOnMainThread(
    SEXP calcDersRangeFun,
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,
    TDers* ders
) {
    SEXP approxSEXP = PROTECT(Rf_allocVector(REALSXP, count));
    std::copy(approxes, approxes + count, REAL(approxSEXP));

    SEXP targetSEXP = PROTECT(Rf_allocVector(REALSXP, count));
    std::copy(targets, targets + count, REAL(targetSEXP));

    SEXP weightSEXP = PROTECT(weights != nullptr ? Rf_allocVector(REALSXP, count) : R_NilValue);
    if (weights != nullptr) {
        std::copy(weights, weights + count, REAL(weightSEXP));
    }

    SEXP call = PROTECT(Rf_lang4(calcDersRangeFun, approxSEXP, targetSEXP, weightSEXP));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(4);
        throw std::runtime_error("catboost: error in R custom objective's calc_ders_range");
    }
    PROTECT(result);

    CB_ENSURE(
        Rf_isMatrix(result) && Rf_isReal(result) && Rf_nrows(result) == count && Rf_ncols(result) == 2,
        "catboost: custom_objective$calc_ders_range must return a numeric matrix with "
            << count << " rows and 2 columns (der1, der2)"
    );
    const double* resultData = REAL(result);
    for (int i = 0; i < count; ++i) {
        ders[i].Der1 = resultData[i];
        ders[i].Der2 = resultData[i + count];
    }

    UNPROTECT(5);
}

// calc_ders_multi(approx, target, weight) -> list(der1 = <numeric vector,
// length approxDim>, der2 = <approxDim x approxDim numeric matrix, or NULL>).
// Shared by both CalcDersMultiClass (targetDim == 1) and CalcDersMultiTarget.
void RunCalcDersMultiOnMainThread(
    SEXP calcDersMultiFun,
    const double* approxData,
    int approxDim,
    const float* targetData,
    int targetDim,
    float weight,
    TVector<double>* ders,
    THessianInfo* der2
) {
    SEXP approxSEXP = PROTECT(Rf_allocVector(REALSXP, approxDim));
    std::copy(approxData, approxData + approxDim, REAL(approxSEXP));

    SEXP targetSEXP = PROTECT(Rf_allocVector(REALSXP, targetDim));
    std::copy(targetData, targetData + targetDim, REAL(targetSEXP));

    SEXP weightSEXP = PROTECT(Rf_ScalarReal(weight));

    SEXP call = PROTECT(Rf_lang4(calcDersMultiFun, approxSEXP, targetSEXP, weightSEXP));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(4);
        throw std::runtime_error("catboost: error in R custom objective's calc_ders_multi");
    }
    PROTECT(result);

    CB_ENSURE(
        TYPEOF(result) == VECSXP && Rf_xlength(result) >= 1,
        "catboost: custom_objective$calc_ders_multi must return list(der1, der2)"
    );
    SEXP der1SEXP = GetListElementByName(result, "der1");
    CB_ENSURE(
        Rf_isReal(der1SEXP) && Rf_xlength(der1SEXP) == approxDim,
        "catboost: custom_objective$calc_ders_multi's 'der1' must be a numeric vector of length "
            << approxDim
    );
    const double* der1Data = REAL(der1SEXP);
    ders->assign(der1Data, der1Data + approxDim);

    if (der2 != nullptr) {
        SEXP der2SEXP = GetListElementByName(result, "der2");
        CB_ENSURE(
            !Rf_isNull(der2SEXP) && Rf_isMatrix(der2SEXP) && Rf_isReal(der2SEXP) &&
                Rf_nrows(der2SEXP) == approxDim && Rf_ncols(der2SEXP) == approxDim,
            "catboost: custom_objective$calc_ders_multi's 'der2' must be a "
                << approxDim << "x" << approxDim << " numeric matrix "
                << "(second derivatives were requested by the current leaf_estimation_method)"
        );
        const double* der2Data = REAL(der2SEXP);
        // THessianInfo::Data layout: for EHessianType::Diagonal, ApproxDimension
        // entries (the diagonal only); for Symmetric, the upper triangle
        // (including the diagonal) stored row-major -- same traversal order
        // Python's _ObjectiveCalcDersMultiClass/_MultiTarget use
        // (_catboost.pyx:1745-1750/1778-1783). R matrices are column-major:
        // element (row, col) lives at der2Data[row + col * approxDim].
        int index = 0;
        if (der2->HessianType == EHessianType::Diagonal) {
            for (int row = 0; row < approxDim; ++row) {
                der2->Data[index++] = der2Data[row + row * approxDim];
            }
        } else {
            for (int row = 0; row < approxDim; ++row) {
                for (int col = row; col < approxDim; ++col) {
                    der2->Data[index++] = der2Data[row + col * approxDim];
                }
            }
        }
    }

    UNPROTECT(5);
}

// --- TCustomObjectiveDescriptor trampolines (called from any thread) -------

void RCalcDersRange(
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,
    TDers* ders,
    void* customData
) {
    auto* context = static_cast<TRCustomObjectiveContext*>(customData);
    TActiveWorkScope workScope(*context->Bridge);
    context->Bridge->Call([&] {
        RunCalcDersRangeOnMainThread(context->CalcDersRangeFun, count, approxes, targets, weights, ders);
    });
}

void RCalcDersMultiClass(
    const TVector<double>& approx,
    float target,
    float weight,
    TVector<double>* ders,
    THessianInfo* der2,
    void* customData
) {
    auto* context = static_cast<TRCustomObjectiveContext*>(customData);
    TActiveWorkScope workScope(*context->Bridge);
    context->Bridge->Call([&] {
        RunCalcDersMultiOnMainThread(
            context->CalcDersMultiFun, approx.data(), approx.ysize(), &target, 1, weight, ders, der2);
    });
}

void RCalcDersMultiTarget(
    TConstArrayRef<double> approx,
    TConstArrayRef<float> target,
    float weight,
    TVector<double>* ders,
    THessianInfo* der2,
    void* customData
) {
    auto* context = static_cast<TRCustomObjectiveContext*>(customData);
    TActiveWorkScope workScope(*context->Bridge);
    context->Bridge->Call([&] {
        RunCalcDersMultiOnMainThread(
            context->CalcDersMultiFun,
            approx.data(), static_cast<int>(approx.size()),
            target.data(), static_cast<int>(target.size()),
            weight, ders, der2);
    });
}

}  // namespace

TMaybe<TCustomObjectiveDescriptor> BuildCustomObjectiveDescriptor(
    SEXP customObjectiveParam,
    TRCallbackBridge* bridge,
    TRCustomObjectiveContext* context
) {
    if (customObjectiveParam == R_NilValue) {
        return Nothing();
    }
    CB_ENSURE(TYPEOF(customObjectiveParam) == VECSXP, "catboost: custom_objective must be a named list");

    context->Bridge = bridge;
    context->CalcDersRangeFun = GetListElementByName(customObjectiveParam, "calc_ders_range");
    context->CalcDersMultiFun = GetListElementByName(customObjectiveParam, "calc_ders_multi");
    CB_ENSURE(
        context->CalcDersRangeFun != R_NilValue || context->CalcDersMultiFun != R_NilValue,
        "catboost: custom_objective must define at least one of calc_ders_range / calc_ders_multi"
    );

    TCustomObjectiveDescriptor descriptor;
    descriptor.CustomData = context;
    if (context->CalcDersRangeFun != R_NilValue) {
        descriptor.CalcDersRange = RCalcDersRange;
    }
    if (context->CalcDersMultiFun != R_NilValue) {
        descriptor.CalcDersMultiClass = RCalcDersMultiClass;
        descriptor.CalcDersMultiTarget = RCalcDersMultiTarget;
    }
    return descriptor;
}

}  // namespace NCatboostR
