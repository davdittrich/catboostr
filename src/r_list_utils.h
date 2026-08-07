// Shared named-R-list lookup, used by both trampoline modules
// (r_custom_objective.cpp, catboost-8z4.89; r_custom_metric.cpp,
// catboost-8z4.90). Genuinely identical in both, so it lives here once
// instead of being copy-pasted.
//
// Must be included after any STL/vendor headers already pulled in by the
// including .cpp/.h: R.h's `#define length(x) Rf_length(x)` (and friends)
// corrupt libstdc++/libc++'s own headers if processed first (catboost-8z4.88
// finding; see r_custom_objective.h for the fuller explanation).
#pragma once

#include <cstring>

#include <Rinternals.h>

namespace NCatboostR {

// R has no public Rf_getListElement helper; this is the standard
// "Writing R Extensions" idiom for looking up a named list element.
inline SEXP GetListElementByName(SEXP list, const char* name) {
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

}  // namespace NCatboostR
