#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <R_ext/Rdynload.h>

SEXP RawScalar_R(void) {
    return Rf_ScalarInteger(42);
}

/* Attempt 1 (FAILED, see report): calling R_init_probepkg(dll) -- the
   cpp11-generated init -- from here after R_registerRoutines(dll, ...,
   RawCallEntries, ...) silently drops the raw entries: R_registerRoutines
   REPLACES a DLL's routine table on each call, it does not merge. Only
   the entries from whichever call ran LAST survive.

   Fix: never call cpp11's generated R_init_probepkg at all (it is already
   dead code -- its symbol name matches the R package name "probepkg", not
   the .so name "libprobe", so dyn.load() never auto-invokes it either).
   Instead declare the individual extern "C" wrapper cpp11::cpp_register()
   emitted in src/cpp11.cpp (_probepkg_cpp11_scalar, which IS externally
   visible) and fold it into ONE combined CallEntries table registered by
   ONE R_registerRoutines call. */
extern SEXP _probepkg_cpp11_scalar(void);

static const R_CallMethodDef CombinedCallEntries[] = {
    {"RawScalar_R",             (DL_FUNC) &RawScalar_R,             0},
    {"_probepkg_cpp11_scalar",  (DL_FUNC) &_probepkg_cpp11_scalar,  0},
    {NULL, NULL, 0}
};

#if defined(_WIN32)
__declspec(dllexport)
#endif
void R_init_libprobe(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CombinedCallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
