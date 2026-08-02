#include <R.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>
#include "catboostr.h"

static const R_CallMethodDef CallEntries[] = {
    {"CatBoostCalcRegularFeatureEffect_R",    (DL_FUNC) &CatBoostCalcRegularFeatureEffect_R,     4},
    {"CatBoostCreateFromFile_R",              (DL_FUNC) &CatBoostCreateFromFile_R,              10},
    {"CatBoostCreateFromMatrix_R",            (DL_FUNC) &CatBoostCreateFromMatrix_R,            15},
    {"CatBoostCV_R",                          (DL_FUNC) &CatBoostCV_R,                           7},
    {"CatBoostDeserializeModel_R",            (DL_FUNC) &CatBoostDeserializeModel_R,             1},
    {"CatBoostDropUnusedFeaturesFromModel_R", (DL_FUNC) &CatBoostDropUnusedFeaturesFromModel_R,  1},
    {"CatBoostEvalMetrics_R",                 (DL_FUNC) &CatBoostEvalMetrics_R,                  9},
    {"CatBoostEvaluateObjectImportances_R",   (DL_FUNC) &CatBoostEvaluateObjectImportances_R,    7},
    {"CatBoostFit_R",                         (DL_FUNC) &CatBoostFit_R,                          3},
    {"CatBoostGetModelParams_R",              (DL_FUNC) &CatBoostGetModelParams_R,               1},
    {"CatBoostGetNumTrees_R",                 (DL_FUNC) &CatBoostGetNumTrees_R,                  1},
    {"CatBoostGetPlainParams_R",              (DL_FUNC) &CatBoostGetPlainParams_R,               1},
    {"CatBoostHashStrings_R",                 (DL_FUNC) &CatBoostHashStrings_R,                  1},
    {"CatBoostIsGroupwiseMetric_R",           (DL_FUNC) &CatBoostIsGroupwiseMetric_R,            1},
    {"CatBoostIsNullHandle_R",                (DL_FUNC) &CatBoostIsNullHandle_R,                 1},
    {"CatBoostIsOblivious_R",                 (DL_FUNC) &CatBoostIsOblivious_R,                  1},
    {"CatBoostOutputModel_R",                 (DL_FUNC) &CatBoostOutputModel_R,                  5},
    {"CatBoostPoolGetBaseline_R",             (DL_FUNC) &CatBoostPoolGetBaseline_R,              1},
    {"CatBoostPoolGetCatFeatureIndices_R",    (DL_FUNC) &CatBoostPoolGetCatFeatureIndices_R,     1},
    {"CatBoostPoolGetEmbeddingFeatureIndices_R", (DL_FUNC) &CatBoostPoolGetEmbeddingFeatureIndices_R, 1},
    {"CatBoostPoolGetFeatureNames_R",         (DL_FUNC) &CatBoostPoolGetFeatureNames_R,          1},
    {"CatBoostPoolGetFeatures_R",             (DL_FUNC) &CatBoostPoolGetFeatures_R,              1},
    {"CatBoostPoolGetGroupIdHash_R",          (DL_FUNC) &CatBoostPoolGetGroupIdHash_R,           1},
    {"CatBoostPoolGetLabel_R",                (DL_FUNC) &CatBoostPoolGetLabel_R,                 1},
    {"CatBoostPoolGetTextFeatureIndices_R",   (DL_FUNC) &CatBoostPoolGetTextFeatureIndices_R,    1},
    {"CatBoostPoolGetWeight_R",               (DL_FUNC) &CatBoostPoolGetWeight_R,                1},
    {"CatBoostPoolHasLabel_R",                (DL_FUNC) &CatBoostPoolHasLabel_R,                 1},
    {"CatBoostPoolNumCol_R",                  (DL_FUNC) &CatBoostPoolNumCol_R,                   1},
    {"CatBoostPoolNumPairs_R",                (DL_FUNC) &CatBoostPoolNumPairs_R,                 1},
    {"CatBoostPoolNumRow_R",                  (DL_FUNC) &CatBoostPoolNumRow_R,                   1},
    {"CatBoostPoolSetBaseline_R",             (DL_FUNC) &CatBoostPoolSetBaseline_R,              2},
    {"CatBoostPoolSetFeatureNames_R",         (DL_FUNC) &CatBoostPoolSetFeatureNames_R,          2},
    {"CatBoostPoolSetGroupId_R",              (DL_FUNC) &CatBoostPoolSetGroupId_R,               2},
    {"CatBoostPoolSetGroupWeight_R",          (DL_FUNC) &CatBoostPoolSetGroupWeight_R,           2},
    {"CatBoostPoolSetPairs_R",                (DL_FUNC) &CatBoostPoolSetPairs_R,                 2},
    {"CatBoostPoolSetPairsWeight_R",          (DL_FUNC) &CatBoostPoolSetPairsWeight_R,           2},
    {"CatBoostPoolSetSubgroupId_R",           (DL_FUNC) &CatBoostPoolSetSubgroupId_R,            2},
    {"CatBoostPoolSetTimestamp_R",            (DL_FUNC) &CatBoostPoolSetTimestamp_R,             2},
    {"CatBoostPoolSetWeight_R",               (DL_FUNC) &CatBoostPoolSetWeight_R,                2},
    {"CatBoostPoolSlice_R",                   (DL_FUNC) &CatBoostPoolSlice_R,                    3},
    {"CatBoostPredictMulti_R",                (DL_FUNC) &CatBoostPredictMulti_R,                 7},
    {"CatBoostPredictVirtualEnsembles_R",     (DL_FUNC) &CatBoostPredictVirtualEnsembles_R,      7},
    {"CatBoostPrepareEval_R",                 (DL_FUNC) &CatBoostPrepareEval_R,                  5},
    {"CatBoostReadModel_R",                   (DL_FUNC) &CatBoostReadModel_R,                    2},
    {"CatBoostSerializeModel_R",              (DL_FUNC) &CatBoostSerializeModel_R,               1},
    {"CatBoostShrinkModel_R",                 (DL_FUNC) &CatBoostShrinkModel_R,                  3},
    {"CatBoostSumModels_R",                   (DL_FUNC) &CatBoostSumModels_R,                    3},
    {"CatBoostVersion_R",                     (DL_FUNC) &CatBoostVersion_R,                      0},
    {NULL, NULL, 0}
};

#if defined(_WIN32)
__declspec(dllexport)
#endif
void R_init_libcatboostr(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
