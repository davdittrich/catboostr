#pragma once
#include <Rinternals.h>

#if defined(_WIN32)
#define EXPORT_FUNCTION __declspec(dllexport) SEXP
#else
#define EXPORT_FUNCTION SEXP
#endif

#if defined(__cplusplus)
extern "C" {
#endif


EXPORT_FUNCTION CatBoostCreateFromFile_R(
    SEXP poolFileParam,
    SEXP cdFileParam,
    SEXP pairsFileParam,
    SEXP graphFileParam,
    SEXP featureNamesFileParam,
    SEXP delimiterParam,
    SEXP numVectorDelimiterParam,
    SEXP hasHeaderParam,
    SEXP threadCountParam,
    SEXP verboseParam
);

EXPORT_FUNCTION CatBoostCreateFromMatrix_R(
    SEXP floatAndCatMatrixParam,
    SEXP targetParam,
    SEXP catFeaturesIndicesParam,
    SEXP textMatrixParam,
    SEXP textFeaturesIndicesParam,
    SEXP pairsParam,
    SEXP graphParam,
    SEXP weightParam,
    SEXP groupIdParam,
    SEXP groupWeightParam,
    SEXP subgroupIdParam,
    SEXP pairsWeightParam,
    SEXP baselineParam,
    SEXP featureNamesParam,
    SEXP classLabelsParam,
    SEXP embeddingListParam,
    SEXP embeddingFeaturesIndicesParam
);

EXPORT_FUNCTION CatBoostHashStrings_R(SEXP stringsParam);

EXPORT_FUNCTION CatBoostCalcCatFeatureHash_R(SEXP stringParam);

EXPORT_FUNCTION CatBoostPoolNumRow_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolNumCol_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostGetNumTrees_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostPoolNumTrees_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostIsOblivious_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostIsGroupwiseMetric_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostPoolSlice_R(
    SEXP poolParam,
    SEXP sizeParam,
    SEXP offsetParam
);

EXPORT_FUNCTION CatBoostPoolSliceSubset_R(
    SEXP poolParam,
    SEXP sizeParam,
    SEXP offsetParam
);

EXPORT_FUNCTION CatBoostFit_R(
    SEXP learnPoolParam,
    SEXP testPoolParam,
    SEXP fitParamsAsJsonParam,
    // P5.1 (catboost-8z4.58): model handle to continue training from, or
    // R_NilValue. Appended as the last argument so existing 3-arg call
    // sites keep compiling; the R wrapper always passes 4 args now.
    SEXP initModelParam
);

EXPORT_FUNCTION CatBoostSumModels_R(
    SEXP modelsParam,
    SEXP weightsParam,
    SEXP ctrMergePolicyParam
);

EXPORT_FUNCTION CatBoostCV_R(
    SEXP fitParamsAsJsonParam,
    SEXP poolParam,
    SEXP foldCountParam,
    SEXP typeParam,
    SEXP partitionRandomSeedParam,
    SEXP shuffleParam,
    SEXP stratifiedParam
);

EXPORT_FUNCTION CatBoostGridSearch_R(
    SEXP gridJsonParam,
    SEXP poolParam,
    SEXP fitParamsAsJsonParam,
    SEXP foldCountParam,
    SEXP partitionRandomSeedParam,
    SEXP shuffleParam,
    SEXP stratifiedParam,
    SEXP trainSizeParam,
    SEXP searchByTrainTestSplitParam,
    SEXP calcCvStatisticsParam,
    SEXP verboseParam
);

EXPORT_FUNCTION CatBoostRandomizedSearch_R(
    SEXP gridJsonParam,
    SEXP poolParam,
    SEXP fitParamsAsJsonParam,
    SEXP nIterParam,
    SEXP foldCountParam,
    SEXP partitionRandomSeedParam,
    SEXP shuffleParam,
    SEXP stratifiedParam,
    SEXP trainSizeParam,
    SEXP searchByTrainTestSplitParam,
    SEXP calcCvStatisticsParam,
    SEXP verboseParam
);

EXPORT_FUNCTION CatBoostOutputModel_R(
    SEXP modelParam,
    SEXP fileParam,
    SEXP formatParam,
    SEXP exportParametersParam,
    SEXP poolParam
);

EXPORT_FUNCTION CatBoostReadModel_R(SEXP fileParam, SEXP formatParam);

EXPORT_FUNCTION CatBoostSerializeModel_R(SEXP handleParam);

EXPORT_FUNCTION CatBoostDeserializeModel_R(SEXP rawParam);

EXPORT_FUNCTION CatBoostPredictMulti_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP verboseParam,
    SEXP typeParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostPredictMulti_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP verboseParam,
    SEXP typeParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostPrepareEval_R(
    SEXP approxParam,
    SEXP typeParam,
    SEXP lossFunctionName,
    SEXP columnCountParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostPredictVirtualEnsembles_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP verboseParam,
    SEXP typeParam,
    SEXP treeCountEndParam,
    SEXP virtualEnsemblesCountParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostShrinkModel_R(
    SEXP modelParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam
);

EXPORT_FUNCTION CatBoostDropUnusedFeaturesFromModel_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostGetModelParams_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostGetPlainParams_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostCalcRegularFeatureEffect_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP fstrTypeParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostEvaluateObjectImportances_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP trainPoolParam,
    SEXP topSizeParam,
    SEXP ostrTypeParam,
    SEXP updateMethodParam,
    SEXP threadCountParam
);

// P4.1 (catboost-8z4.50): catboost.calc_feature_statistics native glue.
EXPORT_FUNCTION CatBoostGetBinarizedStatistics_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP catFeaturesNumsParam,
    SEXP floatFeaturesNumsParam,
    SEXP predictionTypeParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostGetFeatureTypeAndInternalIndex_R(SEXP modelParam, SEXP flatFeatureIndexParam);

EXPORT_FUNCTION CatBoostCalcCatFeaturePerfectHash_R(SEXP modelParam, SEXP valueParam, SEXP featureNumParam);

EXPORT_FUNCTION CatBoostGetCatFeatureValues_R(SEXP poolParam, SEXP flatFeatureIndexParam);

EXPORT_FUNCTION CatBoostIsNullHandle_R(SEXP handleParam);


EXPORT_FUNCTION CatBoostEvalMetrics_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP metricsParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam,
    SEXP evalPeriodParam,
    SEXP threadCountParam,
    SEXP tmpDirParam,
    SEXP resultDirParam
);

EXPORT_FUNCTION CatBoostVersion_R(void);

// P3.1: Pool metadata accessors/mutators (R equivalents of Python Pool's
// get_label/get_weight/set_weight/get_baseline/set_baseline/has_label/
// get_group_id_hash/set_group_id/set_group_weight/set_subgroup_id/
// set_pairs/set_pairs_weight/num_pairs/set_timestamp).

EXPORT_FUNCTION CatBoostPoolHasLabel_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolGetLabel_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolGetWeight_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolSetWeight_R(SEXP poolParam, SEXP weightParam);

EXPORT_FUNCTION CatBoostPoolGetBaseline_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolSetBaseline_R(SEXP poolParam, SEXP baselineParam);

EXPORT_FUNCTION CatBoostPoolGetGroupIdHash_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolSetGroupId_R(SEXP poolParam, SEXP groupIdParam);

EXPORT_FUNCTION CatBoostPoolSetGroupWeight_R(SEXP poolParam, SEXP groupWeightParam);

EXPORT_FUNCTION CatBoostPoolSetSubgroupId_R(SEXP poolParam, SEXP subgroupIdParam);

EXPORT_FUNCTION CatBoostPoolSetPairs_R(SEXP poolParam, SEXP pairsParam);

EXPORT_FUNCTION CatBoostPoolSetPairsWeight_R(SEXP poolParam, SEXP pairsWeightParam);

EXPORT_FUNCTION CatBoostPoolNumPairs_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolSetTimestamp_R(SEXP poolParam, SEXP timestampParam);


// P3.2: Pool feature/shape introspection (R equivalents of Python Pool's
// get_feature_names/set_feature_names/get_features/get_cat_feature_indices/
// get_text_feature_indices/get_embedding_feature_indices). num_row/num_col/
// shape/is_empty_ reuse CatBoostPoolNumRow_R/CatBoostPoolNumCol_R above.

EXPORT_FUNCTION CatBoostPoolGetFeatureNames_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolSetFeatureNames_R(SEXP poolParam, SEXP featureNamesParam);

EXPORT_FUNCTION CatBoostPoolGetFeatures_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolGetCatFeatureIndices_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolGetTextFeatureIndices_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolGetEmbeddingFeatureIndices_R(SEXP poolParam);


// P3.3: Pool quantization support (R equivalents of Python Pool's
// quantize/is_quantized/save_quantization_borders) plus the R equivalent of
// CatBoost CLI's dataset-statistics mode.

EXPORT_FUNCTION CatBoostPoolQuantize_R(SEXP poolParam, SEXP paramsAsJsonParam);

EXPORT_FUNCTION CatBoostPoolIsQuantized_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolSaveQuantizationBorders_R(SEXP poolParam, SEXP outputFileParam);

// P3.4: Pool structural operations (R equivalents of Python Pool's
// train_eval_split/save; slice's native entry point is CatBoostPoolSlice_R
// above, pre-existing).

EXPORT_FUNCTION CatBoostPoolTrainEvalSplit_R(
    SEXP poolParam,
    SEXP hasTimeParam,
    SEXP isClassificationParam,
    SEXP evalFractionParam,
    SEXP saveEvalPoolParam
);

EXPORT_FUNCTION CatBoostPoolSave_R(SEXP poolParam, SEXP fnameParam);

// P4.7: R equivalent of the CLI's `eval-feature` mode (catboost-8z4.56).
EXPORT_FUNCTION CatBoostEvaluateFeatures_R(
    SEXP fitParamsAsJsonParam,
    SEXP poolParam,
    SEXP featuresToEvaluateParam,
    SEXP featureEvalModeParam,
    SEXP offsetParam,
    SEXP foldCountParam,
    SEXP foldSizeUnitParam,
    SEXP foldSizeParam,
    SEXP relativeFoldSizeParam,
    SEXP timeSplitQuantileParam
);

// P4.8: R equivalent of the CLI's `model-based-eval` mode (catboost-8z4.57).
EXPORT_FUNCTION CatBoostModelBasedEval_R(
    SEXP fitParamsAsJsonParam,
    SEXP learnSetPathParam,
    SEXP testSetPathParam,
    SEXP cdPathParam,
    SEXP delimiterParam,
    SEXP hasHeaderParam
);

EXPORT_FUNCTION CatBoostDatasetStatistics_R(
    SEXP poolFileParam,
    SEXP cdFileParam,
    SEXP pairsFileParam,
    SEXP delimiterParam,
    SEXP hasHeaderParam,
    SEXP threadCountParam,
    SEXP borderCountParam,
    SEXP onlyGroupStatisticsParam,
    SEXP onlyLightStatisticsParam,
    SEXP outputPathParam,
    SEXP histogramPathParam
);

// P3.6 follow-up (catboost-8z4.48): native tokenizer/dictionary bridges,
// wrapping NTextProcessing::NTokenizer::TTokenizer and
// NTextProcessing::NDictionary::TDictionary/TDictionaryBuilder/
// TBpeDictionary/TBpeDictionaryBuilder -- replacing the pure-R port in
// R/text_processing.R (catboost-8z4.43).

EXPORT_FUNCTION CatBoostTextTokenizerCreate_R(
    SEXP lowercasingParam,
    SEXP lemmatizingParam,
    SEXP numberProcessPolicyParam,
    SEXP numberTokenParam,
    SEXP separatorTypeParam,
    SEXP delimiterParam,
    SEXP splitBySetParam,
    SEXP skipEmptyParam,
    SEXP tokenTypesParam,
    SEXP subTokensPolicyParam,
    SEXP languagesParam
);

EXPORT_FUNCTION CatBoostTextTokenizerTokenize_R(SEXP tokenizerParam, SEXP stringParam);

EXPORT_FUNCTION CatBoostTextDictionaryFit_R(
    SEXP linesParam,
    SEXP tokenLevelTypeParam,
    SEXP gramOrderParam,
    SEXP skipStepParam,
    SEXP startTokenIdParam,
    SEXP endOfWordPolicyParam,
    SEXP endOfSentencePolicyParam,
    SEXP occurenceLowerBoundParam,
    SEXP maxDictionarySizeParam,
    SEXP dictionaryTypeParam,
    SEXP numBpeUnitsParam,
    SEXP skipUnknownParam
);

EXPORT_FUNCTION CatBoostTextDictionaryApply_R(SEXP dictionaryParam, SEXP linesParam, SEXP unknownTokenPolicyParam);

EXPORT_FUNCTION CatBoostTextDictionarySize_R(SEXP dictionaryParam);

EXPORT_FUNCTION CatBoostTextDictionaryGetTokens_R(SEXP dictionaryParam, SEXP tokenIdsParam);

EXPORT_FUNCTION CatBoostTextDictionaryGetTopTokens_R(SEXP dictionaryParam, SEXP topSizeParam);

EXPORT_FUNCTION CatBoostTextDictionaryUnknownTokenId_R(SEXP dictionaryParam);

EXPORT_FUNCTION CatBoostTextDictionaryEndOfSentenceTokenId_R(SEXP dictionaryParam);

EXPORT_FUNCTION CatBoostTextDictionaryMinUnusedTokenId_R(SEXP dictionaryParam);

EXPORT_FUNCTION CatBoostTextDictionarySave_R(
    SEXP dictionaryParam,
    SEXP dictionaryTypeParam,
    SEXP frequencyDictPathParam,
    SEXP bpePathParam
);

EXPORT_FUNCTION CatBoostTextDictionaryLoad_R(SEXP frequencyDictPathParam, SEXP bpePathParam);

#if defined(__cplusplus)
}
#endif
