#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/borders_io.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/quantization.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/utils.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/cross_validation.h>
#include <catboost/libs/train_lib/eval_feature.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/mvs.h>
#include <catboost/private/libs/algo/plot.h>
#include <catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.h>
#include <catboost/private/libs/documents_importance/docs_importance.h>
#include <catboost/private/libs/documents_importance/enums.h>
#include <catboost/private/libs/options/cross_validation_params.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/feature_eval_options.h>
#include <catboost/private/libs/options/split_params.h>
#include <catboost/private/libs/quantized_pool/serialization.h>
// P4.1: catboost.calc_feature_statistics -- same vendor entry point Python's
// _catboost.pyx _get_binarized_statistics/_get_feature_type_and_internal_index/
// _calc_cat_feature_perfect_hash/_get_cat_feature_values cpdef wrappers call.
#include <catboost/private/libs/quantized_pool_analysis/quantized_pool_analysis.h>
#include <catboost/private/libs/target/data_providers.h>

// P3.6 follow-up (catboost-8z4.48): native tokenizer/dictionary bridges.
#include <library/cpp/langs/langs.h>
#include <library/cpp/text_processing/dictionary/bpe_builder.h>
#include <library/cpp/text_processing/dictionary/bpe_dictionary.h>
#include <library/cpp/text_processing/dictionary/dictionary.h>
#include <library/cpp/text_processing/dictionary/dictionary_builder.h>
#include <library/cpp/text_processing/dictionary/frequency_based_dictionary.h>
#include <library/cpp/text_processing/dictionary/options.h>
#include <library/cpp/text_processing/dictionary/types.h>
#include <library/cpp/text_processing/tokenizer/options.h>
#include <library/cpp/text_processing/tokenizer/tokenizer.h>

// P1.16: compiled-in build identifier for R/libcatboostr version-skew detection.
// Same __vcs_version__.c mechanism (cmake/common.cmake's vcs_info(), fed by
// build/scripts/vcs_info.py + generate_vcs_info.py) already linked into this
// target's own vcs_info(catboostr) call in src/CMakeLists.txt.
#include <library/cpp/svnversion/svnversion.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/mem_copy.h>
#include <util/generic/singleton.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/system/info.h>

#include <algorithm>

#if defined(SIZEOF_SIZE_T)
#undef SIZEOF_SIZE_T
#endif

#include "catboostr.h"


using namespace NCB;


#define R_API_BEGIN()                                                           \
    auto loggingFunc = [](const char* str, size_t len, TCustomLoggingObject) {  \
        TString slicedStr(str, 0, len);                                         \
        Rprintf("%s", slicedStr.c_str());                                       \
    };                                                                          \
    SetCustomLoggingFunction(loggingFunc, loggingFunc);                         \
    *Singleton<TRPackageInitializer>();                                         \
    try {                                                                       \


#define R_API_END()                                                 \
    } catch (std::exception& e) {                                   \
        error(e.what());                                            \
    }                                                               \
    RestoreOriginalLogger();                                        \

typedef TDataProvider* TPoolHandle;
typedef TDataProviderPtr TPoolPtr;

typedef TFullModel* TFullModelHandle;
typedef std::unique_ptr<TFullModel> TFullModelPtr;
typedef const TFullModel* TFullModelConstPtr;


class TRPackageInitializer {
    Y_DECLARE_SINGLETON_FRIEND();
    TRPackageInitializer() {
        ConfigureMalloc();
    }
};

template <typename T>
void _Finalizer(SEXP ext) {
    if (R_ExternalPtrAddr(ext) == NULL) return;
    delete reinterpret_cast<T>(R_ExternalPtrAddr(ext)); // delete allocated memory
    R_ClearExternalPtr(ext);
}

template <typename T>
static TVector<T> GetVectorFromNullableSEXP(SEXP arg, const TStringBuf inputArgName) {
    TVector<T> result;
    if (!Rf_isNull(arg)) {
        result.yresize(length(arg));

        auto convertArg = [&](const auto* arg) {
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = static_cast<T>(arg[i]);
            }
        };
        switch (TYPEOF(arg)) {
            case INTSXP:
                convertArg(INTEGER(arg));
                break;
            case REALSXP:
                convertArg(REAL(arg));
                break;
            case LGLSXP:
                convertArg(LOGICAL(arg));
                break;
            default:
                CB_ENSURE(false, inputArgName << ": unsupported vector type: int, real or logical is required");
        }
    }
    return result;
}

static NJson::TJsonValue LoadFitParams(SEXP fitParamsAsJson) {
    TStringBuf paramsStr(CHAR(asChar(fitParamsAsJson)));
    NJson::TJsonValue result;
    NJson::ReadJsonTree(paramsStr, &result);
    return result;
}

static int UpdateThreadCount(int threadCount) {
    if (threadCount == -1) {
        threadCount = NSystemInfo::CachedNumberOfCpus();
    }
    return threadCount;
}

void SetClassLabels(SEXP classLabelsParam, TDataMetaInfo* metaInfo) {
    if (Rf_isNull(classLabelsParam)) {
        return;
    }
    const int classCount = length(classLabelsParam);
    if (Rf_isInteger(classLabelsParam)) {
        int *ptr_classLabelsParam = INTEGER(classLabelsParam);
        for (auto i : xrange(classCount)) {
            metaInfo->ClassLabels.push_back(NJson::TJsonValue(ptr_classLabelsParam[i]));
        }
    } else if (Rf_isString(classLabelsParam)) {
        for (auto i : xrange(classCount)) {
            metaInfo->ClassLabels.push_back(
                NJson::TJsonValue(CHAR(STRING_ELT(classLabelsParam, i)))
            );
        }
    } else {
        CB_ENSURE(false, "Unsupported class labels data type, only string or integer are supported");
    }
}

template <class TSrc>
void AddTarget(
    const TSrc* srcTarget,
    const size_t targetColumns,
    const size_t targetRows,
    IRawFeaturesOrderDataVisitor* visitor
) {
    for (auto targetIdx : xrange(targetColumns)) {
        TVector<float> target;
        target.yresize(targetRows);
        for (auto docIdx : xrange(targetRows)) {
            target[docIdx] = static_cast<float>(srcTarget[docIdx + targetRows * targetIdx]);
        }
        visitor->AddTarget(
            targetIdx,
            MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(target))
        );
    }
}


extern "C" {

EXPORT_FUNCTION CatBoostCreateFromFile_R(SEXP poolFileParam,
                              SEXP cdFileParam,
                              SEXP pairsFileParam,
                              SEXP graphFileParam,
                              SEXP featureNamesFileParam,
                              SEXP delimiterParam,
                              SEXP numVectorDelimiterParam,
                              SEXP hasHeaderParam,
                              SEXP threadCountParam,
                              SEXP verboseParam) {
    SEXP result = NULL;
    R_API_BEGIN();

    NCatboostOptions::TColumnarPoolFormatParams columnarPoolFormatParams;
    columnarPoolFormatParams.DsvFormat =
        TDsvFormatOptions{
            static_cast<bool>(asLogical(hasHeaderParam)),
            CHAR(asChar(delimiterParam))[0],
            CHAR(asChar(numVectorDelimiterParam))[0]
        };

    TStringBuf cdPathWithScheme(CHAR(asChar(cdFileParam)));
    if (!cdPathWithScheme.empty()) {
        columnarPoolFormatParams.CdFilePath = TPathWithScheme(cdPathWithScheme, "dsv");
    }

    TStringBuf pairsPathWithScheme(CHAR(asChar(pairsFileParam)));
    TStringBuf graphPathWithScheme(CHAR(asChar(graphFileParam)));
    TStringBuf featureNamesPathWithScheme(CHAR(asChar(featureNamesFileParam)));

    TDataProviderPtr poolPtr = ReadDataset(/*taskType*/Nothing(),
                                           TPathWithScheme(CHAR(asChar(poolFileParam)), "dsv"),
                                           !pairsPathWithScheme.empty() ?
                                               TPathWithScheme(pairsPathWithScheme, "dsv-flat") : TPathWithScheme(),
                                           !graphPathWithScheme.empty() ?
                                               TPathWithScheme(graphPathWithScheme, "dsv-flat") : TPathWithScheme(),
                                           /*groupWeightsFilePath=*/TPathWithScheme(),
                                           /*timestampsFilePath=*/TPathWithScheme(),
                                           /*baselineFilePath=*/TPathWithScheme(),
                                           !featureNamesPathWithScheme.empty() ?
                                                TPathWithScheme(featureNamesPathWithScheme, "dsv") : TPathWithScheme(),
                                           /*poolMetaInfoPath=*/TPathWithScheme(),
                                           columnarPoolFormatParams,
                                           TVector<ui32>(),
                                           EObjectsOrder::Undefined,
                                           UpdateThreadCount(asInteger(threadCountParam)),
                                           asLogical(verboseParam),
                                           /*loadSampleIds*/ false,
                                           /*forceUnitAutoPairWeights*/ false,
                                           /*classLabels=*/Nothing());
    result = PROTECT(R_MakeExternalPtr(poolPtr.Get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(poolPtr.Release());
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostCreateFromMatrix_R(SEXP floatAndCatMatrixParam,
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
                                SEXP embeddingFeaturesIndicesParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    // Embedding features arrive as a VECSXP whose elements are (objectCount x embeddingDimension)
    // numeric matrices, one per embedding feature -- an R matrix cell cannot itself hold a vector,
    // so they travel beside the flat float/cat matrix exactly like text features do.
    ui32 embeddingColumns = embeddingListParam == R_NilValue ? 0 :
                       SafeIntegerCast<ui32>(Rf_length(embeddingListParam));
    SEXP dataDim = floatAndCatMatrixParam != R_NilValue ?
                   getAttrib(floatAndCatMatrixParam, R_DimSymbol) :
                   (textMatrixParam != R_NilValue ?
                    getAttrib(textMatrixParam, R_DimSymbol) :
                    getAttrib(VECTOR_ELT(embeddingListParam, 0), R_DimSymbol));
    ui32 dataRows = SafeIntegerCast<ui32>(INTEGER(dataDim)[0]);
    ui32 floatAndCatColumns = floatAndCatMatrixParam == R_NilValue ? 0 :
                       SafeIntegerCast<ui32>(INTEGER(getAttrib(floatAndCatMatrixParam, R_DimSymbol))[1]);
    ui32 textColumns = textMatrixParam == R_NilValue ? 0 :
                       SafeIntegerCast<ui32>(INTEGER(getAttrib(textMatrixParam, R_DimSymbol))[1]);
    ui32 dataColumns = floatAndCatColumns + textColumns + embeddingColumns;
    SEXP targetDim = getAttrib(targetParam, R_DimSymbol);
    ui32 targetRows = 0;
    ui32 targetColumns = 0;
    if (targetDim != R_NilValue) {
        targetRows = SafeIntegerCast<ui32>(INTEGER(targetDim)[0]);
        targetColumns = SafeIntegerCast<ui32>(INTEGER(targetDim)[1]);
    }
    SEXP baselineDim = getAttrib(baselineParam, R_DimSymbol);
    size_t baselineRows = 0;
    size_t baselineColumns = 0;
    if (baselineParam != R_NilValue) {
        baselineRows = static_cast<size_t>(INTEGER(baselineDim)[0]);
        baselineColumns = static_cast<size_t>(INTEGER(baselineDim)[1]);
    }

    auto loaderFunc = [&] (IRawFeaturesOrderDataVisitor* visitor) {
        TDataMetaInfo metaInfo;

        TVector<TString> featureId;
        featureId.reserve(dataColumns);
        if (featureNamesParam != R_NilValue) {
            for (size_t i = 0; i < dataColumns; ++i) {
                featureId.push_back(CHAR(asChar(VECTOR_ELT(featureNamesParam, i))));
            }
        }

        metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
            dataColumns,
            ToUnsigned(GetVectorFromNullableSEXP<int>(catFeaturesIndicesParam, "cat_features_indices"_sb)),
            ToUnsigned(GetVectorFromNullableSEXP<int>(textFeaturesIndicesParam, "text_features_indices"_sb)),
            ToUnsigned(GetVectorFromNullableSEXP<int>(embeddingFeaturesIndicesParam, "embedding_features_indices"_sb)),
            featureId);

        if (!targetColumns) {
            metaInfo.TargetType = ERawTargetType::None;
        } else if (Rf_isInteger(targetParam)) {
            metaInfo.TargetType = ERawTargetType::Integer;
            SetClassLabels(classLabelsParam, &metaInfo);
        } else if (Rf_isReal(targetParam)) {
            metaInfo.TargetType = ERawTargetType::Float;
            CB_ENSURE(
                classLabelsParam == R_NilValue,
                "Specifying class labels is incompatible with real target data"
            );
        } else {
            CB_ENSURE(false, "Unsupported target data type, only real or integer are supported");
        }

        metaInfo.TargetCount = targetColumns;
        metaInfo.BaselineCount = baselineColumns;
        metaInfo.HasGroupId = groupIdParam != R_NilValue;
        metaInfo.HasGroupWeight = groupWeightParam != R_NilValue;
        metaInfo.HasSubgroupIds = subgroupIdParam != R_NilValue;
        metaInfo.HasWeights = weightParam != R_NilValue;
        metaInfo.HasTimestamp = false;
        metaInfo.HasGraph = graphParam != R_NilValue;

        visitor->Start(metaInfo, dataRows, EObjectsOrder::Undefined, {});

        if (Rf_isInteger(targetParam)) {
            AddTarget(INTEGER(targetParam), targetColumns, targetRows, visitor);
        } else if (Rf_isReal(targetParam)) {
            AddTarget(REAL(targetParam), targetColumns, targetRows, visitor);
        }
        TVector<float> weights;
        weights.yresize(metaInfo.HasWeights ? dataRows : 0);
        TVector<float> groupWeights;
        groupWeights.yresize(metaInfo.HasGroupWeight ? dataRows : 0);

        int *ptr_groupIdParam = Rf_isNull(groupIdParam)? nullptr : INTEGER(groupIdParam);
        int *ptr_subgroupIdParam = Rf_isNull(subgroupIdParam)? nullptr : INTEGER(subgroupIdParam);
        double *ptr_weightParam = Rf_isNull(weightParam)? nullptr : REAL(weightParam);
        double *ptr_groupWeightParam = Rf_isNull(groupWeightParam)? nullptr : REAL(groupWeightParam);
        for (ui32 i = 0; i < dataRows; ++i) {
            if (metaInfo.HasGroupId) {
                visitor->AddGroupId(i, static_cast<uint32_t>(ptr_groupIdParam[i]));
            }
            if (metaInfo.HasSubgroupIds) {
                visitor->AddSubgroupId(i, static_cast<uint32_t>(ptr_subgroupIdParam[i]));
            }

            if (weightParam != R_NilValue) {
                weights[i] = static_cast<float>(ptr_weightParam[i]);
            }
            if (groupWeightParam != R_NilValue) {
                groupWeights[i] = static_cast<float>(ptr_groupWeightParam[i]);
            }
        }
        if (metaInfo.HasWeights) {
            visitor->AddWeights(weights);
        }
        if (metaInfo.HasGroupWeight) {
            visitor->SetGroupWeights(std::move(groupWeights));
        }
        if (metaInfo.BaselineCount) {
            TVector<float> baseline;
            baseline.yresize(dataRows);
            double *ptr_baselineParam = Rf_isNull(baselineParam)? nullptr : REAL(baselineParam);
            for (size_t j = 0; j < baselineColumns; ++j) {
                for (ui32 i = 0; i < dataRows; ++i) {
                    baseline[i] = static_cast<float>(ptr_baselineParam[i + baselineRows * j]);
                }
                visitor->AddBaseline(j, baseline);
            }
        }

        double *ptr_floatAndCatMatrixParam = Rf_isNull(floatAndCatMatrixParam)? nullptr : REAL(floatAndCatMatrixParam);
        size_t indexTextMatrix = 0;
        size_t indexEmbeddingMatrix = 0;
        size_t indexFloatAndCatMatrix = 0;
        for (size_t j = 0; j < dataColumns; ++j){
            if (metaInfo.FeaturesLayout->GetExternalFeatureType(j) == EFeatureType::Embedding) {
                SEXP embeddingMatrix = VECTOR_ELT(embeddingListParam, indexEmbeddingMatrix);
                SEXP embeddingDim = getAttrib(embeddingMatrix, R_DimSymbol);
                CB_ENSURE(
                    SafeIntegerCast<ui32>(INTEGER(embeddingDim)[0]) == dataRows,
                    "embedding feature " << j << " has " << INTEGER(embeddingDim)[0]
                        << " rows, data has " << dataRows
                );
                const size_t embeddingSize = static_cast<size_t>(INTEGER(embeddingDim)[1]);
                const double* ptr_embeddingMatrix = REAL(embeddingMatrix);
                TVector<TMaybeOwningConstArrayHolder<float>> embeddingValues;
                embeddingValues.reserve(dataRows);
                for (ui32 i = 0; i < dataRows; ++i) {
                    TVector<float> objectEmbedding;
                    objectEmbedding.yresize(embeddingSize);
                    for (size_t k = 0; k < embeddingSize; ++k) {
                        objectEmbedding[k] = static_cast<float>(ptr_embeddingMatrix[i + dataRows * k]);
                    }
                    embeddingValues.push_back(
                        TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(objectEmbedding))
                    );
                }
                visitor->AddEmbeddingFeature(
                    j,
                    MakeTypeCastArraysHolderFromVector<float, float>(embeddingValues)
                );
                indexEmbeddingMatrix++;
            } else if (metaInfo.FeaturesLayout->GetExternalFeatureType(j) == EFeatureType::Text) {
                TVector<TString> textValues;
                textValues.yresize(dataRows);
                for (ui32 i = 0; i < dataRows; ++i) {
                    textValues[i] = CHAR(STRING_PTR_RO(textMatrixParam)[i + dataRows * indexTextMatrix]);
                }
                visitor->AddTextFeature(j, TMaybeOwningConstArrayHolder<TString>::CreateOwning(std::move(textValues)));
                indexTextMatrix++;
            } else {
                if (metaInfo.FeaturesLayout->GetExternalFeatureType(j) == EFeatureType::Categorical) {
                    TVector<ui32> catValues;
                    catValues.yresize(dataRows);
                    for (ui32 i = 0; i < dataRows; ++i) {
                        catValues[i] =
                            ConvertFloatCatFeatureToIntHash(static_cast<float>(ptr_floatAndCatMatrixParam[i + dataRows * indexFloatAndCatMatrix]));
                    }
                    visitor->AddCatFeature(j, TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(catValues)));
                } else {
                    TVector<float> floatValues;
                    floatValues.yresize(dataRows);
                    for (ui32 i = 0; i < dataRows; ++i) {
                        floatValues[i] = static_cast<float>(ptr_floatAndCatMatrixParam[i + dataRows * indexFloatAndCatMatrix]);
                    }
                    visitor->AddFloatFeature(j, MakeTypeCastArrayHolderFromVector<float, float>(floatValues));
                }
                indexFloatAndCatMatrix++;
            }
        }
        if (graphParam != R_NilValue) {
            CB_ENSURE(pairsParam == R_NilValue, "Only one of 'graph' or 'pairs' options can be set");
            pairsParam = graphParam;
        }
        if (pairsParam != R_NilValue) {
            size_t pairsCount = static_cast<size_t>(INTEGER(getAttrib(pairsParam, R_DimSymbol))[0]);
            TVector<TPair> pairs;
            pairs.reserve(pairsCount);
            double *ptr_pairsWeightParam = Rf_isNull(pairsWeightParam)? nullptr : REAL(pairsWeightParam);
            int *ptr_pairsParam = INTEGER(pairsParam);
            for (size_t i = 0; i < pairsCount; ++i) {
                float weight = 1;
                if (pairsWeightParam != R_NilValue) {
                    weight = static_cast<float>(ptr_pairsWeightParam[i]);
                }
                pairs.emplace_back(
                    static_cast<int>(ptr_pairsParam[i + pairsCount * 0]),
                    static_cast<int>(ptr_pairsParam[i + pairsCount * 1]),
                    weight
                );
            }
            if (metaInfo.HasGraph) {
                visitor->SetGraph(TRawPairsData(std::move(pairs)));
            } else {
                visitor->SetPairs(TRawPairsData(std::move(pairs)));
            }
        }
        visitor->Finish();
    };

    TDataProviderPtr poolPtr = CreateDataProvider(std::move(loaderFunc));

    result = PROTECT(R_MakeExternalPtr(poolPtr.Get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(poolPtr.Release());
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostHashStrings_R(SEXP stringsParam) {
   SEXP result = PROTECT(allocVector(REALSXP, length(stringsParam)));
   double *ptr_result = REAL(result);
   for (int i = 0; i < length(stringsParam); ++i) {
       ptr_result[i] = static_cast<double>(ConvertCatFeatureHashToFloat(CalcCatFeatureHash(TString(CHAR(STRING_ELT(stringsParam, i))))));
   }
   UNPROTECT(1);
   return result;
}

EXPORT_FUNCTION CatBoostPoolNumRow_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarInteger(static_cast<int>(pool->ObjectsGrouping->GetObjectCount()));
    R_API_END();
    return result;
}

EXPORT_FUNCTION CatBoostPoolNumCol_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarInteger(0);
    if (pool->ObjectsGrouping->GetObjectCount() != 0) {
        result = ScalarInteger(static_cast<int>(pool->MetaInfo.GetFeatureCount()));
    }
    R_API_END();
    return result;
}

EXPORT_FUNCTION CatBoostGetNumTrees_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = ScalarInteger(static_cast<int>(model->GetTreeCount()));
    R_API_END();
    return result;
}

// P1.16: returns the build tag compiled into this shared object by vcs_info()
// (GetTag() is defined in the generated __vcs_version__.c, from ARCADIA_TAG =
// `git describe --exact-match --tags HEAD` on the build tree). Empty string
// when the build tree was not checked out exactly at a tag. Used by R's
// .onLoad (R/zzz.R) to detect skew against the installed DESCRIPTION Version.
EXPORT_FUNCTION CatBoostVersion_R(void) {
    SEXP result = NULL;
    R_API_BEGIN();
    result = mkString(GetTag());
    R_API_END();
    return result;
}

// TODO(dbakshee): remove this backward compatibility gag in v0.11
EXPORT_FUNCTION CatBoostPoolNumTrees_R(SEXP modelParam) {
    return CatBoostGetNumTrees_R(modelParam);
}

EXPORT_FUNCTION CatBoostIsOblivious_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = ScalarLogical(static_cast<int>(model->IsOblivious()));
    R_API_END();
    return result;
}

EXPORT_FUNCTION CatBoostIsGroupwiseMetric_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = ScalarLogical(static_cast<int>(IsGroupwiseMetric(model->GetLossFunctionName())));
    R_API_END();
    return result;
}

EXPORT_FUNCTION CatBoostPoolSlice_R(SEXP poolParam, SEXP sizeParam, SEXP offsetParam) {
    SEXP result = NULL;
    size_t size, offset;
    R_API_BEGIN();
    size = static_cast<size_t>(asInteger(sizeParam));
    offset = static_cast<size_t>(asInteger(offsetParam));
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const TRawObjectsDataProvider* rawObjectsData
        = dynamic_cast<const TRawObjectsDataProvider*>(pool->ObjectsData.Get());
    CB_ENSURE(rawObjectsData, "Cannot Slice quantized features data");

    const auto& featuresLayout = *(rawObjectsData->GetFeaturesLayout());

    CB_ENSURE(
        featuresLayout.GetExternalFeatureCount() == featuresLayout.GetFloatFeatureCount(),
        "Dataset slicing error: non-numeric features present, slicing datasets with categorical, text or embedding features is not supported"
    );

    result = PROTECT(allocVector(VECSXP, size));
    ui32 featureCount = pool->MetaInfo.GetFeatureCount();
    auto target = pool->RawTargetData.GetTarget();
    const auto& weights = pool->RawTargetData.GetWeights();


    const size_t sliceEnd = std::min((size_t)pool->GetObjectCount(), offset + size);

    TRangesSubset<ui32>::TBlocks subsetBlocks = { TSubsetBlock<ui32>(TIndexRange<ui32>(offset, sliceEnd), 0) };

    TObjectsGroupingSubset objectsGroupingSubset = GetGroupingSubsetFromObjectsSubset(
        rawObjectsData->GetObjectsGrouping(),
        TArraySubsetIndexing<ui32>(TRangesSubset<ui32>(subsetBlocks[0].GetSize(), std::move(subsetBlocks))),
        EObjectsOrder::Ordered
    );

    TObjectsDataProviderPtr sliceObjectsData = rawObjectsData->GetSubset(
        objectsGroupingSubset,
        GetMonopolisticFreeCpuRam(),
        &NPar::LocalExecutor()
    );

    const TRawObjectsDataProvider& sliceRawObjectsData
        = dynamic_cast<const TRawObjectsDataProvider&>(*sliceObjectsData);

    TVector<double*> rows;
    const auto targetCount = pool->MetaInfo.TargetCount;

    for (size_t i = offset; i < sliceEnd; ++i) {
        ui32 featureCount = pool->MetaInfo.GetFeatureCount();
        SEXP row = PROTECT(allocVector(REALSXP, featureCount + targetCount + 1));
        REAL(row)[targetCount] = weights[i];
        rows.push_back(REAL(row));
        SET_VECTOR_ELT(result, i - offset, row);
    }

    for (auto targetIdx : xrange(targetCount)) {
        if (const ITypedSequencePtr<float>* typedSequence
                = std::get_if<ITypedSequencePtr<float>>(&((*target)[targetIdx])))
        {
            TIntrusivePtr<ITypedArraySubset<float>> subset = (*typedSequence)->GetSubset(
                &objectsGroupingSubset.GetObjectsIndexing()
            );
            subset->ForEach(
                [&rows, targetIdx] (ui32 i, float value) {
                    rows[i][targetIdx] = value;
                }
            );
        } else {
            TConstArrayRef<TString> stringTargetPart = std::get<TVector<TString>>((*target)[targetIdx]);

            for (size_t i = offset; i < sliceEnd; ++i) {
                rows[i - offset][targetIdx] = FromString<double>(stringTargetPart[i]);
            }
        }
    }


    for (auto flatFeatureIdx : xrange(featureCount)) {
        TMaybeData<const TFloatValuesHolder*> maybeFeatureData
            = sliceRawObjectsData.GetFloatFeature(flatFeatureIdx);
        if (maybeFeatureData) {
            if (const auto* arrayColumn = dynamic_cast<const TFloatArrayValuesHolder*>(*maybeFeatureData)) {
                arrayColumn->GetData()->ForEach(
                    [&] (ui32 i, float value) {
                        rows[i][flatFeatureIdx + targetCount + 1] = value;
                    }
                );
            } else {
                CB_ENSURE_INTERNAL(false, "CatBoostPoolSlice_R: Unsupported column type");
            }
        } else {
            for (auto i : xrange(sliceRawObjectsData.GetObjectCount())) {
                rows[i][flatFeatureIdx + targetCount + 1] = 0.0f;
            }
        }
    }

    R_API_END();
    UNPROTECT(size - offset + 1);
    return result;
}

// Mirrors _catboost.pyx's _take_slice() (which backs Python's Pool.slice()):
// builds a new TDataProvider from a row subset via TDataProvider::GetSubset,
// so every column of the pool -- features (numeric, categorical, text,
// embedding), all targets, weights, group ids, subgroup ids, baseline, pairs
// and the feature layout (names) -- is carried through by the core subset
// machinery. This is what catboost.pool.slice() uses; the older
// CatBoostPoolSlice_R above flattens rows into a dense numeric matrix and
// exists only for head()/tail() printing.
EXPORT_FUNCTION CatBoostPoolSliceSubset_R(SEXP poolParam, SEXP sizeParam, SEXP offsetParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const size_t objectCount = static_cast<size_t>(pool->GetObjectCount());
    const size_t offset = std::min(static_cast<size_t>(asInteger(offsetParam)), objectCount);
    const size_t sliceEnd = std::min(objectCount, offset + static_cast<size_t>(asInteger(sizeParam)));

    TRangesSubset<ui32>::TBlocks subsetBlocks
        = { TSubsetBlock<ui32>(TIndexRange<ui32>(offset, sliceEnd), 0) };
    TObjectsGroupingSubset objectsGroupingSubset = GetGroupingSubsetFromObjectsSubset(
        pool->ObjectsGrouping,
        TArraySubsetIndexing<ui32>(
            TRangesSubset<ui32>(subsetBlocks[0].GetSize(), std::move(subsetBlocks))
        ),
        EObjectsOrder::Ordered
    );

    TDataProviderPtr slicedDataProvider = pool->GetSubset(
        objectsGroupingSubset,
        GetMonopolisticFreeCpuRam(),
        &NPar::LocalExecutor()
    );

    result = PROTECT(R_MakeExternalPtr(slicedDataProvider.Get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(slicedDataProvider.Release());

    R_API_END();
    UNPROTECT(1);
    return result;
}

// P3.4: R equivalents of Python Pool's train_eval_split and save (Pool
// structural operations; slice's native entry point is
// CatBoostPoolSliceSubset_R above). train_eval_split has no existing native entry point -- Python's
// Pool.train_eval_split (catboost/python-package/catboost/core.py) calls
// _catboost.pyx's _train_eval_split, which itself calls TrainEvalSplit()
// (catboost/python-package/catboost/helpers.cpp:252). That function lives in
// the python-package tree (not a core lib) and includes Python.h, so it
// cannot be called directly from R; this reimplements its body against the
// same core NCB entry points it itself calls (catboost/libs/data/
// objects_grouping.h's Shuffle/TrainTestSplit/StratifiedTrainTestSplit,
// TDataProvider::GetSubset), which are all already reachable from this file.
// Only the float-typed-target branch of the stratified path is implemented,
// matching this fork's existing precedent at CatBoostPoolGetLabel_R: R's own
// Pool construction paths never produce ERawTargetType::String labels.
EXPORT_FUNCTION CatBoostPoolTrainEvalSplit_R(
    SEXP poolParam,
    SEXP hasTimeParam,
    SEXP isClassificationParam,
    SEXP evalFractionParam,
    SEXP saveEvalPoolParam
) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    bool hasTime = static_cast<bool>(asLogical(hasTimeParam));
    bool isClassification = static_cast<bool>(asLogical(isClassificationParam));
    double evalFraction = asReal(evalFractionParam);
    bool saveEvalPool = static_cast<bool>(asLogical(saveEvalPoolParam));

    CB_ENSURE(evalFraction > 0.0 && evalFraction < 1.0, "eval_fraction must be in (0,1) range");

    TTrainTestSplitParams splitParams;
    splitParams.Shuffle = !hasTime;
    splitParams.Stratified = isClassification;
    splitParams.TrainPart = 1.0 - evalFraction;

    bool shuffle = splitParams.Shuffle
        && pool->ObjectsData->GetOrder() != EObjectsOrder::RandomShuffled;

    TObjectsGroupingSubset postShuffleGroupingSubset;
    if (shuffle) {
        TRestorableFastRng64 rand(splitParams.PartitionRandSeed);
        postShuffleGroupingSubset = NCB::Shuffle(pool->ObjectsGrouping, 1, &rand);
    } else {
        postShuffleGroupingSubset = GetSubset(
            pool->ObjectsGrouping,
            TArraySubsetIndexing<ui32>(TFullSubset<ui32>(pool->ObjectsGrouping->GetGroupCount())),
            EObjectsOrder::Ordered
        );
    }
    TObjectsGroupingPtr postShuffleGrouping = postShuffleGroupingSubset.GetSubsetGrouping();

    TArraySubsetIndexing<ui32> postShuffleTrainIndices;
    TArraySubsetIndexing<ui32> postShuffleTestIndices;

    if (splitParams.Stratified) {
        auto maybeOneDimensionalTarget = pool->RawTargetData.GetOneDimensionalTarget();
        CB_ENSURE(maybeOneDimensionalTarget, "Cannot do stratified split without one-dimensional target data");
        const ITypedSequencePtr<float>* typedSequence
            = std::get_if<ITypedSequencePtr<float>>(&(**maybeOneDimensionalTarget));
        CB_ENSURE(
            typedSequence,
            "CatBoostPoolTrainEvalSplit_R: string labels are not supported for stratified split"
        );
        TVector<float> classesVec(pool->GetObjectCount());
        size_t classesVecIdx = 0;
        (*typedSequence)->ForEach([&classesVec, &classesVecIdx](float value) { classesVec[classesVecIdx++] = value; });
        // Mirrors python-package/catboost/helpers.cpp's TrainEvalSplit: the
        // target array must be re-ordered by the same post-shuffle indexing
        // used to build postShuffleGrouping before stratifying, otherwise
        // StratifiedTrainTestSplit pairs shuffled row positions with
        // pre-shuffle class labels.
        if (shuffle) {
            classesVec = NCB::GetSubset<float>(
                TConstArrayRef<float>(classesVec),
                postShuffleGroupingSubset.GetObjectsIndexing(),
                &NPar::LocalExecutor()
            );
        }
        StratifiedTrainTestSplit(
            *postShuffleGrouping,
            TConstArrayRef<float>(classesVec),
            splitParams.TrainPart,
            &postShuffleTrainIndices,
            &postShuffleTestIndices
        );
    } else {
        TrainTestSplit(*postShuffleGrouping, splitParams.TrainPart, &postShuffleTrainIndices, &postShuffleTestIndices);
    }

    auto getSubset = [&](const TArraySubsetIndexing<ui32>& postShuffleIndexing) {
        return pool->GetSubset(
            GetSubset(
                pool->ObjectsGrouping,
                Compose(postShuffleGroupingSubset.GetGroupsIndexing(), postShuffleIndexing),
                shuffle ? EObjectsOrder::RandomShuffled : EObjectsOrder::Ordered
            ),
            GetMonopolisticFreeCpuRam(),
            &NPar::LocalExecutor()
        );
    };

    TDataProviderPtr trainDataProvider = getSubset(postShuffleTrainIndices);
    TDataProviderPtr evalDataProvider;
    if (saveEvalPool) {
        evalDataProvider = getSubset(postShuffleTestIndices);
    }

    result = PROTECT(allocVector(VECSXP, 2));

    SEXP trainHandle = PROTECT(R_MakeExternalPtr(trainDataProvider.Get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(trainHandle, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(trainDataProvider.Release());
    SET_VECTOR_ELT(result, 0, trainHandle);
    UNPROTECT(1);

    if (saveEvalPool) {
        SEXP evalHandle = PROTECT(R_MakeExternalPtr(evalDataProvider.Get(), R_NilValue, R_NilValue));
        R_RegisterCFinalizerEx(evalHandle, _Finalizer<TPoolHandle>, TRUE);
        Y_UNUSED(evalDataProvider.Release());
        SET_VECTOR_ELT(result, 1, evalHandle);
        UNPROTECT(1);
    } else {
        SET_VECTOR_ELT(result, 1, R_NilValue);
    }

    R_API_END();
    UNPROTECT(1);
    return result;
}

// Mirrors _catboost.pyx _save()/Python's Pool.save(): saves a quantized Pool
// to CatBoost's own binary quantized-pool format via the same core entry
// point Python calls (catboost/private/libs/quantized_pool/serialization.h's
// SaveQuantizedPool(TDataProviderPtr, fname)) -- not the CD/TSV format
// catboost.save_pool (R/catboost.R) writes, which is a different, older,
// human-readable format read back by catboost.load_pool's
// column_description path. Requires the pool to already be quantized, same
// precondition and error message ("Pool is not quantized") as
// BuildSrcDataFromDataProvider (serialization.cpp) enforces for Python.
EXPORT_FUNCTION CatBoostPoolSave_R(SEXP poolParam, SEXP fnameParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    CB_ENSURE(
        dynamic_cast<const TQuantizedObjectsDataProvider*>(pool->ObjectsData.Get()),
        "Pool is not quantized"
    );

    // Manual refcount bump, same pattern as CatBoostPoolQuantize_R above: the
    // R external pointer finalizer deletes this object directly, bypassing
    // intrusive refcounting, so a local TDataProviderPtr must not be allowed
    // to drop the count to 0 and free it out from under the R handle.
    pool->Ref();
    TDataProviderPtr dataProvider(pool);
    SaveQuantizedPool(dataProvider, TString(CHAR(asChar(fnameParam))));
    R_API_END();
    return R_NilValue;
}

// P3.1: R equivalents of Python Pool's metadata accessor/mutator methods
// (catboost/python-package/catboost/_catboost.pyx get_label/get_weight/
// set_weight/get_baseline/set_baseline/has_label/get_group_id_hash/
// set_group_id/set_group_weight/set_subgroup_id/set_pairs/set_pairs_weight/
// num_pairs/set_timestamp). R wrappers are catboost.pool.<snake_case_name>
// in R/catboost.R.

EXPORT_FUNCTION CatBoostPoolHasLabel_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarLogical(pool->MetaInfo.TargetCount > 0);
    R_API_END();
    return result;
}

// Mirrors _catboost.pyx get_label(): reads RawTargetData via GetNumericTarget
// into a pre-sized buffer per target dimension. String targets are out of
// scope for this fork's Pool construction paths (CreateFromMatrix/FromFile
// only ever set ERawTargetType::Integer/Float/None), so unlike the Python
// method this does not need an ERawTargetType::String branch.
EXPORT_FUNCTION CatBoostPoolGetLabel_R(SEXP poolParam) {
    SEXP result = NULL;
    SEXP resultDim = NULL;
    size_t protectedCount = 0;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 targetCount = pool->MetaInfo.TargetCount;
    ui32 objectCount = pool->GetObjectCount();
    CB_ENSURE(
        pool->RawTargetData.GetTargetType() != ERawTargetType::String,
        "CatBoostPoolGetLabel_R: string labels are not supported"
    );
    result = PROTECT(allocVector(REALSXP, (size_t)objectCount * targetCount));
    ++protectedCount;
    if (targetCount > 0) {
        TVector<TVector<float>> targetBuffers(targetCount, TVector<float>(objectCount));
        TVector<TArrayRef<float>> targetRefs(targetCount);
        for (auto targetIdx : xrange(targetCount)) {
            targetRefs[targetIdx] = TArrayRef<float>(targetBuffers[targetIdx]);
        }
        pool->RawTargetData.GetNumericTarget(TArrayRef<TArrayRef<float>>(targetRefs));
        double* ptr_result = REAL(result);
        for (auto targetIdx : xrange(targetCount)) {
            for (auto objectIdx : xrange(objectCount)) {
                ptr_result[objectIdx + (size_t)objectCount * targetIdx] = targetBuffers[targetIdx][objectIdx];
            }
        }
        if (targetCount > 1) {
            resultDim = PROTECT(allocVector(INTSXP, 2));
            ++protectedCount;
            INTEGER(resultDim)[0] = objectCount;
            INTEGER(resultDim)[1] = targetCount;
            setAttrib(result, R_DimSymbol, resultDim);
        }
    }
    R_API_END();
    UNPROTECT(protectedCount);
    return result;
}

// Mirrors _catboost.pyx get_weight(): TWeights::IsTrivial() means "weight
// column was never set", in which case every object's effective weight is 1.
EXPORT_FUNCTION CatBoostPoolGetWeight_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const TWeights<float>& weights = pool->RawTargetData.GetWeights();
    result = PROTECT(allocVector(REALSXP, weights.GetSize()));
    double* ptr_result = REAL(result);
    if (weights.IsTrivial()) {
        std::fill(ptr_result, ptr_result + weights.GetSize(), 1.0);
    } else {
        TConstArrayRef<float> data = weights.GetNonTrivialData();
        for (auto i : xrange(data.size())) {
            ptr_result[i] = data[i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

// Mirrors _catboost.pyx _set_weight()/TDataProviderTemplate::SetWeights().
EXPORT_FUNCTION CatBoostPoolSetWeight_R(SEXP poolParam, SEXP weightParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TVector<float> weights = GetVectorFromNullableSEXP<float>(weightParam, "weight"_sb);
    pool->SetWeights(TConstArrayRef<float>(weights));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx get_baseline(): [approxIdx][objectIdx] -> R matrix
// (objectCount x baselineCount), empty (objectCount x 0) matrix if unset.
EXPORT_FUNCTION CatBoostPoolGetBaseline_R(SEXP poolParam) {
    SEXP result = NULL;
    SEXP resultDim = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 objectCount = pool->GetObjectCount();
    TMaybeData<TBaselineArrayRef> maybeBaseline = pool->RawTargetData.GetBaseline();
    size_t baselineCount = maybeBaseline ? maybeBaseline->size() : 0;
    result = PROTECT(allocVector(REALSXP, (size_t)objectCount * baselineCount));
    double* ptr_result = REAL(result);
    if (maybeBaseline) {
        TBaselineArrayRef baseline = *maybeBaseline;
        for (auto baselineIdx : xrange(baselineCount)) {
            for (auto objectIdx : xrange(objectCount)) {
                ptr_result[objectIdx + (size_t)objectCount * baselineIdx] = baseline[baselineIdx][objectIdx];
            }
        }
    }
    resultDim = PROTECT(allocVector(INTSXP, 2));
    INTEGER(resultDim)[0] = objectCount;
    INTEGER(resultDim)[1] = baselineCount;
    setAttrib(result, R_DimSymbol, resultDim);
    R_API_END();
    UNPROTECT(2);
    return result;
}

// Mirrors _catboost.pyx _set_baseline(): baselineParam is an
// (objectCount x approxDim) R matrix, same [objectIdx, approxIdx] layout
// CatBoostCreateFromMatrix_R already accepts for its baselineParam.
EXPORT_FUNCTION CatBoostPoolSetBaseline_R(SEXP poolParam, SEXP baselineParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 objectCount = pool->GetObjectCount();
    SEXP baselineDim = getAttrib(baselineParam, R_DimSymbol);
    CB_ENSURE(baselineDim != R_NilValue, "baseline must be a matrix");
    ui32 baselineRows = SafeIntegerCast<ui32>(INTEGER(baselineDim)[0]);
    size_t approxDimension = SafeIntegerCast<size_t>(INTEGER(baselineDim)[1]);
    CB_ENSURE(baselineRows == objectCount, "baseline row count must equal pool row count");
    double* ptr_baseline = REAL(baselineParam);

    TVector<TVector<float>> baselineMatrix(approxDimension, TVector<float>(objectCount));
    TVector<TConstArrayRef<float>> baselineMatrixView(approxDimension);
    for (auto approxIdx : xrange(approxDimension)) {
        for (auto objectIdx : xrange(objectCount)) {
            baselineMatrix[approxIdx][objectIdx] =
                static_cast<float>(ptr_baseline[objectIdx + (size_t)objectCount * approxIdx]);
        }
        baselineMatrixView[approxIdx] = baselineMatrix[approxIdx];
    }
    pool->SetBaseline(TBaselineArrayRef(baselineMatrixView.data(), baselineMatrixView.size()));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx get_group_id_hash(): returns the ui64 TGroupId
// stored per object, one decimal string per object (R has no native 64-bit
// integer type; REALSXP's 53-bit mantissa would silently truncate hash
// values above 2^53, so this returns character to stay exact), or R NULL
// if the pool has no group ids.
EXPORT_FUNCTION CatBoostPoolGetGroupIdHash_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TMaybeData<TConstArrayRef<TGroupId>> maybeGroupIds = pool->ObjectsData->GetGroupIds();
    if (maybeGroupIds) {
        TConstArrayRef<TGroupId> groupIds = *maybeGroupIds;
        result = PROTECT(allocVector(STRSXP, groupIds.size()));
        for (auto i : xrange(groupIds.size())) {
            SET_STRING_ELT(result, i, mkChar(ToString<TGroupId>(groupIds[i]).c_str()));
        }
        UNPROTECT(1);
    } else {
        result = R_NilValue;
    }
    R_API_END();
    return result;
}

// Mirrors _catboost.pyx _set_group_id()/CalcGroupIdFor(): groupIdParam is a
// STRSXP of pre-canonicalized tokens (R/catboost.R does the int/string ->
// decimal-string canonicalization Python's get_id_object_bytes_string_
// representation() does), each hashed to a TGroupId exactly as
// CalcGroupIdFor(TStringBuf) does for the Python Pool.
EXPORT_FUNCTION CatBoostPoolSetGroupId_R(SEXP poolParam, SEXP groupIdParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 objectCount = pool->GetObjectCount();
    CB_ENSURE(
        static_cast<ui32>(length(groupIdParam)) == objectCount,
        "group_id length must equal pool row count"
    );
    TVector<TGroupId> groupIds;
    groupIds.reserve(objectCount);
    for (auto i : xrange(objectCount)) {
        groupIds.push_back(CalcGroupIdFor(TStringBuf(CHAR(STRING_ELT(groupIdParam, i)))));
    }
    pool->SetGroupIds(TConstArrayRef<TGroupId>(groupIds));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx _set_group_weight().
EXPORT_FUNCTION CatBoostPoolSetGroupWeight_R(SEXP poolParam, SEXP groupWeightParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TVector<float> groupWeights = GetVectorFromNullableSEXP<float>(groupWeightParam, "group_weight"_sb);
    pool->SetGroupWeights(TConstArrayRef<float>(groupWeights));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx _set_subgroup_id()/CalcSubgroupIdFor(): same
// pre-canonicalized-token convention as CatBoostPoolSetGroupId_R above.
EXPORT_FUNCTION CatBoostPoolSetSubgroupId_R(SEXP poolParam, SEXP subgroupIdParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 objectCount = pool->GetObjectCount();
    CB_ENSURE(
        static_cast<ui32>(length(subgroupIdParam)) == objectCount,
        "subgroup_id length must equal pool row count"
    );
    TVector<TSubgroupId> subgroupIds;
    subgroupIds.reserve(objectCount);
    for (auto i : xrange(objectCount)) {
        subgroupIds.push_back(CalcSubgroupIdFor(TStringBuf(CHAR(STRING_ELT(subgroupIdParam, i)))));
    }
    pool->SetSubgroupIds(TConstArrayRef<TSubgroupId>(subgroupIds));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx set_pairs()/_make_pairs_vector(): pairsParam is an
// (N x 2) or (N x 3) integer/double matrix of (winner_id, loser_id[,
// weight]), 0-indexed object ids -- same convention CatBoostCreateFromMatrix_R
// already uses for its pairsParam. Missing weight column defaults to 1.0.
EXPORT_FUNCTION CatBoostPoolSetPairs_R(SEXP poolParam, SEXP pairsParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    SEXP pairsDim = getAttrib(pairsParam, R_DimSymbol);
    CB_ENSURE(pairsDim != R_NilValue, "pairs must be a matrix");
    size_t pairsCount = SafeIntegerCast<size_t>(INTEGER(pairsDim)[0]);
    int pairsColumns = INTEGER(pairsDim)[1];
    CB_ENSURE(pairsColumns == 2 || pairsColumns == 3, "pairs must have 2 or 3 columns");
    double* ptr_pairs = REAL(pairsParam);

    TVector<TPair> pairs;
    pairs.reserve(pairsCount);
    for (auto i : xrange(pairsCount)) {
        float weight = pairsColumns == 3 ? static_cast<float>(ptr_pairs[i + pairsCount * 2]) : 1.0f;
        pairs.emplace_back(
            static_cast<ui32>(ptr_pairs[i + pairsCount * 0]),
            static_cast<ui32>(ptr_pairs[i + pairsCount * 1]),
            weight
        );
    }
    pool->SetPairs(TConstArrayRef<TPair>(pairs));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx _set_pairs_weight()/GetUngroupedPairs(): keeps the
// existing (winner, loser) ids, replaces only the per-pair weight.
EXPORT_FUNCTION CatBoostPoolSetPairsWeight_R(SEXP poolParam, SEXP pairsWeightParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const TMaybeData<TRawPairsData>& maybePairsData = pool->RawTargetData.GetPairs();
    CB_ENSURE(maybePairsData, "Pool has no pairs, call catboost.pool.set_pairs first");
    const TFlatPairsInfo* oldPairs = std::get_if<TFlatPairsInfo>(&*maybePairsData);
    CB_ENSURE(oldPairs, "Cannot set pairs weight: pairs data is grouped");
    CB_ENSURE(
        static_cast<size_t>(length(pairsWeightParam)) == oldPairs->size(),
        "pairs_weight length must equal num_pairs()"
    );
    double* ptr_pairsWeight = REAL(pairsWeightParam);
    TVector<TPair> newPairs;
    newPairs.reserve(oldPairs->size());
    for (auto i : xrange(oldPairs->size())) {
        newPairs.emplace_back((*oldPairs)[i].WinnerId, (*oldPairs)[i].LoserId, static_cast<float>(ptr_pairsWeight[i]));
    }
    pool->SetPairs(TConstArrayRef<TPair>(newPairs));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx num_pairs()/GetNumPairs().
EXPORT_FUNCTION CatBoostPoolNumPairs_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    size_t numPairs = 0;
    const TMaybeData<TRawPairsData>& maybePairsData = pool->RawTargetData.GetPairs();
    if (maybePairsData) {
        std::visit([&](const auto& pairs) { numPairs = pairs.size(); }, *maybePairsData);
    }
    result = ScalarInteger(static_cast<int>(numPairs));
    R_API_END();
    return result;
}

// Mirrors _catboost.pyx _set_timestamp().
EXPORT_FUNCTION CatBoostPoolSetTimestamp_R(SEXP poolParam, SEXP timestampParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 objectCount = pool->GetObjectCount();
    CB_ENSURE(
        static_cast<ui32>(length(timestampParam)) == objectCount,
        "timestamp length must equal pool row count"
    );
    double* ptr_timestamp = REAL(timestampParam);
    TVector<ui64> timestamps;
    timestamps.reserve(objectCount);
    for (auto i : xrange(objectCount)) {
        timestamps.push_back(static_cast<ui64>(ptr_timestamp[i]));
    }
    pool->SetTimestamps(TConstArrayRef<ui64>(timestamps));
    R_API_END();
    return R_NilValue;
}

// P3.2: R equivalents of Python Pool's feature/shape introspection methods
// (catboost/python-package/catboost/_catboost.pyx get_feature_names/
// _set_feature_names/get_features/get_cat_feature_indices/
// get_text_feature_indices/get_embedding_feature_indices). R wrappers are
// catboost.pool.<snake_case_name> in R/catboost.R. num_row/num_col/shape/
// is_empty_ reuse the pre-existing CatBoostPoolNumRow_R/CatBoostPoolNumCol_R
// (already wrapped by dim.catboost.Pool) instead of adding new C entry
// points for them.

// Mirrors _catboost.pyx get_feature_names(): FeaturesLayout's external
// feature ids, in external (flat) feature order.
EXPORT_FUNCTION CatBoostPoolGetFeatureNames_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TVector<TString> featureIds = pool->MetaInfo.FeaturesLayout->GetExternalFeatureIds();
    result = PROTECT(allocVector(STRSXP, featureIds.size()));
    for (auto i : xrange(featureIds.size())) {
        SET_STRING_ELT(result, i, mkChar(featureIds[i].c_str()));
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

// Mirrors _catboost.pyx _set_feature_names()/TFeaturesLayout::SetExternalFeatureIds().
EXPORT_FUNCTION CatBoostPoolSetFeatureNames_R(SEXP poolParam, SEXP featureNamesParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    ui32 featureCount = pool->MetaInfo.GetFeatureCount();
    CB_ENSURE(
        static_cast<ui32>(length(featureNamesParam)) == featureCount,
        "feature_names length (" << length(featureNamesParam) << ") must equal pool column count (" << featureCount << ")"
    );
    TVector<TString> featureNames;
    featureNames.reserve(featureCount);
    for (auto i : xrange(featureCount)) {
        featureNames.push_back(TString(CHAR(STRING_ELT(featureNamesParam, i))));
    }
    pool->MetaInfo.FeaturesLayout->SetExternalFeatureIds(TConstArrayRef<TString>(featureNames));
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx get_cat_feature_indices(): external (flat) feature
// indices of the categorical features, ascending.
EXPORT_FUNCTION CatBoostPoolGetCatFeatureIndices_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TConstArrayRef<ui32> indices = pool->MetaInfo.FeaturesLayout->GetCatFeatureInternalIdxToExternalIdx();
    result = PROTECT(allocVector(INTSXP, indices.size()));
    for (auto i : xrange(indices.size())) {
        INTEGER(result)[i] = static_cast<int>(indices[i]);
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

// Mirrors _catboost.pyx get_text_feature_indices().
EXPORT_FUNCTION CatBoostPoolGetTextFeatureIndices_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TConstArrayRef<ui32> indices = pool->MetaInfo.FeaturesLayout->GetTextFeatureInternalIdxToExternalIdx();
    result = PROTECT(allocVector(INTSXP, indices.size()));
    for (auto i : xrange(indices.size())) {
        INTEGER(result)[i] = static_cast<int>(indices[i]);
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

// Mirrors _catboost.pyx get_embedding_feature_indices(). Non-empty for Pools
// built with catboost.load_pool(embedding_features = ...) /
// catboost.from_matrix(embedding_features_data = ...), which forward the
// embedding matrices and their flat indices to CatBoostCreateFromMatrix_R.
EXPORT_FUNCTION CatBoostPoolGetEmbeddingFeatureIndices_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TConstArrayRef<ui32> indices = pool->MetaInfo.FeaturesLayout->GetEmbeddingFeatureInternalIdxToExternalIdx();
    result = PROTECT(allocVector(INTSXP, indices.size()));
    for (auto i : xrange(indices.size())) {
        INTEGER(result)[i] = static_cast<int>(indices[i]);
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

// Mirrors _catboost.pyx get_features(): (object_count x feature_count)
// matrix of raw float feature values, 0 for absent (cat/text) columns.
// Errors like the Python oracle if any feature is non-numeric.
EXPORT_FUNCTION CatBoostPoolGetFeatures_R(SEXP poolParam) {
    SEXP result = NULL;
    SEXP resultDim = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const TRawObjectsDataProvider* rawObjectsData
        = dynamic_cast<const TRawObjectsDataProvider*>(pool->ObjectsData.Get());
    CB_ENSURE(rawObjectsData, "CatBoostPoolGetFeatures_R: Pool does not have raw features data, only quantized");
    const auto& featuresLayout = *(rawObjectsData->GetFeaturesLayout());
    CB_ENSURE(
        featuresLayout.GetExternalFeatureCount() == featuresLayout.GetFloatFeatureCount(),
        "CatBoostPoolGetFeatures_R: Pool has non-numeric features, get_features supports only numeric features"
    );
    ui32 objectCount = pool->GetObjectCount();
    ui32 featureCount = pool->MetaInfo.GetFeatureCount();
    result = PROTECT(allocVector(REALSXP, (size_t)objectCount * featureCount));
    double* ptr_result = REAL(result);
    std::fill(ptr_result, ptr_result + (size_t)objectCount * featureCount, 0.0);
    for (auto flatFeatureIdx : xrange(featureCount)) {
        TMaybeData<const TFloatValuesHolder*> maybeFeatureData
            = rawObjectsData->GetFloatFeature(flatFeatureIdx);
        if (maybeFeatureData) {
            if (const auto* arrayColumn = dynamic_cast<const TFloatArrayValuesHolder*>(*maybeFeatureData)) {
                arrayColumn->GetData()->ForEach(
                    [&] (ui32 i, float value) {
                        ptr_result[i + (size_t)objectCount * flatFeatureIdx] = value;
                    }
                );
            } else {
                CB_ENSURE_INTERNAL(false, "CatBoostPoolGetFeatures_R: Unsupported column type");
            }
        }
    }
    resultDim = PROTECT(allocVector(INTSXP, 2));
    INTEGER(resultDim)[0] = objectCount;
    INTEGER(resultDim)[1] = featureCount;
    setAttrib(result, R_DimSymbol, resultDim);
    R_API_END();
    UNPROTECT(2);
    return result;
}

// P3.3: R equivalents of Python Pool's quantize/is_quantized/
// save_quantization_borders (_catboost.pyx _quantize/is_quantized/
// save_quantization_borders) plus the R equivalent of CatBoost CLI's
// dataset-statistics mode.

// Mirrors _catboost.pyx _quantize(): builds quantized objects data from the
// pool's raw data via the same core entry point Python's Pool.quantize()
// calls (catboost/libs/data/quantization.h ConstructQuantizedPoolFromRawPool),
// then swaps it into the pool in place, same as Python does.
EXPORT_FUNCTION CatBoostPoolQuantize_R(SEXP poolParam, SEXP paramsAsJsonParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    CB_ENSURE(
        !dynamic_cast<const TQuantizedObjectsDataProvider*>(pool->ObjectsData.Get()),
        "Pool is already quantized"
    );
    NJson::TJsonValue plainJsonParams = LoadFitParams(paramsAsJsonParam);

    // Manual refcount bump, mirroring CatBoostFit_R's pools.Learn->Ref(): the
    // R external pointer finalizer (_Finalizer<TPoolHandle>) deletes this
    // object directly, bypassing intrusive refcounting, so a local
    // TDataProviderPtr must not be allowed to drop the count to 0 and free it
    // out from under the R handle when this function returns.
    pool->Ref();
    TDataProviderPtr srcData(pool);

    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;
    TQuantizedObjectsDataProviderPtr quantizedObjects =
        ConstructQuantizedPoolFromRawPool(srcData, plainJsonParams, quantizedFeaturesInfo);

    pool->ObjectsData = quantizedObjects;
    pool->MetaInfo.FeaturesLayout = quantizedObjects->GetFeaturesLayout();
    R_API_END();
    return R_NilValue;
}

// Mirrors _catboost.pyx is_quantized().
EXPORT_FUNCTION CatBoostPoolIsQuantized_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    bool isQuantized = dynamic_cast<const TQuantizedObjectsDataProvider*>(pool->ObjectsData.Get()) != nullptr;
    result = ScalarLogical(isQuantized);
    R_API_END();
    return result;
}

// Mirrors _catboost.pyx save_quantization_borders().
EXPORT_FUNCTION CatBoostPoolSaveQuantizationBorders_R(SEXP poolParam, SEXP outputFileParam) {
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const TQuantizedObjectsDataProvider* quantizedObjectsData =
        dynamic_cast<const TQuantizedObjectsDataProvider*>(pool->ObjectsData.Get());
    CB_ENSURE(quantizedObjectsData, "Pool is not quantized");
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo = quantizedObjectsData->GetQuantizedFeaturesInfo();
    SaveBordersAndNanModesToFileInMatrixnetFormat(TString(CHAR(asChar(outputFileParam))), *quantizedFeaturesInfo);
    R_API_END();
    return R_NilValue;
}

// R equivalent of CatBoost CLI's `dataset-statistics` mode. Calls the same
// core library entry point the CLI mode itself calls
// (catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.h
// NCB::CalculateDatasetStatisticsSingleHost, invoked from
// catboost/app/mode_dataset_statistics.cpp) directly -- no shell-out to the
// CLI binary, no argv parsing: the params struct is filled in-process, same
// as CatBoostCreateFromFile_R's TPathWithScheme wiring above. Writes its two
// JSON result files (statistics + histograms) to outputPathParam/
// histogramPathParam; the R wrapper reads them back with jsonlite.
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
) {
    R_API_BEGIN();
    TCalculateStatisticsParams params;

    params.DatasetReadingParams.PoolPath = TPathWithScheme(CHAR(asChar(poolFileParam)), "dsv");

    TStringBuf cdPathWithScheme(CHAR(asChar(cdFileParam)));
    if (!cdPathWithScheme.empty()) {
        params.DatasetReadingParams.ColumnarPoolFormatParams.CdFilePath = TPathWithScheme(cdPathWithScheme, "dsv");
    }
    params.DatasetReadingParams.ColumnarPoolFormatParams.DsvFormat =
        TDsvFormatOptions{static_cast<bool>(asLogical(hasHeaderParam)), CHAR(asChar(delimiterParam))[0]};

    TStringBuf pairsPathWithScheme(CHAR(asChar(pairsFileParam)));
    if (!pairsPathWithScheme.empty()) {
        params.DatasetReadingParams.PairsFilePath = TPathWithScheme(pairsPathWithScheme, "dsv-flat");
    }

    params.ThreadCount = asInteger(threadCountParam);
    params.BorderCount = static_cast<size_t>(asInteger(borderCountParam));
    params.OnlyGroupStatistics = static_cast<bool>(asLogical(onlyGroupStatisticsParam));
    params.OnlyLightStatistics = static_cast<bool>(asLogical(onlyLightStatisticsParam));
    params.OutputPath = TString(CHAR(asChar(outputPathParam)));
    params.HistogramPath = TString(CHAR(asChar(histogramPathParam)));

    NCB::CalculateDatasetStatisticsSingleHost(params);
    R_API_END();
    return R_NilValue;
}

EXPORT_FUNCTION CatBoostFit_R(SEXP learnPoolParam, SEXP testPoolParam, SEXP fitParamsAsJsonParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle learnPool = static_cast<TPoolHandle>(R_ExternalPtrAddr(learnPoolParam));
    TDataProviders pools;
    pools.Learn = learnPool;
    pools.Learn->Ref();

    auto fitParams = LoadFitParams(fitParamsAsJsonParam);
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    if (testPoolParam != R_NilValue) {
        TEvalResult evalResult;
        TPoolHandle testPool = static_cast<TPoolHandle>(R_ExternalPtrAddr(testPoolParam));
        pools.Test.emplace_back(testPool);
        pools.Test.back()->Ref();
        TrainModel(
            fitParams,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            pools,
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            modelPtr.get(),
            {&evalResult}
        );
    }
    else {
        TrainModel(
            fitParams,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            pools,
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            modelPtr.get(),
            {}
        );
    }
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostSumModels_R(SEXP modelsParam,
                         SEXP weightsParam,
                         SEXP ctrMergePolicyParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    const auto weights = GetVectorFromNullableSEXP<double>(weightsParam, "weights"_sb);
    ECtrTableMergePolicy mergePolicy;
    CB_ENSURE(TryFromString<ECtrTableMergePolicy>(CHAR(asChar(ctrMergePolicyParam)), mergePolicy),
        "Unknown value of ctr_table_merge_policy: " << CHAR(asChar(ctrMergePolicyParam)));

    TVector<TFullModelConstPtr> models;
    for (int idx = 0; idx < length(modelsParam); ++idx) {
        TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(VECTOR_ELT(modelsParam, idx)));
        models.push_back(model);
    }
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    SumModels(models, weights, /*modelParamsPrefixes*/{}, mergePolicy).Swap(*modelPtr);
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostCV_R(SEXP fitParamsAsJsonParam,
                  SEXP poolParam,
                  SEXP foldCountParam,
                  SEXP typeParam,
                  SEXP partitionRandomSeedParam,
                  SEXP shuffleParam,
                  SEXP stratifiedParam) {

    SEXP result = NULL;
    size_t metricCount;
    size_t columnCount;

    R_API_BEGIN();
    TPoolPtr pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    pool->Ref();
    auto fitParams = LoadFitParams(fitParamsAsJsonParam);

    TCrossValidationParams cvParams;
    cvParams.FoldCount = asInteger(foldCountParam);
    cvParams.PartitionRandSeed = asInteger(partitionRandomSeedParam);
    cvParams.Shuffle = asLogical(shuffleParam);
    cvParams.Stratified = asLogical(stratifiedParam);

    CB_ENSURE(TryFromString<ECrossValidation>(CHAR(asChar(typeParam)), cvParams.Type),
              "unsupported type of cross_validation: 'Classical', 'Inverted' or 'TimeSeries' was expected");

    TVector<TCVResult> cvResults;

    CrossValidate(
        fitParams,
        TQuantizedFeaturesInfoPtr(nullptr),
        Nothing(),
        Nothing(),
        pool,
        cvParams,
        &cvResults);

    metricCount = cvResults.size();
    TVector<size_t> offsets(metricCount);

    columnCount = 0;
    size_t currentColumnCount = 0;
    for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
        offsets[metricIdx] = columnCount;
        if (cvResults[metricIdx].AverageTrain.size() == 0) {
            currentColumnCount = 2;
        } else {
            currentColumnCount = 4;
        }
        columnCount += currentColumnCount;
    }

    result = PROTECT(allocVector(VECSXP, columnCount));
    SEXP columnNames = PROTECT(allocVector(STRSXP, columnCount));

    for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
        TString metricName = cvResults[metricIdx].Metric;
        size_t numberOfIterations = cvResults[metricIdx].Iterations.size();

        SEXP row_test_mean = PROTECT(allocVector(REALSXP, numberOfIterations));
        SEXP row_test_std = PROTECT(allocVector(REALSXP, numberOfIterations));
        SEXP row_train_mean = NULL;
        SEXP row_train_std = NULL;
        const bool haveTrainResult = (cvResults[metricIdx].AverageTrain.size() != 0);
        if (haveTrainResult) {
            row_train_mean = PROTECT(allocVector(REALSXP, numberOfIterations));
            row_train_std = PROTECT(allocVector(REALSXP, numberOfIterations));
        }

        for (size_t i = 0; i < numberOfIterations; ++i) {
            REAL(row_test_mean)[i] = cvResults[metricIdx].AverageTest[i];
            REAL(row_test_std)[i] = cvResults[metricIdx].StdDevTest[i];
            if (haveTrainResult) {
                REAL(row_train_mean)[i] = cvResults[metricIdx].AverageTrain[i];
                REAL(row_train_std)[i] = cvResults[metricIdx].StdDevTrain[i];
            }
        }

        const size_t offset = offsets[metricIdx];

        SET_VECTOR_ELT(result, offset + 0, row_test_mean);
        SET_VECTOR_ELT(result, offset + 1, row_test_std);

        SET_STRING_ELT(columnNames, offset + 0, mkChar(("test-" + metricName + "-mean").c_str()));
        SET_STRING_ELT(columnNames, offset + 1, mkChar(("test-" + metricName + "-std").c_str()));
        if (haveTrainResult) {
            SET_VECTOR_ELT(result, offset + 2, row_train_mean);
            SET_VECTOR_ELT(result, offset + 3, row_train_std);

            SET_STRING_ELT(columnNames, offset + 2, mkChar(("train-" + metricName + "-mean").c_str()));
            SET_STRING_ELT(columnNames, offset + 3, mkChar(("train-" + metricName + "-std").c_str()));
        }
    }

    setAttrib(result, R_NamesSymbol, columnNames);

    R_API_END();
    UNPROTECT(columnCount + 2);
    return result;
}

// P4.7 (catboost-8z4.56): R equivalent of the CLI's `eval-feature` mode.
// Calls the same core entry point the CLI mode calls
// (EvaluateFeatures, catboost/libs/train_lib/eval_feature.h -- see
// vendor/catboost/catboost/app/mode_eval_feature.cpp), rather than shelling
// out to the CLI binary. catboost-libs-train_lib, which already builds
// eval_feature.cpp, is linked by src/CMakeLists.txt for CatBoostCV_R's sake,
// so no new link dependency is introduced.
//
// The returned list mirrors the columns of the CLI's
// --feature-eval-output-file TSV (ToString(TFeatureEvaluationSummary),
// eval_feature.cpp:58) so the two are directly comparable, but carries full
// double precision instead of the TSV's ~10 significant digits.
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
) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolPtr pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    pool->Ref();
    auto fitParams = LoadFitParams(fitParamsAsJsonParam);

    // EvaluateFeatures is the one core entry point that does NOT resolve feature
    // names / string indices in `ignored_features` itself: TrainModel
    // (train_model.cpp:1628) and CrossValidate (cross_validation.cpp:359,569)
    // both call this converter internally, whereas for eval-feature the CLI mode
    // does it externally before the call (mode_eval_feature.cpp:50). R always
    // serialises ignored_features as an array of strings
    // (prepare_train_export_parameters' I(as.character(...))), so without this
    // any ignored_features value would die inside the option parser with
    // `Can't parse parameter "ignored_features"` -- where catboost.train and
    // catboost.cv accept the very same value.
    ConvertIgnoredFeaturesFromStringToIndices(pool->MetaInfo, &fitParams);

    TVector<ui32> ignoredFeatures;
    if (fitParams.Has("ignored_features")) {
        TJsonFieldHelper<TVector<ui32>>::Read(fitParams["ignored_features"], &ignoredFeatures);
    }

    NCatboostOptions::TFeatureEvalOptions featureEvalOptions;

    TVector<TVector<ui32>> featureSets;
    for (R_xlen_t setIdx = 0; setIdx < xlength(featuresToEvaluateParam); ++setIdx) {
        SEXP featureSetParam = VECTOR_ELT(featuresToEvaluateParam, setIdx);
        TVector<ui32> featureSet;
        for (R_xlen_t i = 0; i < xlength(featureSetParam); ++i) {
            const int featureIdx = INTEGER(featureSetParam)[i];
            CB_ENSURE(featureIdx >= 0, "Tested feature index must be non-negative, got " << featureIdx);
            featureSet.push_back(static_cast<ui32>(featureIdx));
        }
        featureSets.push_back(std::move(featureSet));
    }
    featureEvalOptions.FeaturesToEvaluate = featureSets;

    NCB::EFeatureEvalMode featureEvalMode;
    CB_ENSURE(TryFromString<NCB::EFeatureEvalMode>(CHAR(asChar(featureEvalModeParam)), featureEvalMode),
              "unsupported feature evaluation mode: 'OneVsNone', 'OneVsOthers', 'OneVsAll' or "
              "'OthersVsAll' was expected");
    featureEvalOptions.FeatureEvalMode = featureEvalMode;

    ESamplingUnit foldSizeUnit;
    CB_ENSURE(TryFromString<ESamplingUnit>(CHAR(asChar(foldSizeUnitParam)), foldSizeUnit),
              "unsupported fold size unit: 'Object' or 'Group' was expected");
    featureEvalOptions.FoldSizeUnit = foldSizeUnit;

    featureEvalOptions.Offset = static_cast<ui32>(asInteger(offsetParam));
    featureEvalOptions.FoldCount = static_cast<ui32>(asInteger(foldCountParam));
    featureEvalOptions.FoldSize = static_cast<ui32>(asInteger(foldSizeParam));
    featureEvalOptions.RelativeFoldSize = static_cast<float>(asReal(relativeFoldSizeParam));
    featureEvalOptions.TimeSplitQuantile = asReal(timeSplitQuantileParam);

    // Same guards mode_eval_feature.cpp applies before calling EvaluateFeatures.
    const ui32 featureCount = pool->MetaInfo.GetFeatureCount();
    for (const auto& featureSet : featureSets) {
        for (ui32 feature : featureSet) {
            CB_ENSURE(feature < featureCount,
                      "Tested feature " << feature << " is not present; dataset contains only "
                      << featureCount << " features");
            CB_ENSURE(Count(ignoredFeatures, feature) == 0,
                      "Tested feature " << feature << " should not be ignored");
        }
        CB_ENSURE(Count(featureSets, featureSet) == 1, "All tested feature sets must be different");
    }

    // Defaults, matching the CLI when --cv is not given: the fold layout then
    // comes from featureEvalOptions, not from cross-validation parameters
    // (eval_feature.cpp:1181).
    TCvDataPartitionParams cvParams;

    const auto summary = EvaluateFeatures(
        fitParams,
        featureEvalOptions,
        /*objectiveDescriptor*/ Nothing(),
        /*evalMetricDescriptor*/ Nothing(),
        cvParams,
        pool);

    const size_t setCount = summary.GetFeatureSetCount();
    const size_t metricCount = summary.MetricNames.size();

    SEXP pValue = PROTECT(allocVector(REALSXP, setCount));
    SEXP bestIterations = PROTECT(allocVector(VECSXP, setCount));
    SEXP metricNames = PROTECT(allocVector(STRSXP, metricCount));
    SEXP metricDelta = PROTECT(allocMatrix(REALSXP, static_cast<int>(setCount), static_cast<int>(metricCount)));
    SEXP evaluatedSets = PROTECT(allocVector(VECSXP, setCount));

    for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
        SET_STRING_ELT(metricNames, metricIdx, mkChar(summary.MetricNames[metricIdx].c_str()));
    }
    for (size_t setIdx = 0; setIdx < setCount; ++setIdx) {
        REAL(pValue)[setIdx] = summary.WxTest[setIdx];

        const auto& foldIterations = summary.BestBaselineIterations[setIdx];
        SEXP iterations = PROTECT(allocVector(INTSXP, foldIterations.size()));
        for (size_t foldIdx = 0; foldIdx < foldIterations.size(); ++foldIdx) {
            INTEGER(iterations)[foldIdx] = static_cast<int>(foldIterations[foldIdx]);
        }
        SET_VECTOR_ELT(bestIterations, setIdx, iterations);
        UNPROTECT(1);

        for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
            REAL(metricDelta)[setIdx + metricIdx * setCount] = summary.AverageMetricDelta[setIdx][metricIdx];
        }

        // FeatureSets is empty in OneVsNone mode with no --features-to-evaluate;
        // the CLI leaves the "feature set" column empty in that case.
        SEXP features = PROTECT(allocVector(INTSXP, summary.FeatureSets.empty() ? 0 : summary.FeatureSets[setIdx].size()));
        if (!summary.FeatureSets.empty()) {
            for (size_t i = 0; i < summary.FeatureSets[setIdx].size(); ++i) {
                INTEGER(features)[i] = static_cast<int>(summary.FeatureSets[setIdx][i]);
            }
        }
        SET_VECTOR_ELT(evaluatedSets, setIdx, features);
        UNPROTECT(1);
    }

    result = PROTECT(allocVector(VECSXP, 5));
    SEXP resultNames = PROTECT(allocVector(STRSXP, 5));
    const char* const names[5] = {"p_value", "best_iterations", "metric_names", "metric_delta", "feature_sets"};
    SEXP values[5] = {pValue, bestIterations, metricNames, metricDelta, evaluatedSets};
    for (int i = 0; i < 5; ++i) {
        SET_VECTOR_ELT(result, i, values[i]);
        SET_STRING_ELT(resultNames, i, mkChar(names[i]));
    }
    setAttrib(result, R_NamesSymbol, resultNames);

    R_API_END();
    UNPROTECT(7);
    return result;
}

EXPORT_FUNCTION CatBoostOutputModel_R(SEXP modelParam, SEXP fileParam,
                           SEXP formatParam, SEXP exportParametersParam, SEXP poolParam) {
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    THashMap<ui32, TString> catFeaturesHashToString;
    TVector<TString> featureId;

    if (poolParam != R_NilValue) {
        TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
        catFeaturesHashToString = MergeCatFeaturesHashToString(*pool->ObjectsData.Get());
        featureId = pool->MetaInfo.FeaturesLayout.Get()->GetExternalFeatureIds();
    }

    EModelType modelType;
    CB_ENSURE(TryFromString<EModelType>(CHAR(asChar(formatParam)), modelType),
              "unsupported model type: 'cbm', 'coreml', 'cpp', 'python', 'json', 'onnx' or 'pmml' was expected");

    ExportModel(*model,
                CHAR(asChar(fileParam)),
                modelType,
                CHAR(asChar(exportParametersParam)),
                false,
                &featureId,
                &catFeaturesHashToString
                );
    R_API_END();
    return ScalarLogical(1);
}

EXPORT_FUNCTION CatBoostReadModel_R(SEXP fileParam, SEXP formatParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    EModelType modelType;
    CB_ENSURE(TryFromString<EModelType>(CHAR(asChar(formatParam)), modelType),
              "unsupported model type: 'CatboostBinary', 'AppleCoreML','Cpp','Python','Json','Onnx' or 'Pmml'  was expected");
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    ReadModel(CHAR(asChar(fileParam)), modelType).Swap(*modelPtr);
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostSerializeModel_R(SEXP handleParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle modelHandle = static_cast<TFullModelHandle>(R_ExternalPtrAddr(handleParam));
    const TString raw = SerializeModel(*modelHandle);
    result = PROTECT(allocVector(RAWSXP, raw.size()));
    MemCopy(RAW(result), (const unsigned char*)(raw.data()), raw.size());
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostDeserializeModel_R(SEXP rawParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    DeserializeModel(TMemoryInput(RAW(rawParam), length(rawParam))).Swap(*modelPtr);
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostPredictMulti_R(SEXP modelParam, SEXP poolParam, SEXP verboseParam,
                            SEXP typeParam, SEXP treeCountStartParam, SEXP treeCountEndParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(asChar(typeParam)), predictionType) &&
              !IsUncertaintyPredictionType(predictionType) && predictionType != EPredictionType::InternalRawFormulaVal,
              "Unsupported prediction type: 'Probability', 'LogProbability', 'Class', 'RawFormulaVal', 'Exponent' or 'RMSEWithUncertainty' was expected");
    TVector<TVector<double>> prediction = ApplyModelMulti(*model,
                                                          *pool,
                                                          asLogical(verboseParam),
                                                          predictionType,
                                                          asInteger(treeCountStartParam),
                                                          asInteger(treeCountEndParam),
                                                          UpdateThreadCount(asInteger(threadCountParam)));
    size_t predictionSize = prediction.size() * pool->ObjectsGrouping->GetObjectCount();
    result = PROTECT(allocVector(REALSXP, predictionSize));
    for (size_t i = 0, k = 0; i < pool->ObjectsGrouping->GetObjectCount(); ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            REAL(result)[k++] = prediction[j][i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostPrepareEval_R(SEXP approxParam, SEXP typeParam, SEXP lossFunctionName, SEXP columnCountParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    SEXP dataDim = getAttrib(approxParam, R_DimSymbol);
    size_t dataRows = static_cast<size_t>(INTEGER(dataDim)[0]) / asInteger(columnCountParam);
    TVector<TVector<double>> prediction(asInteger(columnCountParam), TVector<double>(dataRows));
    double *ptr_approxParam = Rf_isNull(approxParam)? nullptr : REAL(approxParam);
    for (size_t i = 0, k = 0; i < dataRows; ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            prediction[j][i] = static_cast<double>(ptr_approxParam[k++]);
        }
    }

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(UpdateThreadCount(asInteger(threadCountParam)) - 1);
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(asChar(typeParam)), predictionType),
              "unsupported prediction type: 'Probability', 'Class' or 'RawFormulaVal' was expected");
    prediction = PrepareEval(predictionType, /* virtualEnsemblesCount*/ 1, CHAR(asChar(lossFunctionName)), prediction, &executor);

    size_t predictionSize = prediction.size() * dataRows;
    result = PROTECT(allocVector(REALSXP, predictionSize));
    double *ptr_result = REAL(result);
    for (size_t i = 0, k = 0; i < dataRows; ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            ptr_result[k++] = prediction[j][i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostPredictVirtualEnsembles_R(SEXP modelParam, SEXP poolParam, SEXP verboseParam,
                            SEXP typeParam, SEXP treeCountEndParam, SEXP virtualEnsemblesCountParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(asChar(typeParam)), predictionType) && IsUncertaintyPredictionType(predictionType),
              "Unsupported virtual ensembles prediction type: 'VirtEnsembles' or 'TotalUncertainty' was expected");
    TVector<TVector<double>> prediction = ApplyUncertaintyPredictions(*model,
                                                                      *pool,
                                                                      asLogical(verboseParam),
                                                                      predictionType,
                                                                      asInteger(treeCountEndParam),
                                                                      asInteger(virtualEnsemblesCountParam),
                                                                      UpdateThreadCount(asInteger(threadCountParam)));
    size_t predictionSize = prediction.size() * pool->ObjectsGrouping->GetObjectCount();
    result = PROTECT(allocVector(REALSXP, predictionSize));
    for (size_t i = 0, k = 0; i < pool->ObjectsGrouping->GetObjectCount(); ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            REAL(result)[k++] = prediction[j][i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostShrinkModel_R(SEXP modelParam, SEXP treeCountStartParam, SEXP treeCountEndParam) {
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    model->Truncate(asInteger(treeCountStartParam), asInteger(treeCountEndParam));
    R_API_END();
    return ScalarLogical(1);
}

EXPORT_FUNCTION CatBoostDropUnusedFeaturesFromModel_R(SEXP modelParam) {
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    model->ModelTrees.GetMutable()->DropUnusedFeatures();
    R_API_END();
    return ScalarLogical(1);
}

EXPORT_FUNCTION CatBoostGetModelParams_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = PROTECT(mkString(model->ModelInfo.at("params").c_str()));
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostGetPlainParams_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = PROTECT(mkString(ToString(GetPlainJsonWithAllOptions(*model)).c_str()));
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostCalcRegularFeatureEffect_R(SEXP modelParam, SEXP poolParam, SEXP fstrTypeParam, SEXP threadCountParam) {
    SEXP result = NULL;
    SEXP resultDim = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TDataProviderPtr pool = Rf_isNull(poolParam) ? nullptr :
                            static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    if (pool) {
        pool->Ref();
    }
    EFstrType fstrType = FromString<EFstrType>(CHAR(asChar(fstrTypeParam)));
    const int threadCount = UpdateThreadCount(asInteger(threadCountParam));
    const bool multiClass = model->GetDimensionsCount() > 1;
    const bool verbose = false;
    // TODO(akhropov): make prettified mode as in python-package
    if (fstrType == EFstrType::ShapInteractionValues) {
        // ShapInteractionValues[featureIdx1][featureIdx2][dim][documentIdx], reordered below to
        // match python-package's (doc[, dim], feature1, feature2) axis order (parity: catboost-8z4.53).
        TVector<TVector<TVector<TVector<double>>>> fstr = CalcShapFeatureInteractionMulti(
            fstrType,
            *model,
            pool,
            /*pairOfFeatures*/ Nothing(),
            threadCount,
            EPreCalcShapValues::Auto,
            /*logPeriod*/ 0,
            ECalcTypeShapValues::Regular
        );
        size_t featuresCount = fstr.size();
        size_t approxDimension = featuresCount > 0 ? fstr[0][0].size() : 0;
        size_t docCount = approxDimension > 0 ? fstr[0][0][0].size() : 0;
        if (multiClass) {
            result = PROTECT(allocVector(REALSXP, docCount * approxDimension * featuresCount * featuresCount));
            double *ptr_result = REAL(result);
            for (size_t f2 = 0; f2 < featuresCount; ++f2) {
                for (size_t f1 = 0; f1 < featuresCount; ++f1) {
                    for (size_t dim = 0; dim < approxDimension; ++dim) {
                        for (size_t doc = 0; doc < docCount; ++doc) {
                            ptr_result[doc + docCount * (dim + approxDimension * (f1 + featuresCount * f2))] = fstr[f1][f2][dim][doc];
                        }
                    }
                }
            }
            PROTECT(resultDim = allocVector(INTSXP, 4));
            INTEGER(resultDim)[0] = docCount;
            INTEGER(resultDim)[1] = approxDimension;
            INTEGER(resultDim)[2] = featuresCount;
            INTEGER(resultDim)[3] = featuresCount;
        } else {
            result = PROTECT(allocVector(REALSXP, docCount * featuresCount * featuresCount));
            double *ptr_result = REAL(result);
            for (size_t f2 = 0; f2 < featuresCount; ++f2) {
                for (size_t f1 = 0; f1 < featuresCount; ++f1) {
                    for (size_t doc = 0; doc < docCount; ++doc) {
                        ptr_result[doc + docCount * (f1 + featuresCount * f2)] = fstr[f1][f2][0][doc];
                    }
                }
            }
            PROTECT(resultDim = allocVector(INTSXP, 3));
            INTEGER(resultDim)[0] = docCount;
            INTEGER(resultDim)[1] = featuresCount;
            INTEGER(resultDim)[2] = featuresCount;
        }
        setAttrib(result, R_DimSymbol, resultDim);
    } else if (fstrType == EFstrType::ShapValues && multiClass) {
        TVector<TVector<TVector<double>>> fstr = GetFeatureImportancesMulti(fstrType,
                                                                            *model,
                                                                            pool,
                                                                            /*referenceDataset*/ nullptr,
                                                                            threadCount,
                                                                            EPreCalcShapValues::Auto,
                                                                            verbose);
        size_t numDocs = fstr.size();
        size_t numClasses = numDocs > 0 ? fstr[0].size() : 0;
        size_t numValues = numClasses > 0 ? fstr[0][0].size() : 0;
        size_t resultSize = numDocs * numClasses * numValues;
        result = PROTECT(allocVector(REALSXP, resultSize));
        double *ptr_result = REAL(result);
        size_t r = 0;
        for (size_t k = 0; k < numValues; ++k) {
            for (size_t j = 0; j < numClasses; ++j) {
               for (size_t i = 0; i < numDocs; ++i) {
                    ptr_result[r++] = fstr[i][j][k];
                }
            }
        }
        PROTECT(resultDim = allocVector(INTSXP, 3));
        INTEGER(resultDim)[0] = numDocs;
        INTEGER(resultDim)[1] = numClasses;
        INTEGER(resultDim)[2] = numValues;
        setAttrib(result, R_DimSymbol, resultDim);
    } else {
        TVector<TVector<double>> fstr = GetFeatureImportances(fstrType,
                                                              *model,
                                                              pool,
                                                              /*referenceDataset*/ nullptr,
                                                              threadCount,
                                                              EPreCalcShapValues::Auto,
                                                              verbose);
        size_t numRows = fstr.size();
        size_t numCols = numRows > 0 ? fstr[0].size() : 0;
        size_t resultSize = numRows * numCols;
        result = PROTECT(allocVector(REALSXP, resultSize));
        double *ptr_result = REAL(result);
        size_t r = 0;
        for (size_t j = 0; j < numCols; ++j) {
            for (size_t i = 0; i < numRows; ++i) {
                ptr_result[r++] = fstr[i][j];
            }
        }
        PROTECT(resultDim = allocVector(INTSXP, 2));
        INTEGER(resultDim)[0] = numRows;
        INTEGER(resultDim)[1] = numCols;
        setAttrib(result, R_DimSymbol, resultDim);
    }
    R_API_END();
    UNPROTECT(2);
    return result;
}

EXPORT_FUNCTION CatBoostEvaluateObjectImportances_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP trainPoolParam,
    SEXP topSizeParam,
    SEXP ostrTypeParam,
    SEXP updateMethodParam,
    SEXP threadCountParam
) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TPoolHandle trainPool = static_cast<TPoolHandle>(R_ExternalPtrAddr(trainPoolParam));
    TString ostrType = CHAR(asChar(ostrTypeParam));
    TString updateMethod = CHAR(asChar(updateMethodParam));
    const bool verbose = false;
    TDStrResult dstrResult = GetDocumentImportances(
        *model,
        *trainPool,
        *pool,
        ostrType,
        asInteger(topSizeParam),
        updateMethod,
        /*importanceValuesSignStr=*/ToString(EImportanceValuesSign::All),
        UpdateThreadCount(asInteger(threadCountParam)),
        verbose
    );
    size_t resultSize = 0;
    if (!dstrResult.Indices.empty()) {
        resultSize += dstrResult.Indices.size() * dstrResult.Indices[0].size();
    }
    if (!dstrResult.Scores.empty()) {
        resultSize += dstrResult.Scores.size() * dstrResult.Scores[0].size();
    }
    result = PROTECT(allocVector(REALSXP, resultSize));
    double *ptr_result = REAL(result);
    size_t k = 0;
    for (size_t i = 0; i < dstrResult.Indices.size(); ++i) {
        for (size_t j = 0; j < dstrResult.Indices[0].size(); ++j) {
            ptr_result[k++] = dstrResult.Indices[i][j];
        }
    }
    for (size_t i = 0; i < dstrResult.Scores.size(); ++i) {
        for (size_t j = 0; j < dstrResult.Scores[0].size(); ++j) {
            ptr_result[k++] = dstrResult.Scores[i][j];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

// P4.1 (catboost-8z4.50): catboost.calc_feature_statistics native glue.
// Thin wrappers around catboost/private/libs/quantized_pool_analysis --
// the SAME vendor functions Python's _catboost.pyx _get_binarized_statistics/
// _get_feature_type_and_internal_index/_calc_cat_feature_perfect_hash/
// _get_cat_feature_values cpdef methods call. All orchestration logic
// (feature resolution, prediction_type defaulting, cat-value-to-hash
// ordering) lives in R (R/catboost.R catboost.calc_feature_statistics),
// mirroring catboost.core.CatBoost.calc_feature_statistics -- these 4
// entry points expose only the primitives that logic needs.
EXPORT_FUNCTION CatBoostGetBinarizedStatistics_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP catFeaturesNumsParam,
    SEXP floatFeaturesNumsParam,
    SEXP predictionTypeParam,
    SEXP threadCountParam
) {
    SEXP result = NULL;
    size_t protectedCount = 0;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));

    TVector<size_t> catFeaturesNums;
    for (int i = 0; i < length(catFeaturesNumsParam); ++i) {
        catFeaturesNums.push_back(static_cast<size_t>(INTEGER(catFeaturesNumsParam)[i]));
    }
    TVector<size_t> floatFeaturesNums;
    for (int i = 0; i < length(floatFeaturesNumsParam); ++i) {
        floatFeaturesNums.push_back(static_cast<size_t>(INTEGER(floatFeaturesNumsParam)[i]));
    }

    EPredictionType predictionType;
    CB_ENSURE(
        TryFromString<EPredictionType>(CHAR(asChar(predictionTypeParam)), predictionType),
        "CatBoostGetBinarizedStatistics_R: unknown prediction type " << CHAR(asChar(predictionTypeParam))
    );
    const int threadCount = UpdateThreadCount(asInteger(threadCountParam));

    TVector<TBinarizedFeatureStatistics> statistics = GetBinarizedStatistics(
        *model, *pool, catFeaturesNums, floatFeaturesNums, predictionType, threadCount
    );

    static const char* const kFieldNames[] = {
        "borders", "binarized_feature", "mean_target", "mean_weighted_target",
        "mean_prediction", "objects_per_bin", "predictions_on_varying_feature"
    };
    const size_t kNumFields = 7;

    result = PROTECT(allocVector(VECSXP, statistics.size()));
    ++protectedCount;

    for (size_t s = 0; s < statistics.size(); ++s) {
        const TBinarizedFeatureStatistics& stat = statistics[s];
        SEXP statList = PROTECT(allocVector(VECSXP, kNumFields));
        ++protectedCount;
        SEXP statNames = PROTECT(allocVector(STRSXP, kNumFields));
        ++protectedCount;

        SEXP borders = PROTECT(allocVector(REALSXP, stat.Borders.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.Borders.size(); ++i) {
            REAL(borders)[i] = stat.Borders[i];
        }
        SET_VECTOR_ELT(statList, 0, borders);

        SEXP binarizedFeature = PROTECT(allocVector(INTSXP, stat.BinarizedFeature.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.BinarizedFeature.size(); ++i) {
            INTEGER(binarizedFeature)[i] = stat.BinarizedFeature[i];
        }
        SET_VECTOR_ELT(statList, 1, binarizedFeature);

        SEXP meanTarget = PROTECT(allocVector(REALSXP, stat.MeanTarget.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.MeanTarget.size(); ++i) {
            REAL(meanTarget)[i] = stat.MeanTarget[i];
        }
        SET_VECTOR_ELT(statList, 2, meanTarget);

        SEXP meanWeightedTarget = PROTECT(allocVector(REALSXP, stat.MeanWeightedTarget.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.MeanWeightedTarget.size(); ++i) {
            REAL(meanWeightedTarget)[i] = stat.MeanWeightedTarget[i];
        }
        SET_VECTOR_ELT(statList, 3, meanWeightedTarget);

        SEXP meanPrediction = PROTECT(allocVector(REALSXP, stat.MeanPrediction.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.MeanPrediction.size(); ++i) {
            REAL(meanPrediction)[i] = stat.MeanPrediction[i];
        }
        SET_VECTOR_ELT(statList, 4, meanPrediction);

        SEXP objectsPerBin = PROTECT(allocVector(INTSXP, stat.ObjectsPerBin.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.ObjectsPerBin.size(); ++i) {
            INTEGER(objectsPerBin)[i] = static_cast<int>(stat.ObjectsPerBin[i]);
        }
        SET_VECTOR_ELT(statList, 5, objectsPerBin);

        SEXP predictionsOnVaryingFeature = PROTECT(allocVector(REALSXP, stat.PredictionsOnVaryingFeature.size()));
        ++protectedCount;
        for (size_t i = 0; i < stat.PredictionsOnVaryingFeature.size(); ++i) {
            REAL(predictionsOnVaryingFeature)[i] = stat.PredictionsOnVaryingFeature[i];
        }
        SET_VECTOR_ELT(statList, 6, predictionsOnVaryingFeature);

        for (size_t i = 0; i < kNumFields; ++i) {
            SET_STRING_ELT(statNames, i, mkChar(kFieldNames[i]));
        }
        setAttrib(statList, R_NamesSymbol, statNames);
        SET_VECTOR_ELT(result, s, statList);
    }

    R_API_END();
    UNPROTECT(protectedCount);
    return result;
}

EXPORT_FUNCTION CatBoostGetFeatureTypeAndInternalIndex_R(SEXP modelParam, SEXP flatFeatureIndexParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TFeatureTypeAndInternalIndex typeAndIndex = GetFeatureTypeAndInternalIndex(*model, asInteger(flatFeatureIndexParam));
    const char* typeStr = "unknown";
    if (typeAndIndex.Type == EFeatureType::Float) {
        typeStr = "float";
    } else if (typeAndIndex.Type == EFeatureType::Categorical) {
        typeStr = "categorical";
    }
    result = PROTECT(allocVector(VECSXP, 2));
    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_VECTOR_ELT(result, 0, mkString(typeStr));
    SET_VECTOR_ELT(result, 1, ScalarInteger(typeAndIndex.Index));
    SET_STRING_ELT(names, 0, mkChar("type"));
    SET_STRING_ELT(names, 1, mkChar("index"));
    setAttrib(result, R_NamesSymbol, names);
    R_API_END();
    UNPROTECT(2);
    return result;
}

EXPORT_FUNCTION CatBoostCalcCatFeaturePerfectHash_R(SEXP modelParam, SEXP valueParam, SEXP featureNumParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TString value = CHAR(asChar(valueParam));
    ui32 hash = GetCatFeaturePerfectHash(*model, value, static_cast<size_t>(asInteger(featureNumParam)));
    result = PROTECT(ScalarReal(static_cast<double>(hash)));
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostGetCatFeatureValues_R(SEXP poolParam, SEXP flatFeatureIndexParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TVector<TString> values = GetCatFeatureValues(*pool, static_cast<size_t>(asInteger(flatFeatureIndexParam)));
    result = PROTECT(allocVector(STRSXP, values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        SET_STRING_ELT(result, i, mkChar(values[i].c_str()));
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

EXPORT_FUNCTION CatBoostIsNullHandle_R(SEXP handleParam) {
    return ScalarLogical(!R_ExternalPtrAddr(handleParam));
}

EXPORT_FUNCTION CatBoostEvalMetrics_R(
        SEXP modelParam,
        SEXP poolParam,
        SEXP metricsParam,
        SEXP treeCountStartParam,
        SEXP treeCountEndParam,
        SEXP evalPeriodParam,
        SEXP threadCountParam,
        SEXP tmpDirParam,
        SEXP resultDirParam) {

    SEXP result = NULL;
    size_t protectedCount = 0;

    R_API_BEGIN()
    auto treeCountStart = asInteger(treeCountStartParam);
    auto treeCountEnd = asInteger(treeCountEndParam);
    auto evalPeriod = asInteger(evalPeriodParam);
    CB_ENSURE(treeCountStart >= 0, "Tree start index should be greater or equal than zero");
    CB_ENSURE(treeCountStart < treeCountEnd, "Tree start index should be less than the tree end index");
    CB_ENSURE(evalPeriod <= (treeCountEnd - treeCountStart), "Eval period should be less or equal than the number of trees");
    CB_ENSURE(evalPeriod > 0, "Eval period should be more than zero");

    size_t metricsParamLen = length(metricsParam);
    size_t treeCount = treeCountEnd - treeCountStart;
    size_t numberOfIterations = treeCount / evalPeriod;
    if (treeCountStart + (numberOfIterations - 1) * evalPeriod != static_cast<size_t>(treeCountEnd) - 1) {
        ++numberOfIterations;
    }

    result = PROTECT(allocVector(VECSXP, metricsParamLen));
    ++protectedCount;
    SEXP metricNames = PROTECT(allocVector(STRSXP, metricsParamLen));
    ++protectedCount;

    for (size_t metricIdx = 0; metricIdx < metricsParamLen; ++metricIdx) {
        SEXP metricScore = PROTECT(allocVector(REALSXP, numberOfIterations));
        ++protectedCount;
        SET_VECTOR_ELT(result, metricIdx, metricScore);
    }

    TFullModelHandle model = static_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = static_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));

    TVector<TString> metricDescriptions;
    TVector<NCatboostOptions::TLossDescription> metricLossDescriptions;
    metricDescriptions.reserve(metricsParamLen);
    metricLossDescriptions.reserve(metricsParamLen);
    for (size_t i = 0; i < metricsParamLen; ++i) {
        TString metricDescription = CHAR(asChar(VECTOR_ELT(metricsParam, i)));
        metricDescriptions.push_back(metricDescription);
        metricLossDescriptions.emplace_back(NCatboostOptions::ParseLossDescription(metricDescription));
    }
    auto metrics = CreateMetrics(metricLossDescriptions, model->GetDimensionsCount());

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(UpdateThreadCount(asInteger(threadCountParam)) - 1);

    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        *model,
        treeCountStart,
        treeCountEnd,
        evalPeriod,
        /*processedIterationsStep=*/50,
        CHAR(asChar(tmpDirParam)),
        metrics,
        &executor
    );

    TRestorableFastRng64 rand(0);
    auto processedDataProvider = CreateModelCompatibleProcessedDataProvider(
        *pool,
        metricLossDescriptions,
        *model,
        GetMonopolisticFreeCpuRam(),
        &rand,
        &executor
    );

    if (plotCalcer.HasAdditiveMetric()) {
        plotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
    }
    if (plotCalcer.HasNonAdditiveMetric()) {
        while (!plotCalcer.AreAllIterationsProcessed()) {
            plotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
            plotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
    }

    TVector<TVector<double>> metricsScore = plotCalcer.GetMetricsScore();
    plotCalcer.SaveResult(CHAR(asChar(resultDirParam)), /*metricsFile=*/"", /*saveMetrics*/ false, /*saveStats=*/true).ClearTempFiles();

    auto metricsResult = CreateMetricsFromDescription(metricDescriptions, model->GetDimensionsCount());
    for (size_t metricIdx = 0; metricIdx < metricsParamLen; ++metricIdx) {
        TString metricName = metricsResult[metricIdx]->GetDescription();
        SEXP metricScoreResult = VECTOR_ELT(result, metricIdx);
        for (size_t i = 0; i < numberOfIterations; ++i) {
            REAL(metricScoreResult)[i] = metricsScore[metricIdx][i];
        }
        SET_STRING_ELT(metricNames, metricIdx, mkChar(metricName.c_str()));
    }

    setAttrib(result, R_NamesSymbol, metricNames);

    R_API_END();
    UNPROTECT(protectedCount);
    return result;
}


// P3.6 follow-up (catboost-8z4.48): native tokenizer/dictionary bridges,
// replacing the pure-R port in R/text_processing.R. Mirrors the method
// surface vendor/catboost/catboost/python-package/catboost/_text_processing.pxi
// wraps around NTextProcessing::NTokenizer::TTokenizer and
// NTextProcessing::NDictionary::TDictionary/TDictionaryBuilder/
// TBpeDictionary/TBpeDictionaryBuilder.

static TVector<TString> GetTokensFromSEXP(SEXP lineTokens) {
    const int tokenCount = length(lineTokens);
    TVector<TString> tokens(tokenCount);
    for (int j = 0; j < tokenCount; ++j) {
        tokens[j] = TString(CHAR(STRING_ELT(lineTokens, j)));
    }
    return tokens;
}

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
) {
    using namespace NTextProcessing::NTokenizer;
    SEXP result = NULL;
    R_API_BEGIN();

    TTokenizerOptions options;
    options.Lowercasing = static_cast<bool>(asLogical(lowercasingParam));
    options.Lemmatizing = static_cast<bool>(asLogical(lemmatizingParam));
    CB_ENSURE(
        TryFromString<ETokenProcessPolicy>(CHAR(asChar(numberProcessPolicyParam)), options.NumberProcessPolicy),
        "catboost.Tokenizer: unsupported number_process_policy '" << CHAR(asChar(numberProcessPolicyParam)) << "'");
    options.NumberToken = TString(CHAR(asChar(numberTokenParam)));
    CB_ENSURE(
        TryFromString<ESeparatorType>(CHAR(asChar(separatorTypeParam)), options.SeparatorType),
        "catboost.Tokenizer: unsupported separator_type '" << CHAR(asChar(separatorTypeParam)) << "'");
    options.Delimiter = TString(CHAR(asChar(delimiterParam)));
    options.SplitBySet = static_cast<bool>(asLogical(splitBySetParam));
    options.SkipEmpty = static_cast<bool>(asLogical(skipEmptyParam));

    if (!Rf_isNull(tokenTypesParam)) {
        options.TokenTypes.clear();
        for (int i = 0; i < length(tokenTypesParam); ++i) {
            ETokenType tokenType;
            CB_ENSURE(
                TryFromString<ETokenType>(CHAR(STRING_ELT(tokenTypesParam, i)), tokenType),
                "catboost.Tokenizer: unsupported token_types entry '" << CHAR(STRING_ELT(tokenTypesParam, i)) << "'");
            options.TokenTypes.insert(tokenType);
        }
    }

    CB_ENSURE(
        TryFromString<ESubTokensPolicy>(CHAR(asChar(subTokensPolicyParam)), options.SubTokensPolicy),
        "catboost.Tokenizer: unsupported sub_tokens_policy '" << CHAR(asChar(subTokensPolicyParam)) << "'");

    if (!Rf_isNull(languagesParam)) {
        options.Languages.clear();
        for (int i = 0; i < length(languagesParam); ++i) {
            options.Languages.push_back(LanguageByNameOrDie(TStringBuf(CHAR(STRING_ELT(languagesParam, i)))));
        }
    }

    NTextProcessing::NTokenizer::TTokenizer* tokenizerPtr =
        new NTextProcessing::NTokenizer::TTokenizer(options);

    result = PROTECT(R_MakeExternalPtr(tokenizerPtr, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<NTextProcessing::NTokenizer::TTokenizer*>, TRUE);

    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextTokenizerTokenize_R(SEXP tokenizerParam, SEXP stringParam) {
    using namespace NTextProcessing::NTokenizer;
    SEXP result = NULL;
    R_API_BEGIN();

    NTextProcessing::NTokenizer::TTokenizer* tokenizer =
        static_cast<NTextProcessing::NTokenizer::TTokenizer*>(R_ExternalPtrAddr(tokenizerParam));
    CB_ENSURE(tokenizer, "catboost.tokenizer.tokenize: tokenizer handle is NULL.");
    TString input(CHAR(asChar(stringParam)));

    TVector<TString> tokens;
    TVector<ETokenType> tokenTypes;
    tokenizer->Tokenize(input, &tokens, &tokenTypes);

    SEXP tokensSexp = PROTECT(allocVector(STRSXP, tokens.size()));
    SEXP typesSexp = PROTECT(allocVector(STRSXP, tokenTypes.size()));
    for (size_t i = 0; i < tokens.size(); ++i) {
        SET_STRING_ELT(tokensSexp, i, mkChar(tokens[i].c_str()));
        SET_STRING_ELT(typesSexp, i, mkChar(ToString(tokenTypes[i]).c_str()));
    }

    result = PROTECT(allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, tokensSexp);
    SET_VECTOR_ELT(result, 1, typesSexp);
    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("tokens"));
    SET_STRING_ELT(names, 1, mkChar("types"));
    setAttrib(result, R_NamesSymbol, names);

    R_API_END();
    UNPROTECT(4);
    return result;
}


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
) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();

    TDictionaryOptions dictOptions;
    CB_ENSURE(
        TryFromString<ETokenLevelType>(CHAR(asChar(tokenLevelTypeParam)), dictOptions.TokenLevelType),
        "catboost.Dictionary: unsupported token_level_type '" << CHAR(asChar(tokenLevelTypeParam)) << "'");
    dictOptions.GramOrder = static_cast<ui32>(asInteger(gramOrderParam));
    dictOptions.SkipStep = static_cast<ui32>(asInteger(skipStepParam));
    dictOptions.StartTokenId = static_cast<NTextProcessing::NDictionary::TTokenId>(asInteger(startTokenIdParam));
    CB_ENSURE(
        TryFromString<EEndOfWordTokenPolicy>(CHAR(asChar(endOfWordPolicyParam)), dictOptions.EndOfWordTokenPolicy),
        "catboost.Dictionary: unsupported end_of_word_policy '" << CHAR(asChar(endOfWordPolicyParam)) << "'");
    CB_ENSURE(
        TryFromString<EEndOfSentenceTokenPolicy>(
            CHAR(asChar(endOfSentencePolicyParam)), dictOptions.EndOfSentenceTokenPolicy),
        "catboost.Dictionary: unsupported end_of_sentence_policy '"
            << CHAR(asChar(endOfSentencePolicyParam)) << "'");

    TDictionaryBuilderOptions builderOptions;
    builderOptions.OccurrenceLowerBound = static_cast<ui64>(asReal(occurenceLowerBoundParam));
    builderOptions.MaxDictionarySize = asInteger(maxDictionarySizeParam);

    EDictionaryType dictionaryType;
    CB_ENSURE(
        TryFromString<EDictionaryType>(CHAR(asChar(dictionaryTypeParam)), dictionaryType),
        "catboost.Dictionary: unsupported dictionary_type '" << CHAR(asChar(dictionaryTypeParam)) << "'");

    const int lineCount = length(linesParam);

    // Matches vendor's own BuildBpeWord/BuildBpeLetter split (library/cpp/
    // text_processing/app_helpers/app_helpers.cpp): Bpe over a Letter-level
    // alphabet builds its merge corpus from *unique* tokens weighted by
    // corpus-wide occurrence count, not from a second per-line pass.
    const bool needTokenCounts =
        (dictionaryType == EDictionaryType::Bpe && dictOptions.TokenLevelType == ETokenLevelType::Letter);

    TDictionaryBuilder alphabetBuilder(builderOptions, dictOptions);
    THashMap<TString, ui64> tokenCounts;
    for (int i = 0; i < lineCount; ++i) {
        TVector<TString> tokens = GetTokensFromSEXP(VECTOR_ELT(linesParam, i));
        alphabetBuilder.Add(TConstArrayRef<TString>(tokens), /*weight*/ 1);
        if (needTokenCounts) {
            for (const auto& token : tokens) {
                ++tokenCounts[token];
            }
        }
    }
    TIntrusivePtr<TDictionary> alphabet = alphabetBuilder.FinishBuilding();

    IDictionary* dictionaryPtr = nullptr;
    if (dictionaryType == EDictionaryType::FrequencyBased) {
        dictionaryPtr = alphabet.Release();
    } else {
        const ui32 numBpeUnits = static_cast<ui32>(asInteger(numBpeUnitsParam));
        const bool skipUnknown = static_cast<bool>(asLogical(skipUnknownParam));
        TBpeDictionaryBuilder bpeBuilder(numBpeUnits, skipUnknown, alphabet);
        if (needTokenCounts) {
            for (const auto& [token, count] : tokenCounts) {
                bpeBuilder.Add(TVector<TStringBuf>({token}), count);
            }
        } else {
            for (int i = 0; i < lineCount; ++i) {
                TVector<TString> tokens = GetTokensFromSEXP(VECTOR_ELT(linesParam, i));
                bpeBuilder.Add(TConstArrayRef<TString>(tokens), /*weight*/ 1);
            }
        }
        TIntrusivePtr<TBpeDictionary> bpeDict = bpeBuilder.FinishBuilding();
        dictionaryPtr = bpeDict.Release();
    }

    result = PROTECT(R_MakeExternalPtr(dictionaryPtr, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<NTextProcessing::NDictionary::IDictionary*>, TRUE);

    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionaryApply_R(SEXP dictionaryParam, SEXP linesParam, SEXP unknownTokenPolicyParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();

    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.apply: dictionary handle is NULL.");
    EUnknownTokenPolicy unknownTokenPolicy;
    CB_ENSURE(
        TryFromString<EUnknownTokenPolicy>(CHAR(asChar(unknownTokenPolicyParam)), unknownTokenPolicy),
        "catboost.dictionary.apply: unsupported unknown_token_policy '"
            << CHAR(asChar(unknownTokenPolicyParam)) << "'");

    const int lineCount = length(linesParam);
    result = PROTECT(allocVector(VECSXP, lineCount));
    for (int i = 0; i < lineCount; ++i) {
        TVector<TString> tokens = GetTokensFromSEXP(VECTOR_ELT(linesParam, i));
        TVector<NTextProcessing::NDictionary::TTokenId> tokenIds;
        dictionary->Apply(TConstArrayRef<TString>(tokens), &tokenIds, unknownTokenPolicy);

        SEXP idsSexp = PROTECT(allocVector(INTSXP, tokenIds.size()));
        for (size_t j = 0; j < tokenIds.size(); ++j) {
            INTEGER(idsSexp)[j] = static_cast<int>(tokenIds[j]);
        }
        SET_VECTOR_ELT(result, i, idsSexp);
        UNPROTECT(1);
    }

    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionarySize_R(SEXP dictionaryParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();
    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.size: dictionary handle is NULL.");
    result = PROTECT(ScalarInteger(static_cast<int>(dictionary->Size())));
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionaryGetTokens_R(SEXP dictionaryParam, SEXP tokenIdsParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();
    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.get_tokens: dictionary handle is NULL.");
    const int n = length(tokenIdsParam);
    result = PROTECT(allocVector(STRSXP, n));
    const int* ids = INTEGER(tokenIdsParam);
    for (int i = 0; i < n; ++i) {
        TString token = dictionary->GetToken(static_cast<NTextProcessing::NDictionary::TTokenId>(ids[i]));
        SET_STRING_ELT(result, i, mkChar(token.c_str()));
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionaryGetTopTokens_R(SEXP dictionaryParam, SEXP topSizeParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();
    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.get_top_tokens: dictionary handle is NULL.");
    TVector<TString> top = dictionary->GetTopTokens(static_cast<ui32>(asInteger(topSizeParam)));
    result = PROTECT(allocVector(STRSXP, top.size()));
    for (size_t i = 0; i < top.size(); ++i) {
        SET_STRING_ELT(result, i, mkChar(top[i].c_str()));
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionaryUnknownTokenId_R(SEXP dictionaryParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();
    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.unknown_token_id: dictionary handle is NULL.");
    result = PROTECT(ScalarInteger(static_cast<int>(dictionary->GetUnknownTokenId())));
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionaryEndOfSentenceTokenId_R(SEXP dictionaryParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();
    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.end_of_sentence_token_id: dictionary handle is NULL.");
    result = PROTECT(ScalarInteger(static_cast<int>(dictionary->GetEndOfSentenceTokenId())));
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionaryMinUnusedTokenId_R(SEXP dictionaryParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();
    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.min_unused_token_id: dictionary handle is NULL.");
    result = PROTECT(ScalarInteger(static_cast<int>(dictionary->GetMinUnusedTokenId())));
    R_API_END();
    UNPROTECT(1);
    return result;
}


EXPORT_FUNCTION CatBoostTextDictionarySave_R(
    SEXP dictionaryParam,
    SEXP dictionaryTypeParam,
    SEXP frequencyDictPathParam,
    SEXP bpePathParam
) {
    using namespace NTextProcessing::NDictionary;
    R_API_BEGIN();

    IDictionary* dictionary = static_cast<IDictionary*>(R_ExternalPtrAddr(dictionaryParam));
    CB_ENSURE(dictionary, "catboost.dictionary.save: dictionary handle is NULL.");
    TString dictionaryType(CHAR(asChar(dictionaryTypeParam)));
    if (dictionaryType == "Bpe") {
        CB_ENSURE(
            !Rf_isNull(bpePathParam), "catboost.dictionary.save: bpe_path is required to save a Bpe dictionary.");
        TBpeDictionary* bpeDictionary = dynamic_cast<TBpeDictionary*>(dictionary);
        CB_ENSURE(bpeDictionary, "catboost.dictionary.save: dictionary is not a Bpe dictionary.");
        bpeDictionary->Save(TString(CHAR(asChar(frequencyDictPathParam))), TString(CHAR(asChar(bpePathParam))));
    } else {
        TFileOutput out(TString(CHAR(asChar(frequencyDictPathParam))));
        dictionary->Save(&out);
    }

    R_API_END();
    return R_NilValue;
}


EXPORT_FUNCTION CatBoostTextDictionaryLoad_R(SEXP frequencyDictPathParam, SEXP bpePathParam) {
    using namespace NTextProcessing::NDictionary;
    SEXP result = NULL;
    R_API_BEGIN();

    IDictionary* dictionaryPtr = nullptr;
    TString freqPath(CHAR(asChar(frequencyDictPathParam)));
    if (!Rf_isNull(bpePathParam)) {
        TString bpePath(CHAR(asChar(bpePathParam)));
        THolder<TBpeDictionary> bpeDictionary = MakeHolder<TBpeDictionary>();
        bpeDictionary->Load(freqPath, bpePath);
        dictionaryPtr = bpeDictionary.Release();
    } else {
        TFileInput in(freqPath);
        TIntrusivePtr<IDictionary> dictionary = IDictionary::Load(&in);
        dictionaryPtr = dictionary.Release();
    }

    result = PROTECT(R_MakeExternalPtr(dictionaryPtr, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<NTextProcessing::NDictionary::IDictionary*>, TRUE);

    R_API_END();
    UNPROTECT(1);
    return result;
}
}
