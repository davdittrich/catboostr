context("test_get_feature_importance_classes.R")

# catboost-8z4.80 differential test: R's catboost.get_feature_importance()
# default type="FeatureImportance" parity against all four Python estimator
# classes -- CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker
# (catboost/python-package/catboost/core.py). get_feature_importance() is
# defined once on the CatBoost base class (core.py:3385) and is not
# overridden by any of the three subclasses, so R's single
# catboost.get_feature_importance() function (R/catboost.R:4013) is the
# parity target for all four CatBoost{,Classifier,Regressor,Ranker}.
# get_feature_importance matrix rows. The default "FeatureImportance" type
# resolves to PredictionValuesChange for non-ranking losses and to
# LossFunctionChange for ranking losses (core.py:3404-3417) -- both variants
# are covered here via the same per-class loss families used by the
# eval_metrics fixture (test_eval_metrics_classes.R): MultiClass via the
# base class, Logloss via Classifier, RMSE via Regressor, YetiRank via
# Ranker. Existing coverage (test_model.R, test_fstr_shap_prediction_diff.R)
# only checks structural shape or the ShapInteractionValues/PredictionDiff
# sub-types, never the default type's actual numeric values against a
# Python oracle -- this file closes that gap.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_get_feature_importance_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "get_feature_importance_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

check_class <- function(class_name, loss_function, group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  expected <- fixture$expected[[class_name]]
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)

  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = inputs$label)
  } else {
    pool <- catboost.load_pool(data, label = inputs$label, group_id = group_id)
  }
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))

  result <- catboost.get_feature_importance(model, pool, type = "FeatureImportance")

  expect_equal(as.numeric(result), as.numeric(expected), tolerance = TOL, info = class_name)
}

test_that("get_feature_importance: CatBoost (base class, MultiClass) matches Python oracle", {
  check_class("CatBoost", "MultiClass")
})

test_that("get_feature_importance: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("get_feature_importance: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("get_feature_importance: CatBoostRanker (YetiRank) matches Python oracle", {
  check_class("CatBoostRanker", "YetiRank", group_id = fixture$inputs$CatBoostRanker$group_id)
})

# catboost-8z4.117 (P10.C): EFstrType.PredictionValuesChange / .LossFunctionChange
# are not separate R capabilities -- they are the two concrete values the
# default type="FeatureImportance" resolves to (see the file banner above,
# core.py:3404-3417): PredictionValuesChange for non-ranking losses,
# LossFunctionChange for ranking losses. Both are already exercised
# byte-for-byte against the Python oracle above via the "FeatureImportance"
# default; these two tests additionally confirm that passing the resolved
# type *by name* is not merely accepted but produces the identical native
# result as the default, for one non-ranking class (RMSE) and the one
# ranking class (YetiRank) respectively -- i.e. the explicit enum value and
# the resolved default are the same call, not a coincidentally similar one.
test_that("get_feature_importance: type = 'PredictionValuesChange' matches the 'FeatureImportance' default (non-ranking loss)", {
  inputs <- fixture$inputs$CatBoostRegressor
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)
  model <- catboost.train(pool, params = c(list(loss_function = "RMSE"), COMMON_PARAMS))

  pvc <- catboost.get_feature_importance(model, pool, type = "PredictionValuesChange")
  default <- catboost.get_feature_importance(model, pool, type = "FeatureImportance")
  expect_equal(pvc, default)
  expect_equal(as.numeric(pvc), as.numeric(fixture$expected$CatBoostRegressor), tolerance = TOL)
})

test_that("get_feature_importance: type = 'LossFunctionChange' matches the 'FeatureImportance' default (ranking loss)", {
  inputs <- fixture$inputs$CatBoostRanker
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label, group_id = inputs$group_id)
  catboost.pool.set_feature_names(pool, inputs$feature_names)
  model <- catboost.train(pool, params = c(list(loss_function = "YetiRank"), COMMON_PARAMS))

  lfc <- catboost.get_feature_importance(model, pool, type = "LossFunctionChange")
  default <- catboost.get_feature_importance(model, pool, type = "FeatureImportance")
  expect_equal(lfc, default)
  expect_equal(as.numeric(lfc), as.numeric(fixture$expected$CatBoostRanker), tolerance = TOL)
})

# catboost-8z4.117: EFstrType.Interaction -- no prior test exercised this
# value at all. Verify the real native output is well-formed: one row per
# unordered feature pair (2 features -> exactly 1 pair), valid 0-based
# indices into the pool's 2 columns, and a finite interaction score.
test_that("get_feature_importance: type = 'Interaction' returns well-formed pairwise scores", {
  inputs <- fixture$inputs$CatBoostRegressor
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  model <- catboost.train(pool, params = c(list(loss_function = "RMSE"), COMMON_PARAMS))

  inter <- catboost.get_feature_importance(model, pool, type = "Interaction")
  expect_equal(colnames(inter), c("feature1_index", "feature2_index", "score"))
  expect_equal(nrow(inter), 1) # only 2 features -> exactly one interacting pair
  expect_true(all(inter[, c("feature1_index", "feature2_index")] %in% c(0, 1)))
  expect_true(all(is.finite(inter[, "score"])))
})

# catboost-8z4.117: EFstrType.ShapValues -- test_model.R's existing
# "catboost.importance with shapvalues for multiclass" test only asserts
# expect_true(TRUE) (no real check). Replace that with the real, well-known
# SHAP invariant: per-row SHAP values (plus the trailing "<base>" bias
# column) must sum exactly to that row's RawFormulaVal prediction, for both
# binary and multiclass models -- this is a mathematical necessity of the
# algorithm (not something a Python-oracle diff is needed to establish) and
# would fail immediately if the native binding were wired wrong.
test_that("get_feature_importance: type = 'ShapValues' sums to the RawFormulaVal prediction (binary)", {
  inputs <- fixture$inputs$CatBoostClassifier
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  model <- catboost.train(pool, params = c(list(loss_function = "Logloss"), COMMON_PARAMS))

  shap <- catboost.get_feature_importance(model, pool, type = "ShapValues")
  raw <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(rowSums(shap), as.numeric(raw), tolerance = 1e-6, check.attributes = FALSE)
})

test_that("get_feature_importance: type = 'ShapValues' sums to the RawFormulaVal prediction (multiclass)", {
  inputs <- fixture$inputs$CatBoost
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  model <- catboost.train(pool, params = c(list(loss_function = "MultiClass"), COMMON_PARAMS))

  shap <- catboost.get_feature_importance(model, pool, type = "ShapValues")
  raw <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  # dim(shap) = (n_objects, n_classes, n_features + 1); verified live
  # (probe against a fixture with n_features != n_classes to disambiguate
  # the two equal-sized dimensions the 2-numeric-feature/3-class case here
  # would otherwise leave ambiguous).
  n_classes <- dim(shap)[2]
  for (k in seq_len(n_classes)) {
    class_sum <- rowSums(shap[, k, ])
    expect_equal(class_sum, as.numeric(raw[, k]), tolerance = 1e-6, check.attributes = FALSE)
  }
})

# catboost-8z4.117: EFstrType.SageValues -- confirmed by reading
# R/catboost.R's catboost.get_feature_importance() dispatch (the final
# `else { stop("Unknown type: ", type) }` branch) that "SageValues" is not
# one of the type strings the R<->native bridge recognizes at all (unlike
# PredictionValuesChange/LossFunctionChange/FeatureImportance/Interaction/
# ShapValues/ShapInteractionValues/PredictionDiff, which each have their own
# branch). This is a genuine capability gap, not a missing test -- verified
# live, not assumed.
test_that("get_feature_importance: type = 'SageValues' is rejected (EFstrType.SageValues has no R binding)", {
  inputs <- fixture$inputs$CatBoostRegressor
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  model <- catboost.train(pool, params = c(list(loss_function = "RMSE"), COMMON_PARAMS))
  expect_error(
    catboost.get_feature_importance(model, pool, type = "SageValues"),
    "Unknown type"
  )
})
