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
