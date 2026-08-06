context("test_drop_unused_features_classes.R")

# catboost-8z4.80 differential test: R's catboost.drop_unused_features()
# parity against all four Python estimator classes -- CatBoost,
# CatBoostClassifier, CatBoostRegressor, CatBoostRanker. drop_unused_features
# is defined once on the CatBoost base class (core.py:3696) and is not
# overridden by any subclass; both languages route to the exact same native
# TModelTrees::DropUnusedFeatures() (catboost/libs/model/model.cpp:626 via
# src/catboostr.cpp:2400's CatBoostDropUnusedFeaturesFromModel_R), so R's
# single catboost.drop_unused_features() function (R/catboost.R:4178) is the
# parity target for all four matrix rows.
#
# drop_unused_features() has no output of its own -- Python's wrapper
# returns None and R's constant TRUE regardless of outcome
# (src/catboostr.cpp:2405) -- so this is a state-mutator row (same pattern
# as the ~45 set_* rows flagged in docs/phase-2/P2.3-disposition-report.md
# section (a)): the real observable is the model's state before/after the
# call, verified via the paired feature-names getter
# (catboost.get_model_feature_names() / model.feature_names_) and via
# predictions staying bit-identical (dropping unused features must not
# change tree evaluation). Each fixture fit uses one strongly-informative
# feature (num1, built directly from that class's own label) plus two
# pure-noise features (noise1, noise2) with depth=1/iterations=3 so the
# fixed-seed, single-threaded, identical-C++-engine build deterministically
# matches between R and Python on which features end up unused.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_drop_unused_features_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "drop_unused_features_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 3, depth = 1, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

check_class <- function(class_name, loss_function, group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  expected <- fixture$expected[[class_name]]
  data <- data.frame(num1 = inputs$num1, noise1 = inputs$noise1, noise2 = inputs$noise2)

  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = inputs$label)
  } else {
    pool <- catboost.load_pool(data, label = inputs$label, group_id = group_id)
  }
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))

  before_names <- catboost.get_model_feature_names(model)
  expect_equal(before_names, expected$before_feature_names, info = paste(class_name, "before names"))

  before_pred <- catboost.predict(model, pool)
  expect_equal(as.numeric(before_pred), as.numeric(expected$before_prediction), tolerance = TOL,
               info = paste(class_name, "before prediction"))

  status <- catboost.drop_unused_features(model)
  expect_true(status)

  after_names <- catboost.get_model_feature_names(model)
  expect_equal(after_names, expected$after_feature_names, info = paste(class_name, "after names"))

  after_pred <- catboost.predict(model, pool)
  expect_equal(as.numeric(after_pred), as.numeric(expected$after_prediction), tolerance = TOL,
               info = paste(class_name, "after prediction"))
}

test_that("drop_unused_features: CatBoost (base class, MultiClass) matches Python oracle", {
  check_class("CatBoost", "MultiClass")
})

test_that("drop_unused_features: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("drop_unused_features: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("drop_unused_features: CatBoostRanker (YetiRank) matches Python oracle", {
  check_class("CatBoostRanker", "YetiRank", group_id = fixture$inputs$CatBoostRanker$group_id)
})

# catboost-8z4.83 regression: ntree_end/ntree_start were dead arguments (the
# native call ignores them; there is no tree-range-limited drop variant
# anywhere in CatBoost -- see src/catboostr.cpp:2400-2405 and
# vendor/catboost/catboost/libs/model/model.cpp:626). The signature was
# simplified to match Python's argument-free drop_unused_features()
# (core.py:3696); confirm the dead parameters are actually gone rather than
# merely unused.
test_that("drop_unused_features: signature has no ntree_end/ntree_start parameters", {
  expect_identical(names(formals(catboost.drop_unused_features)), "model")
  expect_error(catboost.drop_unused_features(NULL, ntree_end = 5), "unused argument")
})
