context("test_model_persistence_classes.R")

# catboost-8z4.82 differential test: R's catboost.save_model()/
# catboost.load_model() parity against all four Python estimator classes --
# CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker
# (catboost/python-package/catboost/core.py). save_model (core.py:3702) and
# load_model (core.py:3745) are each defined once on the CatBoost base class
# and are not overridden by any of the three subclasses, so R's single
# catboost.save_model()/catboost.load_model() pair (R/catboost.R:3662/3614)
# is the parity target for all 8 CatBoost{,Classifier,Regressor,Ranker}.
# {save,load}_model matrix rows.
#
# save_model/load_model have no directly comparable numeric output of their
# own -- the artifact is a binary/JSON model file, not a value -- so, same as
# CatBoost.copy and the CLI's mode:normalize-model (both judged
# final_method: "roundtrip"), parity here is self-consistency: a model's
# predictions must be unchanged after a save-to-disk/load-from-disk cycle.
# test_model.R's existing "model: catboost.load_model"/"model:
# catboost.save_model" tests already cover this for one binary-classification
# fit (cbm/json/coreml formats); this file extends that same round-trip
# check across the other three loss-function families used elsewhere in this
# suite's *_classes.R tests (test_drop_unused_features_classes.R,
# test_shrink_classes.R): MultiClass base, Logloss classifier, RMSE
# regressor, YetiRank ranker.

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent",
                       allow_writing_files = FALSE)

check_class <- function(class_name, loss_function, group_id = NULL) {
  set.seed(42)
  n <- 200
  data <- data.frame(num1 = rnorm(n), num2 = rnorm(n))

  if (loss_function == "MultiClass") {
    label <- sample(c(0, 1, 2), size = n, replace = TRUE)
  } else if (loss_function == "Logloss") {
    label <- sample(c(0, 1), size = n, replace = TRUE)
  } else {
    label <- rnorm(n)
  }

  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = label)
  } else {
    pool <- catboost.load_pool(data, label = label, group_id = group_id)
  }

  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))

  before <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")

  model_path <- tempfile(fileext = ".cbm")
  status <- catboost.save_model(model, model_path)
  expect_true(status, info = paste(class_name, "save_model status"))

  loaded_model <- catboost.load_model(model_path)
  unlink(model_path)

  after <- catboost.predict(loaded_model, pool, prediction_type = "RawFormulaVal")

  expect_equal(as.numeric(after), as.numeric(before), tolerance = TOL,
               info = paste(class_name, "save/load round-trip prediction"))
}

test_that("save_model/load_model: CatBoost (base class, MultiClass) round-trips predictions", {
  check_class("CatBoost", "MultiClass")
})

test_that("save_model/load_model: CatBoostClassifier (Logloss) round-trips predictions", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("save_model/load_model: CatBoostRegressor (RMSE) round-trips predictions", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("save_model/load_model: CatBoostRanker (YetiRank) round-trips predictions", {
  n <- 200
  group_id <- rep(1:20, each = 10)
  check_class("CatBoostRanker", "YetiRank", group_id = group_id)
})
