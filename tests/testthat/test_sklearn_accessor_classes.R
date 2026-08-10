context("test_sklearn_accessor_classes.R")

# catboost-8z4.116 (P10.B) differential test: parity for the sklearn
# accessor family (tree_count_/learning_rate_/random_seed_/
# n_features_in_/get_n_features_in/get_cat_feature_indices/
# get_text_feature_indices/get_embedding_feature_indices/get_all_params/
# get_param/set_feature_names, plus CatBoostClassifier-only
# predict_proba/staged_predict_proba) against all four Python estimator
# classes -- CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker
# (catboost/python-package/catboost/core.py). None of these methods/
# properties are overridden per-subclass in Python, so R's single set of
# accessor functions is the parity target for all four classes; what varies
# per row is only the loss/task shape the model was fit with -- same
# per-class loss families as test_predict_classes.R /
# test_get_feature_importance_classes.R: MultiClass base, Logloss
# classifier, RMSE regressor, YetiRank ranker.
#
# R equivalents used (per catboost-8z4.116 report):
#   tree_count_/learning_rate_/random_seed_  -> model$tree_count /
#     model$learning_rate / catboost.get_plain_params(model)$random_seed
#     (model$random_seed is only ever set by catboost.shrink(), not by
#     catboost.train(), so the plain-params view is the correct getter here)
#   n_features_in_/get_n_features_in()       -> model$feature_count
#   get_cat_feature_indices()/get_text_feature_indices()/
#     get_embedding_feature_indices()        -> catboost.pool.get_*_indices()
#     on the training pool -- valid because these are deterministic
#     functions of the fit pool's feature layout and are never mutated
#     post-fit, so the training pool's indices ARE the model's indices
#   get_all_params()/get_param(key)          -> catboost.get_plain_params()
#     (a flat list, unlike the nested catboost.get_model_params()) restricted
#     to keys explicitly passed at fit time; get_param() only echoes back
#     caller-supplied values (learning_rate, which Python auto-resolves
#     here, is therefore excluded -- see gen_sklearn_accessor_classes_fixture.py)
#   set_feature_names()                      -> catboost.pool.set_feature_names()
#     on the pool BEFORE catboost.train(); R has no post-fit model-level
#     renamer, but the model's feature names are exactly the training pool's
#     names, so pre-fit renaming is the observably equivalent capability
#
# CatBoost{,Classifier,Regressor,Ranker}.fit is NOT re-tested here: it is
# already exercised end-to-end (fit -> predict, RawFormulaVal, all 4 classes)
# by test_predict_classes.R, which is the linked test_id for the 4 .fit rows.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_sklearn_accessor_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "sklearn_accessor_classes.json"),
  simplifyVector = FALSE
)

TOL <- 1e-6
EVAL_PERIOD <- fixture$eval_period

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

build_pool <- function(class_name, group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  data <- data.frame(num1 = unlist(inputs$num1), num2 = unlist(inputs$num2),
                      cat1 = as.factor(unlist(inputs$cat1)))
  label <- unlist(inputs$label)
  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = label)
  } else {
    pool <- catboost.load_pool(data, label = label, group_id = unlist(group_id))
  }
  catboost.pool.set_feature_names(pool, unlist(inputs$feature_names))
  pool
}

check_accessors <- function(class_name, loss_function, group_id = NULL) {
  expected <- fixture$expected[[class_name]]
  pool <- build_pool(class_name, group_id)
  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))
  plain_params <- catboost.get_plain_params(model)

  expect_equal(model$tree_count, expected$tree_count_, info = class_name)
  expect_equal(model$learning_rate, expected$learning_rate_, tolerance = TOL, info = class_name)
  expect_equal(plain_params$random_seed, expected$random_seed_, info = class_name)
  expect_equal(model$feature_count, expected$n_features_in_, info = class_name)
  expect_equal(model$feature_count, expected$get_n_features_in, info = class_name)

  expect_equal(catboost.pool.get_cat_feature_indices(pool),
               as.integer(unlist(expected$get_cat_feature_indices)), info = class_name)
  expect_equal(catboost.pool.get_text_feature_indices(pool), integer(0), info = class_name)
  expect_equal(catboost.pool.get_embedding_feature_indices(pool), integer(0), info = class_name)

  for (key in names(expected$get_all_params)) {
    expect_equal(plain_params[[key]], expected$get_all_params[[key]], tolerance = TOL,
                 info = paste(class_name, "get_all_params", key))
  }
  for (key in names(expected$get_param)) {
    expect_equal(plain_params[[key]], expected$get_param[[key]], tolerance = TOL,
                 info = paste(class_name, "get_param", key))
  }

  # set_feature_names: R has no post-fit renamer, so re-fit on a pool whose
  # names were set to the fixture's new names before training, matching
  # Python's post-fit model.set_feature_names(NEW_FEATURE_NAMES) result.
  renamed_pool <- build_pool(class_name, group_id)
  catboost.pool.set_feature_names(renamed_pool, unlist(fixture$inputs[[class_name]]$new_feature_names))
  renamed_model <- catboost.train(renamed_pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))
  expect_equal(catboost.get_model_feature_names(renamed_model),
               unlist(expected$feature_names_after_set), info = class_name)
}

test_that("sklearn accessors: CatBoost (base class, MultiClass) match Python oracle", {
  check_accessors("CatBoost", "MultiClass")
})

test_that("sklearn accessors: CatBoostClassifier (Logloss) match Python oracle", {
  check_accessors("CatBoostClassifier", "Logloss")
})

test_that("sklearn accessors: CatBoostRegressor (RMSE) match Python oracle", {
  check_accessors("CatBoostRegressor", "RMSE")
})

test_that("sklearn accessors: CatBoostRanker (YetiRank) match Python oracle", {
  check_accessors("CatBoostRanker", "YetiRank", group_id = fixture$inputs$CatBoostRanker$group_id)
})

test_that("predict_proba/staged_predict_proba: CatBoostClassifier matches Python oracle", {
  expected <- fixture$expected$CatBoostClassifier
  pool <- build_pool("CatBoostClassifier")
  model <- catboost.train(pool, params = c(list(loss_function = "Logloss"), COMMON_PARAMS))

  # R's Probability output for binary Logloss is a single column (P(class 1));
  # Python's predict_proba() duplicates this as two columns [P(0), P(1)] --
  # same information, different shape. Compare against Python's 2nd column.
  actual_proba <- as.numeric(catboost.predict(model, pool, prediction_type = "Probability"))
  expected_proba <- do.call(rbind, lapply(expected$predict_proba, function(r) as.numeric(unlist(r))))
  expect_equal(actual_proba, expected_proba[, 2], tolerance = TOL)

  staged <- catboost.staged_predict(model, pool, ntree_start = 0, ntree_end = 0,
                                     eval_period = EVAL_PERIOD, prediction_type = "Probability")
  expected_stages <- expected$staged_predict_proba
  for (i in seq_along(expected_stages)) {
    actual <- as.numeric(staged$nextElem())
    expected_mat <- do.call(rbind, lapply(expected_stages[[i]], function(r) as.numeric(unlist(r))))
    expect_equal(actual, expected_mat[, 2], tolerance = TOL,
                 info = paste("stage", i))
  }
  expect_error(staged$nextElem(), "StopIteration")
})
