context("test_model.R")

check_model_train_simple <- function(target, features, loss_function) {
  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  pool_train <- catboost.load_pool(features[split, ], target[split])
  pool_test <- catboost.load_pool(features[-split, ], target[-split])

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = loss_function,
                 random_seed = 12345,
                 use_best_model = FALSE,
                 allow_writing_files = FALSE)

  model_train_test <- catboost.train(pool_train, pool_test, params)
  prediction_train_test <- catboost.predict(model_train_test, pool_test)

  model_train <- catboost.train(pool_train, NULL, params)
  prediction_train <- catboost.predict(model_train, pool_test)

  expect_equal(prediction_train_test, prediction_train)
}

test_that("model: catboost.train binary classification with double target", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_model_train_simple(target, features, "Logloss")
})

test_that("model: catboost.train binary classification with integer target", {
  target <- sample(c(1L, -1L), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_model_train_simple(target, features, "Logloss")
})

test_that("model: catboost.train binary classification with factor target", {
  target <- as.factor(sample(c(1, -1), size = 1000, replace = TRUE))
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_model_train_simple(target, features, "Logloss")
})

test_that("model: catboost.train binary classification with character target", {
  target <- sample(c("class0", "class1"), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_model_train_simple(target, features, "Logloss")
})

test_that("model: catboost.train with per_float_quantization and ignored_features", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1),
                         f3 = rnorm(length(target), mean = 0, sd = 1),
                         f4 = rnorm(length(target), mean = 0, sd = 1),
                         f5 = rnorm(length(target), mean = 0, sd = 1))

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  pool_train <- catboost.load_pool(features[split, ], target[split])
  pool_test <- catboost.load_pool(features[-split, ], target[-split])

  params <- list(
    iterations = 10,
    loss_function = "Logloss",
    random_seed = 12345,
    use_best_model = FALSE,
    ignored_features = c(1, 3),
    allow_writing_files = FALSE
  )

  params$per_float_feature_quantization <- c("0:border_count=1024")
  catboost.train(pool_train, NULL, params)

  params$per_float_feature_quantization <- c("0:border_count=1024", "1:border_count=1024")
  catboost.train(pool_train, NULL, params)

  # issue with single ignored feature
  params$ignored_features <- c(1)
  catboost.train(pool_train, NULL, params)

  expect_true(TRUE)
})

check_models_with_synonyms <- function(pool, init_params, synonym_params, params_values) {
  params_num = length(synonym_params)
  max_synonym_length <- max(lengths(synonym_params))

  prediction <- NULL
  for (synonym_idx in 1:max_synonym_length) {
    params <- init_params
    for (params_idx in 1:params_num) {
      idx <- synonym_idx
      if (length(synonym_params[[params_idx]]) < synonym_idx) {
        idx <- 1
      }
      params[[synonym_params[[params_idx]][[idx]]]] <- params_values[[params_idx]]
    }

    model <- catboost.train(pool, NULL, params)
    prediction_with_synonyms <- catboost.predict(model, pool)
    if (is.null(prediction)) {
      prediction = prediction_with_synonyms
    } else {
      expect_equal(prediction, prediction_with_synonyms)
    }
  }
}

test_that("model: catboost.train synonyms", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  params <- list(iterations = 10,
                loss_function = "Logloss",
                random_seed = 12345,
                random_state = 12345,
                allow_writing_files = FALSE)

  expect_error(catboost.train(pool, NULL, params),
               "Only one of the parameters [random_seed, random_state] should be initialized.",
               fixed = TRUE)

  synonym_params <- list(
      c('loss_function', 'objective'),
      c('iterations', 'num_boost_round', 'n_estimators', 'num_trees'),
      c('learning_rate', 'eta'),
      c('random_seed', 'random_state'),
      c('l2_leaf_reg', 'reg_lambda'),
      c('depth', 'max_depth'),
      c('rsm', 'colsample_bylevel'),
      c('border_count', 'max_bin'),
      c('verbose', 'verbose_eval')
  )
  params_values <- list('Logloss', 5, 0.04, 12345, 4, 5, 0.5, 32, 20)
  init_params <- list(allow_writing_files = FALSE)
  check_models_with_synonyms(pool, init_params, synonym_params, params_values)

  synonym_params <- list(
    c('min_data_in_leaf', 'min_child_samples'),
    c('max_leaves', 'num_leaves')
  )
  params_values <- list(1, 31)
  init_params <- list(iterations = 5,
                      loss_function = "Logloss",
                      grow_policy = "Lossguide",
                      random_seed = 12345,
                      allow_writing_files = FALSE)
  check_models_with_synonyms(pool, init_params, synonym_params, params_values)
})

test_that("model: catboost.importance", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  feature_importance <- model$feature_importances
  expect_equal(length(feature_importance), ncol(features))
  feature_importance_with_pool <- catboost.get_feature_importance(model, pool)
  expect_equal(feature_importance, feature_importance_with_pool)
})

test_that("model: catboost.importance with shapvalues for multiclass", {
  classes <- c(0, 1, 2, 3, 4)
  target <- sample(classes, size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1),
                         f3 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "MultiClass",
                 random_seed = 12345,
                 use_best_model = FALSE,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  feature_importance <- catboost.get_feature_importance(model = model, pool = pool, type = 'ShapValues')
  expect_true(TRUE)
})


check_train_pred_multiclass <- function(classes, target, features) {
  pool <- catboost.load_pool(features, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)

  prediction <- catboost.predict(model, pool)
  expect_equal(nrow(pool), nrow(prediction))
  expect_equal(length(classes), ncol(prediction))

  prediction_first <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 4)
  prediction_second <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 5)

  staged_preds <- catboost.staged_predict(model, pool, ntree_start = 2, ntree_end = 5, eval_period = 2)

  expect_equal(prediction_first, staged_preds$nextElem())
  expect_equal(prediction_second, staged_preds$nextElem())
}


test_that("model: catboost.train & catboost.predict multiclass with double target", {
  classes <- c(0, 1, 2)
  target <- sample(classes, size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_train_pred_multiclass(classes, target, features)
})

test_that("model: catboost.train & catboost.predict multiclass with integer target", {
  classes <- c(0L, 1L, 2L)
  target <- sample(classes, size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_train_pred_multiclass(classes, target, features)
})

test_that("model: catboost.train & catboost.predict multiclass with factor target", {
  classes <- c(0L, 1L, 2L)
  target <- as.factor(sample(classes, size = 1000, replace = TRUE))
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_train_pred_multiclass(classes, target, features)
})

test_that("model: catboost.train & catboost.predict multiclass with character target", {
  classes <- c("class0", "class1", "class2")
  target <- sample(classes, size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  check_train_pred_multiclass(classes, target, features)
})


is_equal_model_and_load_model <- function(model, pool, file_format = "cbm") {
  prediction <- catboost.predict(model, pool)
  model_path <- "catboost.model"

  catboost.save_model(model, model_path, file_format = file_format)
  loaded_model <- catboost.load_model(model_path, file_format = file_format)

  unlink(model_path)

  loaded_model_prediction <- catboost.predict(loaded_model, pool)

  return (all(abs(prediction - loaded_model_prediction) < 0.00001))
}

test_that("model: catboost.load_model", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  params <- list(iterations = 10,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  expect_true(is_equal_model_and_load_model(model, pool))
})

test_that("model: catboost.save_model", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(feature_0 = rnorm(length(target), mean = 0, sd = 1),
                         feature_1 = rnorm(length(target), mean = 0, sd = 1),
                         feature_2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  params <- list(iterations = 10,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)

  expect_true(is_equal_model_and_load_model(model, pool, file_format = "json"))
  expect_true(is_equal_model_and_load_model(model, pool, file_format = "coreml"))
})

test_that("model: loss_function = multiclass", {
  target <- sample(c(0, 1, 2), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  data$f_logical <- as.factor(data$f_logical)
  data$f_character <- as.factor(data$f_character)

  pool <- catboost.load_pool(data, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool, prediction_type = "Class")

  unique_prediction <- unique(prediction)
  unique_target <- unique(target)

  expect_equal(unique_target[order(unique_target)],
               unique_prediction[order(unique_prediction)])
})

test_that("model: baseline", {
  target <- sample(c(0, 1, 2), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  unique_target <- unique(target)

  data$f_logical <- as.factor(data$f_logical)
  data$f_character <- as.factor(data$f_character)

  pool <- catboost.load_pool(data, target)

  params_baseline <- list(iterations = 3,
                          loss_function = "MultiClass",
                          random_seed = 12345,
                          allow_writing_files = FALSE)

  model_baseline <- catboost.train(pool, NULL, params_baseline)
  baseline <- catboost.predict(model_baseline, pool, prediction_type = "RawFormulaVal")

  expect_equal(nrow(baseline), nrow(pool))
  expect_equal(ncol(baseline), length(unique_target))

  pool_with_baseline <- catboost.load_pool(data, target, baseline = baseline)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345,
                 allow_writing_files = FALSE)

  model <- catboost.train(pool_with_baseline, NULL, params)

  prediction <- catboost.predict(model, pool_with_baseline, prediction_type = "Class")

  unique_prediction <- unique(prediction)

  expect_equal(unique_target[order(unique_target)],
               unique_prediction[order(unique_prediction)])
})

test_that("model: full_history", {
  target <- sample(c(0, 1, 2), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  unique_target <- unique(target)

  data$f_logical <- as.factor(data$f_logical)
  data$f_character <- as.factor(data$f_character)

  pool <- catboost.load_pool(data, target)

  params <- list(iterations = 3,
                 loss_function = "MultiClass",
                 random_seed = 12345,
                 approx_on_full_history = TRUE,
                 boosting_type = "Ordered",
                 allow_writing_files = FALSE)
  model <- catboost.train(pool, NULL, params)
  pred <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")

  expect_true(TRUE)
})

test_that("model: catboost.predict vs catboost.staged_predict", {
  pool_path <- system.file("extdata", "adult_train.1000", package = "catboostr")
  column_description_path <- system.file("extdata", "adult.cd", package = "catboostr")

  pool <- catboost.load_pool(pool_path, column_description = column_description_path)

  params <- list(iterations = 10,
                 loss_function = "Logloss",
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  prediction_first <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 4)
  prediction_second <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 5)

  staged_preds <- catboost.staged_predict(model, pool, ntree_start = 2, ntree_end = 5, eval_period = 2)

  expect_equal(prediction_first, staged_preds$nextElem())
  expect_equal(prediction_second, staged_preds$nextElem())
})

test_that("model: catboost.predict vs stats::predict", {
  pool_path <- system.file("extdata", "adult_train.1000", package = "catboostr")
  column_description_path <- system.file("extdata", "adult.cd", package = "catboostr")

  pool <- catboost.load_pool(pool_path, column_description = column_description_path)

  params <- list(iterations = 10,
                 loss_function = "Logloss",
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  prediction_first <- catboost.predict(model, pool)
  prediction_second <- stats::predict(model, pool)

  expect_equal(prediction_first, prediction_second)
})

test_that("model: save/load by R", {
  train_path <- system.file("extdata", "adult_train.1000", package = "catboostr")
  test_path <- system.file("extdata", "adult_test.1000", package = "catboostr")
  cd_path <- system.file("extdata", "adult.cd", package = "catboostr")
  train_pool <- catboost.load_pool(train_path, column_description = cd_path)
  test_pool <- catboost.load_pool(test_path, column_description = cd_path)
  fit_params <- list(iterations = 4, thread_count = 1, loss_function = "Logloss",
                     allow_writing_files = FALSE)

  model <- catboost.train(train_pool, params = fit_params)
  prediction <- catboost.predict(model, test_pool)

  model_path <- "tmp.rda"
  save(model, file = model_path)
  model <- NULL
  load(model_path)

  prediction_after_save_load <- catboost.predict(model, test_pool)
  expect_equal(prediction, prediction_after_save_load)

  unlink(model_path)
})

test_that("model: saveRDS/readRDS by R", {
  train_path <- system.file("extdata", "adult_train.1000", package = "catboostr")
  test_path <- system.file("extdata", "adult_test.1000", package = "catboostr")
  cd_path <- system.file("extdata", "adult.cd", package = "catboostr")
  train_pool <- catboost.load_pool(train_path, column_description = cd_path)
  test_pool <- catboost.load_pool(test_path, column_description = cd_path)
  fit_params <- list(iterations = 4, thread_count = 1, loss_function = "Logloss",
                     allow_writing_files = FALSE)

  model <- catboost.train(train_pool, params = fit_params)
  prediction <- catboost.predict(model, test_pool)

  model_path <- "tmp.rds"

  saveRDS(model, model_path)
  model <- readRDS(model_path)

  prediction_after_save_load <- catboost.predict(model, test_pool)
  expect_equal(prediction, prediction_after_save_load)

  unlink(model_path)
})

test_that("model: catboost.cv", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 use_best_model = FALSE,
                 allow_writing_files = FALSE)

  fold_count <- 5
  cv_result <- catboost.cv(pool = pool, params = params, fold_count = fold_count)
  print(cv_result)

  expect_true(all(cv_result$train.Logloss.std >= 0))
  expect_true(all(cv_result$test.Logloss.std >= 0))

  expect_true(all(cv_result$train.Logloss.mean >= 0))
  expect_true(all(cv_result$test.Logloss.mean >= 0))
})

test_that("model: catboost.cv with eval_metric=AUC", {
  dataset <- matrix(c(sample(1:100, 20, T),
                      sample(1:100, 20, T),
                      sample(1:100, 20, T)),
                    nrow = 20,
                    ncol = 3,
                    byrow = TRUE)

  label_values <- sample(0:1, 20, T)
  pool <- catboost.load_pool(apply(dataset, 2, as.numeric), label = label_values)
  cv_result <- catboost.cv(pool,
                           params = list(iterations = 10, loss_function = 'Logloss', eval_metric='AUC',
                                         allow_writing_files = FALSE))
  print(cv_result)

  expect_true(all(cv_result$train.Logloss.std >= 0))
  expect_true(all(cv_result$test.Logloss.std >= 0))

  expect_true(all(cv_result$train.Logloss.mean >= 0))
  expect_true(all(cv_result$test.Logloss.mean >= 0))

  expect_true(all(cv_result$test.AUC.std >= 0))
  expect_true(all(cv_result$test.AUC.mean >= 0))
})

# catboost-8z4.NN (flag:--cv row): catboost.cv's `type` argument (Classical/
# Inverted/TimeSeries) is the R-side equivalent of the CLI's `--cv
# type:n;k` descriptor's type component -- this test proves the argument is
# real (reaches native code and changes the fold split) using only R, no CLI
# binary dependency. It deliberately does NOT compare against a CLI oracle:
# investigation (tools/oracle/cli's pinned v1.2.10 binary, `fit --cv
# Classical:{0,1,2};3 --cv-no-shuffle` on this same dataset) showed CLI's
# `--cv` flag (fit/eval-feature/model-based-eval/select-features) is parsed
# into TCvDataPartitionParams at data-load time to pick a single train/test
# split, a different native code path from the TCrossValidationParams-based
# CrossValidate() that catboost.cv binds to (which builds and averages ALL k
# folds) -- so the matrix's matched_r_symbol=catboost.cv is a name-based
# match, not a mechanism match, and flag:--cv stays red (see
# tests/fixtures/parity/closure_overlay.json).
test_that("model: catboost.cv's type argument (Classical vs Inverted) changes the fold split", {
  target <- sample(c(1, -1), size = 60, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))
  pool <- catboost.load_pool(features, target)

  params <- list(iterations = 5,
                 depth = 2,
                 loss_function = "Logloss",
                 random_seed = 42,
                 thread_count = 1,
                 use_best_model = FALSE,
                 allow_writing_files = FALSE)

  cv_classical <- catboost.cv(pool = pool, params = params, fold_count = 3,
                              type = "Classical", shuffle = FALSE)
  cv_inverted <- catboost.cv(pool = pool, params = params, fold_count = 3,
                             type = "Inverted", shuffle = FALSE)

  # Classical trains on k-1 folds and tests on 1 (small test set); Inverted
  # trains on 1 fold and tests on k-1 (large test set) -- same data, same
  # seed, only `type` differs, so an identical result would mean the
  # argument never reached native code.
  expect_false(isTRUE(all.equal(cv_classical$test.Logloss.mean,
                                cv_inverted$test.Logloss.mean)))
  expect_true(all(cv_classical$test.Logloss.std >= 0))
  expect_true(all(cv_inverted$test.Logloss.std >= 0))
})

# catboost-8z4.78 -- closing the catboost.cv parity-matrix row surfaced by
# catboost-8z4.66's pipeline fix. Python's top-level cv() (core.py) calls
# _cv() (_catboost.pyx:6287), which builds a TCrossValidationParams and calls
# CrossValidate() (catboost/libs/train_lib/cross_validation.cpp) directly.
# R's catboost.cv does the same: CatBoostCV_R (src/catboostr.cpp) builds its
# own TCrossValidationParams and calls the identical CrossValidate() entry
# point. Both wrappers are thin field-for-field constructors around the same
# native call, so the per-iteration test/train mean/std columns are expected
# to agree to within cross-architecture floating-point noise (same precedent
# as catboost-8z4.59's grid_search/randomized_search oracle test,
# test_grid_search.R).
#
# This is a *different* capability from the CLI's `--cv` flag (flag:--cv
# row, catboost-8z4.76 / closure_overlay.json): that flag is parsed into a
# distinct TCvDataPartitionParams struct picking a single train/test split,
# a different native code path entirely. This test is Python-cv() vs
# R-catboost.cv only.
#
# Regenerate fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_cv_fixture.py
test_that("model: catboost.cv matches the Python oracle (catboost-8z4.78)", {
  # Cross-architecture FP bound (catboost-8z4.112). The fixture is generated by
  # Python CatBoost on x86_64; macOS CI runs arm64, where the compiler is free
  # to contract multiply-add pairs (FMA) in the boosting/metric accumulation.
  # The columns therefore agree to a few ULP of the *Logloss* value rather than
  # bit-for-bit: worst observed deviation on macos-14 is 4.4e-08 absolute (CI
  # run 31280833801, both macOS jobs), while both x86_64 Linux jobs are exact.
  #
  # The bound is stated in absolute Logloss units because that is the scale the
  # divergence lives on. The std columns are ~1e-2, i.e. the same ~1e-8 absolute
  # noise shows up there as a ~2.6e-06 *relative* difference (cancellation
  # between near-equal fold losses), so a relative tolerance covering it would
  # have to be far looser than what the mean columns actually need.
  # 1e-6 keeps ~20x headroom over the observed divergence and is the same
  # oracle-comparison magnitude already used across the differential suite
  # (TOL in test_compare.R, test_eval_metrics_classes.R, ...).
  ARCH_TOL <- 1e-6

  expect_cv_col <- function(actual, expected, label) {
    expect_length(actual, length(expected))
    expect_lt(max(abs(actual - expected)), ARCH_TOL, label = paste0("max|delta| ", label))
  }

  fixture <- jsonlite::fromJSON(
    testthat::test_path("..", "fixtures", "oracle", "cv.json"),
    simplifyVector = TRUE
  )

  pool <- catboost.load_pool(as.data.frame(fixture$inputs$features), label = fixture$inputs$label)
  params <- as.list(fixture$base_params)
  params$verbose <- NULL
  params$logging_level <- "Silent"

  check_cv <- function(type, expected) {
    result <- catboost.cv(pool,
                           params = params,
                           fold_count = fixture$fold_count,
                           type = type,
                           partition_random_seed = fixture$partition_random_seed,
                           shuffle = fixture$shuffle,
                           stratified = fixture$stratified)

    # data.frame(result) converts CatBoostCV_R's "test-Logloss-mean"-style
    # names to syntactic R names ("test.Logloss.mean") via check.names=TRUE
    # (see test_model.R's earlier catboost.cv tests, e.g.
    # cv_result$train.Logloss.std); the fixture's JSON keys keep the native
    # dashes, so they are indexed with `[[` here instead.
    expect_cv_col(result$test.Logloss.mean, expected[["test-Logloss-mean"]],
                  paste(type, "test-Logloss-mean"))
    expect_cv_col(result$test.Logloss.std, expected[["test-Logloss-std"]],
                  paste(type, "test-Logloss-std"))
    expect_cv_col(result$train.Logloss.mean, expected[["train-Logloss-mean"]],
                  paste(type, "train-Logloss-mean"))
    expect_cv_col(result$train.Logloss.std, expected[["train-Logloss-std"]],
                  paste(type, "train-Logloss-std"))
  }

  check_cv("Classical", fixture$expected$classical)
  check_cv("Inverted", fixture$expected$inverted)
})

test_that("model: catboost.sum_models", {
    target_train <- sample(c(0, 1), size = 20, replace = TRUE)
    features_train <- data.frame(f1 = rnorm(length(target_train), mean = 0, sd = 1),
                                 f2 = rnorm(length(target_train), mean = 0, sd = 1))

    target_test <- sample(c(0, 1, 3, 4), size = 10, replace = TRUE)
    features_test <- data.frame(f1 = rnorm(length(target_test), mean = 0, sd = 1),
                                f2 = rnorm(length(target_test), mean = 0, sd = 1))
    pool_train <- catboost.load_pool(features_train, target_train)
    pool_test <- catboost.load_pool(features_test, target_test)
    params <- list(iterations=100,
                   depth=4,
                   allow_writing_files = FALSE)
    model_train <- catboost.train(pool_train, pool_test, params)
    model_test <- catboost.train(pool_train, pool_test, params)

    list_models <- list(model_train, model_test)
    sum_mod <- catboost.sum_models(list_models, weights=rep(1.0 / length(list_models), length(list_models)))

    prediction_sum_models <- catboost.predict(model_train, pool_test, prediction_type='RawFormulaVal')
    prediction_one_model <- catboost.predict(sum_mod, pool_test, prediction_type='RawFormulaVal')
    expect_equal(prediction_sum_models, prediction_one_model)
})

# catboost-8z4.78 -- closing the catboost.sum_models parity-matrix row
# surfaced by catboost-8z4.66's pipeline fix. Python's top-level
# sum_models() (core.py) constructs a CatBoost() and calls its
# _sum_models() (_catboost.pyx:5884), which calls SumModels() directly on
# the input models' native TFullModel objects. R's catboost.sum_models does
# the same: CatBoostSumModels_R (src/catboostr.cpp) calls the identical
# SumModels() entry point on its own loaded TFullModel handles. Because
# SumModels() operates purely on already-trained model artifacts (not on
# training data), this test loads the *same* two .cbm files the fixture's
# generator trained and summed in Python -- isolating the comparison to
# SumModels() itself rather than conflating it with training parity
# (already covered separately by test_params_precision.R).
#
# Regenerate fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_sum_models_fixture.py
test_that("model: catboost.sum_models matches the Python oracle bit-for-bit (catboost-8z4.78)", {
  fixture <- jsonlite::fromJSON(
    testthat::test_path("..", "fixtures", "oracle", "sum_models.json"),
    simplifyVector = TRUE
  )

  model_a <- catboost.load_model(
    testthat::test_path("..", "fixtures", "oracle", fixture$model_a_file))
  model_b <- catboost.load_model(
    testthat::test_path("..", "fixtures", "oracle", fixture$model_b_file))

  summed <- catboost.sum_models(list(model_a, model_b),
                                 weights = fixture$weights,
                                 ctr_merge_policy = fixture$ctr_merge_policy)

  pred_pool <- catboost.load_pool(as.data.frame(fixture$inputs$features))
  predictions <- catboost.predict(summed, pred_pool)

  expect_equal(as.numeric(predictions), fixture$expected$predictions, tolerance = 1e-12)
})

test_that("model: catboost.eval_metrics", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  pool_train <- catboost.load_pool(features[split, ], target[split])
  pool_test <- catboost.load_pool(features[-split, ], target[-split])

  metric <- 'AUC'
  params <- list(iterations = 10,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 eval_metric = metric,
                 use_best_model = FALSE)

  model <- catboost.train(pool_train, pool_test, params)
  train_metrics <- read.table(file = 'catboost_info/test_error.tsv', sep = '\t', header = TRUE)
  train_metric <- train_metrics[[metric]]

  eval_metrics <- catboost.eval_metrics(model, pool_test, metric)
  eval_metric <- eval_metrics[[metric]]
  expect_equal(train_metric, eval_metric, tolerance = 1e-9)

  eval_metrics <- catboost.eval_metrics(model, pool_test, list(metric))
  eval_metric <- eval_metrics[[metric]]
  expect_equal(train_metric, eval_metric, tolerance = 1e-9)

  eval_metrics <- catboost.eval_metrics(model, pool_test, list(metric), ntree_end = 500)
  eval_metric <- eval_metrics[[metric]]
  expect_equal(train_metric, eval_metric, tolerance = 1e-9)

  eval_metrics <- catboost.eval_metrics(model, pool_test, list(metric), ntree_start = 1, ntree_end = 5, eval_period = 2)
  eval_metric <- eval_metrics[[metric]]
  expect_equal(3, length(eval_metric))
  expect_equal(c(train_metric[2], train_metric[4], train_metric[5]), eval_metric, tolerance = 1e-9)

  eval_metrics <- catboost.eval_metrics(model, pool_test, list(metric), ntree_start = 1, ntree_end = 5, eval_period = 3)
  eval_metric <- eval_metrics[[metric]]
  expect_equal(2, length(eval_metric))
  expect_equal(c(train_metric[2], train_metric[5]), eval_metric, tolerance = 1e-9)

  eval_metrics <- catboost.eval_metrics(model, pool_test, list(metric), ntree_start = 1, ntree_end = 5, eval_period = 15)
  eval_metric <- eval_metrics[[metric]]
  expect_equal(2, length(eval_metric))
  expect_equal(c(train_metric[2], train_metric[5]), eval_metric, tolerance = 1e-9)

  expect_error( catboost.eval_metrics(model, pool_test, list()),
               "No metrics found.", fixed = TRUE)
  expect_error( catboost.eval_metrics(model, pool_test, list(metric), ntree_start = 500),
                "ntree_start should be less than ntree_end.", fixed = TRUE)
  expect_error( catboost.eval_metrics(model, pool_test, list(), ntree_start = -1),
                "ntree_start should be greater or equal zero.", fixed = TRUE)
  expect_error( catboost.eval_metrics(model, pool_test, list(), eval_period = -1),
                "eval_period should be greater than zero.", fixed = TRUE)

  unlink('catboost_info', recursive = TRUE)
})

test_that("model: catboost.virtual_ensembles_predict", {
  docs_count <- 100
  virtual_ensembles_count <- 10
  params <- list(iterations = 60,
                 random_seed = 12345,
                 logging_level = "Silent",
                 allow_writing_files = FALSE)
  target <- sample(c(1, -1), size = docs_count, replace = TRUE)
  features <- data.frame(f1 = rnorm(docs_count, mean = 0, sd = 1),
                         f2 = rnorm(docs_count, mean = 0, sd = 1))
  train_pool <- catboost.load_pool(features, target)
  test_pool <- train_pool


  params[['loss_function']] <- 'Logloss'
  model <- catboost.train(train_pool, params = params)
  expect_error( catboost.virtual_ensembles_predict(model, test_pool, prediction_type = "Probability"),
                "Unsupported virtual ensembles prediction type: 'VirtEnsembles' or 'TotalUncertainty' was expected", fixed = TRUE)

  prediction <- catboost.virtual_ensembles_predict(model, test_pool, prediction_type = "TotalUncertainty",
                                                   virtual_ensembles_count = virtual_ensembles_count)
  expect_equal(dim(prediction), c(docs_count, 2))


  params[['loss_function']] <- 'RMSE'
  model <- catboost.train(train_pool, params = params)
  prediction <- catboost.virtual_ensembles_predict(model, test_pool, prediction_type = "TotalUncertainty",
                                                   virtual_ensembles_count = virtual_ensembles_count)
  expect_equal(dim(prediction), c(docs_count, 2))


  params[['loss_function']] <- 'RMSEWithUncertainty'
  model <- catboost.train(train_pool, params = params)
  prediction <- catboost.virtual_ensembles_predict(model, test_pool, prediction_type = "TotalUncertainty",
                                                   virtual_ensembles_count = virtual_ensembles_count)
  expect_equal(dim(prediction), c(docs_count, 3))


  params[['loss_function']] <- 'RMSEWithUncertainty'
  model <- catboost.train(train_pool, params = params)
  prediction <- catboost.virtual_ensembles_predict(model, test_pool, prediction_type = "VirtEnsembles",
                                                   virtual_ensembles_count = virtual_ensembles_count)
  expect_equal(dim(prediction), c(virtual_ensembles_count, 2, docs_count))


  params[['loss_function']] <- 'Logloss'
  model <- catboost.train(train_pool, params = params)
  prediction <- catboost.virtual_ensembles_predict(model, test_pool, prediction_type = "VirtEnsembles",
                                                   virtual_ensembles_count = virtual_ensembles_count)
  expect_equal(dim(prediction), c(virtual_ensembles_count, 1, docs_count))
})

test_that("model: feature_count attribute", {
  docs_count <- 30

  params <- list(iterations = 20,
                random_seed = 12345,
                logging_level = "Silent",
                allow_writing_files = FALSE,
                loss_function = "RMSE")
  target <- sample(c(1, -1), size = docs_count, replace = TRUE)
  features <- data.frame(f1 = rnorm(docs_count, mean = 0, sd = 1),
                         f2 = rnorm(docs_count, mean = 1, sd = 1),
                         f3 = rnorm(docs_count, mean = 2, sd = 1),
                         f4 = rnorm(docs_count, mean = 3, sd = 1))
  feature_count <- ncol(features)
  group_id <- as.integer(c(rep.int(0, docs_count / 2), rep.int(1, docs_count / 2)))

  train_pool <- catboost.load_pool(features, target)
  model <- catboost.train(train_pool, params = params)
  expect_equal(model$feature_count, feature_count)

  params[['loss_function']] <- "YetiRank"
  train_pool <- catboost.load_pool(features, target, group_id = group_id)
  model <- catboost.train(train_pool, params = params)
  expect_equal(model$feature_count, feature_count)

  # feature count with ignored features
  params$ignored_features <- c(0)
  train_pool <- catboost.load_pool(features, target, group_id = group_id)
  model <- catboost.train(train_pool, params = params)
  expect_equal(model$feature_count, feature_count - 1)
})
