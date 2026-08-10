context("test_fstr_sage_values.R")

# catboost-8z4.125: catboost.get_feature_importance(type = "SageValues") was
# never wired into the dispatch chain (R/catboost.R) even though the native
# entry point (src/catboostr.cpp CatBoostCalcRegularFeatureEffect_R ->
# GetFeatureImportances() -> calc_fstr.cpp EFstrType::SageValues ->
# CalcSageValues(), vendor/catboost/catboost/libs/fstr/sage_values.cpp)
# already handles EFstrType::SageValues unconditionally. The .Call() was
# already reached for type = "SageValues" (no gating on the native call
# itself); only the R-side post-processing switch lacked a branch and fell
# through to stop("Unknown type: ", type), discarding an already-computed
# result. This test is structural/self-consistency only (no Python oracle
# fixture): SAGE values use an internally fixed RNG seed (228) in
# sage_values.cpp, so results are deterministic for a fixed model/pool/build,
# but are not a stable cross-version oracle target.

data <- data.frame(num1 = c(1, 2, 3, 4, 5, 6, 7, 8),
                    num2 = c(8, 7, 6, 5, 4, 3, 2, 1))
label <- c(0, 0, 0, 1, 0, 1, 1, 1)
pool <- catboost.load_pool(data, label = label)

model <- catboost.train(pool, params = list(
  iterations = 10, depth = 2, loss_function = "Logloss",
  random_seed = 42, thread_count = 1, logging_level = "Silent"
))

test_that("get_feature_importance: SageValues is wired and returns one score per feature", {
  result <- catboost.get_feature_importance(model, pool, type = "SageValues")

  expect_equal(dim(result), c(ncol(data), 1))
  expect_equal(rownames(result), colnames(data))
  expect_true(all(is.finite(as.numeric(result))))
})

test_that("get_feature_importance: SageValues requires a pool", {
  expect_error(
    catboost.get_feature_importance(model, type = "SageValues"),
    "pool is required"
  )
})

test_that("get_feature_importance: SageValues rejects multiclass models", {
  multiclass_label <- c(0, 1, 2, 0, 1, 2, 0, 1)
  multiclass_pool <- catboost.load_pool(data, label = multiclass_label)
  multiclass_model <- catboost.train(multiclass_pool, params = list(
    iterations = 10, depth = 2, loss_function = "MultiClass",
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))

  expect_error(
    catboost.get_feature_importance(multiclass_model, multiclass_pool, type = "SageValues")
  )
})
