context("test_eval_metric.R")

# catboost-8z4.124 differential test: catboost.eval_metric() (raw label/approx
# array API, R equivalent of Python's catboost.utils.eval_metric()) must
# agree with catboost.eval_metrics() (model+Pool API) when evaluated on the
# same model's raw predictions over the same pool -- both ultimately compute
# the same IMetric::GetFinalError() over the same stats (src/catboostr.cpp's
# CatBoostEvalMetric_R vs CatBoostEvalMetrics_R), so they must match to
# floating-point precision, not just some tolerance-band oracle.

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")

pool <- catboost.load_pool(POOL_PATH, column_description = CD_PATH, delimiter = ",",
                            has_header = TRUE, thread_count = 1)

# catboost.pool.get_label() rejects this pool's label (file-loaded target
# columns are stored as ERawTargetType::String, see catboostr.cpp's
# CatBoostPoolGetLabel_R -- a pre-existing, unrelated restriction), so read
# the same "target" column straight from the source CSV instead.
csv_label <- read.csv(POOL_PATH, stringsAsFactors = FALSE)$target

model <- catboost.train(pool, params = list(
  loss_function = "Logloss", iterations = 20, depth = 4,
  learning_rate = 0.1, random_seed = 42, thread_count = 1,
  logging_level = "Silent"
))

test_that("catboost.eval_metric matches catboost.eval_metrics on the full ensemble", {
  tree_count <- model$tree_count

  # eval_period = 1 walks the full per-tree learning curve; its last point
  # is the metric using every tree, i.e. the same "full ensemble" value
  # catboost.eval_metric() computes directly from raw predictions below.
  via_model_and_pool <- catboost.eval_metrics(model, pool, "Logloss",
                                               ntree_start = 0, ntree_end = tree_count,
                                               eval_period = 1)
  full_ensemble_value <- tail(via_model_and_pool$Logloss, 1)

  label <- csv_label
  approx <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  via_raw_arrays <- catboost.eval_metric(label, approx, "Logloss")

  expect_equal(length(via_raw_arrays), 1)
  expect_equal(via_raw_arrays[1], full_ensemble_value, tolerance = 1e-9)
})

test_that("catboost.eval_metric rejects a groupwise metric without group_id", {
  label <- csv_label
  approx <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_error(catboost.eval_metric(label, approx, "QueryRMSE"), "requires group data")
})
