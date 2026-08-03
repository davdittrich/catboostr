context("test_compare.R")

# P4.2 (catboost-8z4.51) structural differential test: catboost.compare
# (R/catboost.R). Python's CatBoost.compare(model, data, metrics, ...) only
# draws an interactive Jupyter widget from the data it computes internally
# (self._eval_metrics(...) and model._eval_metrics(...) on the same
# pool/metrics -- catboost/python-package/catboost/core.py:3345-3349) and
# itself returns None: no plot/widget comparison is possible headlessly
# (spec Sec 4.3's "Structural" row explicitly excludes widget rendering).
# catboost.compare() exposes that underlying metrics-diff data structure
# instead, computed the same way, via catboost.eval_metrics() per model --
# both funnel into the same TMetricsPlotCalcer vendor entry point (native
# CatBoostEvalMetrics_R / catboost/libs/metrics) the pinned Python
# catboost==1.2.10 model.eval_metrics() call uses internally, so this test
# compares canonical-JSON field-by-field (each metric's per-iteration vector,
# for both `model` and `other`) against the recorded oracle fixture, within
# tolerance.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_compare_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "compare.json"),
  simplifyVector = TRUE
)
inputs <- fixture$inputs
expected <- fixture$expected

build_pool_and_models <- function() {
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = list(
    iterations = 10, depth = 2, loss_function = "Logloss",
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
  other <- catboost.train(pool, params = list(
    iterations = 10, depth = 4, loss_function = "Logloss",
    random_seed = 7, thread_count = 1, logging_level = "Silent"
  ))
  list(pool = pool, model = model, other = other)
}

fixture_env <- build_pool_and_models()
pool <- fixture_env$pool
model <- fixture_env$model
other <- fixture_env$other

TOL <- 1e-6

test_that("compare: metrics for both models match Python oracle", {
  result <- catboost.compare(model, other, pool, as.list(inputs$metrics))

  expect_s3_class(result, "catboost.compare")
  expect_setequal(names(result), c("model", "other"))

  for (side in c("model", "other")) {
    expect_setequal(names(result[[side]]), inputs$metrics)
    for (metric_name in inputs$metrics) {
      expect_equal(
        as.numeric(result[[side]][[metric_name]]),
        as.numeric(expected[[side]][[metric_name]]),
        tolerance = TOL
      )
    }
  }
})

test_that("compare: rejects a NULL model", {
  expect_error(
    catboost.compare(NULL, other, pool, as.list(inputs$metrics)),
    "You should provide a model"
  )
})

test_that("compare: rejects a NULL other model", {
  expect_error(
    catboost.compare(model, NULL, pool, as.list(inputs$metrics)),
    "You should provide another model"
  )
})

test_that("compare: rejects a NULL pool", {
  expect_error(
    catboost.compare(model, other, NULL, as.list(inputs$metrics)),
    "You should provide data"
  )
})

test_that("compare: rejects NULL metrics", {
  expect_error(
    catboost.compare(model, other, pool, NULL),
    "You should provide metrics"
  )
})

test_that("compare: rejects a non-model `other`", {
  expect_error(
    catboost.compare(model, "not-a-model", pool, as.list(inputs$metrics)),
    "Expected catboost.Model"
  )
})
