context("test_eval_metrics.R")

# mode:eval-metrics (catboost-8z4.77) differential test: catboost.eval_metrics()
# is R's equivalent of CatBoost CLI's `eval-metrics` mode. Both drive the same
# core entry point (catboost/private/libs/algo/plot.h's TMetricsPlotCalcer --
# see src/catboostr.cpp's CatBoostEvalMetrics_R and
# vendor/catboost/catboost/app/mode_eval_metrics.cpp's Run()), so this test
# compares R's output against the pinned CLI v1.2.10 oracle TSV committed at
# tests/fixtures/oracle-cli/eval_metrics.tsv.
#
# Per spec section 4.3, CLI path default tolerance is 1e-9, no bit-exact
# assertion: the CLI's TSV prints ~10 significant digits and eval-metrics
# exposes no --precision flag.
#
# Regenerate the oracle fixture with:
#   tools/oracle/cli/acquire.sh
#   tools/oracle/cli/gen_eval_metrics_fixture.sh

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")
ORACLE_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "eval_metrics.tsv")

# Relative tolerance for the CLI path (spec section 4.3).
CLI_TOLERANCE <- 1e-9

oracle <- read.delim(ORACLE_PATH, check.names = FALSE, stringsAsFactors = FALSE)

pool <- catboost.load_pool(POOL_PATH, column_description = CD_PATH, delimiter = ",",
                            has_header = TRUE, thread_count = 1)

# Mirrors gen_eval_metrics_fixture.sh's fit command exactly.
model <- catboost.train(pool, params = list(
  loss_function = "Logloss", iterations = 20, depth = 4,
  learning_rate = 0.1, random_seed = 42, thread_count = 1,
  logging_level = "Silent"
))

test_that("eval_metrics: per-iteration Logloss/AUC match CatBoost CLI oracle", {
  res <- catboost.eval_metrics(model, pool, metrics = list("Logloss", "AUC"),
                                ntree_start = 0L, ntree_end = 0L, eval_period = 5,
                                thread_count = 1)

  expect_equal(names(res), c("Logloss", "AUC"))
  expect_equal(length(res$Logloss), nrow(oracle))
  expect_equal(length(res$AUC), nrow(oracle))

  expect_equal(res$Logloss, oracle[["Logloss"]], tolerance = CLI_TOLERANCE)
  expect_equal(res$AUC, oracle[["AUC"]], tolerance = CLI_TOLERANCE)
})

test_that("eval_metrics: rejects non-Pool second argument", {
  expect_error(
    catboost.eval_metrics(model, list(), metrics = "Logloss"),
    "Expected catboost.Pool"
  )
})
