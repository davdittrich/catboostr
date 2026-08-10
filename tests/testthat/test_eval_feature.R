context("test_eval_feature.R")

# P4.7 (catboost-8z4.56) differential test: catboost.eval_feature() is the R
# equivalent of the CatBoost CLI's `eval-feature` mode. It calls the same core
# entry point the CLI mode calls (EvaluateFeatures,
# catboost/libs/train_lib/eval_feature.h -- see
# vendor/catboost/catboost/app/mode_eval_feature.cpp), in process, so this test
# compares R's output against the pinned CLI v1.2.10 oracle summary committed
# at tests/fixtures/oracle-cli/eval_feature_summary.tsv.
#
# The Python package has no counterpart for this mode (its nearest name,
# CatBoost.select_features, is a different algorithm and Phase 5 scope), so the
# CLI is the sole oracle. Per spec section 4.3 that means relative tolerance
# 1e-9 and no bit-exact assertion: the CLI's summary TSV prints ~10 significant
# digits and eval-feature has no --precision flag. Justification is recorded in
# tests/fixtures/oracle-cli/eval_feature_metadata.json.
#
# Regenerate the oracle fixture with:
#   tools/oracle/cli/acquire.sh
#   tools/oracle/cli/gen_eval_feature_fixture.sh

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")
ORACLE_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "eval_feature_summary.tsv")

# Relative tolerance for the CLI path (spec section 4.3). testthat's tolerance
# is relative for non-zero expected values.
CLI_TOLERANCE <- 1e-9

# Mirrors gen_eval_feature_fixture.sh's command exactly.
EVAL_PARAMS <- list(
  loss_function = "Logloss",
  iterations = 20,
  depth = 4,
  learning_rate = 0.1,
  random_seed = 42,
  thread_count = 1,
  logging_level = "Silent"
)

oracle <- read.delim(ORACLE_PATH, sep = "\t", check.names = FALSE,
                     colClasses = "character")

load_smoke_pool <- function() {
  catboost.load_pool(POOL_PATH, column_description = CD_PATH,
                     delimiter = ",", has_header = TRUE, thread_count = 1)
}

# eval-feature writes per-fold training directories; keep them out of the
# package tree, exactly as the fixture generator runs in a temp directory.
run_eval_feature <- function(...) {
  train_dir <- file.path(tempdir(), paste0("eval_feature_", as.integer(runif(1, 0, 1e9))))
  dir.create(train_dir, showWarnings = FALSE, recursive = TRUE)
  on.exit(unlink(train_dir, recursive = TRUE), add = TRUE)
  params <- EVAL_PARAMS
  params$train_dir <- train_dir
  catboost.eval_feature(load_smoke_pool(), params = params, ...)
}

test_that("eval_feature: summary matches the CatBoost CLI oracle", {
  res <- run_eval_feature(
    features_to_evaluate = list(0L, 1L),
    eval_mode = "OneVsAll",
    offset = 0,
    fold_count = 2,
    fold_size_unit = "Object",
    fold_size = 10
  )

  # Structure: one row per tested feature set, one metric (Logloss).
  expect_equal(nrow(oracle), 2L)
  expect_equal(length(res$p_value), nrow(oracle))
  expect_equal(res$metric_names, "Logloss")
  expect_equal(dim(res$metric_delta), c(2L, 1L))

  # Numeric columns: relative 1e-9 against the CLI's own printed digits.
  expect_equal(res$p_value, as.numeric(oracle[["p-value"]]),
               tolerance = CLI_TOLERANCE)
  expect_equal(as.numeric(res$metric_delta[, 1]), as.numeric(oracle[["Logloss"]]),
               tolerance = CLI_TOLERANCE)

  # Integer columns: the CLI comma-joins them; exact equality, no tolerance.
  expect_equal(vapply(res$best_iterations, paste, character(1), collapse = ","),
               oracle[["best iteration in each fold"]])
  expect_equal(vapply(res$feature_sets, paste, character(1), collapse = ","),
               oracle[["feature set"]])
})

test_that("eval_feature: timesplit_quantile matches the CatBoost CLI oracle", {
  # catboost-inm (followup to catboost-8z4.115/P10.A): closes
  # flag:--timesplit-quantile. P10.A fix-round-1's justification named the
  # wrong trigger (assumed a nonexistent "--feature-eval-mode TimeSplit"
  # value); verified in catboost/libs/train_lib/eval_feature.cpp:910 that
  # the real trigger is the dataset having a timestamp column
  # (MetaInfo.HasTimestamp), which switches PrepareFolds to
  # PrepareTimeSplitFolds -- the only path that reads TimeSplitQuantile at
  # all. See tools/oracle/cli/gen_eval_feature_timesplit_fixture.sh (8
  # groups, one timestamp per GroupId at 0/10/.../70).
  timesplit_pool_path <- testthat::test_path("..", "fixtures", "oracle-cli", "timesplit_data.csv")
  timesplit_cd_path <- testthat::test_path("..", "fixtures", "oracle-cli", "timesplit_data.cd")
  pool <- catboost.load_pool(timesplit_pool_path, column_description = timesplit_cd_path,
                             delimiter = ",", has_header = TRUE, thread_count = 1)
  timestamps_by_group <- as.matrix(read.delim(
    testthat::test_path("..", "fixtures", "oracle-cli", "timesplit_timestamps.tsv"),
    header = FALSE
  ))
  group_id <- read.csv(timesplit_pool_path)$group_id
  catboost.pool.set_timestamp(pool, timestamps_by_group[match(group_id, timestamps_by_group[, 1]), 2])

  run_timesplit <- function(quantile) {
    train_dir <- file.path(tempdir(), paste0("eval_feature_timesplit_", as.integer(runif(1, 0, 1e9))))
    dir.create(train_dir, showWarnings = FALSE, recursive = TRUE)
    on.exit(unlink(train_dir, recursive = TRUE), add = TRUE)
    catboost.eval_feature(
      pool,
      features_to_evaluate = list(0L, 1L),
      params = list(loss_function = "RMSE", iterations = 10, depth = 3, learning_rate = 0.1,
                    random_seed = 42, thread_count = 1, logging_level = "Silent", train_dir = train_dir),
      eval_mode = "OneVsAll", offset = 0, fold_count = 2, fold_size_unit = "Group", fold_size = 1,
      timesplit_quantile = quantile
    )
  }

  for (spec in list(list(q = 0.5, file = "eval_feature_timesplit_q25.tsv"),
                    list(q = 0.75, file = "eval_feature_timesplit_q75.tsv"))) {
    oracle <- read.delim(testthat::test_path("..", "fixtures", "oracle-cli", spec$file),
                         sep = "\t", check.names = FALSE, colClasses = "character")
    res <- run_timesplit(spec$q)

    expect_equal(res$p_value, as.numeric(oracle[["p-value"]]),
                tolerance = CLI_TOLERANCE, info = spec$file)
    expect_equal(as.numeric(res$metric_delta[, 1]), as.numeric(oracle[["RMSE"]]),
                tolerance = CLI_TOLERANCE, info = spec$file)
    expect_equal(vapply(res$best_iterations, paste, character(1), collapse = ","),
                oracle[["best iteration in each fold"]], info = spec$file)
  }
})

test_that("eval_feature: rejects an out-of-range feature index", {
  expect_error(
    run_eval_feature(features_to_evaluate = list(99L), eval_mode = "OneVsAll",
                     fold_count = 2, fold_size = 10),
    "is not present"
  )
})

test_that("eval_feature: rejects duplicate feature sets", {
  expect_error(
    run_eval_feature(features_to_evaluate = list(0L, 0L), eval_mode = "OneVsAll",
                     fold_count = 2, fold_size = 10),
    "must be different"
  )
})

test_that("eval_feature: rejects an unknown feature evaluation mode", {
  expect_error(
    run_eval_feature(features_to_evaluate = list(0L), eval_mode = "Nonsense",
                     fold_count = 2, fold_size = 10),
    "unsupported feature evaluation mode"
  )
})

test_that("eval_feature: rejects a tested feature that is also ignored", {
  # Fix round 1: mode_eval_feature.cpp errors on this
  # ("Tested feature N should not be ignored"), and the core does not re-check
  # it -- it would silently evaluate a feature that was dropped from the model.
  params <- EVAL_PARAMS
  params$ignored_features <- 1

  expect_error(
    catboost.eval_feature(load_smoke_pool(), features_to_evaluate = list(1L),
                          params = params, eval_mode = "OneVsAll",
                          fold_count = 2, fold_size = 10),
    "should not be ignored"
  )

  # By feature name, the other form the option accepts. Only cat1 is named in
  # smoke.cd, so it is the one feature of this fixture addressable by name.
  params$ignored_features <- "cat1"
  expect_error(
    catboost.eval_feature(load_smoke_pool(), features_to_evaluate = list(2L),
                          params = params, eval_mode = "OneVsAll",
                          fold_count = 2, fold_size = 10),
    "should not be ignored"
  )

  # Negative control: an ignored feature that is not tested stays legal. This
  # also pins the converter fix -- R serialises ignored_features as strings, and
  # EvaluateFeatures (unlike TrainModel/CrossValidate) does not resolve them
  # itself, so before the fix this died with `Can't parse parameter
  # "ignored_features"` for every value, not just clashing ones.
  params$ignored_features <- 2
  params$train_dir <- file.path(tempdir(), "eval_feature_ignored_ok")
  on.exit(unlink(params$train_dir, recursive = TRUE), add = TRUE)
  expect_error(
    catboost.eval_feature(load_smoke_pool(), features_to_evaluate = list(1L),
                          params = params, eval_mode = "OneVsAll",
                          fold_count = 2, fold_size = 10),
    NA
  )
})

test_that("eval_feature: rejects a negative offset and a non-positive fold_count", {
  pool <- load_smoke_pool()
  expect_error(
    catboost.eval_feature(pool, features_to_evaluate = list(0L), params = EVAL_PARAMS,
                          eval_mode = "OneVsAll", offset = -1, fold_count = 2,
                          fold_size = 10),
    "offset must be non-negative"
  )
  expect_error(
    catboost.eval_feature(pool, features_to_evaluate = list(0L), params = EVAL_PARAMS,
                          eval_mode = "OneVsAll", fold_count = 0, fold_size = 10),
    "fold_count must be positive"
  )
})

test_that("eval_feature: requires exactly one of fold_size and relative_fold_size", {
  pool <- load_smoke_pool()
  expect_error(
    catboost.eval_feature(pool, features_to_evaluate = list(0L), params = EVAL_PARAMS,
                          eval_mode = "OneVsAll", fold_count = 2),
    "Exactly one of fold_size and relative_fold_size"
  )
  expect_error(
    catboost.eval_feature(pool, features_to_evaluate = list(0L), params = EVAL_PARAMS,
                          eval_mode = "OneVsAll", fold_count = 2,
                          fold_size = 10, relative_fold_size = 0.25),
    "Exactly one of fold_size and relative_fold_size"
  )
})
