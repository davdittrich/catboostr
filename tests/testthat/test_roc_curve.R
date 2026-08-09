context("test_roc_curve.R")

# P4.6 (catboost-8z4.55) differential test: catboost.get_roc_curve()
# (R/catboost.R), the R port of catboost.utils.get_roc_curve() / the CLI's
# `roc` mode, both of which wrap the same C++ engine
# (catboost/private/libs/algo/roc_curve.cpp TRocCurve).
#
# Two matrix rows:
#  - catboost.utils.get_roc_curve (Python oracle, tolerance 1e-12): a model
#    trained in R with the same seed/params as the pinned Python oracle
#    fixture (Phase 0 established bit-exact cross-language determinism), ROC
#    curve compared point-by-point to catboost==1.2.10's observable output.
#  - mode:roc (CLI oracle, tolerance 1e-9): the pinned CatBoost CLI v1.2.10
#    binary's `roc` mode output on the shared smoke fixture
#    (tests/fixtures/oracle-cli/smoke_data.csv + smoke.cd), compared to a
#    model trained in R with the same params.
#
# Regenerate the Python fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_roc_curve_fixture.py
#
# Regenerate the CLI fixture with (after tools/oracle/cli/acquire.sh):
#   ./tools/oracle/cli/bin/catboost-v1.2.10 fit \
#     --learn-set tests/fixtures/oracle-cli/smoke_data.csv \
#     --column-description tests/fixtures/oracle-cli/smoke.cd \
#     --delimiter , --has-header --loss-function Logloss -i 20 --depth 4 \
#     --learning-rate 0.1 --random-seed 42 -T 1 \
#     --model-file /tmp/smoke_model.cbm --logging-level Silent
#   ./tools/oracle/cli/bin/catboost-v1.2.10 calc \
#     --input-path tests/fixtures/oracle-cli/smoke_data.csv \
#     --column-description tests/fixtures/oracle-cli/smoke.cd \
#     --delimiter , --has-header -m /tmp/smoke_model.cbm \
#     --output-path /tmp/eval_result.tsv \
#     --output-columns Label,RawFormulaVal -T 1
#   ./tools/oracle/cli/bin/catboost-v1.2.10 roc \
#     --eval-file /tmp/eval_result.tsv \
#     --output-path tests/fixtures/oracle-cli/roc_data.tsv -T 1

PY_TOL <- 1e-12
CLI_TOL <- 1e-9

## --- catboost.utils.get_roc_curve (Python oracle) ---

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "roc_curve.json"),
  simplifyVector = TRUE
)
inputs <- fixture$inputs
expected <- fixture$expected

py_pool <- catboost.load_pool(
  data.frame(num1 = inputs$num1, num2 = inputs$num2),
  label = inputs$label,
  feature_names = as.list(inputs$feature_names)
)
py_model <- catboost.train(py_pool, params = list(
  iterations = 10, depth = 2, loss_function = "Logloss",
  random_seed = 42, thread_count = 1, logging_level = "Silent"
))

test_that("get_roc_curve: matches the Python oracle", {
  result <- catboost.get_roc_curve(py_model, py_pool)
  expect_equal(result$fpr, expected$fpr, tolerance = PY_TOL)
  expect_equal(result$tpr, expected$tpr, tolerance = PY_TOL)
  expect_equal(result$threshold, expected$thresholds, tolerance = PY_TOL)
})

test_that("get_roc_curve: rejects a non-Model first argument", {
  expect_error(catboost.get_roc_curve(list(), py_pool), "Expected catboost.Model")
})

test_that("get_roc_curve: rejects a non-Pool second argument", {
  expect_error(catboost.get_roc_curve(py_model, list()), "Expected catboost.Pool")
})

test_that("get_roc_curve: accepts a list of pools, concatenating them", {
  half <- nrow(py_pool) %/% 2
  pool_a <- catboost.load_pool(
    data.frame(num1 = inputs$num1[seq_len(half)], num2 = inputs$num2[seq_len(half)]),
    label = inputs$label[seq_len(half)],
    feature_names = as.list(inputs$feature_names)
  )
  pool_b <- catboost.load_pool(
    data.frame(
      num1 = inputs$num1[(half + 1):length(inputs$num1)],
      num2 = inputs$num2[(half + 1):length(inputs$num2)]
    ),
    label = inputs$label[(half + 1):length(inputs$label)],
    feature_names = as.list(inputs$feature_names)
  )
  result_split <- catboost.get_roc_curve(py_model, list(pool_a, pool_b))
  result_whole <- catboost.get_roc_curve(py_model, py_pool)
  expect_equal(result_split, result_whole, tolerance = PY_TOL)
})

test_that("get_roc_curve: rejects a label that rounds outside {0, 1}", {
  bad_label <- inputs$label
  bad_label[1] <- 2 # rounds to 2L, not a valid binary class
  bad_pool <- catboost.load_pool(
    data.frame(num1 = inputs$num1, num2 = inputs$num2),
    label = bad_label,
    feature_names = as.list(inputs$feature_names)
  )
  expect_error(catboost.get_roc_curve(py_model, bad_pool), "rounded label")
})

## --- mode:roc (CLI oracle) ---

cli_data <- read.csv(
  testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv"),
  header = TRUE
)
cli_cd <- read.table(
  testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd"),
  header = FALSE, sep = "\t", stringsAsFactors = FALSE
)
cli_label_col <- cli_cd[cli_cd$V2 == "Label", "V1"] + 1 # 0-indexed in .cd, 1-indexed in R
cli_cat_col <- cli_cd[cli_cd$V2 == "Categ", "V1"] + 1 # 0-indexed in .cd, 1-indexed in R
# catboost.load_pool() ignores cat_features for data.frame input -- categorical
# columns must be factors instead (see R/catboost.R catboost.from_data_frame).
cli_data[[cli_cat_col]] <- as.factor(cli_data[[cli_cat_col]])
cli_pool <- catboost.load_pool(
  cli_data[, -cli_label_col],
  label = cli_data[, cli_label_col]
)
cli_model <- catboost.train(cli_pool, params = list(
  loss_function = "Logloss", iterations = 20, depth = 4,
  learning_rate = 0.1, random_seed = 42, thread_count = 1,
  logging_level = "Silent"
))

cli_expected <- read.table(
  testthat::test_path("..", "fixtures", "oracle-cli", "roc_data.tsv"),
  header = TRUE, sep = "\t"
)

test_that("get_roc_curve: matches the CatBoost CLI `roc` mode oracle", {
  result <- catboost.get_roc_curve(cli_model, cli_pool)
  expect_equal(result$fpr, cli_expected$FPR, tolerance = CLI_TOL)
  expect_equal(result$tpr, cli_expected$TPR, tolerance = CLI_TOL)
  expect_equal(result$threshold, cli_expected$Threshold, tolerance = CLI_TOL)
})

## --- catboost-8z4.117 (P10.C): catboost.utils.get_fpr_curve / get_fnr_curve /
## select_threshold -- these three Python functions (catboost/python-package/
## catboost/utils.py:400-538) have no dedicated R wrapper, but they compute
## nothing new: get_fpr_curve/get_fnr_curve are pure projections of the same
## (fpr, tpr, thresholds) triple catboost.get_roc_curve() already returns
## (fpr as-is; fnr = 1 - tpr), and select_threshold's native
## _select_threshold (catboost/private/libs/algo/roc_curve.cpp
## TRocCurve::SelectDecisionBoundaryByFalsePositiveRate/
## ByFalseNegativeRate) is a plain upper_bound search over that same curve
## -- reimplemented below directly from that C++ source (not guessed) and
## applied to BOTH the pinned Python oracle curve (fixture$expected) and
## catboost.get_roc_curve()'s live R output, so a match proves R's curve
## data supports the identical selection Python's oracle would make.

select_by_fpr <- function(fpr, threshold, target) {
  # Mirrors TRocCurve::SelectDecisionBoundaryByFalsePositiveRate: points are
  # ascending by FPR (same order get_roc_curve returns); upper_bound finds
  # the first point with fpr > target, then steps back one.
  idx <- which(fpr > target)[1]
  idx <- if (is.na(idx)) length(fpr) else idx - 1
  threshold[max(idx, 1)]
}

select_by_fnr <- function(tpr, threshold, target) {
  # Mirrors TRocCurve::SelectDecisionBoundaryByFalseNegativeRate: the same
  # points read back-to-front (rbegin/rend), where FNR = 1 - TPR is
  # ascending; upper_bound finds the first (in that reversed reading) point
  # with fnr > target, then steps back one.
  fnr <- 1 - tpr
  rev_fnr <- rev(fnr)
  rev_threshold <- rev(threshold)
  idx <- which(rev_fnr > target)[1]
  idx <- if (is.na(idx)) length(rev_fnr) else idx - 1
  rev_threshold[max(idx, 1)]
}

test_that("get_fpr_curve: (thresholds, fpr) projection matches the Python oracle curve", {
  result <- catboost.get_roc_curve(py_model, py_pool)
  # get_fpr_curve(model, data) returns (thresholds, fpr); both components
  # are already part of get_roc_curve()'s output, unprojected.
  expect_equal(result$threshold, expected$thresholds, tolerance = PY_TOL)
  expect_equal(result$fpr, expected$fpr, tolerance = PY_TOL)
})

test_that("get_fnr_curve: (thresholds, 1 - tpr) projection matches the Python oracle curve", {
  result <- catboost.get_roc_curve(py_model, py_pool)
  expect_equal(result$threshold, expected$thresholds, tolerance = PY_TOL)
  expect_equal(1 - result$tpr, 1 - expected$tpr, tolerance = PY_TOL)
})

test_that("select_threshold: FPR-based selection over R's curve matches the same selection over the Python oracle curve", {
  result <- catboost.get_roc_curve(py_model, py_pool)
  for (target in c(0.1, 0.3, 0.5, 0.7)) {
    r_threshold <- select_by_fpr(result$fpr, result$threshold, target)
    py_threshold <- select_by_fpr(expected$fpr, expected$thresholds, target)
    expect_equal(r_threshold, py_threshold, tolerance = PY_TOL, info = target)
  }
})

test_that("select_threshold: FNR-based selection over R's curve matches the same selection over the Python oracle curve", {
  result <- catboost.get_roc_curve(py_model, py_pool)
  for (target in c(0.1, 0.3, 0.5, 0.7)) {
    r_threshold <- select_by_fnr(result$tpr, result$threshold, target)
    py_threshold <- select_by_fnr(expected$tpr, expected$thresholds, target)
    expect_equal(r_threshold, py_threshold, tolerance = PY_TOL, info = target)
  }
})
