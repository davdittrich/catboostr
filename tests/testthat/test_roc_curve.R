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
