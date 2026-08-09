context("test_cli_predict_flag_coverage.R")

# P10.A fix-round-1 (catboost-8z4.115, review findings): closes the calc-mode
# `--prediction-type` and `--tree-count-limit` matrix rows (kind:flag,
# oracle:cli) left untriaged by the original P10.A sweep.
#
# Both flags are exercised by loading the SAME model file (fit once via the
# CLI so R and the CLI calc invocation are scoring an identical binary, not
# two independently-trained models) and comparing R's catboost.predict()
# against a live run of the pinned CLI binary's `calc` mode:
#   - `--prediction-type Probability` <-> catboost.predict(..., prediction_type
#     = "Probability") (non-default value; existing suite coverage only ever
#     exercised the default RawFormulaVal).
#   - `--tree-count-limit N` <-> catboost.predict(..., ntree_end = N)
#     (`--tree-count-limit` truncates the ensemble exactly like `ntree_end`;
#     verified identical via CLI --help text: "limit count of used trees").
#
# Live comparison against the real CLI binary (skip_if_not when the
# gitignored binary is absent), matching test_cli_io_flag_coverage.R's
# precedent.

CLI_BIN <- testthat::test_path("..", "..", "tools", "oracle", "cli", "bin", "catboost-v1.2.10")
testthat::skip_if_not(
  file.exists(CLI_BIN) && file.access(CLI_BIN, mode = 1) == 0,
  paste0("real CatBoost CLI binary not present/executable at ", CLI_BIN,
         " -- run tools/oracle/cli/acquire.sh (per-checkout, gitignored artifact)")
)

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")
CLI_TOLERANCE <- 1e-6

test_that("--prediction-type Probability and --tree-count-limit match the CLI oracle on a shared model", {
  work_dir <- tempfile("cli_predict_flag_")
  dir.create(work_dir)
  on.exit(unlink(work_dir, recursive = TRUE), add = TRUE)
  model_file <- file.path(work_dir, "model.cbm")

  # Fit once via the CLI so both sides score the identical model binary --
  # isolates the assertion to calc-mode's prediction-type/tree-count-limit
  # conversion, not training determinism (already covered elsewhere).
  fit_status <- system2(CLI_BIN, c(
    "fit",
    "--learn-set", POOL_PATH, "--column-description", CD_PATH,
    "--delimiter", ",", "--has-header", "--loss-function", "Logloss",
    "-i", "20", "--logging-level", "Silent", "-T", "1", "--random-seed", "42",
    "--train-dir", work_dir, "--model-file", model_file
  ), stdout = FALSE, stderr = FALSE)
  expect_equal(fit_status, 0L)

  pool <- catboost.load_pool(POOL_PATH, column_description = CD_PATH, delimiter = ",",
                              has_header = TRUE, thread_count = 1)
  model <- catboost.load_model(model_file)

  # --- --prediction-type Probability (non-default) ---
  prob_out <- file.path(work_dir, "pred_prob.tsv")
  calc_status <- system2(CLI_BIN, c(
    "calc",
    "--input-path", POOL_PATH, "--column-description", CD_PATH,
    "--delimiter", ",", "--has-header",
    "-m", model_file, "-o", prob_out, "--prediction-type", "Probability", "-T", "1"
  ), stdout = FALSE, stderr = FALSE)
  expect_equal(calc_status, 0L)

  cli_prob <- read.delim(prob_out)$Probability
  r_prob <- catboost.predict(model, pool, prediction_type = "Probability")
  expect_equal(as.vector(r_prob), cli_prob, tolerance = CLI_TOLERANCE)

  # --- --tree-count-limit 5 (limits the ensemble like ntree_end) ---
  limit_out <- file.path(work_dir, "pred_limit.tsv")
  calc_status2 <- system2(CLI_BIN, c(
    "calc",
    "--input-path", POOL_PATH, "--column-description", CD_PATH,
    "--delimiter", ",", "--has-header",
    "-m", model_file, "-o", limit_out, "--prediction-type", "RawFormulaVal",
    "--tree-count-limit", "5", "-T", "1"
  ), stdout = FALSE, stderr = FALSE)
  expect_equal(calc_status2, 0L)

  cli_limited <- read.delim(limit_out)$RawFormulaVal
  r_limited <- catboost.predict(model, pool, prediction_type = "RawFormulaVal", ntree_end = 5)
  expect_equal(as.vector(r_limited), cli_limited, tolerance = CLI_TOLERANCE)
})
