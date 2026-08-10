context("test_cli_io_flag_coverage.R")

# P10.A (catboost-8z4.115) differential test: closes the shared
# fit/eval-feature/model-based-eval/select-features *output/logging* flag
# family (matrix rows kind:flag, oracle:cli). These flags don't change the
# trained model -- they control where side-effect artifacts get written --
# so the differential assertion is that the SAME flag, given to both
# catboost.train() and the real CLI binary on the same fixture, produces a
# non-empty artifact in both languages (live comparison against the pinned
# CLI binary; skipped, not failed, when the gitignored binary is absent --
# same precedent as test_distributed_training.R).
#
# Closes: --learn-err-log, --test-err-log, --json-log, --profile-log,
# --training-options-file, --snapshot-file, --snapshot-interval,
# --output-borders-file, --use-best-model, --best-model-min-trees, --name,
# --metric-period, --custom-loss/--custom-metric, --eval-metric.
# --trace-log stays red (see test_params_validation.R-style note below):
# not in .catboostr_known_params, rejected as an unknown key.

CLI_BIN <- testthat::test_path("..", "..", "tools", "oracle", "cli", "bin", "catboost-v1.2.10")
testthat::skip_if_not(
  file.exists(CLI_BIN) && file.access(CLI_BIN, mode = 1) == 0,
  paste0("real CatBoost CLI binary not present/executable at ", CLI_BIN,
         " -- run tools/oracle/cli/acquire.sh (per-checkout, gitignored artifact)")
)

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")

test_that("io/logging flags: R and the CLI oracle both accept them and write the documented artifacts", {
  r_dir <- tempfile("cli_io_r_")
  dir.create(r_dir)
  on.exit(unlink(r_dir, recursive = TRUE), add = TRUE)

  pool <- catboost.load_pool(POOL_PATH, column_description = CD_PATH, delimiter = ",",
                              has_header = TRUE, thread_count = 1)
  r_params <- list(
    loss_function = "Logloss", custom_loss = list("AUC"), eval_metric = "AUC",
    iterations = 20, logging_level = "Silent", thread_count = 1, random_seed = 42,
    train_dir = r_dir, name = "probe_run", metric_period = 1,
    use_best_model = TRUE, best_model_min_trees = 2,
    learn_error_log = file.path(r_dir, "learn_error.log"),
    test_error_log = file.path(r_dir, "test_error.log"),
    json_log = file.path(r_dir, "catboost_training.json"),
    profile_log = file.path(r_dir, "profile.log"),
    training_options_file = file.path(r_dir, "training_options.json"),
    snapshot_file = file.path(r_dir, "snapshot.bin"),
    snapshot_interval = 1,
    output_borders = file.path(r_dir, "output_borders.tsv"),
    detailed_profile = TRUE, save_snapshot = TRUE, allow_writing_files = TRUE
  )
  model <- catboost.train(pool, test_pool = pool, params = r_params)
  expect_true(!is.null(model))

  r_artifacts <- c("learn_error.log", "test_error.log", "catboost_training.json",
                    "profile.log", "training_options.json", "snapshot.bin",
                    "output_borders.tsv")
  for (f in r_artifacts) {
    p <- file.path(r_dir, f)
    expect_true(file.exists(p) && file.info(p)$size > 0, info = f)
  }

  cli_dir <- tempfile("cli_io_cli_")
  dir.create(cli_dir)
  on.exit(unlink(cli_dir, recursive = TRUE), add = TRUE)
  model_file <- file.path(cli_dir, "model.cbm")

  status <- system2(CLI_BIN, c(
    "fit",
    "--learn-set", POOL_PATH, "--column-description", CD_PATH,
    "--delimiter", ",", "--has-header", "--loss-function", "Logloss",
    "--custom-metric", "AUC", "--eval-metric", "AUC",
    "-i", "20", "--logging-level", "Silent", "-T", "1", "--random-seed", "42",
    "--train-dir", cli_dir, "--name", "probe_run", "--metric-period", "1",
    "--use-best-model", "true", "--best-model-min-trees", "2",
    "--learn-err-log", file.path(cli_dir, "learn_error.log"),
    "--test-err-log", file.path(cli_dir, "test_error.log"),
    "--json-log", file.path(cli_dir, "catboost_training.json"),
    "--profile-log", file.path(cli_dir, "profile.log"),
    "--training-options-file", file.path(cli_dir, "training_options.json"),
    "--snapshot-file", file.path(cli_dir, "snapshot.bin"),
    "--snapshot-interval", "1",
    "--output-borders-file", file.path(cli_dir, "output_borders.tsv"),
    "--detailed-profile",
    "--model-file", model_file,
    "-t", POOL_PATH
  ), stdout = FALSE, stderr = FALSE)
  expect_equal(status, 0L)

  cli_artifacts <- c("learn_error.log", "test_error.log", "catboost_training.json",
                      "profile.log", "training_options.json", "snapshot.bin",
                      "output_borders.tsv")
  for (f in cli_artifacts) {
    p <- file.path(cli_dir, f)
    expect_true(file.exists(p) && file.info(p)$size > 0, info = f)
  }
})

test_that("flag:--trace-log is rejected as an unknown params key (matrix row stays red)", {
  pool <- catboost.load_pool(POOL_PATH, column_description = CD_PATH, delimiter = ",",
                              has_header = TRUE, thread_count = 1)
  expect_error(
    catboost.train(pool, params = list(
      loss_function = "Logloss", iterations = 2, logging_level = "Silent",
      thread_count = 1, trace_log = tempfile()
    )),
    "Unknown catboost 'params' key.*trace_log"
  )
})
