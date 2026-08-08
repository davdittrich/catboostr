context("test_scale_and_bias.R")

# P5.7 (catboost-8z4.64) differential test: catboost.get_scale_and_bias(),
# catboost.set_scale_and_bias() (R/catboost.R), matching Python's
# _CatBoostBase.get_scale_and_bias()/set_scale_and_bias() (core.py:2422-2429,
# defined once and inherited unchanged by CatBoost/CatBoostClassifier/
# CatBoostRegressor/CatBoostRanker -- one R differential test covers all four
# matrix rows) and the CLI's `normalize-model` mode (mode_normalize_model.cpp),
# which reads/writes the same TFullModel::GetScaleAndBias()/SetScaleAndBias()
# (model.h, scale_and_bias.h) used as `Scale * sumTrees + Bias`.
#
# CLI flag -> R call mapping:
#   --print-scale-and-bias -> catboost.get_scale_and_bias()
#   --set-scale/--set-bias -> catboost.set_scale_and_bias(model, scale, bias)
#   -i/--input-path (auto min-max normalization over a pool; CLI-only, no
#     Python get_scale_and_bias/set_scale_and_bias counterpart) ->
#     catboost.normalize_model_from_pool(model, pool)
#   --output-model / mode:normalize-model (file-to-file roundtrip) ->
#     catboost.save_model()/catboost.load_model() preserve scale/bias
#
# Matrix rows closed by this file:
#   - CatBoost.get_scale_and_bias / CatBoostClassifier.get_scale_and_bias /
#     CatBoostRegressor.get_scale_and_bias / CatBoostRanker.get_scale_and_bias
#     (Python oracle, tolerance 1e-12)
#   - CatBoost.set_scale_and_bias / CatBoostClassifier.set_scale_and_bias /
#     CatBoostRegressor.set_scale_and_bias / CatBoostRanker.set_scale_and_bias
#     (Python oracle, tolerance 1e-12)
#   - mode:normalize-model (CLI oracle, roundtrip, tolerance 1e-9)
#   - flag:--set-scale / flag:--set-bias / flag:--print-scale-and-bias
#     (CLI oracle, tolerance 1e-9)
#   - flag:--input-path/-i (CLI-only residual: no Python equivalent; relative
#     tolerance 1e-9 per Section III of the task brief)
#   - flag:--output-model (CLI oracle: exercised by the save/load roundtrip)
#
# Rows left red (shared with other CLI modes -- not owned by normalize-model,
# out of scope per the ticket's explicit row list): flag:--output-model-format
# (shared family with `metadata set`/`model-sum`), flag:--model-file/-m,
# --model-format, --cd, --delimiter, --has-header, --ignore-csv-quoting,
# --feature-names-path, --pool-metainfo-path, --thread-count/-T,
# --logging-level, --help, --svnrevision (generic multi-mode infra flags).
#
# Regenerate Python fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_scale_and_bias_fixture.py
#
# Regenerate CLI fixture (after tools/oracle/cli/acquire.sh) -- commands
# recorded verbatim in tests/fixtures/oracle-cli/scale_and_bias.json's
# "_regenerate" array.

PY_TOL <- 1e-12
CLI_TOL <- 1e-9

## --- CatBoost{,Classifier,Regressor,Ranker}.get_scale_and_bias/set_scale_and_bias (Python oracle) ---

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "scale_and_bias.json"),
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
  iterations = 10, depth = 2, loss_function = "RMSE",
  random_seed = 42, thread_count = 1, logging_level = "Silent",
  train_dir = tempfile("catboost_train_scale_and_bias_")
))

test_that("get_scale_and_bias: default (post-fit) scale/bias matches Python oracle", {
  sb <- catboost.get_scale_and_bias(py_model)
  expect_equal(sb$scale, expected$scale_before, tolerance = PY_TOL)
  expect_equal(sb$bias, expected$bias_before, tolerance = PY_TOL)
})

test_that("set_scale_and_bias: scalar scale/bias matches Python oracle", {
  catboost.set_scale_and_bias(py_model, 0.8, 0.8)
  sb <- catboost.get_scale_and_bias(py_model)
  expect_equal(sb$scale, expected$scale_after_scalar, tolerance = PY_TOL)
  expect_equal(sb$bias, expected$bias_after_scalar, tolerance = PY_TOL)
})

test_that("mode:normalize-model roundtrip -- scale/bias survive a save/load cycle", {
  model_path <- tempfile(fileext = ".cbm")
  on.exit(unlink(model_path), add = TRUE)
  catboost.save_model(py_model, model_path)
  reloaded <- catboost.load_model(model_path)
  sb <- catboost.get_scale_and_bias(reloaded)
  expect_equal(sb$scale, expected$scale_after_scalar, tolerance = PY_TOL)
  expect_equal(sb$bias, expected$bias_after_scalar, tolerance = PY_TOL)
})

test_that("get_scale_and_bias/set_scale_and_bias reject non-Model first argument", {
  expect_error(catboost.get_scale_and_bias(list()), "Expected catboost.Model")
  expect_error(catboost.set_scale_and_bias(list(), 1, 0), "Expected catboost.Model")
})

test_that("set_scale_and_bias rejects malformed scale/bias arguments", {
  expect_error(catboost.set_scale_and_bias(py_model, "x", 0), "scale must be a single number")
  expect_error(catboost.set_scale_and_bias(py_model, c(1, 2), 0), "scale must be a single number")
  expect_error(catboost.set_scale_and_bias(py_model, 1, "x"), "bias must be numeric")
})

## --- mode:normalize-model / flag:--set-scale / --set-bias / --print-scale-and-bias / --input-path/-i (CLI oracle) ---

cli_fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "scale_and_bias.json"),
  simplifyVector = TRUE
)

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
cli_data[[cli_cat_col]] <- as.factor(cli_data[[cli_cat_col]])
cli_pool <- catboost.load_pool(
  cli_data[, -cli_label_col],
  label = as.character(cli_data[, cli_label_col]),
  feature_names = as.list(c("0", "1", "cat1"))
)

new_cli_model <- function() {
  catboost.train(cli_pool, params = list(
    loss_function = "Logloss", iterations = 20, depth = 4,
    learning_rate = 0.1, random_seed = 42, thread_count = 1,
    logging_level = "Silent"
  ))
}

test_that("flag:--print-scale-and-bias -- freshly trained model's default matches CLI oracle", {
  cli_model <- new_cli_model()
  sb <- catboost.get_scale_and_bias(cli_model)
  expect_equal(sb$scale, cli_fixture$print_only$input_scale, tolerance = CLI_TOL)
  expect_equal(sb$bias, cli_fixture$print_only$input_bias, tolerance = CLI_TOL)
})

test_that("flag:--set-scale/--set-bias -- matches CLI oracle's `normalize-model --set-scale --set-bias`", {
  cli_model <- new_cli_model()
  catboost.set_scale_and_bias(
    cli_model,
    cli_fixture$set_scale_bias$set_scale,
    cli_fixture$set_scale_bias$set_bias
  )
  sb <- catboost.get_scale_and_bias(cli_model)
  expect_equal(sb$scale, cli_fixture$set_scale_bias$output_scale, tolerance = CLI_TOL)
  expect_equal(sb$bias, cli_fixture$set_scale_bias$output_bias, tolerance = CLI_TOL)
})

test_that("flag:--output-model / mode:normalize-model roundtrip -- CLI oracle's set-then-save survives R's save/load", {
  cli_model <- new_cli_model()
  catboost.set_scale_and_bias(cli_model, 0.8, 0.8)
  model_path <- tempfile(fileext = ".cbm")
  on.exit(unlink(model_path), add = TRUE)
  catboost.save_model(cli_model, model_path)
  reloaded <- catboost.load_model(model_path)
  sb <- catboost.get_scale_and_bias(reloaded)
  expect_equal(sb$scale, cli_fixture$set_scale_bias$output_scale, tolerance = CLI_TOL)
  expect_equal(sb$bias, cli_fixture$set_scale_bias$output_bias, tolerance = CLI_TOL)
})

test_that("flag:--input-path/-i -- catboost.normalize_model_from_pool() matches CLI oracle's auto min-max scaling", {
  # CLI-only capability (mode_normalize_model.cpp's PoolPaths branch): no
  # Python get_scale_and_bias/set_scale_and_bias equivalent exists, so this
  # is verified only against the CLI oracle, at the CLI-only relative
  # tolerance (1e-9) called for in Section III of the task brief.
  cli_model <- new_cli_model()
  catboost.normalize_model_from_pool(cli_model, cli_pool)
  sb <- catboost.get_scale_and_bias(cli_model)
  expect_equal(sb$scale, cli_fixture$auto_min_max$output_scale, tolerance = CLI_TOL)
  expect_equal(sb$bias, cli_fixture$auto_min_max$output_bias, tolerance = CLI_TOL)
})

test_that("normalize_model_from_pool errors when the model gives the same result on all docs", {
  # A depth-0 tree has a single leaf, so every tree applies the same
  # constant value to every doc, making RawFormulaVal identical for the
  # whole pool -- exercises mode_normalize_model.cpp's
  # `CB_ENSURE(approx.Min != approx.Max, ...)` guard.
  degenerate_model <- catboost.train(cli_pool, params = list(
    loss_function = "Logloss", iterations = 5, depth = 0,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
  expect_error(
    catboost.normalize_model_from_pool(degenerate_model, cli_pool),
    "Model gives same result on all docs"
  )
})
