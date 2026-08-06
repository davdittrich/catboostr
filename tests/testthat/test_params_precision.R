context("test_params_precision.R")

# catboost-8z4.65 -- prepare_train_export_parameters (R/catboost.R), the
# shared jsonlite::toJSON serialization site behind every catboost.train/
# catboost.cv call, used digits = 10 (jsonlite's default). Any hyperparameter
# with more than 10 significant digits (long-decimal learning_rate,
# l2_leaf_reg, ...) was silently truncated before ever reaching the native
# training call. Same defect class as prepare_grid_json's earlier fix
# (catboost-8z4.60, test_grid_search.R), but at catboost.train's own
# serialization site rather than grid_search's.
#
# Fix: digits = NA (jsonlite's full-round-trip-precision mode), matching
# prepare_grid_json's precedent.
#
# Python never truncates these values -- passed as native doubles, no JSON
# round trip -- so its predictions are ground truth: if R's inbound JSON
# still truncated learning_rate/l2_leaf_reg to 10 significant digits, this
# model would train differently and its predictions would diverge from the
# fixture's by much more than floating-point noise.
#
# Regenerate fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_train_high_precision_fixture.py

PY_TOL <- 1e-6
# ULP-level float64 round-trip noise (jsonlite double -> native double
# parse), not truncation -- truncation at 10 significant digits would show
# up as a ~1e-6 relative error, not ~1e-15. Same order of magnitude as the
# 1e-12 tolerance test_grid_search.R/test_scale_and_bias.R use for their
# Python-oracle comparisons.
EXACT_TOL <- 1e-12

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "train_high_precision.json"),
  simplifyVector = TRUE
)

data <- read.csv(
  testthat::test_path("..", "fixtures", "oracle", "train_high_precision_data.csv"),
  header = TRUE
)
pool <- catboost.load_pool(data[, c("num1", "num2")], label = data$target)

fit_params <- as.list(fixture$params)
fit_params$verbose <- NULL
fit_params$logging_level <- "Silent"

test_that("catboost.train: learning_rate/l2_leaf_reg beyond 10 significant digits round-trip through prepare_train_export_parameters's params JSON without truncation (catboost-8z4.65)", {
  # learning_rate (0.123456789012345, 15 significant digits) and
  # l2_leaf_reg (3.141592653589793, 16 significant digits) are both past
  # jsonlite::toJSON's digits = 10 default truncation point.
  model <- catboost.train(pool, params = fit_params)

  # catboost.get_model_params()$flat_params echoes back the exact params the
  # native training call parsed from R's JSON -- the ModelInfo["params"]
  # round trip proves the value that reached native training, independent of
  # the (unrelated, pre-existing, vendor-native) internal options-tree
  # writer's own float formatting.
  reached_native <- catboost.get_model_params(model)$flat_params
  expect_equal(reached_native$learning_rate, fixture$params$learning_rate, tolerance = EXACT_TOL)
  expect_equal(reached_native$l2_leaf_reg, fixture$params$l2_leaf_reg, tolerance = EXACT_TOL)

  # End-to-end parity: if truncation had silently changed the trained model,
  # predictions would diverge from the Python oracle by much more than
  # floating-point noise.
  predictions <- catboost.predict(model, pool)
  expect_equal(as.numeric(predictions), fixture$predictions, tolerance = PY_TOL)
})
