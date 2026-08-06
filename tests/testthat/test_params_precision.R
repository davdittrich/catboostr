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

# catboost-8z4.75 -- catboost.save_model (R/catboost.R), the
# jsonlite::toJSON(export_parameters, ...) call serializing CoreML/PMML
# export parameters, had no explicit digits argument and so fell back to
# jsonlite's default (digits = 4) -- worse than the digits = 10 default this
# same defect class hit at prepare_train_export_parameters (catboost-8z4.65)
# and prepare_grid_json (catboost-8z4.60). Fix: digits = NA, matching both
# established sites.
#
# export_parameters is passed straight to jsonlite::toJSON with no
# intervening function to call directly (unlike prepare_train_export_
# parameters/prepare_grid_json), and the resulting JSON string is opaque
# once inside the native save call -- it is not echoed back by any R-level
# accessor. The call site's actual `digits` argument is therefore captured
# by mocking jsonlite::toJSON for the duration of one catboost.save_model()
# call and delegating to the real implementation, which is the only way to
# observe -- without touching CoreML/PMML native export support -- whether
# the fix is in place.
test_that("catboost.save_model: export_parameters JSON uses digits = NA, not jsonlite's truncating digits = 4 default (catboost-8z4.75)", {
  target <- sample(c(1, -1), size = 100, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target)))
  pool <- catboost.load_pool(features, target)
  model <- catboost.train(pool, params = list(iterations = 2, loss_function = "Logloss", logging_level = "Silent"))

  real_toJSON <- jsonlite::toJSON
  captured_digits <- "not called"
  testthat::local_mocked_bindings(
    toJSON = function(x, ...) {
      args <- list(...)
      captured_digits <<- if ("digits" %in% names(args)) args$digits else formals(real_toJSON)$digits
      real_toJSON(x, ...)
    },
    .package = "jsonlite"
  )

  model_path <- tempfile(fileext = ".coreml")
  on.exit(unlink(model_path), add = TRUE)
  # 0.123456789012345 has 15 significant digits: digits = 4 (jsonlite's
  # default) would truncate it to "0.1235"; digits = NA round-trips it
  # exactly. file_format = "coreml" is required: "cbm" rejects any
  # export_parameters JSON outright (model_exporter.cpp's userParametersJson
  # empty-check), so it never reaches the toJSON call site's effect. CoreML
  # export parses the JSON but only reads a handful of known
  # "coreml_*" string keys and silently ignores unrecognized ones (see
  # coreml_helpers.cpp's ConfigureMetadata/ConfigureTreeModelIO), so an
  # extra high-precision numeric key exercises the serialization without
  # requiring the native exporter to understand it.
  catboost.save_model(model, model_path, file_format = "coreml", export_parameters = list(some_value = 0.123456789012345))

  expect_true(is.na(captured_digits))
})
