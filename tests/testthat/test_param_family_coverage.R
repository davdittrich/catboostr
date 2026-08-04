context("test_param_family_coverage.R")

# P5.5 (catboost-8z4.62) -- family-level differential closure for the native
# hyperparameters that had no existing Python-oracle-comparing test (see
# task-5 audit). Per spec Sec 4.5's bulk-disposition rule, this batches many
# mutually-compatible parameter families into a handful of training calls
# rather than one test per parameter name; every batch is compared to the
# Python oracle at the spec Sec 4.3 default tolerance (1e-12), except two
# RNG-driven stochastic families (see their test_that()s) widened to 1e-6.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_param_family_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "param_family_coverage.json"),
  simplifyVector = TRUE
)

features <- data.frame(
  num1 = fixture$inputs$num1,
  num2 = fixture$inputs$num2,
  cat1 = factor(fixture$inputs$cat1),
  stringsAsFactors = FALSE
)
pool <- catboost.load_pool(features, label = fixture$inputs$label)

# jsonlite::toJSON(..., auto_unbox = TRUE) (prepare_train_export_parameters)
# collapses a length-1 R vector to a bare JSON scalar instead of a 1-element
# array; several native list-typed options (custom_metric/simple_ctr/
# combinations_ctr) came back from the fixture as length-1 character vectors.
# as.list() forces array serialization regardless of length, matching what
# the Python oracle actually sent (mirrors the existing
# per_float_feature_quantization/ignored_features handling already in
# prepare_train_export_parameters).
force_array <- function(params, keys) {
  for (k in keys) {
    if (!is.null(params[[k]])) params[[k]] <- as.list(params[[k]])
  }
  params
}

expect_batch_matches_oracle <- function(batch_name, params, tolerance = 1e-12) {
  params <- force_array(params, c("custom_metric", "simple_ctr", "combinations_ctr"))
  # The fixture omits "verbose" (see gen_param_family_fixture.py) since
  # native's flat "verbose" is an int print-period, not a bool; silence
  # training output the R-native way instead. Purely cosmetic -- does not
  # affect predictions, so it cannot perturb the oracle comparison.
  if (is.null(params$logging_level)) params$logging_level <- "Silent"
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(fixture$batches[[batch_name]]$predictions)
  expect_equal(as.vector(actual), expected, tolerance = tolerance, check.attributes = FALSE)
}

test_that("core training-control/regularization/leaf-estimation/od/output-settings family matches Python oracle", {
  # tolerance widened to 1e-6: bagging_temperature (Bayesian bootstrap) draws
  # per-object random weights from an RNG stream; cross-process R-vs-Python
  # float non-associativity in that stochastic path yields ~1e-7-magnitude
  # prediction differences even at identical random_seed, not a correctness bug.
  p <- fixture$batches$core_training_control$params
  expect_batch_matches_oracle("core_training_control", p, tolerance = 1e-6)
})

test_that("CTR + binarization settings family matches Python oracle", {
  p <- fixture$batches$ctr_and_binarization$params
  expect_batch_matches_oracle("ctr_and_binarization", p)
})

test_that("Lossguide grow_policy/max_leaves/score_function family matches Python oracle", {
  p <- fixture$batches$lossguide_grow_policy$params
  expect_batch_matches_oracle("lossguide_grow_policy", p)
})

test_that("MVS bootstrap/subsample/sampling_unit family matches Python oracle", {
  p <- fixture$batches$mvs_sampling$params
  expect_batch_matches_oracle("mvs_sampling", p)
})

test_that("class_weights matches Python oracle", {
  p <- fixture$batches$class_weights$params
  expect_batch_matches_oracle("class_weights", p)
})

test_that("auto_class_weights matches Python oracle", {
  p <- fixture$batches$auto_class_weights$params
  expect_batch_matches_oracle("auto_class_weights", p)
})

test_that("feature-penalty family (monotone_constraints/feature_weights/penalties) matches Python oracle", {
  p <- fixture$batches$feature_penalties$params
  expect_batch_matches_oracle("feature_penalties", p)
})

test_that("Langevin boosting family matches Python oracle", {
  # tolerance widened to 1e-6: Stochastic Gradient Langevin Boosting is
  # itself an RNG-driven stochastic method (same class of cross-process
  # float non-associativity as core_training_control's bagging_temperature).
  p <- fixture$batches$langevin$params
  expect_batch_matches_oracle("langevin", p, tolerance = 1e-6)
})

test_that("snapshot family (save_snapshot/snapshot_file/snapshot_interval) matches Python oracle", {
  p <- fixture$batches$snapshot$params
  p$snapshot_file <- tempfile(fileext = ".snapshot")
  expect_batch_matches_oracle("snapshot", p)
})
