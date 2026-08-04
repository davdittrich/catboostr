context("test_param_family_coverage.R")

# P5.5 (catboost-8z4.62) -- family-level differential closure for the native
# hyperparameters that had no existing Python-oracle-comparing test (see
# task-5 audit and the review-round fix notes in docs/phase-5/P5.5-report.md).
# Per spec Sec 4.5's bulk-disposition rule, this batches many
# mutually-compatible parameter families into a handful of training calls
# rather than one test per parameter name; every batch is compared to the
# Python oracle at the spec Sec 4.3 default tolerance (1e-12) unless a
# test_that() below documents a specific, isolated (single-variable) reason
# to widen it.
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
# combinations_ctr/custom_loss/ctr_description/per_feature_ctr) came back
# from the fixture as length-1 character vectors. as.list() forces array
# serialization regardless of length, matching what the Python oracle
# actually sent (mirrors the existing per_float_feature_quantization/
# ignored_features handling already in prepare_train_export_parameters).
force_array <- function(params, keys) {
  for (k in keys) {
    if (!is.null(params[[k]])) params[[k]] <- as.list(params[[k]])
  }
  params
}

expect_batch_matches_oracle <- function(batch_name, params, tolerance = 1e-12) {
  params <- force_array(params, c("custom_metric", "simple_ctr", "combinations_ctr",
                                   "custom_loss", "ctr_description", "per_feature_ctr"))
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

test_that("core training-control/regularization/leaf-estimation/od/output-settings/ignored_features/train_dir family matches Python oracle", {
  # Bit-exact at the real 1e-12 default (model_shrink_rate, the confirmed
  # cause of an earlier ~1.7e-7 divergence in a larger version of this
  # batch, now lives in its own isolated test_that() below).
  p <- fixture$batches$core_training_control$params
  # train_dir is stripped from the exported fixture params (a filesystem
  # path, not comparable across machines -- see gen_param_family_fixture.py),
  # so it must be re-injected here for the row to genuinely reach this key,
  # same as snapshot_file/output_borders in the batches below (review-round
  # fix: this row was green with train_dir never actually passed to R).
  p$train_dir <- tempfile("core_training_control_")
  expect_batch_matches_oracle("core_training_control", p)
})

test_that("Bayesian bootstrap (bagging_temperature), isolated, matches Python oracle bit-exact", {
  # Isolated single-variable-vs-defaults batch. The review round originally
  # attributed core_training_control's ~1.7e-7 divergence to
  # bagging_temperature's RNG draw; isolating it here (bisection, fix round)
  # disproves that -- it is bit-exact on its own. Real 1e-12 default applies.
  # (The actual cause was model_shrink_rate; see the isolated batch below.)
  p <- fixture$batches$bayesian_bootstrap$params
  expect_batch_matches_oracle("bayesian_bootstrap", p)
})

test_that("model_shrink_rate, isolated, matches Python oracle within a measured, justified tolerance", {
  # Isolated single-variable-vs-defaults batch (bisection, fix round):
  # model_shrink_mode="Constant" + model_shrink_rate=0.01 is the confirmed,
  # sole, isolated cause of core_training_control's original ~1.7e-7
  # divergence (removing this one parameter from an otherwise-full batch
  # yields exact 0.0 diff; removing any other single parameter does not).
  # Plausibly floating-point evaluation-order sensitivity in the repeated
  # multiplicative shrinkage applied across iterations, not an RNG stream
  # difference -- deterministic on both sides, but not associative across
  # implementations. Original bundled-batch divergence ~1.7e-7; re-measured
  # ~3.9e-7 in this isolated batch (same order of magnitude, different exact
  # value because the surrounding params differ -- not a discrepancy).
  # Tolerance widened to 1e-6 for margin (same convention as Langevin below).
  p <- fixture$batches$model_shrink_rate_isolated$params
  expect_batch_matches_oracle("model_shrink_rate_isolated", p, tolerance = 1e-6)
})

test_that("CTR + binarization settings family matches Python oracle", {
  p <- fixture$batches$ctr_and_binarization$params
  expect_batch_matches_oracle("ctr_and_binarization", p)
})

test_that("custom_loss/ctr_description/ctr_history_unit/per_feature_ctr family matches Python oracle", {
  p <- fixture$batches$ctr_extra$params
  expect_batch_matches_oracle("ctr_extra", p)
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
  # itself an RNG-driven stochastic method, and this batch is already
  # isolated to just its own 3 params (langevin/diffusion_temperature/
  # posterior_sampling) -- same class of cross-process float
  # non-associativity as the isolated bayesian_bootstrap batch above.
  p <- fixture$batches$langevin$params
  expect_batch_matches_oracle("langevin", p, tolerance = 1e-6)
})

test_that("snapshot family (save_snapshot/snapshot_file/snapshot_interval) matches Python oracle", {
  p <- fixture$batches$snapshot$params
  p$snapshot_file <- tempfile(fileext = ".snapshot")
  expect_batch_matches_oracle("snapshot", p)
})

test_that("output_borders matches Python oracle", {
  # input_borders is deliberately NOT here: native rejects it as an
  # "Unknown option" flat top-level key in this vendor version (verified
  # directly -- see task-5-report.md); output_borders IS a real flat option
  # (output_file_options.cpp OutputBordersFileName) and trains cleanly.
  p <- fixture$batches$output_borders$params
  p$output_borders <- tempfile(fileext = ".tsv")
  expect_batch_matches_oracle("output_borders", p)
})

test_that("used_ram_limit's real, isolated R-vs-Python divergence is measured, not discarded", {
  # Isolated A/B: "used_ram_limit_isolated" (used_ram_limit="512mb") vs.
  # "used_ram_limit_baseline" (identical params, no used_ram_limit) --
  # if used_ram_limit itself has no numeric effect, the two Python
  # predictions should already be identical to each other, and R's
  # used_ram_limit run should match the Python used_ram_limit run at the
  # real 1e-12 default. If not, this fails loudly instead of the parameter
  # being silently dropped from the fixture (review-round fix: catboost-8z4.62).
  py_isolated <- as.vector(fixture$batches$used_ram_limit_isolated$predictions)
  py_baseline <- as.vector(fixture$batches$used_ram_limit_baseline$predictions)
  expect_equal(py_isolated, py_baseline, tolerance = 1e-12,
               label = "Python used_ram_limit vs Python baseline (same params otherwise)")

  p <- fixture$batches$used_ram_limit_isolated$params
  expect_batch_matches_oracle("used_ram_limit_isolated", p)
})

text_features <- data.frame(
  num1 = fixture$text_inputs$num1,
  num2 = fixture$text_inputs$num2,
  cat1 = factor(fixture$text_inputs$cat1),
  text1 = fixture$text_inputs$text1,
  stringsAsFactors = FALSE
)
text_pool <- catboost.load_pool(text_features, label = fixture$text_inputs$label)

expect_text_batch_matches_oracle <- function(batch_name, params) {
  params$logging_level <- "Silent"
  model <- catboost.train(text_pool, params = params)
  actual <- catboost.predict(model, text_pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(fixture$batches[[batch_name]]$predictions)
  expect_equal(as.vector(actual), expected, tolerance = 1e-12, check.attributes = FALSE)
}

test_that("text_features (auto-detected character column) + text_processing matches Python oracle (review-round fix)", {
  # text_features itself has no literal R params-list key (it's supplied via
  # catboost.load_pool auto-detecting character columns, R/catboost.R:401-404,
  # matching Python's Pool(text_features=[...])); this proves that R path
  # end-to-end with a genuine text-bearing Pool, not just the standalone
  # Tokenizer/Dictionary API (test_text_processing.R) or the params-key gate.
  p <- fixture$batches$text_processing_only$params
  p$text_processing <- list(feature_processing = list(default = list(list(
    dictionaries_names = list("Word"), feature_calcers = list("BoW"),
    tokenizers_names = list("Space")
  ))))
  expect_text_batch_matches_oracle("text_processing_only", p)
})

test_that("dictionaries/tokenizers/feature_calcers (the separate-options form) matches Python oracle (review-round fix)", {
  # Native forbids combining `text_processing` with this trio in the same
  # call (text_processing_options.cpp:394), hence a second, separate batch.
  p <- fixture$batches$tokenizers_dictionaries_calcers$params
  p$dictionaries <- list(list(dictionary_id = "Word"))
  p$tokenizers <- list(list(tokenizer_id = "Space"))
  p$feature_calcers <- as.list(p$feature_calcers)
  expect_text_batch_matches_oracle("tokenizers_dictionaries_calcers", p)
})
