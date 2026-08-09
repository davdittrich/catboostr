context("test_cli_hyperparam_coverage.R")

# P10.A (catboost-8z4.115) differential test: closes the shared
# fit/eval-feature/model-based-eval/select-features hyperparameter flag
# family (matrix rows kind:flag, oracle:cli, family owners
# {fit, eval-feature, model-based-eval, select-features}) that P9.1
# (catboost-8z4.113) classified as bucket-C "real gap: implemented-but-
# untested". Every one of these flags is already a `catboost.train()`
# `params` list entry (R/catboost.R's .catboostr_known_params, generated
# from the same machine-readable capability inventory this matrix comes
# from); the gap was test coverage against the CLI oracle specifically, not
# implementation. `catboost.train()`/CLI `fit` both funnel these options
# through the identical native flat-options parser (PlainJsonToOptions),
# so a bit-for-bit prediction match on the shared smoke fixture is a real
# differential test, not an inspection-only closure.
#
# Batched into 4 CLI fit+calc runs instead of one, because several flags are
# mutually exclusive (verified empirically: Ordered boosting rejects
# Lossguide trees; Bayesian bootstrap rejects bagging_temperature's
# alternatives; approx_on_full_history rejects monotone_constraints;
# posterior_sampling rejects an explicit diffusion_temperature; MultiClass-
# only class params reject the other batches' Logloss) -- matches
# test_param_family_coverage.R's existing multi-batch precedent, just against
# the CLI binary instead of the Python oracle.
#
# Regenerate fixtures (after tools/oracle/cli/acquire.sh) with:
#   tools/oracle/cli/gen_hyperparam_coverage_fixture.sh
#   tools/oracle/cli/gen_hyperparam_coverage2_fixture.sh
#   tools/oracle/cli/gen_hyperparam_coverage3_fixture.sh
#   tools/oracle/cli/gen_hyperparam_coverage4_fixture.sh

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")
CLI_TOLERANCE <- 1e-6

pool <- catboost.load_pool(POOL_PATH, column_description = CD_PATH, delimiter = ",",
                            has_header = TRUE, thread_count = 1)

read_predictions <- function(name) {
  path <- testthat::test_path("..", "fixtures", "oracle-cli", name)
  as.matrix(read.delim(path, header = FALSE, sep = "\t")[, -1, drop = FALSE])
}

test_that("batch 1: fold/boosting/od/shrink/langevin/leaf-estimation/tree/ctr/text/class/quantization/misc flags match CLI oracle", {
  params <- list(
    loss_function = "Logloss", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    fold_len_multiplier = 1.5, fold_permutation_block = 2,
    boost_from_average = TRUE, boosting_type = "Ordered",
    od_pval = 0.01, od_wait = 2, od_type = "IncToDec",
    model_shrink_rate = 0.1, model_shrink_mode = "Constant",
    langevin = TRUE, diffusion_temperature = 100,
    rsm = 0.9,
    leaf_estimation_iterations = 2, leaf_estimation_backtracking = "AnyImprovement",
    depth = 3, min_data_in_leaf = 1, l2_leaf_reg = 3,
    bayesian_matrix_reg = 0.1, model_size_reg = 0.5,
    sparse_features_conflict_fraction = 0.0, random_strength = 1.0,
    leaf_estimation_method = "Newton", score_function = "Cosine",
    bootstrap_type = "Bayesian", sampling_unit = "Object", bagging_temperature = 0.5,
    sampling_frequency = "PerTree",
    monotone_constraints = "(1,0,0)", feature_weights = "(1,1,1)",
    penalties_coefficient = 1.0, first_feature_use_penalties = "(0,0,0)",
    per_object_feature_penalties = "(0,0,0)",
    max_ctr_complexity = 2, simple_ctr = list("Borders"), combinations_ctr = list("Borders"),
    per_feature_ctr = list("2:Borders:Prior=0/1"),
    ctr_target_border_count = 1, counter_calc_method = "Full",
    ctr_leaf_count_limit = 100, ctr_history_unit = "Sample",
    one_hot_max_size = 2,
    auto_class_weights = "Balanced", force_unit_auto_pair_weights = TRUE,
    border_count = 32, per_float_feature_quantization = list("0:border_count=32"),
    feature_border_type = "GreedyLogSum", nan_mode = "Min",
    used_ram_limit = "1gb", task_type = "CPU", detailed_profile = TRUE,
    final_ctr_computation_mode = "Default", allow_writing_files = TRUE,
    train_dir = tempfile("cli_hyperparam_batch1_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(read_predictions("hyperparam_coverage_predictions.tsv"))
  expect_equal(as.vector(actual), expected, tolerance = CLI_TOLERANCE)
})

test_that("batch 2: approx-on-full-history / MVS bootstrap+subsample+mvs-reg / posterior-sampling / target-border match CLI oracle", {
  params <- list(
    loss_function = "Logloss", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    boosting_type = "Ordered", approx_on_full_history = TRUE,
    bootstrap_type = "MVS", subsample = 0.8, mvs_reg = 0.2,
    posterior_sampling = TRUE,
    target_border = 0.5,
    train_dir = tempfile("cli_hyperparam_batch2_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(read_predictions("hyperparam_coverage2_predictions.tsv"))
  expect_equal(as.vector(actual), expected, tolerance = CLI_TOLERANCE)
})

test_that("batch 3: grow-policy Lossguide + max-leaves match CLI oracle", {
  params <- list(
    loss_function = "Logloss", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    grow_policy = "Lossguide", max_leaves = 8,
    train_dir = tempfile("cli_hyperparam_batch3_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(read_predictions("hyperparam_coverage3_predictions.tsv"))
  expect_equal(as.vector(actual), expected, tolerance = CLI_TOLERANCE)
})

test_that("batch 5: tokenizers/dictionaries/feature-calcers/embedding-processing match CLI oracle", {
  params <- list(
    loss_function = "Logloss", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    tokenizers = list(list(tokenizer_id = "Space", delimiter = " ")),
    dictionaries = list(list(dictionary_id = "Word", token_level_type = "Word")),
    feature_calcers = list("BoW"),
    embedding_processing = list(),
    train_dir = tempfile("cli_hyperparam_batch5_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(read_predictions("hyperparam_coverage5_predictions.tsv"))
  expect_equal(as.vector(actual), expected, tolerance = CLI_TOLERANCE)
})

test_that("batch 6: text-processing matches CLI oracle", {
  params <- list(
    loss_function = "Logloss", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    text_processing = list(),
    train_dir = tempfile("cli_hyperparam_batch6_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(read_predictions("hyperparam_coverage6_predictions.tsv"))
  expect_equal(as.vector(actual), expected, tolerance = CLI_TOLERANCE)
})

test_that("batch 7: store-all-simple-ctr/ignore-features/has-time/allow-const-label match CLI oracle", {
  params <- list(
    loss_function = "Logloss", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    store_all_simple_ctr = TRUE, ignored_features = list(1), has_time = TRUE,
    allow_const_label = TRUE,
    train_dir = tempfile("cli_hyperparam_batch7_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- as.vector(read_predictions("hyperparam_coverage7_predictions.tsv"))
  expect_equal(as.vector(actual), expected, tolerance = CLI_TOLERANCE)
})

test_that("batch 4: classes-count/class-names/class-weights (MultiClass) match CLI oracle", {
  params <- list(
    loss_function = "MultiClass", iterations = 20, logging_level = "Silent", thread_count = 1,
    random_seed = 42,
    classes_count = 2, class_names = list("0", "1"), class_weights = list(1, 1),
    train_dir = tempfile("cli_hyperparam_batch4_")
  )
  model <- catboost.train(pool, params = params)
  actual <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expected <- read_predictions("hyperparam_coverage4_predictions.tsv")
  expect_equal(as.matrix(actual), expected, tolerance = CLI_TOLERANCE, check.attributes = FALSE)
})
