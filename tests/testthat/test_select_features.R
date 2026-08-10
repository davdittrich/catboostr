context("test_select_features.R")

# P5.3 (catboost-8z4.60) -- catboost.select_features.
#
# Calls the exact same native NCB::SelectFeatures entry point
# (catboost/libs/features_selection/select_features.h) that Python's
# CatBoost.select_features calls via its Cython wrapper (core.py:4822
# self._object._select_features -> cpdef at _catboost.pyx:6031, which calls
# SelectFeatures at :6045), rather than an R-side reimplementation of recursive
# feature elimination -- the elimination order, the per-step retraining and the
# SHAP-based feature strengths all live in that native code, so calling it
# directly is what makes the summary and the final model match the Python oracle
# bit-for-bit rather than merely approximately. Tolerance is therefore the spec
# section 4.3 Python-oracle default of 1e-12, not an override.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_select_features_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "select_features.json"),
  simplifyVector = TRUE
)

learn_pool <- catboost.load_pool(fixture$inputs$learn_features, label = fixture$inputs$learn_label)
test_pool <- catboost.load_pool(fixture$inputs$test_features, label = fixture$inputs$test_label)

base_params <- function() {
  p <- fixture$base_params
  p$verbose <- NULL
  p$logging_level <- "Silent"
  return(p)
}

expect_summary_matches <- function(actual, expected) {
  expect_equal(actual$selected_features, expected$selected_features, check.attributes = FALSE)
  expect_equal(actual$eliminated_features, expected$eliminated_features, check.attributes = FALSE)
  expect_equal(
    actual$loss_graph$removed_features_count,
    expected$loss_graph$removed_features_count,
    check.attributes = FALSE
  )
  expect_equal(
    actual$loss_graph$main_indices,
    expected$loss_graph$main_indices,
    check.attributes = FALSE
  )
  expect_equal(
    actual$loss_graph$loss_values,
    expected$loss_graph$loss_values,
    tolerance = 1e-12, check.attributes = FALSE
  )
}

test_that("select_features: RecursiveByShapValues summary and final model match the Python oracle", {
  expected <- fixture$expected$recursive_by_shap_values

  result <- catboost.select_features(
    learn_pool,
    features_for_select = fixture$features_for_select,
    num_features_to_select = fixture$num_features_to_select,
    test_pool = test_pool,
    params = base_params(),
    algorithm = "RecursiveByShapValues",
    steps = fixture$steps,
    train_final_model = TRUE
  )

  expect_summary_matches(result, expected)

  expect_equal(length(result$selected_features), fixture$num_features_to_select)
  expect_equal(
    sort(c(result$selected_features, result$eliminated_features)),
    fixture$features_for_select
  )

  expect_false(is.null(result$model))
  expect_equal(result$model$tree_count, expected$final_tree_count)
  final_predict <- catboost.predict(result$model, test_pool, prediction_type = "RawFormulaVal")
  expect_equal(final_predict, expected$final_predict, tolerance = 1e-12, check.attributes = FALSE)
})

test_that("select_features: RecursiveByLossFunctionChange with train_final_model = FALSE matches the Python oracle", {
  expected <- fixture$expected$recursive_by_loss_function_change

  result <- catboost.select_features(
    learn_pool,
    features_for_select = fixture$features_for_select,
    num_features_to_select = fixture$num_features_to_select,
    test_pool = test_pool,
    params = base_params(),
    algorithm = "RecursiveByLossFunctionChange",
    steps = fixture$steps,
    train_final_model = FALSE
  )

  expect_summary_matches(result, expected)
  expect_null(result$model)
})

# catboost-8z4.117 (P10.C): EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange
# and EShapCalcType.{Regular,Approximate,Exact} -- no fixture exists for
# these values (gen_select_features_fixture.py only pinned
# RecursiveByShapValues/RecursiveByLossFunctionChange), and regenerating one
# needs the pinned Python catboost binary, which is unavailable in this
# worktree. Following this file's own precedent (the CLI-range-syntax test
# below, which also has no oracle fixture), these are real, non-oracle
# structural/self-consistency tests: the native NCB::SelectFeatures call is
# genuinely exercised end-to-end with each value and must return a
# selection that partitions features_for_select exactly, proving the value
# is accepted and produces a well-formed result (not silently ignored or
# defaulted).
test_that("select_features: algorithm = 'RecursiveByPredictionValuesChange' produces a valid partition", {
  result <- catboost.select_features(
    learn_pool,
    features_for_select = fixture$features_for_select,
    num_features_to_select = fixture$num_features_to_select,
    params = base_params(),
    algorithm = "RecursiveByPredictionValuesChange",
    steps = fixture$steps,
    train_final_model = FALSE
  )
  expect_equal(length(result$selected_features), fixture$num_features_to_select)
  expect_equal(
    sort(c(result$selected_features, result$eliminated_features)),
    sort(fixture$features_for_select)
  )
})

test_that("select_features: shap_calc_type = 'Regular'/'Approximate'/'Exact' each produce a valid partition", {
  for (sct in c("Regular", "Approximate", "Exact")) {
    result <- catboost.select_features(
      learn_pool,
      features_for_select = fixture$features_for_select,
      num_features_to_select = fixture$num_features_to_select,
      params = base_params(),
      algorithm = "RecursiveByShapValues",
      shap_calc_type = sct,
      steps = fixture$steps,
      train_final_model = FALSE
    )
    expect_equal(length(result$selected_features), fixture$num_features_to_select, info = sct)
    expect_equal(
      sort(c(result$selected_features, result$eliminated_features)),
      sort(fixture$features_for_select),
      info = sct
    )
  }
})

test_that("select_features: shap_calc_type = 'Exact' matches the CLI oracle's elimination decision", {
  # catboost-inm (followup to catboost-8z4.115/P10.A): closes the
  # select-features half of flag:--shap-calc-type. The Python-oracle
  # fixture above never varies shap_calc_type away from its "Regular"
  # default, so this differentially tests it directly against the CLI
  # oracle (Python oracle binary unavailable in this worktree) on the
  # shared smoke dataset -- see
  # tools/oracle/cli/gen_select_features_shap_calc_type_fixture.sh.
  #
  # Compares the elimination decision (selected_features/eliminated_features/
  # removed_features_count), which matches the CLI oracle exactly: real
  # evidence that shap_calc_type = "Exact" is wired through to the same
  # feature ranking as the CLI, not merely accepted-and-ignored (the gap
  # this row's justification named). The loss_graph's absolute loss_values
  # do NOT match the CLI oracle at any tolerance tried (~1.6% relative,
  # investigated: same params, same shap_calc_type, same dataset both
  # sides, first-step baseline loss already differs before any feature is
  # removed) -- a distinct, uninvestigated select-features-mode test-set
  # evaluation discrepancy, not something this fixture's scope covers.
  # Filed as catboost-hpk (discovered-from) rather than silently dropped or
  # papered over with a loose tolerance.
  oracle <- jsonlite::fromJSON(
    testthat::test_path("..", "fixtures", "oracle-cli", "select_features_shap_calc_type.json"),
    simplifyVector = TRUE
  )

  smoke_pool <- catboost.load_pool(
    testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv"),
    column_description = testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd"),
    delimiter = ",", has_header = TRUE, thread_count = 1
  )

  base_call <- function(sct) {
    catboost.select_features(
      smoke_pool,
      test_pool = smoke_pool,
      features_for_select = c(0, 1),
      num_features_to_select = 1,
      params = list(
        loss_function = "Logloss", iterations = 20, depth = 4, learning_rate = 0.1,
        random_seed = 42, thread_count = 1, logging_level = "Silent"
      ),
      algorithm = "RecursiveByShapValues",
      shap_calc_type = sct,
      steps = 2,
      train_final_model = FALSE
    )
  }
  result <- base_call("Exact")

  expect_equal(sort(result$selected_features), sort(oracle$selected_features), check.attributes = FALSE)
  expect_equal(sort(result$eliminated_features), sort(oracle$eliminated_features), check.attributes = FALSE)
  expect_equal(
    result$loss_graph$removed_features_count, oracle$loss_graph$removed_features_count,
    check.attributes = FALSE
  )

  # shap_calc_type has a real numeric effect (not accepted-and-ignored):
  # "Approximate"'s intermediate SHAP estimates differ from "Exact"'s.
  result_approx <- base_call("Approximate")
  expect_false(isTRUE(all.equal(
    result$loss_graph$loss_values, result_approx$loss_graph$loss_values
  )))
})

test_that("select_features accepts the CLI range syntax for features_for_select", {
  result <- catboost.select_features(
    learn_pool,
    features_for_select = "0,2-4",
    num_features_to_select = 2,
    params = base_params(),
    steps = 1,
    train_final_model = FALSE
  )
  expect_equal(
    sort(c(result$selected_features, result$eliminated_features)),
    c(0, 2, 3, 4)
  )
})

test_that("select_features rejects missing selection arguments and a non-pool", {
  expect_error(
    catboost.select_features(learn_pool, num_features_to_select = 2, params = base_params()),
    "You should specify features_for_select"
  )
  expect_error(
    catboost.select_features(learn_pool, features_for_select = c(0, 1), params = base_params()),
    "You should specify num_features_to_select"
  )
  expect_error(
    catboost.select_features(42, features_for_select = c(0, 1), num_features_to_select = 1),
    "Expected catboost.Pool"
  )
})
