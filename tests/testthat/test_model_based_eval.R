context("test_model_based_eval.R")

# P4.8 (catboost-8z4.57) -- CLI `model-based-eval` mode parity.
#
# The mode is GPU-only: the CPU trainer refuses it outright
# (train_model.cpp:942, "Model based eval is not implemented for CPU") and
# PlainJsonToOptions only maps its options when task_type is GPU
# (plain_options_helper.cpp:394). This machine has no CUDA device, and neither
# does the pinned CLI oracle binary have one to run against, so the *numeric*
# differential comparison declared for `mode:model-based-eval` in
# tests/fixtures/parity/matrix.dispositioned.json cannot be produced here. Per
# design spec 4.4/9.1 a skipped test never counts as green, so that matrix row
# stays red; see docs/phase-4/P4.8-report.md for the full analysis.
#
# What *is* verifiable without CUDA is everything the R wrapper is responsible
# for. `ModelBasedEval` (train_model.cpp:1533) loads the datasets, validates the
# tested feature indices against the real pool and builds the quantized feature
# info before the GPU learning library is ever requested (train_model.cpp:1439),
# so reaching that GPU error proves the whole option pipeline -- plain-JSON
# option mapping, feature-name/range resolution, ignored-feature conversion,
# column description, delimiter and header handling -- executed correctly.

# Same 40-row CLI fixture every other oracle-cli test uses.
fixture_paths <- function() {
    list(data = testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv"),
         cd = testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd"))
}

mbe <- function(features_to_evaluate = "0", ..., extra_params = list()) {
    fx <- fixture_paths()
    train_dir <- tempfile("mbe")
    dir.create(train_dir)
    params <- modifyList(
        list(iterations = 10, logging_level = "Silent", train_dir = train_dir,
             loss_function = "Logloss", task_type = "GPU"),
        extra_params)
    catboost.model_based_eval(
        learn_set = fx$data,
        test_set = fx$data,
        features_to_evaluate = features_to_evaluate,
        column_description = fx$cd,
        params = params,
        offset = 10,
        experiment_count = 2,
        experiment_size = 5,
        delimiter = ",",
        has_header = TRUE,
        ...
    )
}

test_that("catboost.model_based_eval is exported with the documented signature", {
    expect_true(is.function(catboost.model_based_eval))
    expect_equal(
        names(formals(catboost.model_based_eval)),
        c("learn_set", "test_set", "features_to_evaluate", "baseline_model_snapshot",
          "column_description", "params", "offset", "experiment_count", "experiment_size",
          "use_evaluated_features_in_baseline_model", "delimiter", "has_header")
    )
    # Defaults are the C++ TModelBasedEvalOptions constructor defaults
    # (model_based_eval_options.cpp).
    defaults <- formals(catboost.model_based_eval)
    expect_equal(defaults$baseline_model_snapshot, "baseline_model_snapshot")
    expect_equal(defaults$offset, 1000)
    expect_equal(defaults$experiment_count, 200)
    expect_equal(defaults$experiment_size, 5)
    expect_false(defaults$use_evaluated_features_in_baseline_model)
})

test_that("a CPU task type is rejected with a specific error, never a silent fallback", {
    # Mirrors mode_model_based_eval.cpp:40's bare CB_ENSURE, moved ahead of
    # PlainJsonToOptions so the message names the real cause instead of the
    # CLI's `Unknown option {features_to_evaluate}`.
    expect_error(mbe(extra_params = list(task_type = "CPU")),
                 "Model based eval is not implemented for CPU")
    fx <- fixture_paths()
    train_dir <- tempfile("mbe")
    dir.create(train_dir)
    expect_error(
        catboost.model_based_eval(
            learn_set = fx$data, test_set = fx$data, features_to_evaluate = "0",
            column_description = fx$cd,
            params = list(iterations = 10, logging_level = "Silent",
                          train_dir = train_dir, loss_function = "Logloss"),
            offset = 10, experiment_count = 2, experiment_size = 5,
            delimiter = ",", has_header = TRUE),
        "Model based eval is not implemented for CPU")
})

test_that("GPU is requested honestly: without CUDA the call fails, it does not run on CPU", {
    # Design spec 4.4: a GPU-requested-but-unavailable capability must produce a
    # specific, tested error. On a CUDA host this test's expectation is the
    # opposite branch, guarded below.
    err <- tryCatch({ mbe(); NULL }, error = function(e) conditionMessage(e))
    if (is.null(err)) {
        skip("CUDA device present: this build ran model-based eval for real; the numeric differential comparison is still blocked on a pinned CLI oracle fixture generated on the same hardware (see docs/phase-4/P4.8-report.md).")
    }
    expect_match(err, "Can't load GPU learning library|Environment for task type \\[GPU\\] not found|CUDA")
})

test_that("tested feature indices are validated against the real dataset", {
    # ValidateFeaturesToEvaluate (train_model.cpp:1413) runs after LoadPools, so
    # this also proves the datasets were read with the given cd file, delimiter
    # and header setting.
    expect_error(mbe("99"), "Feature index 99 is too large; dataset has only 3 features")
})

test_that("feature names resolve the same way as for the CLI", {
    # `cat1` is the only named feature in smoke.cd. Resolving it gets past the
    # name converter and all the way to the GPU library check.
    expect_error(mbe("cat1"), "Can't load GPU learning library|Environment for task type|CUDA")
    expect_error(mbe("nosuch"), "String 'nosuch' is not a feature name")
})

test_that("a tested feature may not also be ignored", {
    # mode_model_based_eval.cpp:47-50.
    expect_error(mbe("0", extra_params = list(ignored_features = 0)),
                 "Error: feature 0 is ignored")
    expect_error(mbe("cat1", extra_params = list(ignored_features = "cat1")),
                 "Error: feature 2 is ignored")
    # Negative control: ignoring a feature that is not tested is fine, and gets
    # through to the GPU check.
    expect_error(mbe("0", extra_params = list(ignored_features = 1)),
                 "Can't load GPU learning library|Environment for task type|CUDA")
})

test_that("argument validation happens before any dataset is read", {
    expect_error(mbe(features_to_evaluate = ""), "features_to_evaluate must be a single")
    expect_error(mbe(features_to_evaluate = c("0", "1")), "features_to_evaluate must be a single")
    fx <- fixture_paths()
    expect_error(
        catboost.model_based_eval("no/such/file.csv", fx$data, "0"),
        "learn_set file does not exist")
    expect_error(
        catboost.model_based_eval(fx$data, "no/such/file.csv", "0"),
        "test_set file does not exist")
    expect_error(
        catboost.model_based_eval(fx$data, fx$data, "0", column_description = "no/such.cd"),
        "column_description file does not exist")
    # TModelBasedEvalOptions::Validate(), model_based_eval_options.cpp.
    expect_error(
        catboost.model_based_eval(fx$data, fx$data, "0", offset = 5,
                                  experiment_count = 2, experiment_size = 5),
        "offset must be greater than or equal to experiment_count \\* experiment_size")
})

test_that("differential comparison against the CLI oracle is blocked, not green", {
    skip(paste("BLOCKED: mode:model-based-eval is GPU-only and no CUDA device is available,",
               "so neither the pinned CLI oracle nor catboostr can produce numbers to compare.",
               "The parity-matrix row stays red. See docs/phase-4/P4.8-report.md."))
})
