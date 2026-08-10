context("test_confusion_matrix.R")

# catboost-8z4.117 (P10.C) disposition test: catboost.utils.get_confusion_matrix
# (catboost/python-package/catboost/utils.py:334) has no dedicated R wrapper,
# but its native computation (MakeConfusionMatrix,
# catboost/private/libs/algo/confusion_matrix.cpp:24-57) is exactly:
#   cm[trueLabel * classesCount + predictedLabel] += 1
# for predictedLabel = argmax(RawFormulaVal) per row (GetApproxClass) -- a
# row(=true class)-by-column(=predicted class), ascending-0-based-index
# count table. R already has every piece needed to build the identical
# table with existing, exported functions: catboost.predict(...,
# prediction_type = "Class") for the predicted labels (already proven
# bit-exact against this exact Python oracle fixture's RawFormulaVal output
# in test_predict_classes.R -- Class is that same argmax) and base R's
# table() for the counting. No new R/src capability is implied; this test
# proves the composition is correct, not merely plausible.
#
# Reuses tests/fixtures/oracle/predict_classes.json's CatBoost/MultiClass
# case (same fixture, no regeneration needed): its `expected` field is
# Python's own oracle RawFormulaVal matrix, already oracle-verified
# bit-for-bit against R's predict() in test_predict_classes.R.

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "predict_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

test_that("get_confusion_matrix: predict(Class) + table() reproduces MakeConfusionMatrix's row=true/col=predicted layout", {
  inputs <- fixture$inputs$CatBoost
  expected_raw <- fixture$expected$CatBoost
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = c(list(loss_function = "MultiClass"), COMMON_PARAMS))

  raw <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  # Sanity: R's raw predictions still match the pinned Python oracle (same
  # assertion as test_predict_classes.R); the confusion matrix built below
  # is only meaningful if this holds.
  expect_equal(as.numeric(raw), as.numeric(expected_raw), tolerance = TOL)

  n_classes <- ncol(raw)
  predicted <- catboost.predict(model, pool, prediction_type = "Class")
  # Class must equal argmax(RawFormulaVal) exactly -- this is what
  # GetApproxClass computes natively for MakeConfusionMatrix's predictedLabel.
  argmax_from_raw <- apply(raw, 1, which.max) - 1L
  expect_equal(as.integer(predicted), as.integer(argmax_from_raw))

  true_label <- as.integer(inputs$label)
  cm <- table(
    factor(true_label, levels = 0:(n_classes - 1)),
    factor(predicted, levels = 0:(n_classes - 1))
  )

  # Structural invariants MakeConfusionMatrix guarantees: square, one entry
  # per object, and identical to the same count independently computed from
  # the oracle's own argmax(RawFormulaVal).
  expect_equal(dim(cm), c(n_classes, n_classes))
  expect_equal(sum(cm), length(true_label))
  expected_argmax <- apply(expected_raw, 1, which.max) - 1L
  expected_cm <- table(
    factor(true_label, levels = 0:(n_classes - 1)),
    factor(expected_argmax, levels = 0:(n_classes - 1))
  )
  expect_equal(unclass(cm), unclass(expected_cm), check.attributes = FALSE)
})
