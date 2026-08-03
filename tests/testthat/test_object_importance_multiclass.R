context("test_object_importance_multiclass.R")

# P4.5 (catboost-8z4.54): catboost.get_object_importance MultiClass
# error-path parity (upstream #869).
#
# docs/phase-2/P2.4-report.md Finding 1 (confidence 95) root-caused this to
# catboost/private/libs/documents_importance/ders_helpers.cpp:95,
# GetEvaluateDerivativesFunc: a hardcoded switch over ELossFunction that has
# no MultiClass case and hits CB_ENSURE(false, ...) in its default branch.
# Python's exact reproduced message (P2.4-report.md line 35):
#   "catboost/private/libs/documents_importance/ders_helpers.cpp:95:
#    Error function MultiClass is not supported yet in ostr mode"
#
# This is core-source behavior (vendor/catboost/), not an R wrapper bug: no
# core patch is in scope (catboost-8z4.54 constraint). R's
# R_API_BEGIN/R_API_END (src/catboostr.cpp) already catches the underlying
# std::exception and re-raises it via R's error(e.what()), which propagates
# the C++ exception's message verbatim -- confirmed identical to Python's,
# byte for byte. This test locks that parity in as a regression guard; it
# is not implementing new error translation.

test_that("get_object_importance: MultiClass raises the same ders_helpers.cpp:95 error as Python (#869)", {
  set.seed(1)
  classes <- c(0, 1, 2)
  target <- sample(classes, size = 200, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target)),
                         f2 = rnorm(length(target)),
                         f3 = rnorm(length(target)))
  pool <- catboost.load_pool(features, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345,
                 allow_writing_files = FALSE,
                 logging_level = "Silent")

  model <- catboost.train(pool, NULL, params)

  expect_error(
    catboost.get_object_importance(model, pool, pool, top_size = 3),
    regexp = paste0(
      "catboost/private/libs/documents_importance/ders_helpers.cpp:95: ",
      "Error function MultiClass is not supported yet in ostr mode"
    ),
    fixed = TRUE
  )
})
