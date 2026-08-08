# Shared fixtures for the custom-objective / custom-eval-metric test files
# (test_custom_objective.R, test_custom_eval_metric.R,
# test_custom_objective_metric_search_cv.R). Auto-sourced by testthat
# (helper-*.R convention) before any test file runs.
#
# Only the genuinely byte-identical pieces live here: the synthetic-data
# preamble (set.seed/n/features/label, identical across all three files),
# and rmse_custom_eval_metric (identical in test_custom_eval_metric.R and
# test_custom_objective_metric_search_cv.R).
#
# rmse_custom_objective is deliberately NOT extracted: test_custom_objective.R's
# top-level version and test_custom_objective_metric_search_cv.R's top-level
# version are functionally equivalent but not byte-identical (der1/der2
# intermediates vs. inlined cbind()), so each stays file-local. Each file's
# own common_params()/common_cv_params() helper is likewise deliberately
# differently-shaped (test_custom_eval_metric.R's is deeper/longer for
# early-stopping statistical power) and stays file-local too.

set.seed(20260807)
n <- 300
features <- data.frame(
  x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n), x5 = rnorm(n)
)
label <- with(features, 2 * x1 - 1.5 * x2 + 0.5 * x3 + rnorm(n, sd = 0.3))

# Mirrors TRMSEMetric exactly (metric.cpp:716-748).
rmse_custom_eval_metric <- list(
  evaluate = function(approx, target, weight) {
    w <- if (is.null(weight)) rep(1, length(target)) else weight
    diff <- approx[, 1] - target
    list(error = sum(w * diff^2), weight = sum(w))
  },
  is_max_optimal = function() FALSE,
  get_final_error = function(error) sqrt(error[1] / (error[2] + 1e-38))
)
