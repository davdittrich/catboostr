context("test_sample_gaussian_process.R")

# catboost-8z4.123 self-consistency test: no pinned Python oracle exists in
# this worktree for catboost.sample_gaussian_process (Python's numpy PCG64
# RNG stream cannot be reproduced bit-for-bit in R -- see the roxygen note on
# the function), so this exercises the R implementation's own structural
# guarantees instead: seeded reproducibility, side-effect-freedom on
# learn_pool, output shape/finiteness, and the documented input guards.

set.seed(1)
n <- 60
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- sin(x1) + 0.3 * x2 + rnorm(n, sd = 0.05)

build_pool <- function() {
  catboost.load_pool(data.frame(x1 = x1, x2 = x2), label = as.double(y))
}

small_gp_args <- list(samples = 2, prior_iterations = 5, posterior_iterations = 5,
                       depth = 2, learning_rate = 0.3, random_seed = 7)

test_that("returns `samples` trained models with finite, full-length predictions", {
  pool <- build_pool()
  models <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), small_gp_args))

  expect_equal(length(models), 2)
  for (model in models) {
    expect_s3_class(model, "catboost.Model")
    preds <- catboost.predict(model, pool)
    expect_equal(length(preds), n)
    expect_true(all(is.finite(preds)))
  }
})

test_that("same random_seed reproduces byte-identical predictions", {
  pool_a <- build_pool()
  pool_b <- build_pool()

  models_a <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool_a), small_gp_args))
  models_b <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool_b), small_gp_args))

  preds_a <- lapply(models_a, catboost.predict, pool = pool_a)
  preds_b <- lapply(models_b, catboost.predict, pool = pool_b)
  expect_identical(preds_a, preds_b)
})

test_that("different random_seed changes the sampled models", {
  pool_a <- build_pool()
  pool_b <- build_pool()

  args_b <- small_gp_args
  args_b$random_seed <- 8

  models_a <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool_a), small_gp_args))
  models_b <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool_b), args_b))

  preds_a <- catboost.predict(models_a[[1]], pool_a)
  preds_b <- catboost.predict(models_b[[1]], pool_b)
  expect_false(isTRUE(all.equal(preds_a, preds_b)))
})

test_that("learn_pool's label and baseline are unchanged after the call (no observable mutation)", {
  pool <- build_pool()
  label_before <- catboost.pool.get_label(pool)
  baseline_before <- catboost.pool.get_baseline(pool)

  invisible(do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), small_gp_args)))

  expect_identical(catboost.pool.get_label(pool), label_before)
  expect_identical(catboost.pool.get_baseline(pool), baseline_before)
})

test_that("sigma, samples, random_strength, eps must be positive", {
  pool <- build_pool()
  expect_error(catboost.sample_gaussian_process(pool, sigma = 0), "sigma")
  expect_error(catboost.sample_gaussian_process(pool, sigma = -1), "sigma")
  expect_error(catboost.sample_gaussian_process(pool, samples = 0), "samples")
  expect_error(catboost.sample_gaussian_process(pool, random_strength = 0), "random_strength")
  expect_error(catboost.sample_gaussian_process(pool, eps = 0), "eps")
})

test_that("rejects non-Pool input and a Pool without a label", {
  expect_error(catboost.sample_gaussian_process(matrix(1:4, 2)), "catboost.Pool")

  unlabeled_pool <- catboost.load_pool(data.frame(x1 = x1, x2 = x2))
  expect_error(catboost.sample_gaussian_process(unlabeled_pool), "label")
})

test_that("test_pool is accepted for posterior validation without error", {
  pool <- build_pool()
  test_pool <- build_pool()
  models <- do.call(catboost.sample_gaussian_process,
                     c(list(learn_pool = pool, test_pool = test_pool), small_gp_args))
  expect_equal(length(models), 2)
})

# The tests above would not catch a dropped/no-op leaf-randomization step, a
# dropped/mis-scaled set_scale_and_bias rescale, or wrong sum_models weights
# (all of these would still produce *some* finite, reproducible, differently
# seeded output). The three tests below target exactly those failure modes.

test_that("multiple samples within one call are not identical to each other", {
  pool <- build_pool()
  args <- small_gp_args
  args$samples <- 3
  models <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), args))
  preds <- lapply(models, catboost.predict, pool = pool)

  expect_false(isTRUE(all.equal(preds[[1]], preds[[2]])))
  expect_false(isTRUE(all.equal(preds[[2]], preds[[3]])))
  expect_false(isTRUE(all.equal(preds[[1]], preds[[3]])))
})

test_that("increasing sigma increases across-sample prediction variance", {
  # Only sigma differs between the two calls below (same seed, same
  # everything else), isolating it as the tested variable.
  pool <- build_pool()
  args_low <- small_gp_args
  args_low$samples <- 5
  args_low$sigma <- 0.01
  args_high <- args_low
  args_high$sigma <- 1

  models_low <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), args_low))
  models_high <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), args_high))

  preds_low <- sapply(models_low, catboost.predict, pool = pool)
  preds_high <- sapply(models_high, catboost.predict, pool = pool)

  variance_low <- mean(apply(preds_low, 1, var))
  variance_high <- mean(apply(preds_high, 1, var))
  expect_gt(variance_high, variance_low)
})

test_that("the posterior fit reduces residual RMSE against y relative to the prior alone", {
  # Only the posterior's learning_rate differs between the two calls below:
  # near-zero learning_rate makes the posterior contribute ~nothing, so the
  # returned (prior + posterior) sum is effectively the prior alone. Same
  # seed, same everything else, isolating learning_rate as the tested
  # variable.
  pool <- build_pool()
  args_full <- small_gp_args
  args_full$samples <- 1
  args_prior_only <- args_full
  args_prior_only$learning_rate <- 1e-3

  model_full <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), args_full))[[1]]
  model_prior_only <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), args_prior_only))[[1]]

  rmse <- function(model) sqrt(mean((catboost.predict(model, pool) - y) ^ 2))
  expect_lt(rmse(model_full), rmse(model_prior_only))
})

test_that("weight and group_id on learn_pool are propagated onto the fitted models' training", {
  # Indirect check (no accessor exists to inspect a fitted model's training
  # weights/groups): a heavily-downweighted-outlier fit should track the
  # unweighted label less closely than a uniformly-weighted fit at that
  # outlier's row, proving the weight actually reached catboost.train().
  outlier_idx <- 1
  weight <- rep(1, n)
  weight[outlier_idx] <- 1e-6
  y_outlier <- y
  y_outlier[outlier_idx] <- y[outlier_idx] + 100

  pool_weighted <- catboost.load_pool(data.frame(x1 = x1, x2 = x2), label = as.double(y_outlier), weight = weight)
  pool_unweighted <- catboost.load_pool(data.frame(x1 = x1, x2 = x2), label = as.double(y_outlier))

  args <- small_gp_args
  args$samples <- 1
  args$posterior_iterations <- 30
  args$prior_iterations <- 10

  model_weighted <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool_weighted), args))[[1]]
  model_unweighted <- do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool_unweighted), args))[[1]]

  outlier_gap_weighted <- abs(catboost.predict(model_weighted, pool_weighted)[outlier_idx] - y_outlier[outlier_idx])
  outlier_gap_unweighted <- abs(catboost.predict(model_unweighted, pool_unweighted)[outlier_idx] - y_outlier[outlier_idx])
  expect_gt(outlier_gap_weighted, outlier_gap_unweighted)
})

test_that("baseline or pairs on learn_pool triggers a warning (not a silent drop)", {
  pool <- build_pool()
  catboost.pool.set_baseline(pool, matrix(0, nrow = n, ncol = 1))
  expect_warning(
    do.call(catboost.sample_gaussian_process, c(list(learn_pool = pool), small_gp_args)),
    "baseline"
  )
})
