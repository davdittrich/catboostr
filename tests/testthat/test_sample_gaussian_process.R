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
