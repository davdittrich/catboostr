context("test_version_skew.R")

# P1.16: version-skew detection between the installed R package (DESCRIPTION
# Version) and the tag compiled into libcatboostr via upstream's
# __vcs_version__.c mechanism (CatBoostVersion_R -> GetTag()).
#
# .catboost.check_version_skew() is the pure comparison+error function called
# from .onLoad (R/zzz.R) after fetching the real compiled tag through
# .Call("CatBoostVersion_R"). Testing it directly with independently-chosen
# inputs is a genuine mismatched pair (not two reads of the same source): the
# DESCRIPTION-derived side and the compiled-in side are supplied here as two
# unrelated literals, exactly as they would be two unrelated reads (DESCRIPTION
# vs. the .so) at real load time.

test_that("mismatched compiled tag and package version raises a named, actionable error", {
  pkg_version <- as.character(utils::packageVersion("catboostr"))
  mismatched_tag <- "v9.9.9"
  expect_false(mismatched_tag == paste0("v", pkg_version))

  err <- tryCatch(
    catboostr:::.catboost.check_version_skew(pkg_version, mismatched_tag),
    error = function(e) e
  )

  expect_s3_class(err, "error")
  expect_match(conditionMessage(err), "version skew", fixed = TRUE)
  expect_match(conditionMessage(err), pkg_version, fixed = TRUE)
  expect_match(conditionMessage(err), mismatched_tag, fixed = TRUE)
  expect_match(conditionMessage(err), "[Rr]einstall", perl = TRUE)
})

test_that("matching compiled tag and package version loads silently", {
  pkg_version <- as.character(utils::packageVersion("catboostr"))
  matching_tag <- paste0("v", pkg_version)

  expect_true(
    isTRUE(catboostr:::.catboost.check_version_skew(pkg_version, matching_tag))
  )
  expect_silent(catboostr:::.catboost.check_version_skew(pkg_version, matching_tag))
})

test_that("an empty compiled tag (untagged build tree) does not falsely report skew", {
  pkg_version <- as.character(utils::packageVersion("catboostr"))
  expect_silent(catboostr:::.catboost.check_version_skew(pkg_version, ""))
})
