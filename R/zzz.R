# P1.16: version-skew detection between the installed R package and the
# loaded libcatboostr shared object.
#
# The compiled-in identifier is the build tag from upstream's existing
# __vcs_version__.c mechanism (cmake/common.cmake's vcs_info(), fed by
# build/scripts/vcs_info.py / generate_vcs_info.py), exposed here via the
# fork's own CatBoostVersion_R .Call entry point (src/catboostr.cpp). This is
# an independently-derived value (compiled into the .so from the build
# tree's git tag), not read from DESCRIPTION, so a genuine mismatch is
# observable.

#' @keywords internal
.catboost.version_skew_message <- function(pkg_version, compiled_tag) {
  sprintf(
    paste0(
      "catboostr version skew detected: the installed R package is version "
      , "'%s' (from DESCRIPTION), but the loaded libcatboostr shared object "
      , "was built from tag '%s'. These must match, or R-level code and the "
      , "native library can silently disagree on capabilities. Reinstall "
      , "catboostr so both are rebuilt together from the same source tree "
      , "(e.g. `R CMD INSTALL .` or `install.packages(\"catboostr\")`)."
    ),
    pkg_version, compiled_tag
  )
}

#' @keywords internal
.catboost.check_version_skew <- function(pkg_version, compiled_tag) {
  compiled_version <- sub("^v", "", compiled_tag)
  if (nzchar(compiled_version) && compiled_version != pkg_version) {
    stop(.catboost.version_skew_message(pkg_version, compiled_tag), call. = FALSE)
  }
  invisible(TRUE)
}

.onLoad <- function(libname, pkgname) {
  compiled_tag <- tryCatch(.Call("CatBoostVersion_R"), error = function(e) "")
  pkg_version <- as.character(utils::packageVersion(pkgname, lib.loc = libname))
  .catboost.check_version_skew(pkg_version, compiled_tag)
}
