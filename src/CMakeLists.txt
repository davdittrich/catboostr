# Fork-owned CMakeLists for the catboostr native target.
#
# This file is NOT part of vendor/catboost/ (which is read-only, pinned upstream).
# It is injected into a disposable copy of that tree via add_subdirectory(),
# appended to the copy's root CMakeLists.txt AFTER upstream's own
# add_subdirectory(catboost) chain has already defined every upstream target
# this links against (catboost-8z4.8 spike; docs/superpowers/specs/2026-07-30-
# catboostr-design.md section 4.1). CMAKE_PROJECT_TOP_LEVEL_INCLUDES does not
# work for this: it runs before upstream's targets exist.
#
# Target name is "catboostr" -- matching upstream's own target/NAMESPACE
# useDynLib(libcatboostr) so the produced shared object is libcatboostr.so/.dylib.
# Upstream's own catboost/R-package/src/CMakeLists.txt defines a target of the
# SAME name when CATBOOST_COMPONENTS matches "R-package"/"R-PACKAGE"
# (catboost/R-package/CMakeLists.txt), which would collide with this one.
# The invoking configure step must therefore pass a CATBOOST_COMPONENTS value
# that does NOT match "R-package"/"R-PACKAGE" (e.g. "R-package-fork"), so
# upstream's own R-package/src subdirectory is never added while
# catboost/libs and catboost/private (this target's actual dependencies)
# remain unconditionally configured.

add_shared_library(catboostr)

# R's REAL include path -- never vendor/catboost's contrib/libs/r-lang, which
# lacks Rversion.h and R_ext/Visibility.h. Must be supplied by the caller,
# sourced from `R CMD config --cppflags` / `R.home("include")` on the
# installing user's machine (spec section 4.1, "who compiles what").
if(NOT DEFINED R_INCLUDE_DIR OR R_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR
    "R_INCLUDE_DIR must be set to R's real include directory "
    "(R CMD config --cppflags / R.home(\"include\")). "
    "vendor/catboost's contrib/libs/r-lang is NOT a substitute: it lacks "
    "Rversion.h and R_ext/Visibility.h.")
endif()

target_include_directories(catboostr PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${R_INCLUDE_DIR}
)

target_sources(catboostr PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/catboostr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init.c
)

# Same target list upstream's own catboostr links
# (vendor/catboost/catboost/R-package/src/CMakeLists.linux-x86_64.txt),
# verified working end-to-end by the catboost-8z4.8 spike -- except
# contrib-libs-linux-headers, which is itself platform-gated upstream
# (only appears in the linux-x86_64 variant of that file, not the
# darwin one) and fails "library not found" on macOS if linked
# unconditionally (P1.13 hosted-CI finding, 2026-07-31).
if(NOT APPLE)
  target_link_libraries(catboostr PUBLIC contrib-libs-linux-headers)
endif()

target_link_libraries(catboostr PUBLIC
  contrib-libs-cxxsupp
  yutil
  build-cow-on
  catboost-libs-cat_feature
  catboost-libs-data
  catboost-libs-eval_result
  catboost-libs-fstr
  libs-gpu_config-maybe_have_cuda
  catboost-libs-logging
  catboost-libs-model
  catboost-libs-train_lib
  private-libs-algo
  private-libs-data_util
  private-libs-documents_importance
  private-libs-init
  private-libs-options
)

target_allocator(catboostr
  cpp-malloc-mimalloc
)

if(NOT APPLE)
  target_link_options(catboostr PRIVATE
    -ldl
    -lrt
    -Wl,--no-as-needed
    -fPIC
    -Wl,-z,notext
  )
else()
  target_link_options(catboostr PRIVATE
    -fPIC
  )
endif()

use_export_script(catboostr
  ${CMAKE_CURRENT_SOURCE_DIR}/catboostr.exports
)

# Required: model_export/pmml+onnx helpers and testing/common/env.cpp pull in
# GetArcadiaSourcePath/GetProgramSvnVersion transitively. Omitting this fails
# at link time (catboost-8z4.8 spike, section 4 finding #2), not at configure.
vcs_info(catboostr)
