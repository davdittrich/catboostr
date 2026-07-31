# Fork-owned CMake override (catboost-8z4.10 / P1.2): skip unconditionally
# add_subdirectory()'d unit-test, benchmark, and standalone dev-tool
# directories that -DCATBOOST_COMPONENTS=R-package/R-package-fork does not
# gate, so the vendored source tree does not need to carry their content for
# the R-package build to configure and link.
#
# Root cause: vendor/catboost/catboost/CMakeLists.linux-x86_64.txt
# unconditionally add_subdirectory()s "libs", "private", "tools" (etc.)
# regardless of CATBOOST_COMPONENTS; CATBOOST_COMPONENTS is only consulted
# by the top-level product front-ends' OWN CMakeLists.txt (app, R-package,
# python-package, jvm-packages, spark), each gating a single fixed child.
# There is no upstream switch that reaches into every "ut"/"benchmark"
# leaf spread across catboost/libs, catboost/private/libs, util/*, and
# library/cpp/testing/* (confirmed absent: grepped vendor/catboost/cmake/
# for an existing test/tool switch before writing this file).
#
# This file is NEVER applied to vendor/catboost/ itself (read-only pin). It
# is injected into a disposable copy's configure step via
# -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES (a ";"-joined list also carrying
# cmake/conan_provider.cmake), which CMake includes immediately after the
# project() call in the copy's root CMakeLists.txt -- i.e. before any
# add_subdirectory() in the CATBOOST project has run. CMake macro
# resolution is dynamic-scoped like variables (visible to every directory
# processed after this point, not just the file's own lexical scope), so
# redefining add_subdirectory() here shadows the builtin for the whole
# configure. Redefining a command via function()/macro() causes CMake to
# make the previous (builtin) definition callable via the same name
# prefixed with "_" -- used below to invoke the real add_subdirectory() for
# every directory not on the gate list.
#
# Verified empirically (catboost-8z4.10): none of the paths below appear in
# `ninja -t deps` (the real per-file dependency database from a completed
# `ninja catboostr` build) or in the catboostr CMake target_link_libraries
# list -- i.e. none are a compile or link input for the R-package build.
# library/cpp/testing/common is deliberately EXCLUDED: it IS a real link
# dependency (libcpp-testing-common.a, env.cpp/network.cpp/probe.cpp/
# scope.cpp -- pulled in transitively by model_export/eval_result's use of
# GetArcadiaSourcePath(), per src/CMakeLists.txt's vcs_info(catboostr)
# comment), so only its "benchmark"/"gbenchmark"/"hook"/"unittest"/
# "unittest_main" sibling subdirectories are gated, not their parent.

set(_CATBOOSTR_GATED_TEST_TOOL_DIRS
  "catboost/tools"
  "contrib/restricted/google/benchmark"
  "library/cpp/testing/benchmark"
  "library/cpp/testing/gbenchmark"
  "library/cpp/testing/hook"
  "library/cpp/testing/unittest"
  "library/cpp/testing/unittest_main"
  "catboost/libs/calc_metrics/ut"
  "catboost/libs/carry_model/ut"
  "catboost/libs/dataset_statistics/ut"
  "catboost/libs/data/ut"
  "catboost/libs/helpers/parallel_sort/ut"
  "catboost/libs/helpers/ut"
  "catboost/libs/metrics/ut"
  "catboost/libs/model/ut"
  "catboost/libs/train_lib/ut"
  "catboost/private/libs/algo/ut"
  "catboost/private/libs/algo_helpers/ut"
  "catboost/private/libs/data_util/ut"
  "catboost/private/libs/embedding_features/ut"
  "catboost/private/libs/feature_estimator/ut"
  "catboost/private/libs/functools/ut"
  "catboost/private/libs/options/ut"
  "catboost/private/libs/quantization/ut"
  "catboost/private/libs/quantization_schema/ut"
  "catboost/private/libs/quantized_pool/ut"
  "catboost/private/libs/text_features/ut"
  "catboost/private/libs/text_processing/ut"
  "util/charset/ut"
  "util/datetime/benchmark"
  "util/datetime/ut"
  "util/digest/benchmark"
  "util/digest/ut"
  "util/draft/ut"
  "util/folder/ut"
  "util/generic/ut"
  "util/memory/ut"
  "util/network/ut"
  "util/random/ut"
  "util/stream/ut"
  "util/string/ut"
  "util/system/ut"
  "util/thread/ut"
  "util/ut"
)

function(add_subdirectory _catboostr_gate_dir)
  get_filename_component(_catboostr_gate_abs "${_catboostr_gate_dir}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  file(RELATIVE_PATH _catboostr_gate_rel "${CMAKE_SOURCE_DIR}" "${_catboostr_gate_abs}")
  if(_catboostr_gate_rel IN_LIST _CATBOOSTR_GATED_TEST_TOOL_DIRS)
    message(STATUS "catboostr-fork P1.2: gating out test/tool/benchmark subdir: ${_catboostr_gate_rel}")
  else()
    _add_subdirectory(${ARGV})
  endif()
endfunction()
