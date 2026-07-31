# Fork-owned CMake dependency provider (catboost-8z4.13 / P1.5): the
# Conan-free replacement for vendor/catboost/cmake/conan_provider.cmake.
#
# Why it exists
# -------------
# vendor/catboost's generated CMakeLists.*.txt call `find_package(OpenSSL
# REQUIRED)` and then link the target `openssl::openssl` -- NOT CMake's own
# `OpenSSL::SSL`/`OpenSSL::Crypto`. `openssl::openssl` is a Conan CMakeDeps
# target name; nothing in CMake's FindOpenSSL module produces it. Call sites
# in the catboostr link closure (all under the read-only pin, quoted here so
# the coupling is auditable without opening the tree):
#   vendor/catboost/library/cpp/openssl/init/CMakeLists.linux-x86_64.txt:15,25
#   vendor/catboost/library/cpp/openssl/holders/CMakeLists.linux-x86_64.txt:15
#   vendor/catboost/library/cpp/openssl/method/CMakeLists.linux-x86_64.txt:15
#   vendor/catboost/library/cpp/neh/CMakeLists.linux-x86_64.txt:15
#
# So a Conan-free build has to do exactly two things: (1) stop
# `find_package(OpenSSL REQUIRED)` from failing, and (2) define
# `openssl::openssl`. This file does both, against the pinned,
# SHA256-verified openssl that configure built from
# vendor/thirdparty/openssl-<version>.tar.gz.
#
# Injected into the disposable copy's configure step via
# -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES (a ";"-joined list also carrying
# cmake/gate-test-tool-dirs.cmake). It is NEVER applied to vendor/catboost/
# itself. cmake_language(SET_DEPENDENCY_PROVIDER) may only be called from a
# file included that way -- which is precisely how conan_provider.cmake was
# wired, so this is a like-for-like swap of the provider, not a new mechanism.
#
# Fail-closed: no system-openssl fallback, no "find it somewhere" search. If
# CATBOOSTR_OPENSSL_ROOT is not a real prefix built from the pinned tarball,
# configure stops. An unpinned openssl silently linked into libcatboostr.so is
# the exact hole this ticket closes.

if(NOT DEFINED CATBOOSTR_OPENSSL_ROOT OR CATBOOSTR_OPENSSL_ROOT STREQUAL "")
  message(FATAL_ERROR
    "CATBOOSTR_OPENSSL_ROOT is not set. It must point at the prefix of the "
    "pinned openssl that ./configure built from vendor/thirdparty/. "
    "This build refuses to fall back to a system openssl.")
endif()

set(_catboostr_ssl_inc "${CATBOOSTR_OPENSSL_ROOT}/include")
set(_catboostr_ssl_lib "${CATBOOSTR_OPENSSL_ROOT}/lib/libssl.a")
set(_catboostr_crypto_lib "${CATBOOSTR_OPENSSL_ROOT}/lib/libcrypto.a")

foreach(_f "${_catboostr_ssl_inc}/openssl/ssl.h" "${_catboostr_ssl_lib}" "${_catboostr_crypto_lib}")
  if(NOT EXISTS "${_f}")
    message(FATAL_ERROR
      "Pinned openssl is incomplete: ${_f} is missing. "
      "CATBOOSTR_OPENSSL_ROOT=${CATBOOSTR_OPENSSL_ROOT}")
  endif()
endforeach()

add_library(openssl::openssl INTERFACE IMPORTED GLOBAL)
set_target_properties(openssl::openssl PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_catboostr_ssl_inc}"
  # libssl before libcrypto: libssl references libcrypto, and the link is a
  # single-pass static archive link (see the -fuse-ld=lld note in configure).
  INTERFACE_LINK_LIBRARIES "${_catboostr_ssl_lib};${_catboostr_crypto_lib};${CMAKE_DL_LIBS}"
)
message(STATUS "catboostr-fork P1.5: openssl::openssl -> ${CATBOOSTR_OPENSSL_ROOT} (pinned, SHA256-verified)")

# Satisfy `find_package(OpenSSL REQUIRED)` without letting CMake go looking
# for an unpinned system openssl. Only OpenSSL is claimed; every other
# find_package() falls through to CMake's built-in implementation because
# this provider leaves <PackageName>_FOUND unset for them.
macro(catboostr_provide_dependency method package_name)
  if("${package_name}" STREQUAL "OpenSSL")
    set(OpenSSL_FOUND TRUE)
    set(OPENSSL_FOUND TRUE)
    set(OPENSSL_INCLUDE_DIR "${_catboostr_ssl_inc}")
    set(OPENSSL_LIBRARIES "${_catboostr_ssl_lib};${_catboostr_crypto_lib}")
    set(OPENSSL_VERSION "${CATBOOSTR_OPENSSL_VERSION}")
  endif()
endmacro()

cmake_language(SET_DEPENDENCY_PROVIDER catboostr_provide_dependency
  SUPPORTED_METHODS FIND_PACKAGE)
