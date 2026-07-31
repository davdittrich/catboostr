#!/usr/bin/env bash
# Verify that configure's disposable build-source COPY of vendor/catboost --
# the tree it actually compiles from (configure sec. 3, "SRC_COPY") -- is the
# SAME tree tools/vendor/PRUNED_MANIFEST.sha256 describes. This is the check
# named in P1.13's 2026-07-31 AMENDMENT: the R source tarball ships none of
# vendor/ (.Rbuildignore, P1.6), so hashing tarball contents against the
# manifest would be a vacuous 0-file comparison. What actually needs proving
# is that acquisition, the committed manifest, and the tree the C++ compiler
# reads from are the same bytes -- no silent divergence between the three.
#
# `configure` appends exactly one documented splice to the copy's top-level
# CMakeLists.txt (a blank line, then one add_subdirectory(...) call) to wire
# the R-package-fork target in (configure sec. 3, "Two-argument
# add_subdirectory"). That is the sole expected, intentional divergence from
# the manifest; this script verifies it is EXACTLY that -- nothing more --
# and straight byte-for-byte hashes every other manifested file.
#
# Usage: tools/vendor/verify-build-copy.sh <src_copy_dir> [vendor_src_dir]
#   src_copy_dir   configure's WORKDIR/src-copy (e.g.
#                  $CATBOOSTR_BUILD_DIR/src-copy after a configure run).
#   vendor_src_dir defaults to vendor/catboost -- the fresh acquisition the
#                  copy was made from, needed to know the original (pre-splice)
#                  line count of CMakeLists.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MANIFEST="${REPO_ROOT}/tools/vendor/PRUNED_MANIFEST.sha256"
SRC_COPY="${1:?usage: verify-build-copy.sh <src_copy_dir> [vendor_src_dir]}"
VENDOR_SRC="${2:-${REPO_ROOT}/vendor/catboost}"
CMAKELISTS_REL="./CMakeLists.txt"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "FATAL: manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${SRC_COPY}" ]]; then
  echo "FATAL: build-source copy not found: ${SRC_COPY} -- configure did not reach its copy step (configure sec. 3)." >&2
  exit 1
fi
if [[ ! -f "${VENDOR_SRC}/CMakeLists.txt" ]]; then
  echo "FATAL: ${VENDOR_SRC}/CMakeLists.txt not found -- run tools/vendor/acquire.sh first." >&2
  exit 1
fi

FAIL=0

# 1. The one expected divergence: top-level CMakeLists.txt.
orig_lines=$(wc -l < "${VENDOR_SRC}/CMakeLists.txt")
copy_lines=$(wc -l < "${SRC_COPY}/CMakeLists.txt")
expected_hash=$(grep -F "  ${CMAKELISTS_REL}" "${MANIFEST}" | awk '{print $1}')

if [[ -z "${expected_hash}" ]]; then
  echo "FATAL: ${CMAKELISTS_REL} not present in ${MANIFEST} -- manifest shape changed, re-verify this script." >&2
  exit 1
fi

if [[ "${copy_lines}" -ne $((orig_lines + 2)) ]]; then
  echo "MISMATCH ${CMAKELISTS_REL}: build-source copy has ${copy_lines} lines, expected exactly $((orig_lines + 2)) (original ${orig_lines} + configure's 2-line splice)." >&2
  FAIL=1
else
  splice_blank=$(sed -n "$((orig_lines + 1))p" "${SRC_COPY}/CMakeLists.txt")
  splice_line=$(sed -n "$((orig_lines + 2))p" "${SRC_COPY}/CMakeLists.txt")
  if [[ -n "${splice_blank}" ]] || ! [[ "${splice_line}" =~ ^add_subdirectory\(.*/src\ \$\{CMAKE_BINARY_DIR\}/catboostr-fork-build\)$ ]]; then
    echo "MISMATCH ${CMAKELISTS_REL}: last 2 lines are not configure's documented splice (blank line + add_subdirectory(.../src \${CMAKE_BINARY_DIR}/catboostr-fork-build))." >&2
    echo "  line $((orig_lines + 1)): ${splice_blank}" >&2
    echo "  line $((orig_lines + 2)): ${splice_line}" >&2
    FAIL=1
  fi
fi

prefix_hash=$(head -n "${orig_lines}" "${SRC_COPY}/CMakeLists.txt" | sha256sum 2>/dev/null | awk '{print $1}')
if [[ -z "${prefix_hash}" ]]; then
  prefix_hash=$(head -n "${orig_lines}" "${SRC_COPY}/CMakeLists.txt" | shasum -a 256 | awk '{print $1}')
fi
if [[ "${prefix_hash}" != "${expected_hash}" ]]; then
  echo "MISMATCH ${CMAKELISTS_REL}: content before configure's splice does not match manifest (expected ${expected_hash}, got ${prefix_hash})." >&2
  FAIL=1
fi

# 2. Every other manifested file: bulk byte-for-byte hash verification.
#    (This is the same "read the manifest, trust the tool that wrote it"
#    approach prune.sh itself uses, applied here to configure's copy instead
#    of a fresh prune.sh run.)
CHECK_LOG="$(mktemp)"
trap 'rm -f "${CHECK_LOG}"' EXIT

if command -v sha256sum >/dev/null 2>&1; then
  CHECKER=(sha256sum -c --strict -)
else
  CHECKER=(shasum -a 256 -c --strict -)
fi

rc=0
(
  cd "${SRC_COPY}"
  grep -vF "  ${CMAKELISTS_REL}" "${MANIFEST}" | "${CHECKER[@]}"
) > "${CHECK_LOG}" 2>&1 || rc=$?

OTHER_CHECKED=$(grep -c ': OK$' "${CHECK_LOG}" || true)
echo "*** other manifested files (excluding ${CMAKELISTS_REL}): ${OTHER_CHECKED} OK"

if [[ "${rc}" -ne 0 ]]; then
  echo "*** mismatches / missing (excluding ${CMAKELISTS_REL}):" >&2
  grep -v ': OK$' "${CHECK_LOG}" >&2 || true
  FAIL=1
fi

if [[ "${FAIL}" -ne 0 ]]; then
  echo "*** FAIL: configure's build-source copy does NOT match tools/vendor/PRUNED_MANIFEST.sha256" >&2
  exit 1
fi
echo "*** PASS: configure's build-source copy == manifest (modulo the one documented ${CMAKELISTS_REL} splice)"
