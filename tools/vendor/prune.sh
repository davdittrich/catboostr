#!/usr/bin/env bash
# Prune a fresh vendor/catboost acquisition (tools/vendor/acquire.sh's
# output) down to the subset that configure's from-source build actually
# reads, and write a file-level SHA256 manifest of the result.
#
# This is a BUILD-INPUT determinism/integrity tool, not a second exclusion
# mechanism for the R source tarball: vendor/catboost is NEVER part of the
# tarball (.Rbuildignore ^vendor$ / ^vendor/, asserted in
# docs/phase-1/P1.6-report.md sec. 3) and this script's output is not part
# of it either -- both are build-time-only inputs that `configure` compiles
# from a disposable copy (configure sec. 3) into inst/libs/libcatboostr.so,
# which IS what ships. What this script buys: (a) a smaller, faster local
# build input by removing ~1 GiB of content nothing in the R-package-fork
# build ever reads, and (b) a committed SHA256 manifest P1.13 can re-hash in
# CI and diff, proving the pruned tree a contributor/CI machine builds from
# is byte-for-byte what was audited here -- not eyeballed.
#
# Two prune tiers, both evidenced, neither guessed:
#
#   Tier 1 (CMAKE_GATED): the 44 unit-test/benchmark/tool directories that
#   cmake/gate-test-tool-dirs.cmake (P1.2, catboost-8z4.10) already proved
#   are not a compile or link input for the R-package build (verified there
#   against `ninja -t deps` and the catboostr target_link_libraries list).
#   Deleting them is safe because that cmake file makes CMake skip
#   add_subdirectory() for exactly these paths, unconditionally, on every
#   configure -- read directly out of that file below so the two lists can
#   never drift apart.
#
#   Tier 2 (NEVER_REFERENCED): top-level directories that do not appear as
#   the argument of any add_subdirectory() call anywhere in the pinned tree
#   at all (verified: `grep -rl 'add_subdirectory(<name>' --include
#   'CMakeLists*.txt'` returns zero hits outside the directory's own
#   subtree, for each name below -- see docs/phase-1/P1.6-report.md sec. 5
#   for the exact commands and output). CMake never walks into them, so no
#   cmake-side gating is required to delete them, unlike Tier 1.
#
# Usage: tools/vendor/prune.sh [src_dir] [dest_dir]
#   src_dir  defaults to vendor/catboost (must already be acquired --
#            this script does not fetch anything itself).
#   dest_dir defaults to vendor/catboost-pruned (gitignored via the
#            existing "/vendor/" entry in .gitignore; not committed).
# Deterministic and re-runnable: dest_dir is removed and rebuilt from
# src_dir every invocation, never patched in place.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_DIR="${1:-${REPO_ROOT}/vendor/catboost}"
DEST_DIR="${2:-${REPO_ROOT}/vendor/catboost-pruned}"
GATE_FILE="${REPO_ROOT}/cmake/gate-test-tool-dirs.cmake"
MANIFEST="${REPO_ROOT}/tools/vendor/PRUNED_MANIFEST.sha256"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "FATAL: ${SRC_DIR} does not exist. Run tools/vendor/acquire.sh first." >&2
  exit 1
fi
if [[ ! -f "${GATE_FILE}" ]]; then
  echo "FATAL: ${GATE_FILE} not found -- Tier 1's list of record is missing." >&2
  exit 1
fi

# Tier 1: read the gated dir list directly from cmake/gate-test-tool-dirs.cmake
# (lines of the form '  "path/to/dir"' inside the set(...) block) so this
# script and that file can never silently drift apart.
TIER1_DIRS=()
while IFS= read -r _line; do
  TIER1_DIRS+=("${_line}")
done < <(grep -oE '^\s*"[^"]+"\s*$' "${GATE_FILE}" | tr -d ' "')

if [[ "${#TIER1_DIRS[@]}" -ne 44 ]]; then
  echo "FATAL: expected 44 Tier 1 dirs from ${GATE_FILE}, got ${#TIER1_DIRS[@]}." >&2
  echo "The gate file changed shape -- re-verify before pruning." >&2
  exit 1
fi

# Tier 2: verified zero add_subdirectory() references anywhere in the
# pinned tree outside their own subtree (docs/phase-1/P1.6-report.md sec. 5).
TIER2_DIRS=(
  "catboost/tutorials"
  "catboost/pytest"
  "catboost/benchmarks"
  "catboost/cuda"
  "catboost/dotnet"
  "catboost/node-package"
  "catboost/rust-package"
  "catboost/docs"
  "catboost/debian"
  "catboost/docker"
  "slides"
  "ci"
  "certs"
  "logo"
  "open_problems"
  "bindings"
  ".git"
)

echo "*** pruning ${SRC_DIR} -> ${DEST_DIR}"
rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
cp -a "${SRC_DIR}/." "${DEST_DIR}/"

REMOVED=0
for d in "${TIER1_DIRS[@]}" "${TIER2_DIRS[@]}"; do
  if [[ -e "${DEST_DIR}/${d}" ]]; then
    rm -rf "${DEST_DIR}/${d}"
    REMOVED=$((REMOVED + 1))
  fi
done
echo "*** removed ${REMOVED} / $(( ${#TIER1_DIRS[@]} + ${#TIER2_DIRS[@]} )) listed paths (some Tier 1 entries are test dirs that may not exist on every checkout)"

BEFORE_SIZE=$(du -sh "${SRC_DIR}" | cut -f1)
AFTER_SIZE=$(du -sh "${DEST_DIR}" | cut -f1)
echo "*** size: ${BEFORE_SIZE} -> ${AFTER_SIZE}"

echo "*** writing manifest: ${MANIFEST}"
(
  cd "${DEST_DIR}"
  export LC_ALL=C
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${MANIFEST}"
echo "*** manifest: $(wc -l < "${MANIFEST}") files"
