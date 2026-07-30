#!/usr/bin/env bash
# Acquire the pinned upstream CatBoost source tree as a git submodule-free
# clone at a fixed release tag, verified against a pinned commit SHA.
# Re-runnable: if vendor/catboost already exists at the correct SHA, this
# is a no-op. Fails closed (non-zero exit, no silent partial state) if the
# resolved SHA does not match the pin.
#
# Usage: tools/vendor/acquire.sh [dest_dir]
#   dest_dir defaults to vendor/catboost (gitignored; not committed).
set -euo pipefail

REPO_URL="https://github.com/catboost/catboost.git"
RELEASE_TAG="v1.2.10"
EXPECTED_SHA="b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${1:-${REPO_ROOT}/vendor/catboost}"

resolved_sha() {
  git -C "${DEST_DIR}" rev-parse HEAD
}

if [[ -d "${DEST_DIR}/.git" ]]; then
  actual="$(resolved_sha)"
  if [[ "${actual}" == "${EXPECTED_SHA}" ]]; then
    echo "Already present and verified: ${DEST_DIR} @ ${actual}"
    exit 0
  fi
  echo "FATAL: ${DEST_DIR} exists but resolves to ${actual}, expected ${EXPECTED_SHA}." >&2
  echo "Remove it and re-run, or investigate why it drifted." >&2
  exit 1
fi

mkdir -p "$(dirname "${DEST_DIR}")"
echo "Cloning ${REPO_URL} @ ${RELEASE_TAG} -> ${DEST_DIR}"
git clone --branch "${RELEASE_TAG}" --depth 1 "${REPO_URL}" "${DEST_DIR}"

actual="$(resolved_sha)"
if [[ "${actual}" != "${EXPECTED_SHA}" ]]; then
  echo "FATAL: resolved SHA mismatch. expected=${EXPECTED_SHA} actual=${actual}" >&2
  echo "Upstream tag ${RELEASE_TAG} moved, or the pin in this script is stale." >&2
  exit 1
fi

echo "Verified: ${DEST_DIR}"
echo "  tag:  ${RELEASE_TAG}"
echo "  sha:  ${actual}"
