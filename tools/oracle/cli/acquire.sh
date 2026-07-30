#!/usr/bin/env bash
# Acquire the pinned CatBoost CLI binary (v1.2.10, linux x86_64) as a
# GitHub Release asset. Never builds from source. Re-runnable: verifies
# the checksum of an existing download and skips re-fetching if it
# already matches.
#
# Usage: tools/oracle/cli/acquire.sh [dest_dir]
#   dest_dir defaults to tools/oracle/cli/bin (gitignored; not committed).
set -euo pipefail

RELEASE_TAG="v1.2.10"
ASSET_NAME="catboost-linux-x86_64-1.2.10"
ASSET_URL="https://github.com/catboost/catboost/releases/download/${RELEASE_TAG}/${ASSET_NAME}"
EXPECTED_SHA256="478dc57f4c19de205b19b709fd4c6af93f79753dc462b6bb49d33c920dddec75"
EXPECTED_SIZE=287580072

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${1:-${SCRIPT_DIR}/bin}"
DEST_PATH="${DEST_DIR}/catboost-${RELEASE_TAG}"

mkdir -p "${DEST_DIR}"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

if [[ -f "${DEST_PATH}" ]] && [[ "$(sha256_of "${DEST_PATH}")" == "${EXPECTED_SHA256}" ]]; then
  echo "Already present and verified: ${DEST_PATH}"
else
  echo "Downloading ${ASSET_URL} -> ${DEST_PATH}"
  curl -fL --retry 3 -o "${DEST_PATH}.part" "${ASSET_URL}"
  mv "${DEST_PATH}.part" "${DEST_PATH}"
fi

ACTUAL_SIZE=$(stat -c%s "${DEST_PATH}")
ACTUAL_SHA256=$(sha256_of "${DEST_PATH}")

if [[ "${ACTUAL_SIZE}" != "${EXPECTED_SIZE}" ]]; then
  echo "FATAL: size mismatch. expected=${EXPECTED_SIZE} actual=${ACTUAL_SIZE}" >&2
  exit 1
fi

if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "FATAL: sha256 mismatch. expected=${EXPECTED_SHA256} actual=${ACTUAL_SHA256}" >&2
  exit 1
fi

chmod +x "${DEST_PATH}"
echo "Verified: ${DEST_PATH}"
echo "  size:   ${ACTUAL_SIZE}"
echo "  sha256: ${ACTUAL_SHA256}"
