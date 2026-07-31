#!/usr/bin/env bash
# Acquire the third-party sources that participate in producing
# inst/libs/libcatboostr.so but do NOT live inside the pinned
# vendor/catboost snapshot: the openssl library the R link unconditionally
# pulls in, and the two code generators (ragel, yasm) the yutil target
# invokes at build time.
#
# Upstream CatBoost resolves all three through Conan
# (vendor/catboost/conanfile.py:16,19,21). Conan is a network-dependent
# resolver with no integrity pin visible to this repository, and it is
# forbidden in the install path (spec 4.8 / catboost-8z4.13). This script is
# the replacement: exact version, exact upstream URL, exact SHA256, verified
# fail-closed. See SOURCES.md for the inventory and the provenance of each
# SHA256 below.
#
# Re-runnable: an already-present tarball whose SHA256 matches is a no-op; an
# already-present tarball whose SHA256 does NOT match is a hard error (it is
# never silently re-downloaded over).
#
# Usage: tools/vendor/acquire-thirdparty.sh [dest_dir]
#   dest_dir defaults to vendor/thirdparty (gitignored; not committed).
set -euo pipefail

# name|version|sha256|url
PACKAGES=(
  "openssl|3.5.7|a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8|https://github.com/openssl/openssl/releases/download/openssl-3.5.7/openssl-3.5.7.tar.gz"
  "ragel|6.10|5f156edb65d20b856d638dd9ee2dfb43285914d9aa2b6ec779dac0270cd56c3f|https://www.colm.net/files/ragel/ragel-6.10.tar.gz"
  "yasm|1.3.0|3dce6601b495f5b3d45b59f7d2492a340ee7e84b5beca17e48f862502bd5603f|https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${1:-${REPO_ROOT}/vendor/thirdparty}"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "${DEST_DIR}"

for entry in "${PACKAGES[@]}"; do
  IFS='|' read -r name version sha url <<<"${entry}"
  file="${DEST_DIR}/${name}-${version}.tar.gz"

  if [[ -f "${file}" ]]; then
    actual="$(sha256_of "${file}")"
    if [[ "${actual}" == "${sha}" ]]; then
      echo "Already present and verified: ${file}"
      continue
    fi
    echo "FATAL: ${file} exists but its SHA256 does not match the pin." >&2
    echo "  expected=${sha}" >&2
    echo "  actual  =${actual}" >&2
    echo "Refusing to overwrite it. Investigate, then delete it and re-run." >&2
    exit 1
  fi

  tmp="${file}.part"
  rm -f "${tmp}"
  echo "Fetching ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --proto '=https' --tlsv1.2 -o "${tmp}" "${url}"
  else
    wget -q -O "${tmp}" "${url}"
  fi

  actual="$(sha256_of "${tmp}")"
  if [[ "${actual}" != "${sha}" ]]; then
    rm -f "${tmp}"
    echo "FATAL: SHA256 mismatch for ${name} ${version}." >&2
    echo "  url     =${url}" >&2
    echo "  expected=${sha}" >&2
    echo "  actual  =${actual}" >&2
    echo "The download was discarded. Do NOT use it." >&2
    exit 1
  fi
  mv "${tmp}" "${file}"
  echo "Verified: ${file}"
  echo "  version: ${version}"
  echo "  sha256:  ${sha}"
done

echo
echo "All third-party sources present and SHA256-verified in ${DEST_DIR}."
