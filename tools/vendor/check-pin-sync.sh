#!/usr/bin/env bash
# Asserts the three third-party pins (openssl, ragel, yasm) declared in
# `configure` §3b agree exactly (version + SHA256) with the same three
# pins declared in `tools/vendor/acquire-thirdparty.sh`'s PACKAGES array.
#
# Why the duplication exists at all (do not "fix" it by removing one
# copy): `configure` must run standalone against an already-unpacked
# vendor/thirdparty tree, where `tools/` may not exist -- see P1.5
# (catboost-8z4.13) and its report section 9. That duplication is a real
# drift hazard: a future version bump could edit one file and not the
# other. This script is the fail-closed check for that drift.
#
# Usage: tools/vendor/check-pin-sync.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURE="${REPO_ROOT}/configure"
ACQUIRE="${REPO_ROOT}/tools/vendor/acquire-thirdparty.sh"

FAIL=0

for name in openssl ragel yasm; do
  upper=$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')

  cfg_version=$(grep -E "^${upper}_VERSION=" "$CONFIGURE" | head -n1 | sed -E 's/^[A-Z0-9_]+="([^"]*)"$/\1/')
  cfg_sha256=$(grep -E "^${upper}_SHA256=" "$CONFIGURE" | head -n1 | sed -E 's/^[A-Z0-9_]+="([^"]*)"$/\1/')

  acq_line=$(grep -E "^\s*\"${name}\|" "$ACQUIRE" | head -n1)
  acq_version=$(printf '%s' "$acq_line" | awk -F'|' '{print $2}')
  acq_sha256=$(printf '%s' "$acq_line" | awk -F'|' '{print $3}')

  if [[ -z "$cfg_version" || -z "$cfg_sha256" ]]; then
    echo "FATAL: ${upper}_VERSION/${upper}_SHA256 not found in configure." >&2
    FAIL=1
    continue
  fi
  if [[ -z "$acq_version" || -z "$acq_sha256" ]]; then
    echo "FATAL: '${name}|...' entry not found in ${ACQUIRE}'s PACKAGES array." >&2
    FAIL=1
    continue
  fi

  if [[ "$cfg_version" != "$acq_version" ]]; then
    echo "FATAL: ${name} version mismatch: configure=${cfg_version} acquire-thirdparty.sh=${acq_version}" >&2
    FAIL=1
  fi
  if [[ "$cfg_sha256" != "$acq_sha256" ]]; then
    echo "FATAL: ${name} SHA256 mismatch: configure=${cfg_sha256} acquire-thirdparty.sh=${acq_sha256}" >&2
    FAIL=1
  fi
  if [[ "$cfg_version" == "$acq_version" && "$cfg_sha256" == "$acq_sha256" ]]; then
    echo "OK: ${name} ${cfg_version} (${cfg_sha256}) agrees in both files."
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "*** Pin drift detected between configure and tools/vendor/acquire-thirdparty.sh." >&2
  exit 1
fi

echo "*** All third-party pins agree between configure and acquire-thirdparty.sh."
