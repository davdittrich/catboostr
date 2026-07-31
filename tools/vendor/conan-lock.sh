#!/usr/bin/env bash
# MAINTAINER-SIDE ONLY. Nothing in the install path calls this, and nothing in
# the install path calls Conan at all (catboost-8z4.13 / P1.5): `./configure`
# builds openssl, ragel and yasm from the pinned, SHA256-verified tarballs in
# vendor/thirdparty/ instead. Conan survives here purely as an auditing tool
# (spec 4.1), to answer one question:
#
#   what would upstream's own conanfile.py resolve to today, and has that
#   drifted away from what SOURCES.md says we ship?
#
# Two subcommands:
#   verify (default)  resolve the graph WITH --lockfile conan.lock and fail if
#                     the lock cannot satisfy it (i.e. upstream's recipes
#                     moved under us). Read-only w.r.t. conan.lock.
#   update            regenerate conan.lock. Review the diff by hand; a change
#                     here is a supply-chain event, not a chore.
#
# Requires `conan` (Conan 2.x) on PATH. Install it in a throwaway venv --
# never system-wide, and never as a build dependency of this package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONANFILE="${REPO_ROOT}/vendor/catboost/conanfile.py"
LOCKFILE="${REPO_ROOT}/conan.lock"
MODE="${1:-verify}"

if ! command -v conan >/dev/null 2>&1; then
  echo "FATAL: conan not on PATH. This is a maintainer-only tool; install it" >&2
  echo "in a throwaway virtualenv (python -m venv .venv && .venv/bin/pip install conan)." >&2
  exit 1
fi
if [[ ! -f "${CONANFILE}" ]]; then
  echo "FATAL: ${CONANFILE} not found. Run tools/vendor/acquire.sh first." >&2
  exit 1
fi

# Keep every Conan write out of the repository and out of $HOME's default
# cache. vendor/catboost is a read-only pin; Conan is happy to scribble in a
# source tree if you let it.
export CONAN_HOME="${CONAN_HOME:-$(mktemp -d "${TMPDIR:-/tmp}/catboostr-conan.XXXXXX")}"
[[ -f "${CONAN_HOME}/profiles/default" ]] || conan profile detect --force >/dev/null

case "${MODE}" in
  verify)
    [[ -f "${LOCKFILE}" ]] || { echo "FATAL: ${LOCKFILE} missing." >&2; exit 1; }
    echo "Resolving ${CONANFILE} against ${LOCKFILE}"
    # No --lockfile-partial: the default is strict, i.e. anything the lock
    # cannot pin is an error. That strictness is the whole point.
    conan graph info "${CONANFILE}" --lockfile "${LOCKFILE}" >/dev/null
    echo "OK: conan.lock still satisfies upstream's conanfile.py."
    echo
    echo "Reminder: the lock records what UPSTREAM asks for. What this package"
    echo "actually links is in SOURCES.md, and the two are deliberately not"
    echo "identical (openssl is pinned newer than upstream's 3.0.15)."
    ;;
  update)
    conan lock create "${CONANFILE}" --lockfile-out "${LOCKFILE}"
    echo "Rewrote ${LOCKFILE}. Diff it before committing."
    ;;
  *)
    echo "usage: $0 [verify|update]" >&2
    exit 2
    ;;
esac
