#!/usr/bin/env bash
# Regenerate the full capability inventory in pipeline order (see README.md).
# Requires vendor/catboost (tools/vendor/acquire.sh) and the CLI oracle binary
# (tools/oracle/cli/acquire.sh) to already be present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${SCRIPT_DIR}/parse_namespace.py"
uv run --project "${REPO_ROOT}/tools/oracle" python "${SCRIPT_DIR}/introspect_python.py"
python3 "${SCRIPT_DIR}/enumerate_cli.py"
python3 "${SCRIPT_DIR}/compute_diff.py"
python3 "${SCRIPT_DIR}/spec_crosscheck.py"
