"""Parse upstream R-package/NAMESPACE into export()/S3method() records.

Run: python3 tools/parity/parse_namespace.py
Reads: vendor/catboost (acquired via tools/vendor/acquire.sh; gitignored)
Writes: tests/fixtures/parity/r_surface.json
"""
import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "vendor" / "catboost"
NAMESPACE_PATH = VENDOR_DIR / "catboost" / "R-package" / "NAMESPACE"
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "parity" / "r_surface.json"
EXPECTED_TAG = "v1.2.10"

if not VENDOR_DIR.is_dir():
    sys.exit(
        f"vendor/catboost is absent. Run tools/vendor/acquire.sh to clone upstream "
        f"at {EXPECTED_TAG} before running this script."
    )

upstream_sha = subprocess.run(
    ["git", "-C", str(VENDOR_DIR), "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()

exports = []
s3methods = []
with open(NAMESPACE_PATH) as f:
    for line in f:
        line = line.strip()
        m = re.match(r"^export\(([^)]+)\)$", line)
        if m:
            exports.append(m.group(1))
        m = re.match(r"^S3method\(([^,]+),([^)]+)\)$", line)
        if m:
            s3methods.append({"generic": m.group(1), "class": m.group(2)})

out = {
    "source": "vendor/catboost/catboost/R-package/NAMESPACE",
    "upstream_tag": EXPECTED_TAG,
    "upstream_sha": upstream_sha,
    "exports": exports,
    "s3methods": s3methods,
    "export_count": len(exports),
    "s3method_count": len(s3methods),
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"exports={len(exports)} s3methods={len(s3methods)} upstream_sha={upstream_sha}")
