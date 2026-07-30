"""Parse upstream R-package/NAMESPACE into export()/S3method() records.

Run: python3 tools/parity/parse_namespace.py
Writes: tests/fixtures/parity/r_surface.json
"""
import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
NAMESPACE_PATH = REPO_ROOT / "vendor" / "catboost" / "catboost" / "R-package" / "NAMESPACE"
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "parity" / "r_surface.json"

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
    "exports": exports,
    "s3methods": s3methods,
    "export_count": len(exports),
    "s3method_count": len(s3methods),
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"exports={len(exports)} s3methods={len(s3methods)}")
