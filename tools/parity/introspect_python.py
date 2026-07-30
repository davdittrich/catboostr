"""Machine-introspect the catboost Python package surface. No hand-typed names.

Run: uv run --project tools/oracle python tools/parity/introspect_python.py
Writes: tests/fixtures/parity/python_surface.json
"""
import enum
import inspect
import json
import pathlib
import sys

import catboost

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "parity" / "python_surface.json"

entries = []
private_excluded = 0
non_introspectable = 0

TOP_LEVEL_CLASSES = ["CatBoost", "CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker", "Pool", "FeaturesData"]
SUBMODULES = ["utils", "eval", "datasets", "text_processing"]

# stdlib/__future__ objects that leak into catboost's submodule namespaces via
# `from enum import Enum` / `from __future__ import print_function` — not a
# catboost capability, just an import side effect.
NON_CATBOOST_MODULES = {"enum", "__future__"}


def sig_str(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return None


def expand_enum_members(entries, qualname, cls_name, obj):
    """Enum classes carry their real capability surface as members (e.g.
    EFstrType.ShapInteractionValues, eval.EvalType.SeqRem), not as the class
    name itself. Every scan path that records a class must call this or its
    members silently vanish -- the exact failure this ticket exists to catch."""
    if not (inspect.isclass(obj) and issubclass(obj, enum.Enum)):
        return
    for member in obj:
        entries.append({
            "qualified_name": f"{qualname}.{member.name}",
            "kind": "enum_member",
            "owner": cls_name,
            "signature": None,
            "introspectable": True,
        })


def record(qualname, kind, owner, obj):
    global non_introspectable
    sig = None
    introspectable = True
    if kind in ("function", "method"):
        sig = sig_str(obj)
        if sig is None:
            introspectable = False
            non_introspectable += 1
    entries.append({
        "qualified_name": qualname,
        "kind": kind,
        "owner": owner,
        "signature": sig,
        "introspectable": introspectable,
    })


# 1. Top-level module members
for name in dir(catboost):
    if name.startswith("_"):
        private_excluded += 1
        continue
    obj = getattr(catboost, name)
    if inspect.isclass(obj):
        record(f"catboost.{name}", "class", None, obj)
        expand_enum_members(entries, f"catboost.{name}", name, obj)
    elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
        record(f"catboost.{name}", "function", None, obj)
    elif inspect.ismodule(obj):
        continue  # handled separately if in SUBMODULES
    else:
        record(f"catboost.{name}", "attribute", None, obj)

# 2. Public methods/properties of key classes
for cls_name in TOP_LEVEL_CLASSES:
    cls = getattr(catboost, cls_name, None)
    if cls is None:
        continue
    for name in dir(cls):
        if name.startswith("_"):
            private_excluded += 1
            continue
        try:
            obj = getattr(cls, name)
        except Exception:
            non_introspectable += 1
            entries.append({
                "qualified_name": f"{cls_name}.{name}",
                "kind": "unknown",
                "owner": cls_name,
                "signature": None,
                "introspectable": False,
            })
            continue
        if isinstance(inspect.getattr_static(cls, name, None), property):
            record(f"{cls_name}.{name}", "property", cls_name, obj)
        elif inspect.isclass(obj):
            record(f"{cls_name}.{name}", "attribute", cls_name, obj)
            expand_enum_members(entries, f"{cls_name}.{name}", name, obj)
        elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj) or callable(obj):
            # callable() catches Cython/C-extension bound methods on Pool/
            # FeaturesData that isfunction/ismethod/isbuiltin all miss --
            # sig_str() inside record() will correctly mark these
            # introspectable:false when inspect.signature() can't resolve them,
            # rather than silently mistagging them as non-callable attributes.
            record(f"{cls_name}.{name}", "method", cls_name, obj)
        else:
            record(f"{cls_name}.{name}", "attribute", cls_name, obj)

# 3. Training-parameter set accepted by CatBoost.__init__ / fit.
# __init__ is a dunder (leading underscore) and thus excluded from the generic
# dir()-based member scan above; the brief requires the parameter set as its
# own explicit enumeration regardless. CatBoost.__init__ itself only takes a
# `params` dict passthrough, so the real named hyperparameter surface lives on
# the concrete estimator subclasses' __init__ (also named in the brief's
# surface list) plus CatBoost.fit's data-binding kwargs.
for cls_name, meth_name in [
    ("CatBoost", "__init__"),
    ("CatBoost", "fit"),
    ("CatBoostClassifier", "__init__"),
    ("CatBoostRegressor", "__init__"),
    ("CatBoostRanker", "__init__"),
]:
    cls = getattr(catboost, cls_name)
    meth = getattr(cls, meth_name)
    sig = sig_str(meth)
    if sig is None:
        non_introspectable += 1
        entries.append({
            "qualified_name": f"{cls_name}.{meth_name}(params)",
            "kind": "parameter_set",
            "owner": cls_name,
            "signature": None,
            "introspectable": False,
        })
        continue
    try:
        params = inspect.signature(meth).parameters
    except (ValueError, TypeError):
        params = {}
    for pname, p in params.items():
        if pname in ("self",) or pname.startswith("_"):
            if pname.startswith("_"):
                private_excluded += 1
            continue
        entries.append({
            "qualified_name": f"{cls_name}.{meth_name}(param={pname})",
            "kind": "parameter",
            "owner": f"{cls_name}.{meth_name}",
            "signature": str(p),
            "introspectable": True,
        })

# 4. Submodules
for sub_name in SUBMODULES:
    try:
        submod = getattr(catboost, sub_name, None)
        if submod is None:
            import importlib
            submod = importlib.import_module(f"catboost.{sub_name}")
    except ImportError:
        continue
    for name in dir(submod):
        if name.startswith("_"):
            private_excluded += 1
            continue
        obj = getattr(submod, name)
        if inspect.isclass(obj):
            if getattr(obj, "__module__", "") in NON_CATBOOST_MODULES:
                continue  # stdlib/__future__ re-export leak, not a catboost capability
            record(f"catboost.{sub_name}.{name}", "class", sub_name, obj)
            expand_enum_members(entries, f"catboost.{sub_name}.{name}", name, obj)
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            record(f"catboost.{sub_name}.{name}", "function", sub_name, obj)
        elif inspect.ismodule(obj):
            continue
        else:
            if getattr(type(obj), "__module__", "") in NON_CATBOOST_MODULES:
                continue  # e.g. __future__._Feature instances like print_function
            record(f"catboost.{sub_name}.{name}", "attribute", sub_name, obj)

output = {
    "catboost_version": catboost.__version__,
    "python_version": sys.version.split()[0],
    "entries": entries,
    "private_excluded_count": private_excluded,
    "non_introspectable_count": non_introspectable,
    "total_count": len(entries),
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"total_count={len(entries)} private_excluded={private_excluded} non_introspectable={non_introspectable}")
