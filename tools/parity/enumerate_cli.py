"""Enumerate CLI modes and per-mode flags from --help output only.

Run: python3 tools/parity/enumerate_cli.py
Writes: tests/fixtures/parity/cli_surface.json
"""
import json
import pathlib
import re
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = REPO_ROOT / "tools" / "oracle" / "cli" / "bin" / "catboost-v1.2.10"
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "parity" / "cli_surface.json"

# Flag tokens look like: --long-name, -x, or grouped {-x|--long-name}
FLAG_RE = re.compile(r"(--[A-Za-z0-9][A-Za-z0-9_-]*|(?<![A-Za-z0-9-])-[A-Za-z](?![A-Za-z0-9-]))")


def run_help(args):
    r = subprocess.run([str(BIN)] + args, capture_output=True, text=True, timeout=30)
    return r.stdout + r.stderr


def top_level_modes():
    out = run_help(["--help"])
    modes = []
    in_modes = False
    for line in out.splitlines():
        if line.strip() == "Modes:":
            in_modes = True
            continue
        if in_modes:
            m = re.match(r"^\s{2}(\S[\w-]*)\s{2,}(.*)$", line)
            if m:
                modes.append((m.group(1), m.group(2).strip()))
            elif line.strip() == "" or line.startswith("To "):
                if line.startswith("To "):
                    break
    return modes


def extract_flags(help_text):
    """Extract flag entries from an OPTIONS-style --help block, verbatim.
    Each entry line is one flag (possibly with multiple aliases, e.g. -f/--learn-set).
    Returns a list of alias-token lists, one per option-line entry, so no name is
    dropped while a flag_count based on entries doesn't double-count aliases."""
    entries = []
    for line in help_text.splitlines():
        if not line.startswith("  "):
            continue
        stripped = line.strip()
        if not stripped or not (stripped.startswith("-") or stripped.startswith("{")):
            continue
        leading_tok = stripped.split(None, 1)[0]
        brace_m = re.match(r"^\{([^}]*)\}$", leading_tok)
        if brace_m:
            aliases = FLAG_RE.findall(brace_m.group(1))
        else:
            aliases = FLAG_RE.findall(leading_tok)
        if aliases:
            entries.append(sorted(set(aliases)))
    return entries


def sub_modes(mode_args):
    """Return (submode_name, desc) list if this mode's --help shows a 'Modes:' section
    instead of 'Options:' (i.e. it's a mode-of-modes like `metadata`)."""
    out = run_help(mode_args + ["-?"])
    if re.search(r"^Modes:\s*$", out, re.M):
        subs = []
        in_modes = False
        for line in out.splitlines():
            if line.strip() == "Modes:":
                in_modes = True
                continue
            if in_modes:
                m = re.match(r"^\s{2}(\S[\w-]*)\s{2,}(.*)$", line)
                if m:
                    subs.append((m.group(1), m.group(2).strip()))
                elif line.strip() == "" or line.startswith("To "):
                    if line.startswith("To "):
                        break
        return subs
    return None


result = {"modes": []}

ver_out = run_help(["--version"])
result["cli_version_raw"] = ver_out.strip()

modes = top_level_modes()
total_flags = 0
for mode_name, mode_desc in modes:
    entry = {"mode": mode_name, "description": mode_desc, "submodes": [], "flags": []}
    subs = sub_modes([mode_name])
    if subs:
        for sub_name, sub_desc in subs:
            sub_help = run_help([mode_name, sub_name, "-?"])
            sub_flags = extract_flags(sub_help)
            entry["submodes"].append({
                "submode": sub_name,
                "description": sub_desc,
                "flags": sub_flags,
            })
            total_flags += len(sub_flags)
    else:
        help_text = run_help([mode_name, "-?"])
        flags = extract_flags(help_text)
        entry["flags"] = flags
        total_flags += len(flags)
    result["modes"].append(entry)

result["mode_count"] = len(modes)
result["flag_count_total"] = total_flags

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"modes={len(modes)} total_flag_entries={total_flags}")
for e in result["modes"]:
    if e["submodes"]:
        for s in e["submodes"]:
            print(f"  {e['mode']} {s['submode']}: {len(s['flags'])} flag entries")
    else:
        print(f"  {e['mode']}: {len(e['flags'])} flag entries")
