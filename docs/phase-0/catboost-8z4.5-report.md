# catboost-8z4.5: Generate the machine-derived capability inventory — report

## Summary

Enumerated the Python surface (introspection), the CLI surface (`--help`
parsing), and the R export surface (`NAMESPACE` parsing) by program output
only, computed the Python/CLI-vs-R diff, and cross-checked the design spec's
Section 2 hand-written gap list against the generated inventory. All outputs
are committed as JSON under `tests/fixtures/parity/`; the generator scripts
that produced them are committed under `tools/parity/` and are re-runnable.

## 1. Input verification

- Python oracle: `uv run --project tools/oracle python -c ...` reports
  `catboost.__version__ == "1.2.10"`, resolved from
  `tools/oracle/.venv/lib/python3.14/site-packages/catboost/__init__.py`.
- CLI oracle: `tools/oracle/cli/bin/catboost-v1.2.10 --version` reports
  `Branch: tags/v1.2.10`, commit `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`
  (matches the pinned SHA from catboost-8z4.4).
- R NAMESPACE: `vendor/catboost/catboost/R-package/NAMESPACE`, present,
  parsed to 19 `export()` + 8 `S3method()` entries — matches the brief's
  stated baseline exactly.

All three inputs present; no STOP condition.

## 2. Python surface introspection

Script: `tools/parity/introspect_python.py`. Method: `dir()` + `inspect`
over the top-level `catboost` module, the six named classes (`CatBoost`,
`CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`, `Pool`,
`FeaturesData`), the `__init__`/`fit` parameter sets, the four submodules
(`utils`, `eval`, `datasets`, `text_processing`), and — added after an
initial gap was caught by hand — the members of every top-level `enum.Enum`
subclass (`EFstrType`, `EShapCalcType`, `EFeaturesSelectionAlgorithm`,
`EFeaturesSelectionGrouping`). Without that last step, `EFstrType` itself
was recorded as a class but its members (`ShapInteractionValues`,
`PredictionDiff`, ...) — both explicitly named in spec §2 — were silently
absent. This is exactly the failure mode the ticket exists to catch, caught
against my own tool rather than the target list.

- **756** total entries (leading-underscore members excluded).
- **559** private (leading-underscore) members excluded — counted, not dropped.
- **0** non-introspectable (every function/method in this build yielded an
  `inspect.signature`; the guard's fallback path is implemented but did not
  trigger).
- Breakdown by kind: 375 parameter, 214 method, 40 function, 40 property,
  39 class, 32 attribute, 16 enum_member.
- `CatBoost.__init__` itself only takes a `params` dict passthrough (not a
  named hyperparameter surface); the real 100+-name hyperparameter surface
  lives on `CatBoostClassifier`/`Regressor`/`Ranker.__init__`, which are also
  named in the brief's surface list — both are enumerated.

Output: `tests/fixtures/parity/python_surface.json`.

## 3. CLI surface enumeration

Script: `tools/parity/enumerate_cli.py`. Parses `--help` output only
(top-level `Modes:` block, then each mode's own `-?` output; `metadata` has
its own nested `Modes:` block with 4 submodes, parsed the same way).

- **15** top-level modes (not 8 — the brief's stated fact is confirmed
  independently): `fit`, `calc`, `dataset-statistics`, `fstr`, `ostr`,
  `eval-metrics`, `eval-feature`, `metadata` (4 submodes: `set`, `get`,
  `dump`, `dump-feature-names`), `model-sum`, `run-worker`, `roc`,
  `model-based-eval`, `normalize-model`, `select-features`, `dump-options`.
- **804** flag entries total (one entry per option line; grouped aliases
  like `{-f|--learn-set}` are kept together as one entry with both alias
  strings preserved, not double-counted as two flags).

Output: `tests/fixtures/parity/cli_surface.json`.

## 4. R NAMESPACE parsing

Script: `tools/parity/parse_namespace.py`. Regex over
`export(...)` / `S3method(...,...)` lines only.

- **19** exports, **8** S3 methods (`dim`, `dimnames`, `head`, `predict`,
  `print`×2, `summary`, `tail`) — exact match to the brief's stated baseline.

Output: `tests/fixtures/parity/r_surface.json`.

## 5. Diff computation

Script: `tools/parity/compute_diff.py`. Matching rule, fixed and
deterministic (no per-row judgment call): `normalize(name) = lowercase,
strip all non-alphanumeric characters`; a candidate is COVERED if its
normalized last-component name exactly equals, or is an exact-substring
(≥4 chars, either direction) of, a normalized R export suffix or S3 generic
name. Everything else is a gap, tagged with its source oracle.

- **1439** total gap rows: **673** Python-only, **766** CLI-only.
- **140** candidates matched an R export/S3-generic by this rule.

Caveat, recorded here rather than used to filter rows: R's `catboost.train`
takes an untyped `params` named list, so none of the 375 individual
`CatBoostClassifier`/etc. hyperparameter names has a discrete 1:1 R export to
match against by construction — they still appear as gap rows
(`kind: parameter`) per the no-filtering guard, inflating the Python-only
count. Whether R's list-passthrough mechanism already "covers" a given named
hyperparameter is a semantic question for Phase 2, not something this
enumeration step is positioned to decide.

Output: `tests/fixtures/parity/capability_diff.json`.

## 6. Cross-check against spec §2

Script: `tools/parity/spec_crosscheck.py`. Spec §2
(`docs/superpowers/specs/2026-07-30-catboostr-design.md:25-30`) was split
into 18 discrete capability claims and searched (same normalize+substring
rule) against the full Python+CLI corpus, independent of R-coverage status.

**Items in §2 the generated inventory did NOT find (1):**
- `sparse/CSR pool input` — no member/parameter/flag *name* contains
  `csr`/`sparse` referring to sparse-matrix input (`sparse_features_conflict_fraction`
  is a real but unrelated parameter, correctly not counted as a match).
  Caveat: CSR support in Python is a *type*-level capability of `Pool`'s
  `data` argument (accepts `scipy.sparse` matrices per upstream docs), not a
  named symbol — invisible to name-based introspection by construction, not
  necessarily a spec factual error. Flagging honestly rather than silently
  resolving it either way.

The other 17 claims (`embedding features`, `timestamps`, `grid_search`,
`randomized_search`, `select_features`, `calc_feature_statistics`,
`ShapInteractionValues`, `PredictionDiff`, `plot_tree`, training-progress
plotting, `model.compare`, explicit pool quantization, text
tokenizer/dictionary configuration, `init_model` continued training, custom
loss/metric callbacks, GPU training, distributed training) all resolved to
concrete program-derived symbols, e.g. `catboost.MultiRegressionCustomObjective`
/ `catboost.MultiTargetCustomObjective` for "custom callbacks",
`catboost.EFstrType.ShapInteractionValues` / `.PredictionDiff` (found only
after the enum-member fix in step 2), and CLI mode `run-worker` for
"distributed training".

**Items the generated inventory found that §2 never mentioned (1199):**
1199 of the 1439 diff rows have no plausible substring link back to any of
the 18 §2 terms — i.e. real Python/CLI capabilities absent from R that the
hand-written list never named at all. This is the number the ticket exists
to produce. Full list: `tests/fixtures/parity/spec_crosscheck.json` →
`gaps_missing_from_spec` (1199 entries; a sample of the first ~15 illustrates
the range — top-level classes/errors like `CatBoostError`, enum members like
`EFeaturesSelectionAlgorithm.RecursiveByShapValues`, methods like
`CatBoost.calc_leaf_indexes`, `CatBoost.feature_importances_`,
`CatBoost.get_all_params`, plus the bulk of the 673 individual hyperparameter
names and 766 CLI flags).

## Guard verification

- Enumeration by introspection/`--help` only: every entry in the three
  surface files traces to a script call, no hand-typed name — verified by
  reading the three generator scripts, which contain zero literal capability
  names outside comments/docstrings.
- Private-excluded and non-introspectable counts recorded, not dropped:
  559 / 0 respectively (see §2 above).
- Diff records provenance per gap (`oracle: python|cli` field on every row).
- No gap ranked, filtered, or judged for importance — `compute_diff.py`
  applies one uniform matching rule with no per-capability branching.
- `git -C vendor/catboost status --porcelain` — one untracked file,
  `CMakeUserPresets.json`, referencing a path in this session's scratch
  directory. Not created by this task (no `cmake`/`conan`/write command was
  run against `vendor/` here); the scratch directory contains sibling
  `cbuild*/` directories from a concurrent process sharing the same session
  scratch path, most plausibly another Phase-0 ticket's native-build probe
  running in parallel. Left untouched — out of this ticket's write scope
  (`tests/fixtures/parity/` and `tools/parity/` only) and not something I
  authored.

## Files

- `tools/parity/introspect_python.py` — Python surface generator.
- `tools/parity/enumerate_cli.py` — CLI surface generator.
- `tools/parity/parse_namespace.py` — R NAMESPACE parser.
- `tools/parity/compute_diff.py` — diff computation.
- `tools/parity/spec_crosscheck.py` — §2 cross-check.
- `tests/fixtures/parity/python_surface.json` — 756 entries.
- `tests/fixtures/parity/cli_surface.json` — 15 modes, 804 flag entries.
- `tests/fixtures/parity/r_surface.json` — 19 exports, 8 S3 methods.
- `tests/fixtures/parity/capability_diff.json` — 1439 gap rows.
- `tests/fixtures/parity/spec_crosscheck.json` — cross-check numbers + full
  `gaps_missing_from_spec` list.

## Section V output

```toon
task_id: catboost-8z4.5
success: true
data:
  python_version_used: "1.2.10"
  cli_version_used: "tags/v1.2.10 (commit b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084)"
  namespace_source: "vendor/catboost/catboost/R-package/NAMESPACE"
  python_surface_count: 756
  python_private_excluded_count: 559
  python_non_introspectable_count: 0
  cli_mode_count: 15
  cli_flag_count: 804
  r_export_count: 19
  r_s3method_count: 8
  gap_count_total: 1439
  gap_count_python_only: 673
  gap_count_cli_only: 766
  spec_claims_not_found: ["sparse/CSR pool input"]
  gaps_missing_from_spec: ["1199 entries -- see tests/fixtures/parity/spec_crosscheck.json:gaps_missing_from_spec for full list"]
  inventory_path: "tests/fixtures/parity/{python_surface,cli_surface,r_surface}.json"
  diff_path: "tests/fixtures/parity/capability_diff.json"
verdict:
  inventory_complete: true
  spec_hand_list_was_accurate: false
  reasoning: "Spec §2's hand-written list correctly named 17/18 of the capability categories it claimed, but the machine inventory found 1199 additional real gap rows (individual methods, enum members, hyperparameters, CLI flags) it never mentioned -- confirming the ticket's premise that a hand-curated list under-enumerates. One §2 claim (sparse/CSR pool input) could not be confirmed by name-based introspection alone; it is a type-signature-level capability, not a named symbol, and is flagged rather than resolved."
  confidence: 85
error_log: null
```

---

## Fix round 1 of 5

Review found two Critical defects, both reproduced against my own tools.
Fixed both, re-ran all five generators, re-verified determinism.

### Critical 1 — enum blind spot in the submodule scan path

`introspect_python.py` step 4 (submodule scan over `utils`/`eval`/`datasets`/
`text_processing`) recorded enum classes (`catboost.eval.EvalType`,
`LabelMode`, `ScoreType`) as bare classes without expanding their members,
even though step 1 (top-level module scan) already did this for `EFstrType`
etc. Root cause: the enum-expansion logic lived inline in step 1 only, never
factored out, so step 4 never got it.

Fix: extracted `expand_enum_members()` and call it from every site that
records a class (step 1's module scan, step 2's class-member scan, step 4's
submodule scan) — one shared function instead of duplicated inline logic, so
a fourth scan path added later can't reintroduce the same gap. Verified by
direct introspection that all 8 real enum members now appear:
`EvalType.{SeqRem,SeqAdd,SeqAddAndAll,All}`, `LabelMode.{AddFeature,
IgnoreFeature}`, `ScoreType.{Abs,Rel}` — `LabelMode` has no `All` member; the
reviewer's 8-name list pooled members across all three enums, not one each.

Audited for further blind spots: a broad recursive scan (script used only
for the audit, not committed — one-off check, not a generator) over every
class reachable from the six named classes plus four submodules found
exactly the same 8 enum classes the fixed script now covers, `catboost.
eval.Enum` (removed, see Minor below), and nothing else. No third blind spot
found.

### Critical 2 — coverage heuristic was an unaudited filter

`compute_diff.py`'s substring match (≥4 chars, either direction) was
replaced with **exact normalized match only**. Verified against all six
named false positives (`--has-header`↔`head()`, `--detailed-profile`↔
`tail()`, `model_shrink_rate`/`model_shrink_mode`↔`shrink`, `--name`↔
`dimnames()`, `metadata get`↔`get_feature_importance`) — none appear in the
new `covered` list. `covered_count` dropped from 140 to 48, i.e. exactly the
92 false positives the reviewer counted are now gap rows instead of
silently-dropped matches. `capability_diff.json` now carries a `covered`
array (48 rows, each with `matched_r_symbol`) alongside `gaps`, so every
match — not just every drop — is auditable the same way
`private_excluded_count` already was.

Corrected inflation caveat: under exact matching, **0 of the 375** individual
hyperparameter rows match an R export exactly (previously miscounted at
"355 of 375" under the substring rule, which is now deleted). All 375 remain
gap rows for the mechanistic reason already stated — R takes an untyped
`params` list, no per-parameter export exists to match against by
construction — not because of any residual heuristic imprecision.

### Minor fixes applied

- `catboost.eval.Enum` (module `enum`) and the `catboost.eval.print_function`
  `__future__` re-export leak are now filtered out in step 4 via a
  `NON_CATBOOST_MODULES = {"enum", "__future__"}` check on `__module__` — 2
  fewer padded rows.
- Cython/C-extension bound methods on `Pool`/`FeaturesData` (e.g.
  `get_baseline`, `quantize`, `set_timestamp`, `is_quantized`) are now
  tagged `kind: "method"` instead of `"attribute"`: added `callable(obj)` as
  a catch-all alongside `isfunction`/`ismethod`/`isbuiltin`, since Cython
  bound methods (`cython_function_or_method`) satisfy none of the three.
  `getset_descriptor` C-level properties (`Pool.shape`, `Pool.is_empty_`)
  correctly remain `attribute` — verified non-callable, so unaffected.

### Re-run results (all five generators)

```
introspect_python.py: total_count=762 (was 756, +6) private_excluded=559 (unchanged) non_introspectable=0
enumerate_cli.py:      modes=15 total_flag_entries=804 (unchanged -- CLI script untouched)
parse_namespace.py:    exports=19 s3methods=8 (unchanged -- R script untouched)
compute_diff.py:       gap_total=1537 (was 1439, +98) python_only=719 (was 673, +46) cli_only=818 (was 766, +52) covered=48 (was 140, -92)
spec_crosscheck.py:    spec_claims_not_found_count=1 (unchanged) gaps_missing_from_spec_count=1281 (was 1199, +82)
```

Re-ran the full pipeline twice and diffed `sha256sum` of all five JSON
outputs — byte-identical both times; determinism holds after the fix.
`git -C vendor/catboost status --porcelain` — clean (empty), confirming
`vendor/` untouched by this fix round.

**Gap total moved up by 98, as expected: +92 from Critical 2** (false
"covered" rows reclassified as gaps — exactly the 92 the reviewer counted)
**+6 net from Critical 1** (+8 real enum members newly enumerated as gap
rows, −2 for the two removed stdlib leak rows, which had themselves
contributed to prior gap and non-gap counts). Both fixes push the total in
the same direction the brief requires: when in doubt, a row stays a gap
rather than getting silently resolved either way.

### Files changed this round

- `tools/parity/introspect_python.py` — shared enum expansion, module-leak
  filter, `callable()` kind-tagging fix.
- `tools/parity/compute_diff.py` — exact-match-only coverage rule, `covered`
  array added to output.
- `tests/fixtures/parity/python_surface.json`,
  `tests/fixtures/parity/capability_diff.json`,
  `tests/fixtures/parity/spec_crosscheck.json` — regenerated.
  `cli_surface.json` / `r_surface.json` unchanged (their generators were not
  touched this round).

### Updated Section V

```toon
task_id: catboost-8z4.5
success: true
data:
  python_version_used: "1.2.10"
  cli_version_used: "tags/v1.2.10 (commit b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084)"
  namespace_source: "vendor/catboost/catboost/R-package/NAMESPACE"
  python_surface_count: 762
  python_private_excluded_count: 559
  python_non_introspectable_count: 0
  cli_mode_count: 15
  cli_flag_count: 804
  r_export_count: 19
  r_s3method_count: 8
  gap_count_total: 1537
  gap_count_python_only: 719
  gap_count_cli_only: 818
  spec_claims_not_found: ["sparse/CSR pool input"]
  gaps_missing_from_spec: ["1281 entries -- see tests/fixtures/parity/spec_crosscheck.json:gaps_missing_from_spec for full list"]
  inventory_path: "tests/fixtures/parity/{python_surface,cli_surface,r_surface}.json"
  diff_path: "tests/fixtures/parity/capability_diff.json"
verdict:
  inventory_complete: true
  spec_hand_list_was_accurate: false
  reasoning: "Fix round 1 corrected two completeness defects: an enum-member blind spot in the submodule scan path (fixed by factoring expansion into one shared function called from every class-recording site) and a substring coverage heuristic that silently filtered 92 of 140 covered rows as false positives (replaced with exact-match-only, with every match now recorded in an auditable `covered` list). Gap total moved up 98 (1439 -> 1537), the correct direction per the brief's over-reporting bias. Determinism re-verified byte-identical across two full re-runs."
  confidence: 90
error_log: null
```
