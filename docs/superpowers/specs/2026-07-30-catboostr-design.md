# catboostr — Design Spec

**Date:** 2026-07-30
**Status:** Approved (design), pending implementation plan
**Upstream pinned at:** catboost/catboost `v1.2.10` (released 2026-02-19)

## 1. Goal

Ship `catboostr`: a fork of the CatBoost R package that reaches feature parity with the
CatBoost Python package and CLI, treats R as a first-class target rather than a
transliteration of the Python API, and is eventually accepted on CRAN.

Parity is defined as an equivalence claim, not a feature checklist: for every capability,
R and Python must produce numerically equivalent results from equivalent inputs, verified
by a differential test.

## 2. Problem Statement

The upstream R package is a second-class citizen:

- **Missing capability.** Absent from R but present in Python/CLI: embedding features,
  sparse/CSR pool input, timestamps, `grid_search`, `randomized_search`, `select_features`,
  `calc_feature_statistics`, `ShapInteractionValues`, `PredictionDiff`, `plot_tree`,
  training-progress plotting, `model.compare`, explicit pool quantization, text
  tokenizer/dictionary configuration, `init_model` continued training, custom R
  loss/metric callbacks, GPU training, distributed training.
- **Broken capability.** `catboost.get_object_importance` fails for MultiClass
  (upstream #869). Multi-target losses (MultiRMSE, MultiLogloss) are reported unusable,
  though the pool path does accept matrix labels — root cause unconfirmed and must be
  established by test, not by reading.
- **Broken distribution.** Never on CRAN (#439), not on r-universe (#1846). The upstream
  `configure` downloads a prebuilt `libcatboostr` from GitHub Releases (#724) — a flat
  violation of CRAN Policy, which forbids network access during installation. Roughly nine
  open issues are pure install/build breakage. Users end up on stale binaries that lack
  functions the current source exports.
- **Blocked ecosystem.** `bonsai` declined a CatBoost engine explicitly because the R
  package is not on CRAN. `vetiver` excludes it. `DALEX` has no native support. Only
  `mlr3extralearners` integrates, because it tolerates a non-CRAN dependency. Every
  missing integration traces to the same root cause: CRAN absence.

### Evidence

| Claim | Evidence |
| :--- | :--- |
| 20 exported R functions; `from_matrix`/`from_data_frame`/`from_file` unexported | upstream `R-package/NAMESPACE`, `R/catboost.R:118,141,243` |
| Multi-column labels do reach the core | `R/catboost.R:164` (`label <- as.matrix(label)`); `src/catboostr.cpp:147-161,254,302` |
| Shipped `libcatboostr` is CPU-only | zero CUDA references in `R-package/CMakeLists.txt` / `configure` |
| `configure` fetches a binary over the network | upstream issue #724 |
| Install size ~141 MB | secondary source, confidence 55 — **must be measured in Phase 0** |

## 3. Decisions

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Package name | `catboostr` | No CRAN name collision if upstream ever submits; matches the existing `libcatboostr` shared object. |
| Fork shape | Standalone repo, pinned core | Upstream enters as a submodule at a release tag. Avoids inheriting a multi-GB monorepo and permanent merge conflicts. |
| API compatibility | Strict superset | Every upstream `catboost.*` name and signature preserved. Nothing that works today breaks, including `mlr3extralearners`. |
| Sequencing | r-universe first, CRAN as phase 2 | A usable install in weeks instead of quarters; CRAN submission happens once, against a frozen API, rather than repeatedly against a moving one. |
| Glue layer | Existing raw `.Call` untouched; new entry points in cpp11 | cpp11 is header-only, vendorable, ALTREP-aware, and removes PROTECT boilerplate across the many new entry points parity requires. |
| GPU | Separate non-CRAN binary channel | CRAN and r-universe runners have no CUDA. Parity and CRAN cannot coexist in one artifact. |

## 4. Architecture

### 4.1 Native core acquisition — two modes, one code path

`configure` selects how `libcatboost` is obtained. The upstream binary-download mechanism
is deleted outright.

- **`vendored`** (default; release, r-universe, CRAN): builds the pinned core in-tree via
  CMake. No network access at any point during installation.
- **`prebuilt`** (development): `--with-catboost=/path` or `CATBOOST_LIB_DIR` links an
  already-built library.

Both modes compile the same `src/`, run the same R code, and pass the same tests. The
`prebuilt` mode exists because a full rebuild makes iteration impossible during feature
work — it is a development necessity, not user-facing flexibility. CRAN hardening later
reduces to proving `vendored` is portable and small enough; it is not a rewrite.

### 4.2 R API — two layers over one core

1. **Compatibility layer.** Upstream `catboost.*` names and signatures, preserved exactly.
2. **Idiomatic layer.** Real `predict()` S3 methods, formula and data.frame handling via
   `hardhat`, conventional `print`/`summary`. This is the concrete content of "R as a
   first-class citizen" — the package should behave like an R package.

Both layers call the same native core. Neither reimplements the other.

### 4.3 Differential test harness (the oracle)

Every parity claim is a test: identical data, parameters, and seed, run through Python
CatBoost and through `catboostr`, compared within tolerance.

- Python runs in a pinned `uv`-managed virtualenv. The local interpreter is 3.14.6 and
  CatBoost 1.2.10 predates it, so the harness must pin a Python version that has wheels.
- CI does not require Python. The harness records golden fixtures; CI compares against
  those. Python re-runs only when the pinned upstream version moves.

This converts "X doesn't work" into a file:line failure, and produces a parity matrix in
which every row is green, red, or explicitly out of scope.

## 5. Phases

Each phase becomes an epic. Phase 0 is a hard gate.

| # | Phase | Gate condition |
| :--- | :--- | :--- |
| 0 | **Build spike.** Build upstream's R package from the pinned tag. Measure compile time, artifact size, tarball size. Prove cpp11 and raw `.Call` registration can coexist in one `R_init_`. | If vendoring proves intractable, the approach is revisited with numbers before anything is built on it. |
| 1 | Repo skeleton, dual-mode `configure`, CI green under r-universe constraints. | Package installs from a clean checkout with no network. |
| 2 | Differential harness + fixtures. Every gap-inventory claim becomes a passing or failing test. | Root cause established for multi-target and virtual ensembles. |
| 3 | Data/Pool parity: multi-target labels, embeddings, sparse/CSR, timestamps, quantized pools, text tokenizers. | Differential tests green. |
| 4 | Analysis parity: `ShapInteractionValues`, `PredictionDiff`, `calc_feature_statistics`, object-importance MultiClass fix, `plot_tree`. | Differential tests green. |
| 5 | Training-control parity: `init_model`, grid/randomized search, `select_features`, full parameter surface documented and validated. | Differential tests green. |
| 6 | Custom R loss/metric callback bridge. | Highest risk; isolated deliberately. May prove infeasible in Python's form — see Risks. |
| 7 | Idiomatic layer: `predict()` S3, hardhat formula interface, print/summary, training-progress plots. | |
| 8 | Ecosystem: parsnip engine, mlr3 learner, DALEX/iml, vetiver. | |
| 9 | Distribution: r-universe live, GPU release channel, pkgdown site, vignettes. | |
| 10 | CRAN hardening and submission. | |

Phases 0-2 are foundational and gate everything after them. Phases 9-10 are where the
CRAN goal actually lands; that is quarters of work, not weeks.

## 6. Alternatives Considered

| Alternative | Why not chosen |
| :--- | :--- |
| **CRAN-first** (solve vendoring and size before any feature work) | Nothing installable for a long time; feature work blocked behind packaging archaeology. Retained as the phase-10 destination. |
| **Thin package + system `libcatboost`** (the `sf`/GDAL model) | CRAN-legal with a tiny tarball and fast compile, but no distro ships `libcatboost`, so we would own conda-forge and homebrew feedstocks and every user hits an install wall. Trades one packaging problem for another. |
| **Full monorepo fork** | Inherits a multi-GB repo and permanent upstream merge conflicts for no gain — the R package needs the core's source, not its history. |
| **Wrap the CatBoost CLI** | Rejected. CRAN forbids bundling standalone executables at this size; no precedent exists for a full ML engine being CLI-shelled from R; loses in-memory pools. |
| **`reticulate` over the Python package** | Instant parity, but inherits Python environment management on top of R's, forfeits direct C++ performance, and does not solve CRAN availability. Acceptable only as a bounded stopgap for Python-only features, never for core train/predict/pool paths. |
| **Rcpp instead of cpp11** | Heavier compile and header cost across many new translation units. |
| **extendr (Rust)** | Would add a third toolchain (rustc + cargo) atop CMake, C++, and Python for no net gain against a C++ core. |

## 7. Risks

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Vendored build too large or slow for CRAN | Kills the stated end goal | Phase 0 measures it before anything depends on it. Fallback: the system-library model from §6. |
| cpp11 and raw `.Call` registration conflict over `R_init_libcatboostr` | Forces a glue-layer decision late | Proven or disproven in Phase 0. Fallback: raw `.Call` throughout, which is what xgboost does. |
| Custom R loss/metric callback infeasible | Phase 6 does not ship | Isolated as its own phase. R's single-threaded evaluator constrains any callback design; the constraint is documented rather than worked around silently. |
| Python 1.2.10 has no wheels for an available Python | Blocks the oracle | Pin the Python version in the harness venv; fall back to building CatBoost Python from the same pinned source. |
| Upstream bumps break the fork | Ongoing maintenance cost | Submodule moves are deliberate and gated on a full differential suite run. |

## 8. Out of Scope

- Modifying upstream CatBoost core source. The submodule is read-only; anything requiring
  core changes is filed upstream and tracked, not patched locally.
- Distributed/MPI training beyond confirming whether it is reachable from R at all.
- Publishing under the name `catboost`.
