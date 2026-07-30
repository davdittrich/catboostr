# catboostr — Design Spec

**Date:** 2026-07-30
**Status:** Approved (design), pending implementation plan
**Upstream pinned at:** catboost/catboost `v1.2.10` (released 2026-02-19)

## 1. Goal

Ship `catboostr`: a fork of the CatBoost R package in which **every capability available in
the CatBoost CLI or Python package is also available in R**, and which is eventually
accepted on CRAN.

"R as a first-class citizen" is defined, by explicit user instruction, as exactly that
parity — nothing more. Idiomatic-R additions that do not correspond to a CLI or Python
capability are out of scope (see §8).

Parity is an equivalence claim, not a feature checklist: for every capability, R and
Python/CLI must produce equivalent results from equivalent inputs, verified by a
differential test.

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
  open issues are pure install/build breakage. Users end up on stale binaries lacking
  functions the current source exports. This is the confirmed root cause of the reported
  "virtual ensembles are not available": `catboost.virtual_ensembles_predict` is in
  upstream's NAMESPACE, so the user's installed build predated it.

### Evidence

| Claim | Evidence |
| :--- | :--- |
| 19 exported R functions; `from_matrix`/`from_data_frame`/`from_file` unexported | upstream `R-package/NAMESPACE`, `R/catboost.R:118,141,243` |
| Multi-column labels do reach the core | `R/catboost.R:165` (`label <- as.matrix(label)`; line 164 is the `if` guard); `src/catboostr.cpp:147-161,254,302` |
| Shipped `libcatboostr` is CPU-only | zero CUDA references in `R-package/CMakeLists.txt` / `configure` |
| `configure` fetches a binary over the network | upstream issue #724 |
| Install size ~141 MB | secondary source, confidence 55 — **must be measured in Phase 0** |

## 3. Decisions

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Package name | `catboostr` | No CRAN name collision if upstream ever submits; matches the existing `libcatboostr` shared object. |
| Fork shape | Standalone repo, pinned core | Upstream enters as a submodule at a release tag. Avoids inheriting a multi-GB monorepo and permanent merge conflicts. |
| API compatibility | Strict superset | Every upstream `catboost.*` name and signature preserved. Nothing that works today breaks. |
| Scope boundary | CLI/Python parity only | Per explicit user instruction. No new API surface that lacks a CLI or Python counterpart. |
| Sequencing | r-universe first, CRAN as phase 2 | A usable install in weeks instead of quarters; CRAN submission happens once, against a frozen API. |
| Glue layer | Existing raw `.Call` untouched; new entry points in cpp11 | cpp11 is header-only, vendorable, and removes PROTECT boilerplate across many new entry points. **Contingent on the Phase 0 registration probe.** |
| GPU | Separate non-CRAN binary channel, identical R API | CRAN and r-universe runners have no CUDA. Parity and CRAN cannot coexist in one artifact. |

## 4. Architecture

### 4.1 Native core acquisition — two modes, one code path

**What upstream actually does today** (verified at tag `v1.2.10`, not assumed):

- `R-package/configure` has **zero** occurrences of `cmake`. It never invokes CMake, Make,
  or a compiler. It obtains `libcatboostr` by exactly three routes: an environment variable
  pointing at a prebuilt library, a pre-existing `src/libcatboostr.{so,dylib}` in the
  checkout, or a network download (`configure:1756-1796`, `catboost_download_dynlib`).
- The only from-source path reachable from `R-package/` is `src/Makefile` →
  `Makefile.inner`, which shells out to `ya make`, Yandex's proprietary build tool.
- The R package is pulled into the native build by `add_subdirectory(R-package)` from the
  platform-specific top-level file (`catboost/CMakeLists.linux-x86_64.txt:19`), and that
  tree is driven by `build/build_native.py`. That script hardcodes `-G Ninja`
  (`build_native.py:582`) and injects Conan into the CMake configure step via
  `-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=.../cmake/conan_provider.cmake`
  (`build_native.py:585`). Conan resolves third-party dependencies over the network the
  moment `cmake` runs.

**Consequence:** no network-free, Conan-free, from-source build of the CatBoost core exists
anywhere in the upstream repository. Producing one is **new engineering**, not a
configuration or hardening pass. Neither `ninja` nor `conan` is installed in the target
environment, which sharpens rather than softens the problem.

**Target design.** `configure` is rewritten, not patched. The binary-download mechanism is
deleted outright.

- **`vendored`** (default; release, r-universe, CRAN): builds the pinned core in-tree with
  no network access at any point during installation. Requires decoupling the CMake tree
  from Conan's dependency provider by vendoring the third-party dependencies Conan
  currently fetches. Phase 0 establishes the size and tractability of that decoupling;
  Phase 1 performs it.
- **`prebuilt`** (development): `--with-catboost=/path` or `CATBOOST_LIB_DIR` links an
  already-built library.

Both modes compile the same `src/`, run the same R code, and pass the same tests. The
`prebuilt` mode exists because a full rebuild makes iteration impossible during feature
work — a development necessity, not user-facing flexibility.

If Phase 0 finds the Conan decoupling intractable, the fallback is the system-library model
from §6, which moves the native build out of the R package entirely.

### 4.2 R API — one layer

Upstream `catboost.*` names and signatures preserved exactly; new capabilities added as
new `catboost.*` functions mirroring their Python/CLI counterparts. Upstream's existing S3
methods (`predict.catboost.Model`, `print`/`summary.catboost.Model`, the `catboost.Pool`
methods) are retained as-is because they already exist — retaining them is compatibility,
not new surface.

No `hardhat` blueprint layer, no recipes/parsnip/mlr3/DALEX/vetiver adapters, no
alternative idiomatic API. These have no CLI or Python counterpart and are out of scope
per §8.

### 4.3 Differential test harness (the oracle)

Every parity claim is a test: identical data, parameters, and seed, run through Python
CatBoost (or the CLI, where the capability is CLI-only) and through `catboostr`, compared
within tolerance.

- Python runs in a pinned `uv`-managed virtualenv. The local interpreter is 3.14.6 and
  CatBoost 1.2.10 predates it, so the harness must pin a Python version that has wheels.
- CI does not require Python. The harness records golden fixtures; CI compares against
  those. Python re-runs only when the pinned upstream version moves.

**Verification method by output type.** Numeric tolerance comparison does not cover every
capability, so each capability declares its method:

| Output type | Method |
| :--- | :--- |
| Numeric (predictions, importances, metrics) | Elementwise comparison within a declared tolerance. |
| Model artifacts | Round-trip: save in R, load in Python, compare predictions; and the reverse. |
| Structural (`plot_tree`, `calc_feature_statistics`, progress logs) | Both sides serialize to a **canonical JSON form** with a field mapping declared per capability in the parity matrix; comparison is field-by-field, numeric leaves within tolerance. Rendering gets a separate smoke test asserting it runs and returns an object of the expected class. The Phase 2 harness owns defining the serializer and the mapping; a capability without a declared mapping cannot be marked green. |
| Text/console output | Snapshot tests whose snapshot files are **generated from the oracle's output**, never from R's. A snapshot authored from R output only proves R is self-consistent, which is not the claim being made. Snapshots are regenerated from the oracle whenever the pinned upstream version moves. |
| Error paths | Assert on error class and message, including GPU-requested-but-unavailable (§4.4). |

**Two oracles, not one.** Python is the reference for capabilities Python exposes. For
CLI-only capabilities — distributed training via `run-worker` being the flagship case — the
reference is the CatBoost CLI binary at the same pinned version, driven over files. Both
oracles are pinned, both are stood up in Phase 0, and a capability is compared against
whichever oracle actually exposes it.

### 4.4 GPU parity

GPU is a capability the CLI and Python have, so it is in scope.

**Hardware reality.** CatBoost's GPU backend is CUDA-only; there is no ROCm or HIP support.
The development machine has an AMD GPU, so it cannot run CatBoost on GPU at all — this is
not a driver or toolchain gap that can be closed locally. Neither can CRAN or r-universe
runners, which have no CUDA. Consequently GPU parity cannot be verified anywhere currently
available to this project, and the verification route is an explicit open decision recorded
in §9.

Regardless of how verification is resourced:

- GPU differential tests are **tagged and skipped** when no CUDA device is present, and
  **required** on a CUDA-capable runner. Parity is claimed only from a run on real
  hardware; a skipped test never counts as green.
- On a CPU-only build, `task_type="GPU"` must fail with a specific, tested error naming the
  GPU-enabled channel — not a crash and not a silent fallback to CPU. A silent CPU fallback
  would make GPU parity results meaningless.
- The GPU-enabled artifact is built from the same tree and ships via GitHub Releases.

### 4.5 Regression protection for the strict-superset promise

The claim "nothing that works today breaks" is verified, not asserted: upstream's own
`R-package` test suite is vendored into the fork and must pass unmodified at every phase
gate. A change that requires editing an upstream test halts and reports rather than
silently rewriting the test.

## 5. Phases

Each phase becomes an epic. Phase 0 is a hard gate.

| # | Phase | Gate condition |
| :--- | :--- | :--- |
| 0 | **Build spike.** Build upstream's R package from the pinned tag. Measure compile time, artifact size, tarball size. Prove cpp11 and raw `.Call` registration can coexist in one `R_init_`. Stand up the Python oracle. | If vendoring proves intractable, the approach is revisited with numbers before anything is built on it. |
| 1 | Repo skeleton, dual-mode `configure`, upstream test suite vendored and green, CI green under r-universe constraints. | Installs from a clean checkout with no network. |
| 2 | Differential harness + fixtures + the parity matrix. Every gap-inventory claim becomes a passing or failing test. | Root cause established for multi-target and any other reported breakage. |
| 3 | Data/Pool parity: multi-target labels, embeddings, sparse/CSR, timestamps, quantized pools, text tokenizers. | Differential tests green. |
| 4 | Analysis parity: `ShapInteractionValues`, `PredictionDiff`, `calc_feature_statistics`, object-importance MultiClass fix (#869), `plot_tree`, `model.compare`. | Differential tests green, structural method per §4.3. |
| 5 | Training-control parity: `init_model`, grid/randomized search, `select_features`, virtual ensembles verified end-to-end, full parameter surface documented and validated. | Differential tests green. |
| 6 | Custom R loss/metric callback bridge. | Highest risk; isolated deliberately. See Risks. |
| 7 | GPU parity: build variant, tests on CUDA hardware, tested failure mode on CPU-only builds. | Differential tests green **on a real GPU runner**; skips do not count. |
| 8 | Distributed training parity (CLI `run-worker` equivalent reachable from R). | Differential test against the CLI. |
| 9 | Distribution: r-universe live, GPU release channel, documentation and vignettes covering the full parity surface. | Installable from r-universe. |
| 10 | CRAN hardening and submission. | Accepted. |

Phases 0-2 are foundational and gate everything after them. Phases 9-10 are where the CRAN
goal lands; that is quarters of work, not weeks.

## 6. Alternatives Considered

| Alternative | Why not chosen |
| :--- | :--- |
| **Parity + idiomatic R layer + ecosystem integration** (hardhat blueprints, parsnip engine, mlr3 learner, DALEX, vetiver) | Rejected on explicit user instruction: "First class support in R means that all functions / capabilities available for the cli and python version are also available in R. Beyond that is scope creep." Roughly doubles API surface and test burden for capabilities with no CLI/Python counterpart. |
| **CRAN-first** (solve vendoring and size before any feature work) | Nothing installable for a long time; feature work blocked behind packaging archaeology. Retained as the phase-10 destination. |
| **Thin package + system `libcatboost`** (the `sf`/GDAL model) | CRAN-legal with a tiny tarball and fast compile, but no distro ships `libcatboost`, so we would own conda-forge and homebrew feedstocks and every user hits an install wall. Trades one packaging problem for another. Retained as the Phase 0 fallback if vendoring proves intractable. |
| **Full monorepo fork** | Inherits a multi-GB repo and permanent upstream merge conflicts for no gain — the R package needs the core's source, not its history. |
| **Wrap the CatBoost CLI** | CRAN forbids bundling standalone executables at this size; no precedent for a full ML engine being CLI-shelled from R; loses in-memory pools. |
| **`reticulate` over the Python package** | **The strongest rejected alternative — rejected by user decision, not on technical grounds.** An earlier version of this spec dismissed it as "does not solve CRAN availability". That was false: `keras3` and `tensorflow` are on CRAN, wrap Python via reticulate as their primary mechanism, and satisfy the no-network-during-install rule by deferring Python setup to a post-install user-invoked step. Honestly assessed, reticulate delivers every capability in §1 far sooner, tracks upstream releases automatically, avoids the Conan/vendoring engineering entirely, and makes custom R loss/metric *easier* (reticulate passes R functions as Python callables) rather than "may prove infeasible". Its one real cost is a Python runtime dependency. The user was shown this comparison explicitly and chose the native fork; that independence from Python is the deciding requirement. |
| **Rcpp instead of cpp11** | Heavier compile and header cost across many new translation units. |
| **extendr (Rust)** | Would add a third toolchain atop CMake, C++, and Python for no net gain against a C++ core. |

## 7. Risks

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Vendored build too large or slow for CRAN | Kills the stated end goal | Phase 0 measures it before anything depends on it. Fallback: the system-library model from §6. |
| **No network-free build path exists upstream.** Conan is wired into the CMake configure step; the only R-reachable source build uses Yandex's `ya make`. | Largest single unknown in the plan. Phase 1 becomes build engineering, not packaging. | Phase 0 enumerates every Conan-provided dependency and reports the vendoring cost before Phase 1 is planned. Fallback: system-library model (§6). |
| `ninja` and `conan` absent from the target environment | Blocks even reproducing upstream's own build | Phase 0 reports what the build actually requires rather than installing tools ad hoc. |
| cpp11 and raw `.Call` registration conflict over `R_init_libcatboostr` | Forces a glue-layer decision late | Proven or disproven in Phase 0. Fallback: raw `.Call` throughout, as xgboost does. |
| Custom R loss/metric callback infeasible | Phase 6 does not ship | Isolated as its own phase. R's single-threaded evaluator constrains any callback design; the constraint is documented rather than worked around silently. |
| **No CUDA hardware exists anywhere in this project.** Development machine has an AMD GPU; CatBoost has no ROCm backend; CRAN and r-universe runners have no CUDA. | GPU parity unverifiable with current resources | GPU parity is claimed only from a real-hardware run. Phase 7 reports blocked rather than claiming green from skipped tests. Resourcing decision open — see §9. |
| Python 1.2.10 has no wheels for an available Python | Blocks the oracle | Pin the Python version in the harness venv; fall back to building CatBoost Python from the same pinned source. |
| Upstream bumps break the fork | Ongoing maintenance cost | Submodule moves are deliberate and gated on a full differential suite run plus the vendored upstream test suite. |

## 8. Out of Scope

Per explicit user instruction, anything without a CLI or Python counterpart:

- `hardhat`/`recipes` blueprint handling, `parsnip`/`bonsai` engine, `mlr3` learner,
  `DALEX`/`iml` explainers, `vetiver` deployment.
- Any redesigned or alternative "idiomatic" R API beyond the upstream `catboost.*` surface
  plus its existing S3 methods.

Also out of scope:

- Modifying upstream CatBoost core source. The submodule is read-only; anything requiring
  core changes is filed upstream and tracked, not patched locally.
- Publishing under the name `catboost`.

## 9. Open Decisions

### 9.1 How GPU parity gets verified

CatBoost's GPU backend is CUDA-only. The development machine has an AMD GPU, so no
CatBoost GPU execution is possible locally. The plan forbids counting skipped tests as
green, so Phase 7 cannot be completed without resolving this. Options, none yet chosen:

1. **Rented CUDA CI.** A GPU runner (cloud CI or a spot instance) executes the GPU
   differential suite on release tags only, to bound cost. Gives genuine verification.
2. **Ship GPU unverified, labelled.** Build the GPU variant, mark every GPU row in the
   parity matrix as "implemented, unverified", and say so in the documentation. Honest, but
   the project then knowingly ships a capability nobody has run.
3. **Declare GPU out of scope.** Contradicts the parity definition in §1, and would need to
   be an explicit user decision recorded here.

Until this is decided, Phase 7 stays planned but unstarted, and no GPU claim is made.
