# catboostr — Design Spec

**Date:** 2026-07-30
**Status:** Revised against the Phase 1 build spike (2026-07-31). The design-review blockers that
were empirical are now answered by measurement; §10 records which, and what remains deferred.
Build model, dependency surface and CRAN size are measured facts, not assumptions.

> **AMENDMENT 2026-07-31 — CRAN dropped as a distribution goal.** User decision, made after
> catboost-8z4.32's research (`docs/phase-1/catboost-8z4.32-research.md`) found that every
> CRAN-related section below (§4.1b size gate, §5 gate 1a, §6 thin-package alternative, §7
> tarball-size risk row, §9.0 30MB trigger) assumed the pruned CatBoost C++ core would ship
> inside the CRAN source tarball — engineering that was never actually built (the running
> architecture instead treats `vendor/catboost` as an externally-acquired, gitignored
> dependency the user obtains via `tools/vendor/acquire.sh`). Legally shipping a pruned
> tarball WAS confirmed permitted (CatBoost is Apache-2.0, License §4 explicitly allows
> redistributing modified Source-form Derivative Works under standard conditions), but the
> user's stated preference is that users install required build dependencies
> (CMake/compiler/Python3/Perl, plus running the acquire script) themselves rather than the
> package vendoring and shipping everything invisibly — "easier to maintain the package."
> Distribution is now **r-universe or GitHub only, built from a git checkout — not a CRAN
> source tarball.** Every CRAN-specific passage below is a historical record of the reasoning
> that led here, not a current requirement; it is left unedited in place rather than rewritten,
> so the amendment is auditable against what it supersedes. See `catboost-8z4` (epic) for the
> current, authoritative success criteria.

## 1. Goal

Ship `catboostr`: a fork of the CatBoost R package in which **every capability available in
the CatBoost CLI or Python package is also available in R**. CRAN acceptance is explicitly
NOT a goal (amendment above) — r-universe and/or GitHub, built from a git checkout, are the
distribution targets.

"R as a first-class citizen" is defined, by explicit user instruction, as exactly that
parity — nothing more. Idiomatic-R additions that do not correspond to a CLI or Python
capability are out of scope (see §8).

Parity is an equivalence claim, not a feature checklist: for every capability, R and
Python/CLI must produce equivalent results from equivalent inputs, verified by a
differential test.

## 2. Problem Statement

The upstream R package is a second-class citizen:

- **Missing capability.** Absent from R but present in Python/CLI: embedding features,
  sparse/CSR pool input (name-matching in the Phase 0 machine-generated inventory did not
  find a Python/CLI symbol containing "csr" — the capability is asserted from CatBoost's
  documented `Pool` constructor accepting scipy sparse matrices, not from the generated
  inventory; see `tools/parity/spec_crosscheck.py`'s `spec_claims_not_found`), timestamps,
  `grid_search`, `randomized_search`, `select_features`,
  `calc_feature_statistics`, `ShapInteractionValues`, `PredictionDiff`, `plot_tree`,
  training-progress plotting, `model.compare`, explicit pool quantization, text
  tokenizer/dictionary configuration, `init_model` continued training, custom R
  loss/metric callbacks, GPU training, distributed training.
- **Broken capability.** `catboost.get_object_importance` fails for MultiClass
  (upstream #869). Multi-target losses (MultiRMSE, MultiLogloss) are reported unusable,
  though the pool path does accept matrix labels — root cause unconfirmed and must be
  established by test, not by reading.
- **Broken distribution.** Never on CRAN (#439), not on r-universe (#1846). The upstream
  `catboost/R-package/configure` downloads a prebuilt `libcatboostr` at install time —
  verified directly from source at `configure:1793` (`catboost_download_dynlib`), not
  inferred from an issue report. This is a flat violation of CRAN Policy, which forbids
  network access during installation. (Issue #724 is sometimes cited for this; it should not
  be. It is a closed 2019 report of a transient 404 against v0.13, not a tracking issue for
  the design flaw. The source is the evidence.) Roughly nine open issues are pure
  install/build breakage. Users end up on stale binaries lacking
  functions the current source exports. This is the confirmed root cause of the reported
  "virtual ensembles are not available": `catboost.virtual_ensembles_predict` is in
  upstream's NAMESPACE, so the user's installed build predated it.

### Evidence

| Claim | Evidence |
| :--- | :--- |
| 19 exported R functions; `from_matrix`/`from_data_frame`/`from_file` unexported | upstream `R-package/NAMESPACE`, `R/catboost.R:118,141,243` |
| Multi-column labels do reach the core | `R/catboost.R:165` (`label <- as.matrix(label)`; line 164 is the `if` guard); `src/catboostr.cpp:147-161,254,302` |
| Shipped `libcatboostr` is CPU-only | zero CUDA references in `R-package/CMakeLists.txt` / `configure` |
| `configure` fetches a binary over the network | `catboost/R-package/configure:1793`, `catboost_download_dynlib` (source, not issue report) |
| ~~Install size ~141 MB~~ | **REFUTED in Phase 0.** Measured ~34 MB installed, `libcatboostr.so` 32.9 MB. The 141 MB figure came from a secondary source at confidence 55 and was wrong by ~4×. |

## 3. Decisions

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Package name | `catboostr` | No CRAN name collision if upstream ever submits; matches the existing `libcatboostr` shared object. |
| Fork shape | Standalone repo, pinned core | Upstream enters as a **gitignored acquired tree**, not a git submodule — `.gitignore` carries `/vendor/` and there is no `.gitmodules`. `tools/vendor/acquire.sh` fetches tag `v1.2.10` at SHA `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084` and fails closed on mismatch. Avoids inheriting a multi-GB monorepo and permanent merge conflicts. **Consequence for CI and for any fresh clone: `vendor/` is EMPTY until `acquire.sh` runs**, so every build job must acquire first. Earlier drafts of this spec called it a submodule; that was wrong. |
| Build model | **The fork defines its OWN CMake target** (`add_shared_library`) linking the same upstream targets by name plus the fork's own sources, injected into the disposable copy by appending `add_subdirectory(fork-src)` to the root `CMakeLists.txt` **after** upstream's own `add_subdirectory(catboost)`. **Measured, not assumed:** `-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES` does NOT work for this — it runs too early, before upstream's targets exist, so `target_link_libraries` cannot resolve them by identity. | Corrects two earlier wrong models. Upstream's `catboostr` target hardcodes its source list in the read-only submodule, so the fork's C++ cannot join it — but the R-Makevars alternative would have required hand-reproducing whole-archive semantics for 28 `.global` archives (`cmake/common.cmake:149-166`) plus the mimalloc allocator and export script. Owning a CMake target edits no submodule file and lets CMake resolve whole-archive, allocator, export script and link order for free. **VERIFIED by the spike** (catboost-8z4.8): built, loaded via `dyn.load`, entry point called, and 23/23 `.global` archives confirmed wrapped in `-Wl,--whole-archive` in the actual link line. |
| API compatibility | Strict superset | Every upstream `catboost.*` name and signature preserved. Nothing that works today breaks. |
| Scope boundary | CLI/Python parity only | Per explicit user instruction. No new API surface that lacks a CLI or Python counterpart. |
| Sequencing | Release at R1, immediately after the CPU capability phases | Ship to r-universe and submit to CRAN once Phases 1-5 are green, then continue parity as minor releases. See §5. |
| Platform scope for R1 | **Linux and macOS. Windows deferred.** | CRAN builds Windows for every accepted package, and MSVC-produced C++ static libraries cannot link into a mingw-built R DLL (different mangling, runtime and C++ ABI); upstream's Windows CMake path targets MSVC/clang-cl while R on Windows uses Rtools mingw-w64. Solving that is real work that would gate R1 on an unmeasured assumption. R1 therefore targets Linux and macOS; full CRAN acceptance follows once Windows is solved, and the spec says so rather than discovering it in Phase 1. |
| Glue layer | **Raw `.Call` throughout. Decided; no spike.** | The glue is 28 entry points. Upstream already provides the exception safety cpp11 is usually bought for (`R_API_BEGIN`/`R_API_END`, `src/catboostr.cpp:45-58`). cpp11 would buy less PROTECT boilerplate at the cost of a new dependency, the combined-table registration trap, and a maintainer footgun this spec itself called a trap. Zero new mechanism wins. This is what xgboost does. §4.7's trap is thereby designed out rather than mitigated. |
| GPU | Separate non-CRAN binary channel, identical R API | CRAN and r-universe runners have no CUDA. Parity and CRAN cannot coexist in one artifact. |

## 4. Architecture

### 4.1 Native core acquisition — two modes, one code path

**What upstream actually does today** (verified at tag `v1.2.10`, not assumed):

- `catboost/R-package/configure` has **zero** occurrences of `cmake`. It never invokes
  CMake, Make, or a compiler. It obtains `libcatboostr` by exactly three routes: the
  `CATBOOST_DYNLIB` environment variable, a pre-existing `src/libcatboostr.{so,dylib}` in
  the checkout, or a network download (`catboost/R-package/configure:1756-1796`, download
  call at :1793).
- The only from-source path reachable from `catboost/R-package/` is `src/Makefile` →
  `Makefile.inner` (13 lines), which shells out to `ya make`, Yandex's proprietary build
  tool.
- The R package is pulled into the native build by `add_subdirectory(R-package)` from the
  platform-specific top-level file (`catboost/CMakeLists.linux-x86_64.txt:19`), and that
  tree is driven by `build/build_native.py`. That script hardcodes `-G Ninja`
  (`build_native.py:582`) and injects Conan into the CMake configure step via
  `-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=.../cmake/conan_provider.cmake`
  (`build_native.py:585`). Conan resolves third-party dependencies over the network the
  moment `cmake` runs.

**MEASURED IN PHASE 0 (catboost-8z4.1) — this section's earlier pessimism is refuted.**
The core builds cleanly from source once `ninja` and `conan` are present, and both are pip
wheels installable into a throwaway venv, which is not a system-package install:

| Measurement | Value |
| :--- | :--- |
| Build result | **4265/4265 targets, zero errors**, `libcatboostr.so` linked |
| Wall-clock, 32 cores, network ON | **164 s** (run `run1_network_on`; log trimmed at `docs/phase-0/run1_network_on.trimmed.log`) |
| `libcatboostr.so` | **32.9 MB** unstripped |
| Installed R library | **~34 MB** — the ~141 MB claim is **refuted**, by roughly 4× |
| R source tarball | 0.164 MB (C++ excluded; not representative post-vendoring) |
| Conan closure | **13 packages**, of which exactly **2 are link dependencies of `libcatboostr.so`: openssl and zlib** (zlib transitively via openssl). All 11 others are `context=build`. `bzip2` and `pcre` carry `libs=True` only as edges of the **swig** node — they link into the code generator, not the shipped library. |

**Offline build (run `run6_offline_final`; log trimmed at
`docs/phase-0/run6_offline_final.trimmed.log`) — built with zero network access during the
build itself**, every proxy pointed at a black hole and the Conan remote disabled. This
demonstrates the build *step* needs no network given a pre-populated Conan cache and a venv
that already has `ninja`, `conan`, NumPy, and Cython installed — it does **not** demonstrate
a from-scratch, no-network-ever install on a clean machine (populating that cache and venv
still requires network access once, upstream, before this build runs). That from-scratch
case remains Phase 1 work (see §7):

```
configure_rc=0  build_rc=0  total 174 s
[4265/4265] Linking CXX shared library .../libcatboostr.so
network attempts: 0    hard failures: 0
libcatboostr.so: 34,467,216 bytes — byte-identical in size to the network-ON build
```

Three things had to be true, all now known:

1. **Disable the Conan remote.** Conan probes `center2.conan.io` during CMake *configure*
   even with a fully warm cache — a binary-availability check, not a download.
   `conan remote disable conancenter` (or `-nr` / `--no-remote`) clears it; all 13 packages
   then resolve from cache.
2. **Satisfy or gate the `hnsw` Python component.** `library/python/CMakeLists.txt:15` calls
   `add_subdirectory(hnsw)` **unconditionally** — unlike the R-package directory, it is not
   gated by `CATBOOST_COMPONENTS`. An R-only configure therefore descends into it and
   demands NumPy, then Cython. Supplying both in the build venv works; gating the component
   out is the cleaner Phase 1 fix.
3. **Pass the flags upstream's own build path passes.** `-DCMAKE_POSITION_INDEPENDENT_CODE=On`,
   `-DCATBOOST_COMPONENTS=R-package`, `-DHAVE_CUDA=no`. Omitting PIC produces a link failure
   (`relocation R_X86_64_TPOFF32 against _mi_heap_default cannot be used with -shared`) that
   looks like an upstream defect but is purely an invocation error.

**Status: tractable, established by demonstration.** §6's system-library fallback is no
longer the expected outcome. Phase 1's remaining work is vendoring openssl and zlib
portably and making the three conditions above reproducible in `configure`.

**Measurement caveats.** The 32.9 MB figure is an **unstripped, debug-info-bearing** binary,
installed via the `CATBOOST_DYNLIB` copy-in path rather than compiled through R's own
build-and-strip pipeline. A stripped release build would be smaller, so the direction favours
the size refutation rather than undermining it. Sizes are MiB, not decimal MB.

**Constraint discovered while measuring:** any from-source build writes
`CMakeUserPresets.json` into the source tree unconditionally (Conan 2.x `CMakeToolchain`
behaviour, not flag-controlled). The pinned upstream snapshot must therefore be treated as
read-only and built from a disposable copy, never in place.

**Target design.** `configure` is rewritten, not patched. The binary-download mechanism is
deleted outright.

#### Who compiles what (decided, then VERIFIED by the Phase 1 spike)

Phase 0 built upstream's **unmodified** `catboostr` CMake target and installed the result by
`CATBOOST_DYNLIB` copy-in. That never exercised the fork's own glue-compilation model, and
the model the spec first implied does not work: the `catboostr` target hardcodes its source
list to exactly `catboostr.cpp` and `init.c`
(`vendor/catboost/catboost/R-package/src/CMakeLists.linux-x86_64.txt:57-61`), duplicated
across 8 machine-generated platform files, all inside the submodule §8 declares read-only.
Adding fork sources or editing `init.c` there is impossible without violating that rule.

A second model — R's own `Makevars` compiling the glue and linking CMake-built static
libraries — was then written into this spec and is also **rejected**. It would have required
hand-reproducing whole-archive semantics for the 28 `.global` archives that
`add_global_library_for` creates (`cmake/common.cmake:149-166`), plus the allocator and the
generated linker version script. Getting that subtly wrong produces a package that builds,
installs, loads, and then fails at runtime with missing model formats or metrics.

**Decision: the fork defines its own CMake target.** `add_shared_library(catboostr)` in a
fork-owned `CMakeLists.txt`, listing the fork's own sources and linking upstream's targets by
name. It is injected by appending `add_subdirectory` to the copied tree's root
`CMakeLists.txt` **after** upstream's own `add_subdirectory(catboost)` — not via
`-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES`, which the spike proved runs too early for
`target_link_libraries` to resolve upstream targets by identity.

**Verified by the spike (catboost-8z4.8):** the target builds, `dyn.load`s into R, and its
entry point runs. The actual `ninja -v` link line wraps **23 of 23** `.global` archives in
`-Wl,--whole-archive`, and two independent object-factory registrars were exercised —
`TTrainerFactory` through a live training run, `TModelLoaderFactory` through a JSON
export/reload that reproduced the original prediction exactly. CMake resolves whole-archive,
allocator, export script and link order for free; no submodule file is edited.

Consequences and obligations:

- The glue compiles against the **installing user's** R headers, not the stale vendored
  `contrib/libs/r-lang` set (which lacks `Rversion.h` and `R_ext/Visibility.h`). `configure`
  passes R's include path explicitly; it does not rely on upstream's vendored copy.
- `configure` must forward R's compiler configuration (`R CMD config CC`, `CXX`, `CFLAGS`,
  `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`) into the CMake invocation, which CRAN requires of any
  package that shells out to another build system.
- `SystemRequirements` must declare **CMake (>= 3.15), C++20, and Python3** — the spike
  established that Python3 is load-bearing at build time regardless of Conan
  (`cmake/common.cmake:3`; it generates `__vcs_version__.c`, which is compiled into the
  target, and the linker version script).
- The generator must be selected explicitly rather than inheriting upstream's hardcoded
  `-G Ninja`: Ninja may not exist on a user's machine, so Makefiles are the portable default
  with Ninja used when present. Only the Ninja path has been exercised so far.
- `-DCMAKE_POSITION_INDEPENDENT_CODE=On` is mandatory. Omitting it produces a
  `relocation R_X86_64_TPOFF32 against _mi_heap_default` link failure, which is a missing
  flag rather than an upstream defect.

#### Modes

- **`vendored`** (default; release, r-universe, CRAN): builds the pinned core with no
  network access at any point during installation.
- **`prebuilt`** (development): reuses an already-built library via the **existing upstream
  spelling `CATBOOST_DYNLIB`** (`configure:1756-1796`) — no new environment variable and no
  new flag is invented for a mechanism upstream already has. It must print the resolved path
  and its SHA256 during configure, and release builds must assert vendored mode, so a stale
  or unintended library cannot be substituted silently.

Both modes run the same R code and pass the same tests. `prebuilt` exists because a full
rebuild makes iteration impossible during feature work — a development necessity, not
user-facing flexibility.

#### Conan is not an install-time dependency

The spec previously contradicted itself: §4.1 eliminated Conan while §7 proposed vendoring
its cache. Resolved: **Conan must not appear anywhere in the end-user install path.** Conan
is a Python application; keeping it would make Python and pip install-time dependencies,
which is fatal for CRAN. The vendored build therefore carries the two link dependencies
(openssl, zlib) as pinned, checksum-verified vendored sources found via plain
`find_package`/direct compilation, with no Conan mediation. Conan remains a *maintainer-side*
tool for reproducing upstream's own build during development only, and while it is used there
it must be driven from a committed `conan.lock`.

**Tractability is established for building, NOT for CRAN shipping.** Both builds completed
4265/4265 with zero errors and only two link dependencies need vendoring. But "builds" and
"is CRAN-shippable" are different claims and the spec previously conflated them — see the
size risk in §7 and the reinstated contingency in §6.

### 4.1b Phase 1 spike results (2026-07-31) — measured

Three spikes answered the empirical questions design review raised. Reports:
`docs/phase-1/catboost-8z4.{6,7,8}-report.md`.

**Build model — VIABLE.** A fork-owned CMake target linking upstream's targets by name builds,
loads and runs. The whole-archive hazard is real and CMake handles it: the actual `ninja -v`
link line wraps **23 of 23** `.global` archives in `-Wl,--whole-archive`. Two independent
object-factory registrars were exercised — `TTrainerFactory` through a live training run and
`TModelLoaderFactory` through a JSON export/reload that reproduced the original prediction
exactly. Raw `.Call` with plain `extern "C"` symbols worked; no cpp11, no `init.c` needed for
the spike. Injection is `add_subdirectory` appended **after** upstream's, not
`CMAKE_PROJECT_TOP_LEVEL_INCLUDES`.

**Dependency surface — 4 invoked, not 13 and not 2.**

| Package | Status |
| :--- | :--- |
| openssl, zlib, ragel, yasm | **INVOKED** in an R-package-only build |
| swig, bzip2, pcre, autoconf, automake, bison, flex, gnu-config, m4 | **RESOLVED_ONLY** — absent from the 84,789-line `build.ninja`; consumed inside Conan's own recipe sandbox to compile ragel/swig |

**openssl is reachable only through code R never calls.** Traced with `ninja -t query`:
`catboostr → train_lib → private/libs/distributed → library/cpp/par → library/cpp/neh →
openssl`. It enters solely via the distributed-training network transport. **Component-gating
`private/libs/distributed` looked like the single largest simplification available to
Phase 1** — it would drop openssl from the link and retire the CVE-tracking liability with it.
**REFUTED 2026-07-31, and the user then accepted openssl.** Plan review established the
coupling is at SOURCE level, not link level: `catboost/libs/train_lib/train_model.cpp:18-19`
hard-includes `private/libs/distributed/{master,worker}.h` and `:1099-1102` constructs a
`TMasterContext` unconditionally. Severing the link edge leaves undefined references; keeping
it keeps openssl. Removing it would require editing the read-only pinned tree, which §8
forbids. openssl is therefore ACCEPTED into the R1 link, pinned and CVE-tracked (see the §7
risk row). This also removes the tension with Phase 8's distributed-training parity, which the
gating plan would have had to unwind later. Note the
tension with §5's Phase 8 (distributed training parity): gating the subtree for the R1 build
does not preclude re-enabling it later, but the two must be reconciled deliberately.

**Python3 IS required at install time.** An earlier claim in this spec said removing Conan
removed Python; that was wrong. `cmake/common.cmake:3` requires it, and it is load-bearing at
build time: `vcs_info.py`/`generate_vcs_info.py` generate `__vcs_version__.c`, which is
compiled into the target, and `export_script_gen.py` generates the linker version script for
`libcatboostr.so`. `SystemRequirements` must list Python3 and CMake.

**`-DCUSTOM_ALLOCATORS=Off` works** — 858/858 steps, mimalloc absent from the link line, the
resulting `.so` loads into R. This removes the `_mi_heap_default` TPOFF32 relocation hazard and
process-wide `malloc` interposition for one flag. Caveat: load-tested only, not yet a full
train/predict round-trip.

**CRAN size — CONDITIONALLY viable.** 3902 unique translation units. Pruned tarball
**16.078 MiB** (16,858,704 bytes). **CORRECTION 2026-07-31: earlier drafts of this spec
compared that against a '5 MiB CRAN guidance'. That figure was wrong.** The CRAN Repository
Policy says source tarballs "should if possible not exceed **10MB**", and packages well above
it are live on CRAN today (`rcdklibs` 19M, `fastrmodels` 16M, `acss.data` 14M). The real
overshoot is ~1.7×, not 3.2×, and the mechanism for exceeding the figure is a justification in
`cran-comments.md`, which policy explicitly contemplates for C++ and Rust packages. The wrong
number was propagated into a design decision before anyone checked the policy text. About 19.15 MiB of the 94.9 MiB uncompressed pruned
tree is build-system plumbing: the full CMakeLists tree plus 87 unconditionally-configured
test/tool/benchmark directories that `-DCATBOOST_COMPONENTS=R-package` does not gate. Reducing
that needs build-system changes and is the obvious next lever.

**Prune-set validation caveat.** A successful configure and dry run were *not* sufficient: the
first real build failed on a Ragel include chain invisible to both `compile_commands.json` and
ninja's dep records. Prune sets must be validated by an actual build. Only
Linux/clang/no-CUDA/R-package was exercised.

**Unreconciled discrepancy, recorded rather than smoothed:** the spike measured the full tree
at 870.2 MiB and `contrib` at 178.0 MiB, against the ~163 MiB / 141.4 MiB figures this spec
had been citing. The measurement scopes evidently differ. The pruned-tarball number is the one
that matters and it was measured directly, but the baseline discrepancy is unexplained.

### 4.2 R API — one layer

Upstream `catboost.*` names and signatures preserved exactly. Upstream's existing S3 methods
(`predict.catboost.Model`, `print`/`summary.catboost.Model`, the `catboost.Pool` methods) are
retained as-is because they already exist — retaining them is compatibility, not new surface.

**A capability does not automatically become a function.** The naive rule "mirror each
Python/CLI capability as a new `catboost.*` function" produces a broken API, because much of
the parity surface is not method-shaped. The mapping rule is:

| Upstream shape | Becomes in R |
| :--- | :--- |
| Python method / CLI mode | A `catboost.*` function, `verb_object` snake_case, matching upstream's dominant style (`get_feature_importance`, `drop_unused_features`). Never a dotted name — `catboost.compare(model, other, ...)`, **not** `model.compare`, because dots collide with R's S3 dispatch already used by `predict.catboost.Model`. |
| **Enum member** (e.g. `EFstrType.ShapInteractionValues`, `EFstrType.PredictionDiff`) | An accepted **value of an existing argument** — `catboost.get_feature_importance(..., type = "ShapInteractionValues")`. Not a new export. `python_surface.json` records these as `catboost.EFstrType.*`; §2 lists them by bare name for readability, which must not be read as a mandate for top-level functions. |
| **Training hyperparameter** (139 distinct) | A documented key of the existing `params` list — see below. Not an export. |
| **CLI flag** (224 distinct alias-sets) | Either an argument of the R function wrapping that mode, or a `params` key. Universal flags (`--help`, `--svnrevision`, `--thread-count`) are not capabilities and are flagged `universal: true` in the inventory. |

**The `params` list is the weak point, and Phase 5 owns it.** Upstream routes every
hyperparameter through an untyped `params = list()` (`R/catboost.R:1550`) with no validation
of unknown keys before JSON serialisation (`:1622`). A mistyped key is silently accepted. The
Phase 5 gate "full parameter surface documented and validated" is only meaningful with a
named mechanism, so it is fixed here:

- **Documented** = a generated `@param`-level reference for all 139 distinct hyperparameters,
  derived from the machine inventory (§4.5), not hand-written.
- **Validated** = `catboost.train`/`catboost.cv` check supplied `params` names against that
  generated list and error on unknown keys, with a documented escape hatch for
  forward-compatibility with a newer core.

No `hardhat` blueprint layer, no recipes/parsnip/mlr3/DALEX/vetiver adapters, no alternative
idiomatic API. These have no CLI or Python counterpart and are out of scope per §8.

### 4.3 Differential test harness (the oracle)

Every parity claim is a test: identical data, parameters, and seed, run through Python
CatBoost (or the CLI, where the capability is CLI-only) and through `catboostr`, compared
within tolerance.

- Python runs in a pinned `uv`-managed virtualenv. CatBoost 1.2.10 does publish
  `catboost-1.2.10-cp314-cp314-manylinux2014_x86_64.whl`, so the local 3.14.6 interpreter
  is usable and no source build of the Python package is required.
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

**Tolerance policy — a rule, not a per-test choice.** "Within a declared tolerance" is a hole
unless the declaration is governed: a red differential test can always be made green by
widening its own tolerance, which converts differential testing into tolerance-fitting.
Therefore:

- **Defaults, derived from first principles, not from observed failures.** Python path:
  relative `1e-12`, because Phase 0 established bit-exact reproducibility there. CLI path:
  relative `1e-9`, derived from the measured ~10 significant digits of `calc`'s text output
  (below). These are the defaults; a test that passes at default declares nothing.
- **Any override must be recorded with a first-principles justification** — an appeal to
  floating-point accumulation order, a documented precision limit of the output format, a
  known algorithmic non-determinism. "The test failed at 1e-12 and passes at 1e-6" is not a
  justification; it is the failure mode this rule exists to prevent.
- **Tolerances are fixed before the R implementation exists.** Widening one after
  implementation is a design change requiring the same recorded justification and review, not
  a test tweak.

**Precision ceiling on CLI comparisons (measured in Phase 0, catboost-8z4.4).** The CLI's
`calc` mode has no `--precision` flag and emits roughly 10 significant digits, not a
full float64 repr round-trip. Python fixtures are full precision; CLI fixtures cannot be.
Therefore bit-exact comparison is available on the Python path only. Every CLI-only
capability must declare a tolerance at or above the CLI's own output precision, and no test
may assert bit-exactness against CLI text output. If a CLI-only capability ever needs
tighter comparison, the extraction path must change (via the binary model or another mode) —
that is a design change, not a tolerance tweak.

**The parity matrix is the join table, and it is a real artifact.** Phase 0 produced three
tools that currently coexist rather than compose: the inventory writes
`tests/fixtures/parity/`, the Python oracle writes `tests/fixtures/oracle/`, the CLI oracle
writes `tests/fixtures/oracle-cli/`, and nothing connects a capability to the oracle that
verifies it. Phase 2's first deliverable is the matrix that joins them, one row per inventory
entry:

`inventory_row_id → oracle (python | cli | none) → verification method (§4.3 table) → tolerance → test id → state (green | red | out-of-scope + reason)`

This is the artifact §4.3 refers to when it says "declared per capability", it is what makes
the three Phase 0 tools one substrate, and it is the input to every phase after it. A
capability with no row cannot be claimed; a row with no test cannot be green.

**RED-GREEN, concretely.** For a new capability — `select_features` is representative:
1. Generate the oracle fixture (Python or CLI per the matrix row) and commit it.
2. Write the R differential test against that fixture. It fails: the R function does not exist.
3. Implement the R function and any needed glue.
4. Test passes at the default tolerance. Matrix row flips to green with its test id recorded.

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
- The GPU-enabled artifact is built from the same tree and ships via GitHub Releases,
  **with integrity requirements, because this is native code distributed outside a curated
  registry**: a `SHA256SUMS` file published alongside every asset, a documented user-side
  verification command, and any install helper must verify before loading, never
  install-then-hope. A tampered Release asset is a shared object loaded into the user's R
  session with full process privileges. The project already implements exactly this pattern
  on the consumer side (`tools/oracle/cli/acquire.sh`); it must apply it to its own
  publishing channel.
- Other failure modes get the same rigour as the GPU one, since §2 documents a real
  version-skew symptom in the wild (the "virtual ensembles missing" report, root-caused to a
  stale binary): native library load failure, R-package/`libcatboostr` version skew, and
  unsupported loss function each need a specific, tested error rather than a crash or a
  silent wrong answer.

### 4.5 The capability inventory is machine-generated, never hand-curated

The parity matrix is only as complete as its input list. A hand-written inventory — including
the one in §2 of this spec — silently omits whatever nobody happened to notice, which makes
the goal in §1 unfalsifiable for the omitted capability.

Therefore the inventory is generated, not written. **Built and run in Phase 0
(catboost-8z4.5); the generator tools live under `tools/parity/`, see
`tools/parity/README.md` for the pipeline order and regeneration command
(`tools/parity/run.sh`):**

- **Python surface**: introspect the installed `catboost` module — every public module
  member, every public method on `CatBoost`, `CatBoostClassifier`, `CatBoostRegressor`,
  `CatBoostRanker`, `Pool`, and every documented training parameter
  (`tools/parity/introspect_python.py`).
- **CLI surface**: every mode from the binary's `--help`, and every flag within each mode
  (`tools/parity/enumerate_cli.py`).
- **R surface**: every `export()` in the fork's NAMESPACE plus every registered S3 method
  (`tools/parity/parse_namespace.py`).

The parity matrix is the *diff* of those three sets (`tools/parity/compute_diff.py`). §2's
list is treated as a hypothesis to be checked against the generated inventory
(`tools/parity/spec_crosscheck.py`), not as the source of truth. Any capability present in
the Python or CLI set and absent from the R set is automatically a matrix row — green, red,
or explicitly out-of-scope with a recorded reason. A capability may not be silently absent.

**What Phase 0 found:** 1537 raw gap rows (719 Python-only, 818 CLI-only, 48 covered),
inflated roughly 2× by duplication across CLI modes and Python classes — 725 after
deduplication (224 distinct CLI flag alias-sets, 139 distinct Python parameter names; two
universal non-capability CLI flags, `--help` and `--svnrevision`, are marked rather than
dropped). Of the spec's own 18 hand-written §2 claims, 17 are confirmed present in the
generated inventory. Full numbers and methodology: `tools/parity/README.md`.

The inventory is regenerated whenever the pinned upstream version moves, so a newly added
upstream capability appears as a new red row rather than going unnoticed.

**725 rows are not 725 work items, and Phase 2 must not become 725 tickets.** Roughly 363 of
the 725 are parameter and flag surface (139 distinct hyperparameters + 224 distinct CLI flag
alias-sets) that cannot match an R export by construction, because R routes hyperparameters
through an untyped `params` list — `tools/parity/compute_diff.py:17-24` documents this
honestly and the spec now states it too. The bulk-disposition rule for Phase 2:

- **Parameter/flag rows** are dispositioned as a family, not individually: one differential
  test per parameter family that reaches the core through `params`, plus the generated
  documentation and validation described in §4.2. They do not become per-name tickets.
- **Enum-member rows** attach to the function whose argument they are (§4.2), not to
  tickets of their own.
- **Universal flags** (`--help`, `--svnrevision`, and similar) are marked non-capabilities.
- **Method- and mode-shaped rows** are the genuine feature work, and only these become
  per-capability tickets.

Phase 2 produces the dispositioned matrix; **the classification gets explicit user sign-off
before any Phase 3+ epic is written**, because that classification is what determines whether
the remaining work is weeks or quarters.

### 4.6 Regression protection for the strict-superset promise

The claim "nothing that works today breaks" is verified, not asserted: upstream's own
`R-package` test suite is vendored into the fork and must pass at every phase gate. A change
that requires editing an upstream test halts and reports rather than silently rewriting the
test.

**One carve-out, because "unmodified" is literally unsatisfiable after the §3 rename.**
Upstream's suite calls `library(catboost)` and `test_check("catboost")`
(`tests/testthat.R:2`) and uses `catboost::` at
`tests/testthat/test_caret_parameter_tuning.R:49` and
`tests/testthat/test_on_trimmed_adult_dataset.R:6`. A **scripted, mechanical package-name
rewrite** is therefore permitted and is the only permitted edit. Every other change halts and
reports. The rewrite is a script, not hand-editing, so drift is visible.

### 4.7 Native routine registration — the required mechanism

**Status: the trap below is DESIGNED OUT, not mitigated.** §3 decides raw `.Call` throughout,
so the fork ships exactly one registration source and the double-registration hazard cannot
arise. The spike confirmed the practical half: the fork target built with plain `extern "C"`
entry points and needed neither cpp11 nor a generated `init.c`.

The mechanism is recorded here because it is a real property of R that any future decision to
add a second registration source (cpp11, Rcpp, a generated table) must respect. It was
established empirically in Phase 0 (catboost-8z4.2) and independently reproduced in review.

Established empirically in Phase 0 (catboost-8z4.2) and independently reproduced in review.
This is a correctness constraint, not a style preference:

**R replaces, rather than merges, the routine table per DLL.** Calling `R_registerRoutines`
twice — once from a hand-written `init.c` and once from cpp11's generated
`R_init_<package>` — silently drops the first table. The failure is silent: the package
compiles, installs, and loads, and the missing entry points only fail at call time.

The working mechanism:

1. One combined `static const R_CallMethodDef` table containing both the raw `.Call` entries
   and the cpp11 entries.
2. One `R_registerRoutines` call, from the hand-written `init.c`.
3. cpp11's individual `extern "C"` wrapper symbols are forward-declared in `init.c`.
4. cpp11's generated init function is never invoked. It is dead code here anyway, because
   its symbol name binds to the package name (`catboostr`) while the shared object is named
   `libcatboostr` — the two differ, so `dyn.load` never auto-invokes it.
5. `R_useDynamicSymbols(dll, FALSE)` is retained, as upstream sets it. Both entry kinds
   remain reachable under it. `R CMD check` produced no registration NOTEs (relevant to
   upstream #778).

**Maintenance cost, stated explicitly because it is a trap.** `cpp11::cpp_register()` does
**not** update the combined table. Adding a cpp11 function requires two manual edits to
`init.c` — a forward declaration and a table row. A maintainer who runs `cpp_register()` and
assumes the function is wired will get a symbol that exists but is unreachable.

**The mitigation is an enforced check, not documentation.** Prose cannot stop a failure the
spec itself calls a trap. The required guard is three lines of testthat, added the moment a
second registration source exists: assert that every symbol named in an `R/*.R` `.Call()`
invocation appears in `getDLLRegisteredRoutines("libcatboostr")$.Call`. That fails loudly on
exactly the silent-drop mode the probe found, needs no code generation, and no table
generator is justified until the entry-point count actually makes hand-maintenance
burdensome.

### 4.8 Supply-chain requirements

This package compiles code from thirteen third-party sources on the end user's machine and
will be distributed through a public registry. The Python side of the project already carries
192 sha256 pins in `tools/oracle/uv.lock`; the C++ side — the code that actually ships inside
`libcatboostr.so` — currently carries none. That asymmetry is the largest security hole in the
design, and closing it is a Phase 1 requirement, not a later hardening pass.

Requirements, all modelled on patterns the repo already implements correctly
(`tools/vendor/acquire.sh`, `tools/oracle/cli/acquire.sh`):

1. **Every vendored third-party source is pinned and verified**: exact upstream URL, exact
   version, `SHA256`, and a fail-closed check. No exceptions, because these end up compiled
   into the shipped binary.
2. **A committed `conan.lock`** capturing recipe revisions and package ids, used with
   `--lockfile` for as long as Conan is used anywhere in the project. `swig`, `ragel`,
   `bison`, `flex` and `m4` execute during the build and emit source that gets compiled —
   a tampered recipe is code execution on the build host.
3. **`SHA256SUMS` published for every GPU Release asset**, with a documented verification
   command and verify-before-load in any install helper (§4.4).
4. **`.Rbuildignore`** excluding `docs/`, `tools/`, `.beads/`, `.claude/`, `.codex/`,
   `AGENTS.md`, `CLAUDE.md` before the first tarball is built, so development tooling and
   evidence logs never reach a distributed artifact.
5. **`uv run --frozen`** (or `--locked`) everywhere the oracle is invoked. Plain `uv run`
   silently re-resolves and rewrites `uv.lock` if `pyproject.toml` drifts, quietly defeating
   the strongest supply-chain control currently in the repo.
6. **A `SOURCES.md` inventory** listing every external artifact the project ingests —
   upstream tag and SHA, CLI asset and SHA256, Python lock, Conan lock, each vendored
   dependency and its SHA256. This is a lightweight SBOM and is the first thing a CRAN
   reviewer or a downstream security team will ask for.

**Trust boundary, stated so it is a decision rather than an omission.** A user of the source
package trusts that they compiled from a SHA-pinned source tree; the build is not
bit-reproducible and no published hash lets them compare their `libcatboostr.so` against
anyone else's. That is the normal position for a source-installed CRAN package and is
accepted. It is *not* acceptable for the GPU binary channel, which is why requirement 3
exists.

## 5. Phases

Phase 0 is complete. **Phases 3-8 are PROVISIONAL**: their contents are determined by the
Phase 2 classification of the 725-row matrix (§4.5), which has not happened yet. The rows
below name the capabilities currently believed to belong in each phase; they are not a
committed work breakdown, and no Phase 3+ epic is written before the classification exists
and is signed off. Treating this table as a costed multi-quarter plan would be fiction.

| # | Phase | Gate condition |
| :--- | :--- | :--- |
| 0 | **COMPLETE.** Build spike, both oracles, capability inventory, cpp11 registration probe. | Passed. Evidence in `docs/phase-0/`. |
| 1 | **Build engineering.** Fork-owned CMake target (§4.1, verified by the spike): a `configure` that copies the pinned tree, appends the fork's `add_subdirectory` after upstream's, and drives CMake with the measured flag set. Gate the `hnsw` component so an R-only configure needs neither NumPy nor Cython (`library/python/CMakeLists.txt:15` is an unconditional `add_subdirectory`). Gate the 87 unconditionally-configured test/tool/benchmark directories. **openssl is NOT gated out** — see §4.1b; it is accepted, pinned and CVE-tracked. Pin and checksum-verify every third-party source that survives. Remove Conan from the install path. Declare Python3 and CMake in `SystemRequirements`. Commit `.Rbuildignore` and `conan.lock`. | A mechanical no-network install test passes **in a fresh container** with no pre-populated Conan cache and no pre-existing build venv, asserting zero network attempts across `R CMD INSTALL`; the upstream R test suite passes against that from-source build (§4.6's baseline, currently untested); **plus the size and build-time gates below.** |
| 1a | **Size gate. MEASURED 2026-07-31: 16.078 MiB pruned** (§4.1b). | ~1.7× the CRAN policy figure of 10MB, not the 3.2× an earlier draft claimed against a wrong 5 MiB number. Not disqualifying — `rcdklibs` (19M) and `fastrmodels` (16M) are live on CRAN. Reduction is still worth pursuing: `xgboost` (1.5M) and `lightgbm` (1.7M) vendor comparable C++ ML cores and stay small by **amalgamating and stripping unused source**, not by linking system libraries. A re-measurement after pruning plus a drafted `cran-comments.md` justification is the Phase 1 exit criterion. |
| 1b | **Glue layer. DECIDED, no spike: raw `.Call` throughout** (§3, §4.7). | Closed. The spike built the fork target with plain `extern "C"` entry points and needed neither cpp11 nor a generated `init.c`, so §4.7's double-registration trap is designed out rather than mitigated. |
| 1c | **Build-time gate.** Measure `R CMD INSTALL` wall-clock on a 2-core machine resembling a CRAN check runner. | Measured number recorded. Build time is the second structural CRAN rejection cause and currently has no measurement and no abort threshold; the threshold is set once the first number exists. |
| 2 | **Classification.** Differential harness + fixtures + the parity matrix as a real join table (§4.3), seeded from the machine-generated inventory. Every one of the 725 rows dispositioned per §4.5's bulk rule. | Matrix exists with every row in a state; **user signs off the classification**; root cause established for multi-target and any other reported breakage. This gate is what converts Phases 3-8 from provisional to planned. |
| 3 | *(Provisional)* Data/Pool parity: multi-target labels, embeddings, sparse/CSR, timestamps, quantized pools, text tokenizers. Includes the CLI's `dataset-statistics` mode. | Differential tests green at default tolerance. |
| 4 | *(Provisional)* Analysis parity: `calc_feature_statistics`, object-importance MultiClass fix (#869), `plot_tree`, `catboost.compare`, and the `ShapInteractionValues`/`PredictionDiff` **argument values** (§4.2 — not new functions). Includes the CLI's `eval-feature`, `roc`, `model-based-eval` modes. | Differential tests green; structural method per §4.3. |
| 5 | *(Provisional)* Training-control parity: `init_model`, grid/randomized search, `select_features`, virtual ensembles verified end-to-end, the generated parameter documentation and validation (§4.2). Includes the CLI's `metadata` and `normalize-model` modes. | Differential tests green. |
| **R1** | **FIRST PUBLIC RELEASE — r-universe + CRAN submission.** CPU-only. Ships the strict-superset upstream surface plus everything green through Phase 5. Documentation and vignettes cover the parity matrix **as it stands**, with red rows listed honestly as not-yet-available. | Installable from r-universe; CRAN submitted. **This is where the user's stated goal is delivered**, not Phase 10. |
| 6 | *(Provisional)* Custom R loss/metric callback bridge. Highest risk; isolated deliberately. | Differential tests green, or a recorded infeasibility finding. |
| 7 | *(Provisional)* GPU parity. **Blocked on the §9.1 hardware decision.** | Differential tests green **on real CUDA hardware**; skips never count. Does not gate R1. |
| 8 | *(Provisional)* Distributed training parity (CLI `run-worker`). | Differential test against the CLI. |
| 9+ | Each subsequent parity phase ships as a **minor release against the live CRAN package**, not as a gate in front of it. | Per-phase differential tests green. |

**Sequencing rationale.** An earlier draft placed r-universe at Phase 9 and CRAN at Phase 10,
behind every parity phase — including Phase 7, which is blocked on hardware nobody has. That
contradicted §3's own "usable install in weeks, not quarters" and put the user's only stated
success criterion last, gated on the least certain work. R1 moves the release to just after
the CPU capability phases: the packaging risk that actually threatens CRAN is resolved in
Phase 1, so there is no reason to sit on it for quarters. Parity then continues against a
shipped package, where unscoped work is a backlog rather than a blocker.

## 6. Alternatives Considered

| Alternative | Status |
| :--- | :--- |
| **Thin package + system `libcatboost`** (the `sf`/GDAL model) | **REFUTED 2026-07-31 by evidence — no longer a contingency.** Selected by the user on 2026-07-31, then withdrawn the same day when research showed it cannot work. No mainstream package manager ships a `libcatboost` development package (shared library plus headers): not Debian, Ubuntu, Fedora, Homebrew, conda-forge, or vcpkg — only source and Python wheels. CRAN policy states software is installed on its Debian check machines only when it is available from Debian repositories for 'testing', and that bundling or vendoring sources is usually the faster path. A `SystemRequirements: libcatboost` would therefore leave the package **uninstallable on CRAN's own check machines** — a harder failure than an oversized tarball, because it fails at build rather than at review. Corroborating precedent: of `xgboost`, `lightgbm`, `duckdb`, `arrow`, `torch`, `tensorflow` and `keras3`, **none** declares a system library CRAN is expected to pre-install; `lightgbm` actively **removed** system-library linking in v3.0.0. |
| **Raw `.Call` throughout, no binding framework** | **CHOSEN (§3).** Upstream already has the exception safety cpp11 is usually bought for: `R_API_BEGIN`/`R_API_END` wrap every entry point in try/catch and route to `error()` (`vendor/catboost/catboost/R-package/src/catboostr.cpp:45-58`). Choosing cpp11 buys less PROTECT boilerplate but costs a new dependency, the combined-table trap (§4.7), and a documented maintainer footgun. This is what xgboost does. An earlier draft listed this only as a fallback, then as a peer option to be settled by a spike; it is the decision. The spike incidentally confirmed it works — the fork target built with plain `extern "C"` entry points. |
| **Parity + idiomatic R layer + ecosystem integration** (hardhat, parsnip, mlr3, DALEX, vetiver) | Rejected on explicit user instruction: "First class support in R means that all functions / capabilities available for the cli and python version are also available in R. Beyond that is scope creep." |
| **CRAN-first** (solve vendoring and size before any feature work) | Partially adopted. Phase 1 and 1a now front-load exactly the packaging and size risk, without blocking feature work behind the whole CRAN process. |
| **Full monorepo fork** | Inherits a multi-GB repo and permanent upstream merge conflicts for no gain — the R package needs the core's source, not its history. |
| **Wrap the CatBoost CLI** | CRAN forbids bundling standalone executables at this size; no precedent for a full ML engine being CLI-shelled from R; loses in-memory pools. |
| **`reticulate` over the Python package** | **The strongest rejected alternative — rejected by user decision, not on technical grounds.** An earlier version of this spec dismissed it as "does not solve CRAN availability". That was false: `keras3` and `tensorflow` are on CRAN, wrap Python via reticulate as their primary mechanism, and satisfy the no-network-during-install rule by deferring Python setup to a post-install user-invoked step. Honestly assessed, reticulate delivers every capability in §1 far sooner, tracks upstream releases automatically, avoids the vendoring engineering entirely, and makes custom R loss/metric *easier* rather than "may prove infeasible". Its one real cost is a Python runtime dependency. The user was shown this comparison explicitly and chose the native fork; that independence from Python is the deciding requirement. |
| **Rcpp instead of cpp11** | Heavier compile and header cost across many new translation units. |
| **extendr (Rust)** | Would add a third toolchain atop CMake, C++, and Python for no net gain against a C++ core. |

## 7. Risks

Retired risks are removed rather than left as answered questions; a register where half the
rows are settled stops being read.

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Vendored source tarball ~1.7× the CRAN policy figure** | MEASURED at 16.078 MiB pruned (§4.1b) against policy's 10MB. Not fatal and not unprecedented (`rcdklibs` 19M, `fastrmodels` 16M are live). Requires a written justification at submission. | Levers, in order: gate the 87 unconditionally-configured test/tool/benchmark directories (~19 MiB of plumbing); then amalgamate and strip unused source on the `xgboost`/`lightgbm` model, which is how comparable packages reach 1.5-1.7M. §6's thin-package model is REFUTED and is no longer a fallback — if the vendored route fails, there is no cheaper packaging alternative, only a smaller tarball. |
| **CRAN check-time limits on build machines** | Rejection. The 164 s / 174 s figures are this machine's, on 32 cores. CRAN and r-universe runners have far fewer. | Measure build time on a constrained runner during Phase 1; treat compile time as a CRAN acceptance criterion, not an afterthought. |
| **No integrity pinning on the C++ dependency chain** | A tampered Conan recipe or vendored source is arbitrary code execution on the build host and a silent backdoor in what users install. `swig`, `ragel`, `bison`, `flex`, `m4` are code generators that execute during the build and emit compiled source. The Python side has 192 sha256 pins; the C++ side has none. | Commit a `conan.lock` for as long as Conan is used anywhere; give every vendored dependency the pin-and-verify contract already used by `tools/vendor/acquire.sh` and `tools/oracle/cli/acquire.sh`; record every external artifact in a `SOURCES.md` inventory. |
| **Vendoring openssl creates a standing CVE liability** | A frozen openssl in a CRAN package ages badly, and this is now a CERTAINTY rather than a risk: the user accepted openssl into the R1 link on 2026-07-31 after §4.1b's gating route was refuted. | Configuring it out is NOT available (see §4.1b — the coupling is source-level). Mitigation is therefore: pin the exact version with a SHA256 and a fail-closed check, record the CVE-tracking obligation explicitly in `SOURCES.md`, and re-check it at every upstream version bump. Note this is a RECORD, not a monitoring process — no recurring mechanism is specified, and that gap is deliberate and acknowledged. |
| **The §4.7 registration trap** | Latent, not live: with raw `.Call` throughout there is one registration source and the trap cannot fire. It returns the moment anyone adds a second one (cpp11, Rcpp, a generated table), and its failure mode is silent — compiles, installs, loads, fails only at call time. | The §3 decision removes it. If a second source is ever introduced, §4.7's enforced testthat check is a precondition of that change, not a follow-up. |
| **Custom R loss/metric callback infeasible** | Phase 6 does not ship. | Isolated as its own phase. R's single-threaded evaluator constrains any callback design; whether a `thread_count=1`-only callback counts as parity is an open question that must be answered before Phase 6 is planned, not during it. |
| **No CUDA hardware exists anywhere in this project** | GPU parity unverifiable. Dev machine is AMD; CatBoost has no ROCm backend; CRAN and r-universe runners have no CUDA. | GPU parity claimed only from a real-hardware run; skipped tests never count as green. Resourcing decision deferred to Phase 7 by user decision — see §9.1. Does not gate the R1 release. |
| **Upstream bumps break the fork** | Ongoing maintenance cost. | Submodule moves are deliberate and gated on a full differential suite run plus the vendored upstream test suite, with the inventory regenerated so new upstream capabilities appear as new red rows. |

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

**Carve-out, so §3's strict-superset promise and this section do not collide:**
`catboost.caret` is an existing upstream export with no CLI or Python counterpart, and
upstream ships a test for it. It is **retained as pre-existing legacy** under the same
compatibility logic §4.2 applies to the existing S3 methods. Retaining what already exists is
not the same as adding an ecosystem adapter, which remains out of scope.

## 9. Open Decisions

### 9.0 When this project stops

Every phase has a pass condition; none had a stop condition, which against a 725-row surface
and a multi-quarter horizon is how projects run indefinitely. Explicit abandonment and
descope triggers:

| Trigger | Response |
| :--- | :--- |
| Phase 1a pruned tarball exceeds 30 MB | Escalate to the user. NOTE: §6's thin-package contingency is REFUTED, so this trigger no longer has a fallback route attached — the only remaining responses are a smaller tarball or abandoning CRAN. |
| Two consecutive CRAN rejections on grounds that are structural rather than fixable (size, build time, bundled sources) | Stop pursuing CRAN. Ship r-universe only, and record that the stated goal was not reachable on this architecture. |
| Phase 6 (custom R loss/metric) proves infeasible after one honest attempt | Record the infeasibility with evidence, mark those matrix rows permanently out-of-scope, continue. It does not block anything else. |
| GPU hardware still unresourced when Phases 1-6 are done | Take the §9.1 decision then, on the terms below. GPU never blocks R1. |
| Upstream ships its own feature-complete R package or accepts these changes | Stop. The fork's purpose is gone; contribute upstream instead. |

The R1 release exists partly so that abandonment after it still leaves the user with a
working, installed, useful package rather than nothing.


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

**Decision (2026-07-30): deferred to Phase 7 by explicit user choice, with a time-box.** If the decision is still open when Phases 1-6 are complete, it defaults to option 2 (ship GPU implemented-but-unverified, every GPU matrix row labelled unverified in the documentation) rather than stalling the project — unless the user chooses otherwise at that point. Phases 0-6 are
unaffected, so the decision is taken when it becomes actionable rather than now. Until then
Phase 7 stays planned but unstarted, and no GPU claim is made anywhere.

Consequence to close when Phase 7 starts: the epic's GPU success criterion is written
unconditionally ("verified by differential tests run on real CUDA hardware"). If option 2 or
3 is chosen, that criterion must be amended in the same change, or the epic becomes
internally contradictory.

## 10. Spike results and remaining work

**§10.1-10.3 are ANSWERED by the 2026-07-31 spike — see §4.1b for the measurements.** In
summary: the fork-owned CMake target is viable and registrars survive (Q1); the shared object
name and init symbol need no non-standard override (Q3); the dependency surface is 4 invoked
packages with openssl reachable only through the distributed subtree (Q4, Q5); Python3 is
required at install time, contradicting an earlier claim in this spec (Q6); the pruned tarball
is 16.078 MiB, conditionally viable (Q7). `-DCUSTOM_ALLOCATORS=Off` works (Q2).

**Still unanswered and carried into Phase 1:** build time on a 2-core CRAN-like runner (Q8),
and whether upstream's vendored R test suite passes against a from-source build (Q9) — it
currently passes only against a downloaded prebuilt library, so the §4.6 regression gate has
an untested baseline.

The original question text is retained below for provenance.

## 10-original. Open questions as posed before the spike

Design review round 2 returned NEEDS_REVISION from all five reviewers. The blockers below are
**empirical** — no amount of spec revision answers them, and two successive build models were
written into this document with confidence and then demolished. The spike answers them by
measurement; the spec is revised once afterwards, against facts.

### 10.1 Build model (blocking everything else)

1. Does a fork-owned CMake target linking upstream's targets by name actually produce a
   working `libcatboostr.so`? Specifically, do the 28 `.global` archives created by
   `add_global_library_for` (`vendor/catboost/cmake/common.cmake:149-166`) link with their
   static registrars intact, or are they silently dropped — producing a package that builds,
   installs, loads, and fails at runtime with missing model formats or metrics?
2. Does `-DCUSTOM_ALLOCATORS=Off` (`cmake/common.cmake:333`) swap mimalloc for the system
   allocator cleanly? If so it removes the `_mi_heap_default` TPOFF32 relocation failure and
   the risk of an R shared object interposing `malloc`/`free` process-wide. Measure the
   performance cost once.
3. What is the shared object named — `libcatboostr` (preserving upstream's
   `useDynLib(libcatboostr)` and every existing `.Call` site) or `catboostr`? This determines
   the init symbol and whether any double-registration hazard exists at all.

### 10.2 Dependency surface (determines Phase 1's real size)

4. Which of the 13 Conan packages are actually *invoked* in a `CATBOOST_COMPONENTS=R-package`
   build, rather than merely resolved? Phase 0 always had all 13 present, so this was never
   isolated. `ragel` and `yasm` are confirmed invoked via `util`
   (`util/CMakeLists.linux-x86_64.txt:589,595`); the rest are unverified.
5. **Does openssl configure out of the R-package component?** If it does, vendoring collapses
   to zlib alone — and R itself already links zlib and exposes it via `R CMD config`, so the
   vendoring scope could reach **zero**. This is the single largest simplification available
   and it is one measurement.
6. `cmake/common.cmake:3` is an unconditional `find_package(Python3 REQUIRED)`. So removing
   Conan does **not** remove Python from the install path — an earlier claim in §4.1 was wrong
   on this. What is the actual minimum install-time toolchain, and what belongs in
   `SystemRequirements`?

### 10.3 CRAN viability (can fail the whole route)

7. What is the pruned vendored source tarball size? Unpruned C/C++ source is ~163 MB against
   CRAN's guidance (recorded here as 5 MB; the policy text actually says 10MB — see §4.1b). A configure-only CMake run yields a compile database naming every
   translation unit the target compiles — that is the prune set, and it needs none of Phase
   1's other work. **This measurement must run first**, so an abort costs a day rather than a
   phase.
8. What is the build time on a 2-core machine resembling a CRAN check runner? 164 s on 32
   cores is roughly 1.5 CPU-hours; build time is the second structural CRAN rejection cause
   and currently has no abort threshold.
9. Does upstream's vendored R test suite — the §4.6 strict-superset regression gate — actually
   pass against a from-source build? It currently passes against a *downloaded prebuilt*
   library. A green baseline established before Phase 1 is worth more than the same suite
   discovered red mid-phase.

### 10.4 Recorded review findings not yet folded into the design

These are accepted as valid and are deferred to the single post-spike revision, not dropped:

- **Integrity of the shipped artifact.** Nothing verifies `tarball == prune(pinned_tree)`;
  §4.8 pins everything going in and nothing coming out. Needs a committed prune script, a
  file-level SHA256 manifest, and CI reproduction from a fresh acquire.
- **GPU channel signing.** A `SHA256SUMS` file co-located with the asset shares its write
  credential and does not defend against whoever can write the release. Needs a detached
  signature or build attestation with the trust root shipped through the CRAN package.
- **`CATBOOST_DYNLIB` in the shipped `configure`** is an env-var-selected shared object loaded
  into R at install time — same class as the deleted download, minus the network. Decide
  whether the distributed `configure` refuses it outright.
- ~~**`uv run` without `--frozen`**~~ — **WITHDRAWN 2026-07-31: the finding was false.** All
  three cited sites already carry `--frozen` (`tools/parity/run.sh:11`,
  `tools/oracle/gen_smoke_fixture.py:5`, `tools/oracle/cli/gen_smoke_fixture.sh:28`), as do
  `tools/parity/introspect_python.py:3` and `tools/oracle/README.md:20`. Verified by grep
  across `tools/`. The requirement in §4.8.6 stands as a rule for new invocations; there is
  no outstanding work. Recorded rather than deleted because it was propagated into a Phase 1
  ticket (`catboost-8z4.18`, closed invalid) before anyone checked it.
- **`.Rbuildignore` as a denylist** already misses `.agents/` and `tests/fixtures/` (652K).
  Replace with a built-tarball manifest assertion.
- **Uncovered inventory kinds.** §4.2's mapping table has no rule for `property` (40 rows:
  `best_iteration_`, `tree_count_`, `evals_result_`, `classes_`, `feature_names_`, …),
  `attribute` (11 rows), `class` (38) or `submode` (4). None appears in any phase. §4.5 says
  a capability may not be silently absent; today this slice is.
- **Generator kind-tagging bug.** `catboost.utils.compute_wx_test`,
  `compute_training_options` and `fspath` are real functions tagged `attribute`. Mapping by
  `kind` plus a tagging bug drops capabilities by construction.
- **Hyperparameter documentation is not machine-derivable.** `python_surface.json` carries
  only name and default; `introspect_python.py` captures no docstrings. A "generated
  `@param` reference" cannot explain what `l2_leaf_reg` does. Either capture docstrings or
  state honestly where the prose comes from.
- **No version-skew detection exists.** §4.4 demands a tested error for R-package/
  `libcatboostr` skew, but no `.onLoad` check or version handshake is designed anywhere.
- **Tolerance default vs its own evidence.** The 1e-12 Python default cites Phase 0
  bit-exactness that `docs/phase-0/fix-wave-report.md:56-70` explicitly disclaims as holding
  only for a 40-row fixture.
- **Parity matrix should extend `capability_diff.json`**, not become a second store, and needs
  its own enforced check: a green row whose test id does not resolve is the same silent
  failure §4.7 guards against.
- **Licence provenance.** Vendoring dozens of third-party sources requires
  `inst/COPYRIGHTS`/`LICENSE.note` and `Authors@R` holders — a routine CRAN rejection cause,
  currently absent. `SOURCES.md` is a security SBOM and does not discharge it.
- **30 MB abort threshold is asserted, not derived** (and was compared against a guidance figure that was itself wrong — see §4.1b), sitting above the CRAN guidance it
  cites, so the gate can pass while the goal still fails.
- **§9.0 lacks its most likely stop condition**: Phase 2's classification coming back
  multi-quarter and the user declining to fund it.
- **R1 wording overstates** — §1's goal is CRAN acceptance *and* parity; R1 delivers the CRAN
  half with red rows documented.
