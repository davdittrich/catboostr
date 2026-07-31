# catboost-8z4.10 (P1.2) — Gate the unconditionally-configured test/tool/benchmark directories

Date: 2026-07-31
Ticket: catboost-8z4.10 (hermetic — read in full via `bd show catboost-8z4.10 --json`)
Repo: `/home/dd/Gemini/catboost`, branch `phase-0`
Result: SUCCESS. All Section VI DoD items met.

## 1. Baseline (Step 1)

Disposable copy of `vendor/catboost` (`.git` stripped), `src/CMakeLists.txt`
(P1.3's fork target) injected via `add_subdirectory()` appended to the copy's
root `CMakeLists.txt` — same injection method P1.3 proved. Configured with
the ticket's proven invocation (`CATBOOST_COMPONENTS=R-package-fork` to avoid
the target-name collision documented in P1.3):

```
$ cmake -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=$VENV/bin/ninja \
  -DCMAKE_TOOLCHAIN_FILE=$SRC/build/toolchains/clang.toolchain \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=$SRC/cmake/conan_provider.cmake \
  -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCATBOOST_COMPONENTS=R-package-fork \
  -DHAVE_CUDA=no -DPython3_EXECUTABLE=$VENV/bin/python -DCYTHON_EXE=$VENV/bin/cython
configure_rc=0
```

`ninja -n catboostr` dry-run: 4265 total build-graph nodes (matches the P1.6
spike exactly). Real `ninja catboostr -j32`: `build_rc=0`,
`libcatboostr.so` (34,496,296 bytes).

**Directories enumerated from the configure output (Step 1's instruction:
"do not trust the count as a list").** Traced every `add_subdirectory()` call
with `cmake --trace-expand --trace-redirect=trace.log`, resolved caller-dir +
argument to 392 distinct directories, cross-referenced against the real
`ninja -t deps` dependency database from the completed build above. From
those 392, classified by directory-name/path pattern
(`ut`, `benchmark`, `tests?`, `for_testing`, `testing`, or under
`catboost/tools/`), then verified each candidate against the real deps db —
see Step 2 for the one correction this verification produced.

The P1.6 spike's own `missing_dirs.txt` (87 directories, still present on
disk under the spike's scratch tree) independently confirms this set: it is
the union of "not in the compile closure, and configure fails without it."
Of those 87, this ticket's scope (test/tool/benchmark only, per the ticket
title) covers 44 (the other 43 are unused-but-real product-feature
directories — e.g. `catboost/libs/monoforest`, `catboost/private/libs/
hyperparameter_tuning`, `contrib/libs/{pugixml,re2}` — genuinely dead code
from `catboostr`'s perspective but not test/tool/benchmark by category, and
therefore out of scope here).

## 2. Gating mechanism (Step 2)

Grepped `vendor/catboost/cmake/` for an existing test/tool switch first, per
the ticket's instruction — none exists (`BUILD_TESTING`, `SKIP_UT`, no `ut()`/
`add_ut()` macro defined anywhere in `cmake/`). `-DCATBOOST_COMPONENTS`
itself is not it either: it is consulted only inside each top-level product
front-end's own `CMakeLists.txt` (`app`, `R-package`, `python-package`,
`jvm-packages`, `spark` — each with its own single-child `if(...IN_LIST
CATBOOST_COMPONENTS...)` guard). The parent
`catboost/CMakeLists.linux-x86_64.txt` unconditionally
`add_subdirectory()`s `libs`, `private`, `tools`, etc., and those in turn
unconditionally `add_subdirectory()` their own `ut`/`benchmark` children —
confirmed by reading `catboost/R-package/CMakeLists.txt`,
`catboost/app/CMakeLists.txt`, `catboost/tools/CMakeLists.txt` directly.

New fork-owned file: **`cmake/gate-test-tool-dirs.cmake`** (not under
`vendor/`). It redefines `add_subdirectory()` as a `function()` — CMake's
documented one-level override mechanism makes the previous (builtin)
definition callable via `_add_subdirectory()` — against a fixed list of 44
directory paths. Injected via
`-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="<gate-file>;<conan_provider.cmake>"`
(a `;`-joined list CMake includes right after `project()`, i.e. before any
`add_subdirectory()` in the CATBOOST project has run — unlike the
`src/CMakeLists.txt` injection, which P1.3 found had to happen later since
it links against targets `add_subdirectory(catboost)` defines).

**One correction the real-deps-db check produced before finalizing the
list:** `library/cpp/testing` cannot be gated wholesale.
`library/cpp/testing/common` (`env.cpp`, `network.cpp`, `probe.cpp`,
`scope.cpp` → `libcpp-testing-common.a`) is a genuine link dependency,
referenced 9 times in the real `ninja -t deps` database — this is exactly
the `GetArcadiaSourcePath()`/`vcs_info()` dependency `src/CMakeLists.txt`'s
own comment already flags. Only `library/cpp/testing`'s `benchmark`,
`gbenchmark`, `hook`, `unittest`, and `unittest_main` siblings are gated;
`common` is not touched.

Final 44-entry gate list (`cmake/gate-test-tool-dirs.cmake`,
`_CATBOOSTR_GATED_TEST_TOOL_DIRS`):

```
catboost/tools
contrib/restricted/google/benchmark
library/cpp/testing/{benchmark,gbenchmark,hook,unittest,unittest_main}
catboost/libs/{calc_metrics,carry_model,dataset_statistics,data,helpers,helpers/parallel_sort,metrics,model,train_lib}/ut
catboost/private/libs/{algo,algo_helpers,data_util,embedding_features,feature_estimator,functools,options,quantization,quantization_schema,quantized_pool,text_features,text_processing}/ut
util/{charset,datetime,digest,draft,folder,generic,memory,network,random,stream,string,system,thread}/ut  (+ util/datetime/benchmark, util/digest/benchmark, util/ut)
```

**Verification before applying**, not after: wrote
`check_deps.py`/`gate_list.txt`, cross-referenced all 44 entries against the
1,756,151-line real `ninja -t deps` database from the completed baseline
build in Step 1. Result: **0/44 gate entries referenced** — none is a
compile or link input for `catboostr`. (`library/cpp/testing/common` was
excluded from the list precisely because this check flagged it — see
above.)

## 3. Re-configure and full build (Step 3)

**3a. Unpruned tree, gate applied (proves the gate itself doesn't break
configure/build):**

```
$ cmake ... -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$GATE;$SRC/cmake/conan_provider.cmake" ...
configure_rc=0
-- catboostr-fork P1.2: gating out test/tool/benchmark subdir: catboost/private/libs/functools/ut
-- catboostr-fork P1.2: gating out test/tool/benchmark subdir: catboost/tools
[... 44 "gating out" lines total, grep -c = 44]
$ ninja -n catboostr   # dry run
4265/4265 steps resolved (identical to the ungated baseline)
$ ninja catboostr -j32
build_rc=0
libcatboostr.so: 34,496,296 bytes
```

**3b. Pruned tree (the deliverable): physically removed all 44 gated
directories' content** (not just skipped at configure time) from a copy of
the P1.6 spike's own saved 94.901 MiB pruned tree (still on disk, verified
byte-identical to the spike's reported baseline before touching it:
`du -sb` → 99,511,153 bytes / 13,371 files, exact match) —
`removed_dirs=44 removed_files=950 removed_bytes=3,643,602 (3.475 MiB)`.
Re-configured and rebuilt from scratch on this smaller tree with the same
gate file:

```
$ cmake -S $PRUNED -B $BUILD ... -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$GATE;$PRUNED/cmake/conan_provider.cmake" ...
configure_rc=0   (44/44 "gating out" lines again — the physically-absent
                  directories are never even attempted)
$ ninja -n catboostr
4265/4265 steps resolved
$ ninja catboostr -j32
build_rc=0
libcatboostr.so: 34,496,312 bytes
```

A **full real build to completion**, not a configure-only or dry-run-only
pass, per the ticket's explicit prior-failure warning (the P1.6 Ragel
include-chain gap was invisible to both `compile_commands.json` and
`ninja -t deps`, and only surfaced on a real build attempt). None of the 44
gated directories touch Ragel, protobuf `.proto` imports, or
`configure_file()` templates — those categories are unrelated, already-known
gaps from the P1.6 spike, and are untouched by this ticket's prune set.

## 4. `catboostr` links, loads in R, trains, and predicts (Step 4)

Through the **pruned + gated** tree's `libcatboostr.so`
(`build3/catboostr-fork-build/libcatboostr.so`, 34,496,312 bytes), via the
real R wrapper functions (`R/catboost.R`, not hand-rolled `.Call` argument
lists), same method P1.3 used:

```r
pool  <- catboost.load_pool(train_data, label = train_label)
model <- catboost.train(pool, params = list(iterations = 2, learning_rate = 0.5,
                                             loss_function = "Logloss", thread_count = 1))
pred  <- catboost.predict(model, pool)
```

Output:

```
0:	learn: 0.6554074	total: 45.7ms	remaining: 45.7ms
1:	learn: 0.6204904	total: 45.7ms	remaining: 0us
train_predict_prediction: -0.150897441329512,0.150897441329512
train_predict_tree_count: 2
train_predict_ok: TRUE
```

Real 2-iteration boosting through `catboost.train` → `CatBoostFit_R`,
prediction through `catboost.predict` → `CatBoostPredictMulti_R`, both
resolved via `dyn.load()` against the pruned+gated `.so` — not stubs.

## 5. Re-measurement and delta against the spike baseline (Step 5)

| Quantity | Baseline (P1.6 spike) | This ticket (pruned + gated) | Delta |
|---|---|---|---|
| Uncompressed tree size | 99,511,153 B (94.901 MiB) | 95,867,547 B (91.42 MiB) | −3,643,606 B (−3.475 MiB, −3.66%) |
| File count | 13,371 | 12,421 | −950 |
| `catboostr` ninja build-graph nodes | 4,265 | 4,265 | 0 (gated dirs were never TU inputs — confirmed §2) |
| Unique compiled TUs | 3,902 | 3,902 | 0 |
| Compressed tarball (`tar czf`) | 16,858,704 B (16.078 MiB) | 16,075,929 B (15.331 MiB / 16.076 MB decimal) | −782,775 B (−0.747 MiB, −4.64%) |

**Against CRAN's policy figure of 10 MB (retracted 5 MiB figure per spec
4.1b superseded):** the pruned+gated tarball is 16.076 MB (decimal) /
15.331 MiB — still **1.53×–1.61× over** the 10 MB guidance, down from
1.61×–1.69× over at the spike baseline. This ticket's gate removes real
bytes (3.475 MiB uncompressed, 0.747 MiB compressed) but the TU/compile
closure (47.70 MiB compiled sources + 30.25 MiB header closure, per the
P1.6 spike, untouched by this ticket) remains the dominant cost — the
43 non-test/tool/benchmark directories the spike also found unnecessary
(`monoforest`, `hyperparameter_tuning`, `pugixml`, `re2`, etc.) are the next
largest lever and are explicitly out of scope for this ticket's title.

## 6. Guards re-checked (Step 6 / DoD)

- `git -C vendor/catboost status --porcelain` — empty, checked before this
  session's first command and again just before writing this report.
- No writing git command run (add/commit/push/checkout -b/stash) at any
  point.
- Nothing installed system-wide: reused the already-warm venv
  (`ninja`, `conan` 2.31.1, `cython`, `python` 3.12.13) and `~/.conan2`
  package cache from a prior session task; no new installs.
- All builds ran against disposable copies under scratch
  (`$SCRATCH/p1.2/{src1,src2,pruned2}`), never in `vendor/catboost/` itself;
  `CMakeUserPresets.json` only ever appeared in those disposable copies and
  was stripped from the measured pruned tree before tarring.
- Touched only: `cmake/gate-test-tool-dirs.cmake` (new) and this report —
  no `vendor/` edits, no edits to `src/CMakeLists.txt` (P1.3's file, read
  but not modified).

## Section V — machine-readable result

```toon
task_id: P1.2
success: true
data:
  dirs_gated: 44
  tu_count_before: 3902
  tu_count_after: 3902
  tarball_bytes_after: 16075929
  full_build_succeeded: true
report_path: docs/phase-1/catboost-8z4.10-report.md
vendor_clean: true
error_log: null
```

## Concerns for the caller

1. **44 gated, not 87.** The ticket's reference data cites the P1.6 spike's
   87-directory count; this ticket's title scopes it to test/tool/benchmark
   directories specifically, which this investigation independently
   measured at 44 of those 87 (the remaining 43 are unused product-feature
   directories, not test/tool/benchmark — flagged for a follow-up ticket,
   not silently included here or silently dropped from the record).
2. Still **over CRAN's 10 MB tarball guidance** (16.08 MB decimal /
   15.33 MiB) after this gate. The larger levers (compiled-source and
   header-closure bytes, and the 43 unused-but-real directories noted
   above) are unchanged by this ticket by design (out of scope per its
   title) and would need a separate ticket.
3. The gate list was verified against the real `ninja -t deps` database
   (post-build, genuine `-MM`-style scan) for C/C++ header/link
   dependencies, per the ticket's own standard. It was **not** re-verified
   against Ragel/protobuf/`configure_file()` invisible-dependency classes
   the P1.6 spike found — none of the 44 gated paths intersects those
   categories (verified by directory name/path, not by a second real build
   probing that specific failure mode), and the real full builds in both
   §3a and §3b completed without hitting any such gap, which is the
   strongest available evidence short of exhaustively re-deriving the P1.6
   spike's reactive-failure process for this specific prune set.

Confidence: 92. Every count above is from an executed command's output
quoted in this report (or in the still-on-disk scratch logs referenced by
path); the one lower-confidence item is concern 3 (absence of a
Ragel/protobuf-specific regression probe beyond the two full builds that
already completed clean).
