# catboost-8z4.8 — Phase 1 spike: fork-owned CMake target proof

**Status:** DONE. Result: the fork-owned-target design is viable. Both
registrar tests pass — no static registrars were dropped.

## 1. What was built

A disposable copy of `vendor/catboost` at
`/tmp/claude-1000/-home-dd-Gemini-catboost/7cc8a254-e8a7-4829-9fda-c9fbdf6e1bf6/scratchpad/catboost-8z4.8/copy`
(940 MiB, `cp -r`, `.git` stripped). Inside that copy only:

- `fork-src/CMakeLists.txt` — new file. Defines `add_shared_library(catboostr_fork)`,
  linking the **exact same target list** upstream's `catboostr` links
  (`vendor/catboost/catboost/R-package/src/CMakeLists.linux-x86_64.txt:16-38`):
  `contrib-libs-linux-headers`, `contrib-libs-cxxsupp`, `yutil`, `build-cow-on`,
  `catboost-libs-cat_feature`, `catboost-libs-data`, `catboost-libs-eval_result`,
  `catboost-libs-fstr`, `libs-gpu_config-maybe_have_cuda`, `catboost-libs-logging`,
  `catboost-libs-model`, `catboost-libs-train_lib`, `private-libs-algo`,
  `private-libs-data_util`, `private-libs-documents_importance`,
  `private-libs-init`, `private-libs-options`, plus `target_allocator(... cpp-malloc-mimalloc)`
  and `vcs_info(catboostr_fork)` (needed once discovered missing — see §4).
- `fork-src/fork_entry.cpp` — new file, one fork-authored C++ TU, raw `.Call`
  glue only (no cpp11 anywhere).
- `CMakeLists.txt` (the copy's **root** file) — one line appended at the very
  end, after the existing platform-`include()` chain:
  `add_subdirectory(fork-src)`.

**Injection method used:** `add_subdirectory` into the disposable copy
(the ticket's second-preference option), not
`-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES`. Reason: `TOP_LEVEL_INCLUDES` runs
*before* `project()`/before any upstream target exists, so
`target_link_libraries(catboostr_fork PUBLIC catboost-libs-model ...)` would
silently degrade to bare linker-flag strings (CMake only resolves a bare name
as a *target* — carrying its `INTERFACE` whole-archive properties — if that
target is already defined at the point of the call). Appending
`add_subdirectory(fork-src)` after `add_subdirectory(catboost)` (already the
last line of the platform-specific include) guarantees every upstream target,
including all 24 `.global` archives, exists first. This is a deliberate,
reproducible choice, not an accident — and it still satisfies the ticket's
"never edit a file inside the pinned tree" guard because the edited
`CMakeLists.txt` lives only in the disposable copy.

`git -C vendor/catboost status --porcelain` → empty (verified after the full
run). Nothing in the pinned tree was touched.

## 2. Configure + build invocation

Exact Phase-0 invocation, unmodified:

```
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<copy>/build/toolchains/clang.toolchain \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=<copy>/cmake/conan_provider.cmake \
  -DCMAKE_POSITION_INDEPENDENT_CODE=On \
  -DCATBOOST_COMPONENTS=R-package \
  -DHAVE_CUDA=no \
  -DPython3_EXECUTABLE=<venv>/bin/python \
  -DCYTHON_EXE=<venv>/bin/cython \
  <copy>
ninja catboostr_fork
```

Configure succeeded on the first try (conan offline cache warm, per Phase 0).
`ninja catboostr_fork` needed two fix/rebuild cycles (§4) before it linked
cleanly.

## 3. Result: the fork-authored entry point (DoD item 1)

`fork_entry.cpp` exports `Fork_TrainPredict_R()`: builds an in-memory 4-row/
1-feature pool, trains a real 2-iteration model via upstream's
`TrainModel()` (the actual boosting core, not a stub), then calls
`model.CalcFlat()` on the training input. Loaded from R via raw `dyn.load()` +
`.Call()` (no NAMESPACE, no `useDynLib`, no cpp11):

```r
dyn.load("<build>/fork-src/libcatboostr_fork.so")
.Call("Fork_TrainPredict_R")
```

Console output (verbatim, deterministic given the fixed seed/data/params):

```
Learning rate set to 0.5
0:	learn: 0.4000000	test: 0.4000000	best: 0.4000000 (0)	total: 45.7ms	remaining: 45.7ms
1:	learn: 0.3200000	test: 0.3200000	best: 0.3200000 (1)	total: 45.7ms	remaining: 0us

bestTest = 0.32
bestIteration = 1
```

Returned R value:

```r
List of 3
 $ prediction                    : num 0.32
 $ tree_count                    : int 2
 $ cpu_trainer_factory_registered: logi TRUE
```

`prediction == 0.32`, `tree_count == 2` — matches the expected outcome of a
2-iteration fit at `bestTest = 0.32`. Not a constant: this exercises the real
data-provider builder, `TrainModel()`, boosting, and the CPU model evaluator.

## 4. The registrar test (DoD item 2 — the point of the ticket)

**What was exercised:** `NCB::TModelLoaderFactory` (declared
`vendor/catboost/catboost/libs/model/model_import_interface.h:32`), whose
Json registrar (`TJsonModelLoaderRegistrator`,
`vendor/catboost/catboost/libs/model/model_export/model_import.cpp:31`) lives
inside the **`libs-model-model_export.global`** static archive
(`add_global_library_for(libs-model-model_export.global, libs-model-model_export)`,
`vendor/catboost/catboost/libs/model/model_export/CMakeLists.linux-x86_64.txt:55`).
This is a *different* `.global` archive than the one used by ordinary
`Calc()`/training (`catboost-libs-model.global`,
`catboost-libs-train_lib.global`) — deliberately chosen because it is the
kind of registrar that can be silently dropped independently of the "obvious"
path, exactly the ticket's stated hazard ("everything looks fine until it
isn't").

`Fork_JsonRegistrarTest_R(path)`:
1. Trains the same tiny model.
2. `ExportModel(model, path, EModelType::Json)` — a plain function call
   (not registrar-dependent) writes the file.
3. Checks `NCB::TModelLoaderFactory::Has(EModelType::Json)` **before**
   touching the loader pointer — this avoids ever dereferencing a null
   `THolder` if the registrar had been dropped (upstream's own
   `ReadModel()` in `model.cpp:76-85` does not guard this and would
   segfault on a dropped registrar; the guarded check was a deliberate
   safety choice for this spike so the result is legible instead of a
   crash).
4. If registered: constructs the loader, reads the file back, re-predicts,
   and compares tree count + prediction against the original.

Result (verbatim R output):

```r
List of 5
 $ json_loader_registered: logi TRUE
 $ read_succeeded        : logi TRUE
 $ prediction_matched    : logi TRUE
 $ original_prediction   : num 0.32
 $ error_message         : chr ""
```

`file.exists(path)` → `TRUE`, size 13941 bytes. **The registrar survived.**
Json round-trip reproduced the exact same prediction (`0.32`) and tree count.

## 5. Link-line inspection (not assumed)

Captured from `ninja -v catboostr_fork`'s actual clang++ invocation (the full
command line the linker ran, not `link.txt` — Ninja doesn't emit one).
24 distinct `.global.a` archives are pulled into the final link; **23 of
them** are wrapped in `-Wl,--whole-archive ... -Wl,--no-whole-archive`
(`catboost-libs-train_lib.global.a` is linked via its `.wholearchive`
INTERFACE target so it appears once whole-archived and additionally through
a transitive plain reference later in the list — CMake dedupes the actual
archive member inclusion; the whole-archive wrapping is what determines
whether static initializers survive, and that wrapping is present for the
entire set observed). None of the 23 wrapped archives were linked as plain
(non-whole-archive) static libraries. Representative excerpt:

```
-Wl,--whole-archive .../catboost/libs/model/libcatboost-libs-model.global.a -Wl,--no-whole-archive
-Wl,--whole-archive .../catboost/libs/model/model_export/liblibs-model-model_export.global.a -Wl,--no-whole-archive
-Wl,--whole-archive .../catboost/libs/train_lib/libcatboost-libs-train_lib.global.a -Wl,--no-whole-archive
-Wl,--whole-archive .../catboost/private/libs/init/libprivate-libs-init.global.a -Wl,--no-whole-archive
```

This is the direct, causal explanation for why both registrar tests passed:
the `.global` → `.wholearchive` INTERFACE property
(`add_global_library_for`, `vendor/catboost/cmake/common.cmake:157-165`)
propagated transitively from `libs-model-model_export` /
`catboost-libs-train_lib` (linked PUBLIC by `catboost-libs-model` and
`catboost-libs-train_lib` respectively, which our fork target links
directly) all the way to our target, exactly as the decided design
predicted — because our target is a real node in the *same* CMake
configure/target graph as upstream, not a separately-configured build
linking upstream's archives by path.

## 6. Failures encountered and fixed (both were spike bugs, not design bugs)

1. **Header-order compile error.** `Rinternals.h` included before catboost/
   libcxx headers poisoned `contrib/libs/cxxsupp/libcxx/include/__locale`
   (`length` macro collision, and later `SIZEOF_SIZE_T`). Fixed by including
   catboost headers first, `#undef SIZEOF_SIZE_T`, then `Rinternals.h` last —
   this exactly mirrors the ordering already used by upstream's own
   `vendor/catboost/catboost/R-package/src/catboostr.cpp`.
2. **Missing `vcs_info()` call.** First successful compile failed to *link*
   with:
   ```
   ld.lld: error: undefined hidden symbol: GetArcadiaSourcePath
   ld.lld: error: undefined hidden symbol: GetProgramSvnVersion
   ```
   (referenced by `model_export`'s `pmml_helpers.cpp`/`onnx_helpers.cpp` and
   `testing/common/env.cpp`, both pulled in transitively). Upstream's own
   `catboostr` CMakeLists calls `vcs_info(catboostr)` at the end for exactly
   this reason; I had omitted it. Added `vcs_info(catboostr_fork)` — fixed.
3. **JSON param type.** `params.InsertValue("verbose", false)` threw
   `TCatBoostException: Can't parse parameter "verbose" with value: false`
   inside `TrainModel()` on first R run (not a build-time issue — a runtime
   options-parsing quirk). Dropped the unnecessary `verbose` key.

None of these three were related to the whole-archive/registrar hazard the
ticket exists to test; they were ordinary "new C++ file in this codebase"
friction, now resolved.

## 7. Naming / init symbol (DoD item, ticket step 7)

Our spike's shared object: **`libcatboostr_fork.so`** (from CMake target
name `catboostr_fork`; `add_shared_library` applies CMake's default `lib`
prefix). We used `_fork` to keep it unambiguous and non-colliding in a
concurrent-agent checkout; no code required renaming.

Upstream's own convention (confirmed from
`vendor/catboost/catboost/R-package/src/init.c:40` and
`.../NAMESPACE: useDynLib(libcatboostr, .registration = TRUE)`): the CMake
target is literally named `catboostr` (not `libcatboostr`), which
CMake's default lib-prefix turns into `libcatboostr.so` — and the init
function is `R_init_libcatboostr` (matching the DLL name given to
`useDynLib`, i.e. the `.so` basename minus extension, **not** the CMake
target name). **No non-standard override is needed** to keep the final name
`libcatboostr`: a production fork target only needs to be named `catboostr`
(exactly as upstream's is) and provide an `R_init_libcatboostr` entry point if
it wants to go through R's `useDynLib(..., .registration = TRUE)` path. This
spike deliberately bypassed that path (raw `dyn.load()` + `.Call("<symbol>")`
by name, per the ticket's instruction), so no `init.c`/`R_init_*` was written
or needed here — R's `.Call()` resolved `Fork_TrainPredict_R` /
`Fork_JsonRegistrarTest_R` directly via `dlsym()` because they're plain
`extern "C"` exported symbols (no `-fvisibility=hidden` is set anywhere in
the toolchain/common.cmake, so default ELF visibility already exports them —
verified no export-script/`use_export_script()` was needed for this to work).

## 8. Allocator (ticket step 8)

Did not test `-DCUSTOM_ALLOCATORS=Off`. `target_allocator(catboostr_fork
cpp-malloc-mimalloc)` was used (mirroring upstream) and linked/ran without
incident; `CUSTOM_ALLOCATORS` defaults to `On`
(`vendor/catboost/cmake/common.cmake:333`). Not blocking per ticket
instruction; flagged as untested, not verified-working-without.

## 9. Timing

- First cold `ninja catboostr_fork` (built essentially the full upstream
  dependency graph from scratch: `train_lib`, `train_lib.global`, `model`,
  `model.global`, `data`, `eval_result`, `model_export`, etc. — nothing was
  pre-cached, no ccache present on this host) — **failed** on the header-order
  bug above after 3m47s (227.2s) wall / 55m32s user / 32 cores.
- Second `ninja catboostr_fork` (header-order fix only; everything else
  cached) — failed at link (missing `vcs_info`) after 5.5s wall.
- Third `ninja catboostr_fork` (added `vcs_info`, plus incremental
  reconfigure) — **succeeded**, 1.7s wall.
- Total wall time from first configure to a working `.so`: **≈234s** across
  three build invocations (227.2 + 5.5 + 1.7), i.e. one real full build plus
  two near-instant incremental fix cycles. Bytes: `libcatboostr_fork.so` =
  46,661,888 bytes = 44.50 MiB.

## Section V — Output Schema (Strict TOON)

```toon
task_id: catboost-8z4.8
success: true
data:
  injection_method: "add_subdirectory(fork-src) appended to the disposable copy's root CMakeLists.txt, after add_subdirectory(catboost) so all upstream targets already exist"
  pinned_tree_untouched: true
  build_succeeded: true
  link_line_has_whole_archive: true
  global_archives_linked_count: 24
  so_name: "libcatboostr_fork.so"
  init_symbol_expected: "R_init_libcatboostr (upstream convention; not registered in this spike -- raw dyn.load()+.Call() used instead, per ticket instruction)"
  dyn_load_succeeded: true
  entry_point_returned_expected: true
  registrar_test_description: "Exported a tiny trained model to Json (non-default format) via ExportModel(), then read it back solely through NCB::TModelLoaderFactory::Construct(EModelType::Json) -- a registrar (TJsonModelLoaderRegistrator) that lives inside the libs-model-model_export.global static archive, a different .global unit than the one used by ordinary Calc()/training. Verified prediction and tree count matched the original after the round trip."
  registrar_test_passed: true
  build_wall_seconds: 234.4
  errors_verbatim: [
    "In file included from .../contrib/libs/cxxsupp/libcxx/include/__locale:871:9: error: expected ';' at end of declaration list",
    "TCatBoostException: catboost/private/libs/options/json_helper.h:41: Can't parse parameter \"verbose\" with value: false",
    "ld.lld: error: undefined hidden symbol: GetArcadiaSourcePath",
    "ld.lld: error: undefined hidden symbol: GetProgramSvnVersion"
  ]
verdict:
  fork_owned_target_viable: true
  reasoning: "A fork-owned CMake target added via add_subdirectory into a disposable copy of the pinned tree -- built and linked AFTER all upstream targets are defined -- correctly resolves target_link_libraries() by real target identity, which is what lets CMake propagate each linked library's .global.wholearchive INTERFACE link options transitively. The actual link line showed 23/23 relevant .global archives whole-archive-wrapped, zero dropped. Two independent registrar-dependent code paths (CPU trainer factory in catboost-libs-train_lib.global, and Json model loader in libs-model-model_export.global -- a different archive) both worked end to end from R via raw .Call, with no cpp11 and no edits inside vendor/catboost/. The three failures hit were ordinary new-file integration friction (header order, a missing vcs_info() call, a JSON param type), not instances of the whole-archive hazard the ticket exists to catch."
  confidence: 92
error_log: null
```
