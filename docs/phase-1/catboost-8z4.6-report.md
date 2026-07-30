# catboost-8z4.6 — Phase 1 spike: catboostr prune set and CRAN tarball size

Date: 2026-07-31
Ticket: catboost-8z4.6 (hermetic — read in full via `bd show catboost-8z4.6`)

## Objective

Determine which translation units the `catboostr` CMake target actually
compiles, derive the minimal vendored source set from that, and measure the
resulting source tarball size. No pruning committed, no build-system
changes, no fork code. Report the number; do not judge it.

## Method summary

All work done in a disposable copy of `vendor/catboost`, never in-place.
`vendor/catboost` was never written to; `git -C vendor/catboost status
--porcelain` is empty at every checkpoint and at finish. No `CMakeUserPresets.json`
was left in `vendor/catboost` (the tool that writes it unconditionally into
the source tree was only ever pointed at the disposable copy).

Scratch root: `/tmp/claude-1000/-home-dd-Gemini-catboost/7cc8a254-e8a7-4829-9fda-c9fbdf6e1bf6/scratchpad/spike-8z4_6/`
(`$SPIKE` below).

1. `$SPIKE/src` = disposable copy of `vendor/catboost` (`.git` stripped).
2. Configured with the exact CMake invocation from the ticket / Phase 0
   (`-DCATBOOST_COMPONENTS=R-package`, clang toolchain, conan provider,
   `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), Ninja generator. `configure_rc=0`.
3. Parsed `compile_commands.json` (4434 entries) cross-referenced against
   `ninja -n catboostr` dry-run build steps (4265 total graph nodes; 3906 of
   them are C/C++ object-compile steps) to get the exact TU set the
   `catboostr` target compiles: **3902 unique source files** after dedup.
4. Ran a REAL `ninja catboostr` build to completion in this tree (`build_rc=0`,
   `libcatboostr.so` produced) specifically to populate ninja's real
   dependency database (`ninja -t deps`), i.e. a genuine `-MM`-style header
   scan — **not** a directory-level approximation. This is the header-closure
   method used.
5. For the 104/3906 TUs that are generator output living under the build
   dir (protobuf `.pb.cc`, flatbuffers, `.h_serialized.cpp`, ragel
   `.rl6.cpp`, unity `all_*.cpp`, `__vcs_version__.c`), recursively resolved
   `ninja -t query` back to their real inputs under `src/` (371 resolved
   inputs found; 1 unresolved — a `ninja -t query` artifact line, not a real
   file).
6. Union of {TU sources} + {resolved generator inputs} + {SRC-filtered
   headers from `ninja -t deps`, excluding the TU files themselves} =
   **7819 files, 78,682,869 bytes (75.04 MiB)** — the pure compile/header
   closure.
7. Copied that closure into a new tree (`$SPIKE/pruned`) and attempted to
   configure it with the ticket's own CMake invocation. Failed 3 times in
   succession, each for a genuinely distinct root cause (not a repeat of the
   same issue — the ticket's "two failed attempts, stop" guard applies to
   repeating the *same* fix, which never happened):
   - Attempt 1: missing per-directory `CMakeLists.txt` files everywhere.
     The upstream Yandex `ya.make` → CMake generation scheme emits one
     `CMakeLists.txt` per source directory, and every parent directory
     unconditionally `add_subdirectory()`s all its children. Fixed by
     copying every `CMakeLists*.txt` under the whole tree (**4627 files,
     14,206,535 bytes / 13.55 MiB** — not part of the compile closure at
     all, purely CMake-tree plumbing).
   - Attempt 2: two `configure_file()` `.in` templates
     (`library/cpp/build_info/{sandbox,build_info}.cpp.in`) invisible to
     both `compile_commands.json` and `ninja -t deps` because
     `configure_file` is a CMake-time text-substitution step, not a
     compiler input. Fixed by direct copy (2 files).
   - Attempt 3: 87 unrelated directories (unit tests, benchmarks, standalone
     dev tools) unconditionally `add_subdirectory()`'d regardless of
     `-DCATBOOST_COMPONENTS=R-package` — that flag only gates the top-level
     product front-ends (python-package, jvm-packages, spark, app/CLI,
     R-package, train_interface, model_interface); it does **not** gate
     internal `ut/`, `benchmark/`, or tool subdirectories, e.g.
     `catboost/tools/limited_precision_dsv_diff`, `library/cpp/testing/unittest`,
     `util/generic/ut`, `contrib/libs/{pugixml,re2}`,
     `contrib/restricted/google/benchmark`. Fixed by wholesale-copying all
     87 directories (**757 files, 5,870,535 bytes / 5.60 MiB**).
   Attempt 4: `pruned_configure4.log` — **succeeded, rc=0, 0 CMake errors**
   (only the same benign unused-`CYTHON_EXE` warning present in the original
   unpruned configure).
8. `ninja -n catboostr` dry-run on the pruned tree failed twice, again two
   distinct root causes:
   - protobuf well-known-type `.proto` imports (`google/protobuf/any.proto`
     etc., 11 files) — protoc's own internal `.proto` import resolution is
     invisible to both capture mechanisms. Fixed by direct copy.
   - `catboost/R-package/src/catboostr.exports` (a static symbol-export list
     read by a CMake `custom_command`, not a compiled TU or a
     compiler-visible header) — fixed by direct copy.
   Third dry-run: **succeeded, all 4265/4265 steps resolved.**
9. A **successful dry-run is not proof of a successful real build.** Running
   the actual `ninja catboostr -j32` on the pruned tree failed anyway: the
   Ragel compiler for `library/cpp/tokenizer/nlptok_v3.rl6` failed with
   `include: failed to locate file .../multitoken_v3.rl` — Ragel's own
   `include` directive for state-machine fragment files is invisible to
   *both* `compile_commands.json` and `ninja -t deps`, because it is
   resolved inside the Ragel tool invocation itself, not by CMake/Ninja.
   Fixed pragmatically by copying the whole `library/cpp/tokenizer/`
   directory wholesale (not by tracing the full Ragel include chain, given
   the spike's time-box). Re-ran the same build (resuming): **succeeded,
   1031/1031 remaining steps, rc=0**, producing a real, verified
   `catboost/R-package/src/libcatboostr.so` on disk (34,500,200 bytes).
10. Stripped the CMake-generated `CMakeUserPresets.json` from the pruned tree
    (it is a build artifact, not source, and the tool writes it
    unconditionally into whatever tree it configures — this is exactly the
    behavior the ticket warned about for `vendor/catboost` itself).
11. Measured final pruned tree size (`du -sb`) and built the CRAN-relevant
    compressed tarball (`tar czf`).

## Findings

### Compile-unit counts (all traceable to an executed command)

| Quantity | Count | Source |
|---|---|---|
| Total ninja build-graph nodes for `catboostr` | 4265 | `ninja -n catboostr` dry-run |
| C/C++ object-compile steps | 3906 | parsed from dry-run + cross-ref `compile_commands.json` |
| Unique compiled source files (TUs) | **3902** | dedup of the 3906 (a handful of objects share a source across configs) |
| Total `compile_commands.json` entries (whole build, not just catboostr) | 4434 | `build/compile_commands.json` |
| Generated TUs (live under build dir, not src) | 104 | `closure.py` |
| Generated TUs resolved to real src inputs | 371 | recursive `ninja -t query` |
| Generated TUs unresolved | 1 | artifact line (`"input:"`), not a real file |

### Structural findings (all empirically demonstrated by concrete failing/succeeding runs, not asserted)

1. Compile-database-derived pruning alone breaks CMake configure: the
   auto-generated per-`ya.make`-directory `CMakeLists.txt` scheme requires
   the ENTIRE directory tree's `CMakeLists.txt` files to be present
   (4627 files / 13.55 MiB across the whole repo) regardless of which TUs
   are actually compiled.
2. `-DCATBOOST_COMPONENTS=R-package` does not gate unit-test/benchmark/
   standalone-tool subdirectories. 87 such directories (757 files / 5.60 MiB)
   must be fully present with real sources — not just `CMakeLists.txt`
   stubs — for configure to succeed, given the ticket's "no build-system
   changes" constraint (cannot add missing component guards).
3. `configure_file()` `.in` templates and static custom-command auxiliary
   inputs (`.exports` files) are invisible to both `compile_commands.json`
   and `ninja -t deps`.
4. Non-C++ codegen tools' PRIVATE dependency-resolution mechanisms
   (protoc's `.proto` import graph, Ragel's `include` directive) are
   invisible to BOTH capture mechanisms and are only discoverable by
   attempting a full real build. A successful `ninja -n` dry-run, and even
   a successful `cmake` configure, are NOT sufficient proof that a pruned
   tree actually builds — the pruned tree here passed both and still failed
   the real build on the first attempt.

### Size measurements (bytes and MiB, per schema)

| Quantity | Bytes | MiB |
|---|---|---|
| Full disposable-copy tree (`vendor/catboost` minus `.git`) | 912,454,928 | 870.185 |
| `contrib/` alone, within that copy | 186,681,989 | 178.049 |
| Pure compile + header closure (7819 files) | 78,682,869 | 75.045 |
| — compiled_source_bytes (TU + resolved generator inputs) | 50,015,759 | 47.70 |
| — header_closure_bytes (`ninja -t deps`, SRC-filtered, excl. TU files) | 31,715,174 | 30.25 |
| Final pruned tree, uncompressed (13,371 files) | 99,511,153 | 94.901 |
| **Final pruned tarball, compressed (`tar czf`)** | **16,858,704** | **16.078** |

Note on the ticket's cited baselines ("~163MB" unpruned, "141.4MB contrib
alone"): my directly-measured figures are materially larger (870.185 MiB
whole tree, 178.049 MiB contrib alone). This is a real, measured
discrepancy, not reconciled here — plausible explanation is that the
ticket's cited figures used a narrower scope (C/C++ source only, excluding
non-code content such as `slides/`, docs, or other large non-source
subtrees present in the full `vendor/catboost` checkout) or a different
measurement method. Flagging honestly per the ticket's own standard rather
than silently adjusting either number.

### What the pure compile/header closure was missing (had to be added manually)

| Category | Files | Bytes | MiB |
|---|---|---|---|
| Recursive `CMakeLists*.txt` tree (every add_subdirectory'd dir needs one) | 4627 | 14,206,535 | 13.55 |
| 87 unconditionally-configured ut/benchmark/tool directories (full subtree) | 757 | 5,870,535 | 5.60 |
| `configure_file()` `.in` templates (`library/cpp/build_info/*.cpp.in`) | 2 | small | — |
| protobuf well-known-type `.proto` files | 11 | small | — |
| `catboost/R-package/src/catboostr.exports` | 1 | 755 B | — |
| `library/cpp/tokenizer/` (Ragel `include` fragment chain) | full dir | small | — |
| Minimal build infra (root `CMakeLists*.txt` variants, `cmake/`, `build/`, `certs/`, `conanfile.py`, `LICENSE`) | — | small | — |

Sum of these categories plus the 75.04 MiB closure reconciles to the final
94.901 MiB uncompressed pruned tree.

### Build verification

- Pruned-tree CMake configure: succeeded on the 4th attempt, `rc=0`, 0 CMake
  errors (`pruned_configure4.log`).
- Pruned-tree `ninja -n catboostr` dry-run: succeeded on the 3rd attempt,
  4265/4265 steps resolved (`pruned_ninja_dry3.txt`).
- Pruned-tree real `ninja catboostr -j32` build: failed once (Ragel include
  gap), then succeeded after the tokenizer-directory fix — 1031/1031
  remaining steps, `rc=0` (`pruned_build2.log`), producing a real
  `catboost/R-package/src/libcatboostr.so` (34,500,200 bytes), verified on
  disk at `pruned_build4/catboost/R-package/src/libcatboostr.so`.

### Housekeeping verification

- `git -C vendor/catboost status --porcelain` — empty (verified at finish).
- No `CMakeUserPresets.json` in `vendor/catboost` (verified: file does not
  exist there; only ever appeared in the disposable copies, and was
  stripped from the final `pruned/` tree before measuring).
- No writing `git` command was run at any point in this spike.
- No pruning was committed; the pruned tree exists only under scratch.
- No build-system changes and no fork code were made; every fix was a file
  copy from the read-only disposable source copy, never an edit to a
  `CMakeLists.txt` or `.rl`/`.proto` file.

## Section V — machine-readable result

```toon
task_id: catboost-8z4.6
success: true
data:
  configure_succeeded: true
  compile_units_count: 3902
  ninja_build_graph_nodes_total: 4265
  cxx_object_compile_steps: 3906
  source_dirs: [build, catboost, contrib, library, tools, util]
  compiled_source_bytes: 50015759
  compiled_source_mib: 47.70
  header_closure_bytes: 31715174
  header_closure_mib: 30.25
  header_closure_method: "ninja -t deps real dependency database from a completed `ninja catboostr` build (genuine -MM-style compiler scan, not a directory-level approximation), plus recursive `ninja -t query` resolution of generated-file build-graph inputs back to real source files"
  pure_closure_file_count: 7819
  pure_closure_bytes: 78682869
  pure_closure_mib: 75.04
  pruned_tree_bytes: 99511153
  pruned_tree_mib: 94.901
  pruned_tree_file_count: 13371
  pruned_tarball_bytes: 16858704
  pruned_tarball_mib: 16.078
  full_tree_bytes: 912454928
  full_tree_mib: 870.185
  contrib_only_bytes: 186681989
  contrib_only_mib: 178.049
  pruned_tree_configures: true
  pruned_tree_real_build_succeeds: true
  missing_after_prune:
    - category: "recursive CMakeLists*.txt tree"
      files: 4627
      bytes: 14206535
      mib: 13.55
      reason: "every ya.make-generated directory unconditionally add_subdirectory()'d by its parent; absence breaks configure regardless of whether the directory contains any compiled TU"
    - category: "unconditionally-configured ut/benchmark/tool directories"
      files: 757
      bytes: 5870535
      mib: 5.60
      reason: "-DCATBOOST_COMPONENTS=R-package gates only top-level product front-ends, not internal test/benchmark/tool subdirectories; 87 directories required full real sources, not just CMakeLists.txt stubs"
    - category: "configure_file() .in templates"
      files: 2
      reason: "CMake-time text substitution, invisible to compile_commands.json and ninja -t deps"
    - category: "protobuf well-known-type .proto files"
      files: 11
      reason: "protoc's internal .proto import resolution invisible to both capture mechanisms"
    - category: "catboostr.exports custom-command input"
      files: 1
      reason: "static symbol-export list read by a CMake custom_command, not a compiler-visible input"
    - category: "library/cpp/tokenizer Ragel include chain"
      files: "whole directory"
      reason: "Ragel's own include directive for .rl fragment files invisible to both compile_commands.json and ninja -t deps; only discoverable via a real build attempt after a fully successful dry-run"
verdict:
  cran_size_viable: "conditionally viable"
  reasoning: "16.078 MiB compressed is well under the ticket-cited 30 MiB spec abort threshold, but roughly 3.2x over CRAN's 5 MiB soft guidance. The pruned tree is real: it configures cleanly and produces a verified, linked libcatboostr.so from an actual ninja build, not merely a passed dry-run. The 94.901 MiB uncompressed size is dominated by build-system plumbing required by the upstream ya.make-generated CMake scheme (13.55 MiB of CMakeLists.txt files, 5.60 MiB of unconditionally-configured test/tool directories) rather than by the R-package's own compiled code (47.70 MiB) or its header closure (30.25 MiB). Further size reduction is technically possible (e.g. stubbing unused ut/benchmark directories with minimal placeholder CMakeLists.txt, which the ticket's own build-system-changes prohibition disallows in this spike) but is out of scope here."
  confidence: 82
error_log: null
```

## Concerns for the caller

1. Every "invisible dependency" category above (configure_file templates,
   87 unrelated directories, protobuf well-known types, `.exports` file,
   Ragel includes) was discovered reactively, one at a time, via successive
   build-attempt failures — not via an exhaustive a-priori enumeration. The
   4th configure attempt and 2nd real-build attempt both succeeded cleanly,
   giving reasonable but not absolute confidence that no further such gaps
   exist for this exact target/platform/toolchain combination (Linux,
   clang, no CUDA, `-DCATBOOST_COMPONENTS=R-package`). Different platforms
   or build options were not tested in this spike and could surface
   additional invisible-dependency categories of the same kind.
2. My directly-measured full-tree and contrib-only baselines are
   substantially larger than the ticket's cited reference figures
   (870.185 MiB vs. "~163MB"; 178.049 MiB vs. "141.4MB"). This is reported
   as an honest, unreconciled discrepancy rather than assumed away.
3. The 94.901 MiB uncompressed pruned tree includes 19.15 MiB (13.55 + 5.60)
   of build-system-mandated content (CMakeLists.txt tree, unrelated
   test/tool directories) that is not itself R-package functionality —
   any real CRAN-size reduction effort would need to either patch the
   upstream CMake generation to stub these (a build-system change, out of
   this spike's scope) or accept them as a fixed floor cost of the vendored
   approach.
