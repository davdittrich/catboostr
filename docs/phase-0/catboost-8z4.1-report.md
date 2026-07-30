# catboost-8z4.1 — Vendored source build measurement spike

Date: 2026-07-30
Environment: R 4.6.1 (/usr/bin/R), cmake 3.x (system), Python 3.14.6, 32 cores, 60 GB RAM.
`ninja` and `conan` confirmed NOT installed (`which ninja`, `which conan` both empty). Neither was installed during this task, per guard.

## 1. Clone

`git clone --branch v1.2.10 --depth 1 https://github.com/catboost/catboost.git vendor/catboost`

- Resolved SHA: `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084` — matches the SHA specified in the brief exactly.
- Checkout size: 1295 MB (`du -sm vendor/catboost`).
- `git -C vendor/catboost status --porcelain` is empty (verified at end of task too — nothing under `vendor/catboost/` was touched).

## 2. Conan-provided dependencies (core deliverable)

From `vendor/catboost/conanfile.py` (repo-root conanfile, full text read):

| name | version | via |
|---|---|---|
| openssl | 3.0.15 | `requirements()` → `self.requires("openssl/3.0.15")` (line 16) |
| ragel | 6.10 | `build_requirements()` → `self.tool_requires("ragel/6.10")` (line 19) |
| swig | 4.0.2 | `build_requirements()` → `self.tool_requires("swig/4.0.2")` (line 20) |
| yasm | 1.3.0 | `build_requirements()` → `self.tool_requires("yasm/1.3.0")` (line 21) |

4 direct dependencies. Each of these is a Conan Center recipe, each of which has its own transitive dependency graph (e.g. openssl has none typically, but ragel/swig/yasm are typically self-contained tool packages) — Conan will resolve and potentially build all of these plus any transitive deps the first time `find_package()` triggers `conan_provide_dependency` (see §3). No transitive list was enumerated — Conan itself is not runnable in this environment (not installed), so `conan graph info` could not be run to expand it. This is the true "vendoring cost surface": at minimum 4 packages, each requiring either a prebuilt binary from Conan Center (network) or a from-source Conan recipe build (also normally network-fetched for the recipe itself, then locally compiled).

## 3. Ninja/Conan wiring — file:line, and bypass-by-flag check

`vendor/catboost/build/build_native.py`, function that constructs the CMake configure command (~line 578-586):

```python
cmake_cmd = [
    'cmake',
    source_root_dir,
    '-B', opts.build_root_dir,
    '-G', 'Ninja',                                                          # line 582
    f'-DCMAKE_BUILD_TYPE={opts.build_type}',
    f'-DCMAKE_TOOLCHAIN_FILE={cmake_target_toolchain}',
    f'-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={source_root_dir}/cmake/conan_provider.cmake'   # line 585
]
```

- `-G Ninja` (line 582) is an unconditional list literal — no `if`/option branch anywhere in the file selects a different generator. Confirmed by grepping the whole file for `'-G'`, `generator`, `skip_conan`, `no_conan`, `use_system` — only the one hit at line 582. **Not bypassable by a documented flag.** The full CLI arg surface (`--dry-run --verbose --build-root-dir --build-type --rebuild --keep-going --conan-build-profile --conan-host-profile --msvs-* --macosx-version-min --have-cuda --cuda-* --android-ndk-root-dir --cmake-extra-args --parallel --targets`, all enumerated from the `Option()` table at file top) contains nothing that changes generator or disables Conan; `--cmake-extra-args` could in principle append `-G Unix Makefiles` a second time but the first `-G Ninja` is already positioned earlier in argv, and CMake takes the first `-G` — untested, out of scope, and still leaves the Conan provider include in place regardless.
- `CMAKE_PROJECT_TOP_LEVEL_INCLUDES=.../cmake/conan_provider.cmake` (line 585) is likewise unconditional. `conan_provider.cmake` installs a CMake dependency provider (`cmake_language(SET_DEPENDENCY_PROVIDER conan_provide_dependency SUPPORTED_METHODS FIND_PACKAGE)`, lines 663-666) that fires on the *first* `find_package()` call anywhere in the CMake tree and shells out to `conan install ... --format=json` (line 496, inside `conan_install()`, called from `conan_provide_dependency` at lines 583-660). There is no CMake cache variable to disable the provider short of not passing `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` at all — which `build_native.py` always passes.
- **Verdict:** neither Ninja nor Conan is bypassable by a documented flag in `build_native.py`. Bypassing either requires either (a) invoking `cmake` directly instead of through `build_native.py`, still requires providing a real generator and installing `ninja`/`make`+something, and still hits the Conan provider on `find_package()`, or (b) patching `build_native.py`/`conan_provider.cmake` — explicitly out of scope for this ticket.
- Separately, `build_native.py` also invokes Conan directly (not via the CMake provider) at lines 349-363 (`conan_install_cmd = ['conan', ...]`, `cmd_runner.run(conan_install_cmd)` at line 363) for a different code path (build-environment/tool bootstrap for cross-compilation). Also unconditional, also a network fetch point.

## 4. Section I claims — confirmed/refuted, file:line

| Claim | Verdict | Evidence |
|---|---|---|
| `configure` contains zero `cmake` occurrences and never invokes a compiler | **CONFIRMED** | `grep -c cmake vendor/catboost/catboost/R-package/configure` → `0` (2528-line file, full grep). Dynlib resolution logic at `catboost/R-package/configure:1756-1799`: checks `$CATBOOST_DYNLIB` env var (1759-1772) → checks pre-existing `src/libcatboostr.so` (1774-1785) → downloads via `R/install.R` (1787-1799), download call at line 1793 (`catboost_download_dynlib("src/")`). No compiler invocation anywhere in the file. |
| The only from-source path reachable from `catboost/R-package/` is `src/Makefile` → `Makefile.inner` → `ya make` | **CONFIRMED, with a material addition** | `catboost/R-package/src/Makefile` (12 lines): `libcatboostr.so: $(MAKE) -f Makefile.inner`. `catboost/R-package/src/Makefile.inner`: `all: chmod +rx $(THIS_DIR)/../../../ya; $(THIS_DIR)/../../../ya make -r ...`. This resolves to `<repo-root>/ya`. **Addition: the `ya` binary/script does not exist anywhere in this checkout** (`find . -maxdepth 1 -iname 'ya*'` at repo root and inside `catboost/` both empty). This path is not merely "requires Yandex's proprietary tool" — the tool is entirely absent from the public tag, so `ya make` fails immediately with "no such file" before any Yandex-infrastructure question even arises. Separately, `catboost/R-package/CMakeLists.txt` (auto-generated from `ya.make`, comment says so) *does* wire the R-package into the top-level CMake/Ninja/Conan graph via `add_subdirectory(src)` when `R-package` is in `CATBOOST_COMPONENTS` — but that path is reachable only by invoking the top-level `build/build_native.py`/`cmake` directly, not by anything `configure` or `src/Makefile` themselves call. So the claim is correct as scoped ("reachable from catboost/R-package/'s own configure/Makefile") but there IS a second from-source path if you go around the R package's own build scripts — the CMake/Ninja/Conan one, whose network dependency is the one measured in §5-6 below. |
| `build_native.py` hardcodes Ninja and injects Conan | **CONFIRMED** | See §3, lines 582 and 585. |

## 5. Offline build attempt

Command (from repo root, `vendor/catboost/`):

```
python3 build/build_native.py --targets catboostr --build-root-dir <scratch>/cbuild
```

(First attempt included `--parallel 32`; `build_native.py`'s parallel option is not threaded onto the `cmake` configure invocation in this code path and CMake rejected it as an unknown argument — a script-side bug, not a build-system finding. Removed the flag; parallelism is a build-step not configure-step concern anyway. This does not count against the two-failed-attempt guard since it was a usage correction, not a build failure.)

Actual CMake invocation logged by the script:
```
cmake vendor/catboost -B <scratch>/cbuild -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vendor/catboost/build/toolchains/clang.toolchain \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=vendor/catboost/cmake/conan_provider.cmake \
  -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCATBOOST_COMPONENTS=R-package -DHAVE_CUDA=no
```

Failure (verbatim tail):
```
CMake Error: CMake was unable to find a build program corresponding to "Ninja".  CMAKE_MAKE_PROGRAM is not set.  You probably need to select a different build tool.
CMake Error: CMAKE_ASM_COMPILER not set, after EnableLanguage
-- Configuring incomplete, errors occurred!
```
followed by Python `subprocess.CalledProcessError` (script's `check=True` propagating the non-zero exit).

**This failure happens at CMake generator selection — before CMake reaches any `find_package()` call, so the Conan dependency provider is never invoked and no network fetch occurs.** `offline_build_attempted: true`, `offline_build_succeeded: false`. `missing_tools: [ninja]` is the proximate blocker; `conan` is also absent and would block the very next step even if `ninja` were present (would fail at `find_program(CONAN_COMMAND "conan" REQUIRED)` inside `conan_provide_dependency`, `conan_provider.cmake:587`).

This is a clean, deterministic, single-attempt result — not a flaky failure needing a retry. Per guard, no system packages (`ninja`, `conan`) were installed to work around it.

## 6. Step 6 (Conan pre-population retry) — not applicable

Step 6 applies only "if step 5 fails purely on Conan dependency resolution." It did not: it failed at Ninja generator selection, upstream of Conan entirely. Since neither `ninja` nor `conan` may be installed (guard), and Conan resolution was never reached, there is no way to separate "cannot build offline" from "cannot build at all" in this environment — both are blocked by the same missing-tool wall, and installing either tool is out of scope. `deps_prepopulated: false`.

## 7. Measurements

- `build_wall_seconds`: null (build never started compiling).
- `cores_used`: 32 (available, `nproc`; not exercised by a build since none progressed past configure).
- `installed_size_mb`: null — not measurable; the ~141 MB installed-size claim from the design spec is **neither confirmed nor refuted**, because the build never produced installed artifacts. It cannot be measured in this environment without installing `ninja`/`conan`, which the guards forbid.
- `libcatboostr_size_mb`: null (never built).
- `source_tarball_mb`: measured as a bonus, independent check (does not require `libcatboostr.so`): `R CMD build vendor/catboost/catboost/R-package` from the scratch dir succeeded (R itself doesn't need cmake/ninja/conan to package R sources) and produced `catboost_1.2.10.tar.gz`, **171,853 bytes ≈ 0.164 MB**. This is the R-wrapper-source-only tarball (R/, man/, tests/, DESCRIPTION, etc.) with no compiled `libcatboostr.so` inside and, critically, no vendored C++ core sources — it is NOT representative of what a real from-source-vendored CRAN package tarball would weigh once the CatBoost C++ core is pulled in (that would be a large fraction of the 1295 MB checkout, filtered down by build-relevant subdirs only). Do not conflate this number with a post-vendoring tarball estimate.

## 8. `ya make` viability assessment (report only, not attempted)

Even disregarding the missing-tool constraints on the CMake path, `ya make` is not viable in this environment or on this public tag at all:
- The `ya` launcher script that `Makefile.inner` shells out to (`<repo-root>/ya`) does not exist anywhere in the tag `v1.2.10` checkout (verified: no `ya`/`ya.*` at repo root or under `catboost/`).
- `ya` is Yandex's internal monorepo build tool; the public mirror does not ship it, and (per public knowledge of the tool) it self-bootstraps additional binary components from Yandex-internal infrastructure on first invocation, which is both a network dependency AND infra this environment (and CRAN infra) has no path to reach at all, unlike Conan Center which is at least a public package index.
- What it would require: obtaining the `ya` binary from Yandex (not distributed via GitHub releases for this repo), plus network access to Yandex's internal package/artifact servers for `ya`'s own dependency resolution — a strictly harder and less transparent problem than the already-intractable Conan network dependency. Not a viable path for a CRAN-targeting fork under any circumstances.

## 9. Verdict

`vendored_build_tractable`: **false, as measured in this environment; genuinely uncertain even with tools present**, because the build never got far enough (generator selection failure) to reach the actual object of measurement — the Conan network-fetch surface. What IS established with certainty:
- The Conan dependency surface is small in *direct* count (4 packages: openssl 3.0.15, ragel 6.10, swig 4.0.2, yasm 1.3.0) but each pulls its own recipe + transitive deps from Conan Center over the network, unmeasured here since Conan itself isn't installed.
- Neither Ninja nor Conan is avoidable via any documented `build_native.py` flag — vendoring means either patching the build scripts to accept a different generator/dependency-resolution path (real engineering, explicitly out of scope for this ticket) or pre-vendoring build-time tool binaries (ragel, swig, yasm) and pinning/vendoring openssl, then still needing a Ninja-generator-compatible offline path or patching `-G Ninja` to something else (e.g. `Unix Makefiles`, which ships with `cmake`/`make` typically already present).
- `ya make` is a dead end — the tool itself is absent from the public repo.

**Estimated vendoring cost** (qualitative, since the actual compile-time/size numbers could not be measured here): non-trivial but bounded. Concretely it requires: (1) patching `build_native.py` to drop the hardcoded `-G Ninja` (or vendor `ninja` — it's a tiny static binary, easy to vendor/pin, unlike Conan); (2) replacing `CMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake` with either vendored copies of openssl/ragel/swig/yasm resolved via plain `find_package`/`CMAKE_PREFIX_PATH` (no Conan), or a frozen local Conan cache shipped in the package (CRAN would very likely reject a frozen binary cache bundled in a source package, so the `find_package`-without-Conan route is the CRAN-viable one); (3) re-testing that CMake's `R-package` component target actually builds standalone once decoupled from Conan/Ninja. This is real, multi-day engineering, not a config flag — consistent with the brief's framing that "creating [a network-free path] is new engineering," not a hardening pass.

`recommend_fallback_to_system_library`: **true, provisionally** — given (a) the from-source path requires non-trivial build-script surgery no matter what, (b) `ya make` is dead, and (c) CRAN forbids network access during install outright, a system-library / vendored-prebuilt-binary model (the same model `catboost/R-package/configure` already falls back to today for `CATBOOST_DYNLIB`/pre-existing `.so`) is the more CRAN-tractable path unless Phase 1 explicitly commits to patching `build_native.py` and vendoring the 4 direct Conan deps' sources. This ticket does not decide that; it sizes the alternative honestly for Phase 1 to weigh.

`confidence`: 85 — the wiring and claims analysis is high-confidence (exact file:line evidence, reproduced failure). The main open unknown, honestly flagged, is the *actual* compile time/size once past Conan, and the *actual* size of the transitive Conan dependency graph beyond the 4 direct packages — neither measurable here without installing forbidden tools.

---

```toon
task_id: catboost-8z4.1
success: true
data:
  upstream_tag: v1.2.10
  upstream_sha: b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084
  checkout_size_mb: 1295
  claims_verified:
    configure_has_no_cmake: true
    only_source_path_is_ya_make: true
    build_native_hardcodes_ninja_and_conan: true
    corrections: ["ya binary itself is absent from the public v1.2.10 checkout (no ya/ya.* at repo root or under catboost/) — ya make fails before any Yandex-infra question arises","catboost/R-package/CMakeLists.txt (auto-generated from ya.make) does wire R-package into the top-level CMake/Ninja/Conan graph via add_subdirectory(src) when R-package is in CATBOOST_COMPONENTS, reachable only by invoking build_native.py/cmake directly, not via configure or src/Makefile"]
  conan_dependencies:
    - {name: openssl, version: 3.0.15}
    - {name: ragel, version: 6.10}
    - {name: swig, version: 4.0.2}
    - {name: yasm, version: 1.3.0}
  conan_bypassable_by_flag: false
  ninja_bypassable_by_flag: false
  network_fetches:
    - {file: catboost/R-package/R/install.R, line: 27, command: "download.file(url, dest_fpath, mode='wb')  # prebuilt libcatboostr.so from GitHub releases"}
    - {file: cmake/conan_provider.cmake, line: 496, command: "execute_process(COMMAND ${CONAN_COMMAND} install ... --format=json)"}
    - {file: build/build_native.py, line: 363, command: "cmd_runner.run(conan_install_cmd)  # direct 'conan' invocation, cmd built at line 349"}
  offline_build_attempted: true
  offline_build_succeeded: false
  deps_prepopulated: false
  build_wall_seconds: null
  cores_used: 32
  installed_size_mb: null
  libcatboostr_size_mb: null
  source_tarball_mb: 0.164
  missing_tools: [ninja, conan]
  ya_make_viability: "Not viable: the ya launcher script that Makefile.inner shells out to (<repo-root>/ya) does not exist anywhere in the public v1.2.10 tag. Even if obtained, ya is Yandex's internal monorepo tool that self-bootstraps further binary components from Yandex-internal infrastructure, which is neither public nor reachable, and is a strictly worse network dependency than Conan Center. Not viable for a CRAN-targeting fork under any circumstances."
  failing_command: "cmake vendor/catboost -B <scratch>/cbuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=vendor/catboost/build/toolchains/clang.toolchain -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=vendor/catboost/cmake/conan_provider.cmake -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCATBOOST_COMPONENTS=R-package -DHAVE_CUDA=no"
  failing_output_tail: "CMake Error: CMake was unable to find a build program corresponding to \"Ninja\".  CMAKE_MAKE_PROGRAM is not set.  You probably need to select a different build tool.\nCMake Error: CMAKE_ASM_COMPILER not set, after EnableLanguage\n-- Configuring incomplete, errors occurred!"
verdict:
  vendored_build_tractable: false
  estimated_vendoring_cost: "Multi-day engineering, not a flag flip: (1) patch build_native.py to drop hardcoded -G Ninja (vendoring ninja itself is cheap, it's a small static binary); (2) replace the CMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake injection with plain find_package against vendored/system copies of openssl 3.0.15, ragel 6.10, swig 4.0.2, yasm 1.3.0 (a frozen Conan cache is not CRAN-viable to ship); (3) validate the R-package CMake component builds standalone once decoupled from Conan/Ninja. Actual compile time/size beyond this point unmeasured here — build never got past CMake generator selection."
  recommend_fallback_to_system_library: true
  reasoning: "Both from-source paths are blocked: (a) the CMake/Ninja/Conan path hits a missing-tool wall (ninja, conan) before even reaching Conan's network fetch, and structurally requires patching two hardcoded, non-flag-bypassable injection points (build_native.py:582, :585) to ever be network-free; (b) the ya make path is dead on this tag because the ya tool itself is not shipped in the public repo, and even if it were, it is a proprietary Yandex tool with further internal-infra dependencies, categorically worse than Conan for CRAN purposes. configure already has a working fallback model (env var CATBOOST_DYNLIB or pre-existing src/libcatboostr.so) that is the natural fit for a system-library/vendored-prebuilt-binary strategy, consistent with CRAN policy against network access during install, without requiring the from-source build-script surgery in (a)."
  confidence: 85
error_log: null
```

---

# Fix round 1 — actual measurements with ninja/conan in a throwaway venv

Brief corrected: the "no ninja/conan" guard was over-restrictive; both ship as pip wheels and installing them into a local, throwaway virtualenv is not a system-wide install. Executed per corrected brief §IV.5.

## 0. Tooling provisioning (local venv, not system-wide)

```
uv venv /tmp/.../scratchpad/venv --python 3.12
uv pip install --python /tmp/.../scratchpad/venv/bin/python ninja conan
```
Result: `ninja 1.13.0`, `conan 2.31.1` installed, both pip wheels, entirely inside the scratch venv. Nothing installed system-wide, no sudo, no pacman/apt/dnf. `conan profile detect --force` run once to generate a host profile (gcc 16, libstdc++11, x86_64, Release).

Conan's cache (`CONAN_HOME`) was pointed at `<scratch>/conan_home` (not `~/.conan2`) to keep every write inside `vendor/catboost/` + scratch, per the boundary guard — the default `~/.conan2` location was created once by an initial `conan profile detect` and immediately deleted (`rm -rf /home/dd/.conan2`) before any dependency data landed there.

## 1. Conan transitive dependency closure (`conan graph info conanfile.py --format=json`, network ON)

The 4 direct dependencies reported in the original report (openssl 3.0.15, ragel 6.10, swig 4.0.2, yasm 1.3.0) are not the full cost surface. `conan graph info` resolves the real closure — **13 unique packages**:

| name | version |
|---|---|
| openssl | 3.0.15 |
| zlib | 1.3.2 |
| autoconf | 2.71 |
| automake | 1.16.5 |
| bison | 3.8.2 |
| bzip2 | 1.0.8 |
| flex | 2.6.4 |
| gnu-config | cci.20210814 |
| m4 | 1.4.19 |
| pcre | 8.45 |
| ragel | 6.10 |
| swig | 4.0.2 |
| yasm | 1.3.0 |

(zlib is a shared transitive dep of both the `openssl` requirement and several `tool_requires`; the rest are transitive build-tool dependencies of `ragel`/`swig`/`autoconf`-family recipes.) This is the real vendoring surface Phase 1 would have to replace or vendor — roughly 3x the direct-dependency count.

## 2. Run (a) — Network ON, primary measurement run

Command (venv `ninja`/`conan` prepended to PATH, `CONAN_HOME` set to scratch):
```
python3 build/build_native.py --targets catboostr --build-root-dir <scratch>/cbuild
```
(First attempt included `--parallel 32` and hit the same `CMake Error: Unknown argument --parallel` script bug already documented in the original report — `build_native.py` mis-threads `--parallel` onto the `cmake` configure invocation rather than the `ninja` build invocation. Not a build-system finding; dropped the flag and reran immediately. Ninja auto-detects all 32 cores for the build step with no `-j` needed, so this cost nothing.)

**Result: SUCCESS.** Full log tail: `[4265/4265] Linking CXX shared library catboost/R-package/src/libcatboostr.so`, `exit_status=0 elapsed_seconds=164`.

- `build_wall_seconds`: **164** (2m44s), wall-clock via `date +%s` around the whole `build_native.py` invocation (configure + Conan install/build + ninja build).
- `cores_used`: **32** (ninja auto-detected, no explicit `-j`; confirmed by machine `nproc`).
- `libcatboostr_size_mb`: `<scratch>/cbuild/catboost/R-package/src/libcatboostr.so` = **34,467,216 bytes ≈ 32.9 MB**.
- `peak_rss`: not measured. No GNU `/usr/bin/time -v` present on this system (`ls /usr/bin/time` → no such file) and installing it would be a system package; a manual `/proc/<pid>/status` polling loop is exactly the polling pattern this environment's tooling forbids for waiting-on-completion, and setting one up for a 164-second build is disproportionate effort for an explicitly optional ("if available") field. Not a blocker for the ticket's gating question.
- **Installed size** (via `R CMD INSTALL --library=<scratch>/rlib <scratch copy of R-package>` with `CATBOOST_DYNLIB=<built .so>`, install succeeded, `exit_status=0`): `<scratch>/rlib/catboost` = **34 MB** (`du -sm`), of which the installed `libs/libcatboostr.so` alone is 34,467,216 bytes (identical file, just copied).
- **~141 MB installed-size claim from the design spec: REFUTED.** Measured installed size is ~34 MB, roughly 4x smaller than the design spec's estimate. (Scope note: this build only targets the `catboostr` R-package component, not the full CatBoost CLI/Python/all-tools build — the 141 MB figure may have been estimated against a larger build scope. Flagging the discrepancy rather than guessing which scope the original 141 MB referred to.)
- `source_tarball_mb`: **0.164 MB** (171,864 bytes) — re-measured via `R CMD build` on the now-configured (post-install, with `libcatboostr.so` present) package copy; identical size to the original report's pre-build measurement. R CMD build's packaging step strips compiled objects (`* cleaning src`) before tarring, so the source tarball never carries the compiled library regardless of build state — confirmed, not just assumed, by running it both ways.

**Side-effect finding (new, material to the "never edit vendor/catboost" guard):** the build silently wrote one file into `vendor/catboost/CMakeUserPresets.json` — auto-generated by Conan's `CMakeToolchain`/preset integration at `CMAKE_SOURCE_DIR` regardless of `-B <build-root-dir>`. This is not something the invocation controls via a flag; it is unconditional Conan 2.x behavior. Detected via `git -C vendor/catboost status --porcelain` after the build (`?? CMakeUserPresets.json`), and removed immediately (`rm`) to restore the pristine snapshot — confirmed clean again afterward. **This means any from-source CMake+Conan build of this repo, even a fully successful and fully offline one, mutates the nominally read-only source tree by default** — a fact Phase 1 needs to design around (e.g. always build from a disposable copy, never the canonical vendored snapshot) regardless of the network question.

## 3. Run (b) — Network OFF, Conan cache warm from run (a)

Isolation method: bogus `http_proxy`/`https_proxy`/`HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:1` (nothing listening) rather than a network namespace (`unshare -net` blocked by this session's shell allowlist and not worth escalating for a one-shot check) — sufficient because both Conan (Python `requests`) and R's `download.file` (libcurl) honor proxy env vars, and any attempted connection fails fast and loud rather than hanging.

Command: identical `build_native.py` invocation, fresh `--build-root-dir`, same warm `CONAN_HOME`.

**Result: FAILED, cleanly and immediately (elapsed_seconds=1).**

Failing command:
```
cmake vendor/catboost -B <scratch>/cbuild2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vendor/catboost/build/toolchains/clang.toolchain \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=vendor/catboost/cmake/conan_provider.cmake \
  -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCATBOOST_COMPONENTS=R-package -DHAVE_CUDA=no
```

Failing output tail:
```
    ... (openssl, zlib, autoconf, automake, bison, bzip2, flex, gnu-config, m4, pcre, ragel, swig, yasm all resolved from "Cache")
Connecting to remote 'conancenter' anonymously
ERROR: Failed checking for binary 'bzip2/1.0.8:215859dd67e540c0068a46cd8c8f792d14f85182' in remote 'conancenter': remote not available
ERROR: HTTPSConnectionPool(host='center2.conan.io', port=443): Max retries exceeded with url: /v1/ping
  (Caused by ProxyError('Unable to connect to proxy', ... Connection refused))
Unable to connect to remote conancenter=https://center2.conan.io
CMake Error at cmake/conan_provider.cmake:508 (message):
  Conan install failed='1'
```

**This is the single most important isolated finding of this fix round.** Every package resolved from the local cache by *name* ("Cache" status for all 13), yet Conan still attempted to reach `center2.conan.io` to double-check binary availability for at least one package (`bzip2`) before proceeding — this is `conan install`'s default remote-consultation behavior, not a missing-artifact problem. `build_native.py`'s `CONAN_INSTALL_ARGS` defaults to `--build=missing` (`cmake/conan_provider.cmake:693`) with no `-nr`/`--no-remote`. **A warm cache alone does not make this build network-free** — it additionally requires passing `-nr`/`--no-remote` (or equivalent offline flags) to the Conan invocation, which is a `build_native.py` patch, i.e. real engineering, not a config toggle available today. This directly confirms and sharpens the original report's conclusion: no flag exists today to get a network-free build; it must be built.

## 4. Updated Section V TOON (supersedes the block above for run-1/run-2 fields; all other fields carry over unless corrected)

```toon
task_id: catboost-8z4.1
success: true
data:
  upstream_tag: v1.2.10
  upstream_sha: b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084
  checkout_size_mb: 1295
  claims_verified:
    configure_has_no_cmake: true
    only_source_path_is_ya_make: true
    build_native_hardcodes_ninja_and_conan: true
    corrections: ["ya binary itself is absent from the public v1.2.10 checkout (no ya/ya.* at repo root or under catboost/) — ya make fails before any Yandex-infra question arises","catboost/R-package/CMakeLists.txt (auto-generated from ya.make) does wire R-package into the top-level CMake/Ninja/Conan graph via add_subdirectory(src) when R-package is in CATBOOST_COMPONENTS, reachable only by invoking build_native.py/cmake directly, not via configure or src/Makefile","even a fully successful, fully offline-capable build silently writes CMakeUserPresets.json into vendor/catboost/ (Conan 2.x CMakeToolchain default behavior, not flag-controlled) — detected via git status and removed; Phase 1 must always build from a disposable copy, never the canonical snapshot"]
  conan_dependencies:
    - {name: openssl, version: 3.0.15}
    - {name: zlib, version: 1.3.2}
    - {name: autoconf, version: 2.71}
    - {name: automake, version: 1.16.5}
    - {name: bison, version: 3.8.2}
    - {name: bzip2, version: 1.0.8}
    - {name: flex, version: 2.6.4}
    - {name: gnu-config, version: cci.20210814}
    - {name: m4, version: 1.4.19}
    - {name: pcre, version: 8.45}
    - {name: ragel, version: 6.10}
    - {name: swig, version: 4.0.2}
    - {name: yasm, version: 1.3.0}
  conan_bypassable_by_flag: false
  ninja_bypassable_by_flag: false
  network_fetches:
    - {file: catboost/R-package/R/install.R, line: 27, command: "download.file(url, dest_fpath, mode='wb')  # prebuilt libcatboostr.so from GitHub releases"}
    - {file: cmake/conan_provider.cmake, line: 496, command: "execute_process(COMMAND ${CONAN_COMMAND} install ... --format=json)  # also does an unconditional remote-availability check even with a fully warm local cache, per run (b)"}
    - {file: build/build_native.py, line: 363, command: "cmd_runner.run(conan_install_cmd)  # direct 'conan' invocation, cmd built at line 349"}
  offline_build_attempted: true
  offline_build_succeeded: false
  deps_prepopulated: true
  build_wall_seconds: 164
  cores_used: 32
  installed_size_mb: 34
  libcatboostr_size_mb: 32.9
  source_tarball_mb: 0.164
  missing_tools: []
  ya_make_viability: "Not viable: the ya launcher script that Makefile.inner shells out to (<repo-root>/ya) does not exist anywhere in the public v1.2.10 tag. Even if obtained, ya is Yandex's internal monorepo tool that self-bootstraps further binary components from Yandex-internal infrastructure, which is neither public nor reachable, and is a strictly worse network dependency than Conan Center. Not viable for a CRAN-targeting fork under any circumstances."
  failing_command: "cmake vendor/catboost -B <scratch>/cbuild2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=vendor/catboost/build/toolchains/clang.toolchain -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=vendor/catboost/cmake/conan_provider.cmake -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCATBOOST_COMPONENTS=R-package -DHAVE_CUDA=no"
  failing_output_tail: "Connecting to remote 'conancenter' anonymously\nERROR: Failed checking for binary 'bzip2/1.0.8:215859dd67e540c0068a46cd8c8f792d14f85182' in remote 'conancenter': remote not available\nERROR: HTTPSConnectionPool(host='center2.conan.io', port=443): Max retries exceeded with url: /v1/ping (Caused by ProxyError('Unable to connect to proxy', ...))\nUnable to connect to remote conancenter=https://center2.conan.io\nCMake Error at cmake/conan_provider.cmake:508 (message):\n  Conan install failed='1'"
verdict:
  vendored_build_tractable: true
  estimated_vendoring_cost: "Now bounded by real numbers: the R-package-only build compiles in 164s wall-clock on 32 cores and installs to ~34 MB (libcatboostr.so alone is ~33 MB) once Conan resolves 13 packages (openssl, zlib, autoconf, automake, bison, bzip2, flex, gnu-config, m4, pcre, ragel, swig, yasm). Compile cost is trivial; the actual vendoring cost is entirely in replacing Conan's network dependency: (1) patch build_native.py's Conan invocation to add -nr/--no-remote (or equivalent) — confirmed necessary in run (b), a warm cache alone is NOT sufficient because conan install still probes the remote for at least one package's binary; (2) vendor or pin the 13-package closure as local sources/binaries CRAN can build without touching network; (3) handle CMakeUserPresets.json being written into the source tree on every build (found in run (a)) so Phase 1's build process never treats vendor/catboost as writable. None of this is exotic engineering — the compile itself is fast and small — but it is real, multi-file build-script surgery, not a config flag."
  recommend_fallback_to_system_library: false
  reasoning: "Reversing the prior round's provisional recommendation now that real numbers exist: the actual build is fast (164s), small (34 MB installed), and the Conan dependency surface, while larger than first estimated (13 packages, not 4), is a well-known, fully public set of Conan Center recipes with no exotic requirements. Network isolation was refuted only by Conan's default remote-probe behavior, not by an unavailable artifact — a targeted flag/patch (-nr/--no-remote plus offline vendoring of the 13 recipes) closes the gap. This is materially cheaper than reworking catboostr around a prebuilt-binary/system-library distribution model, which carries its own CRAN and cross-platform packaging burden. Recommend Phase 1 pursue the from-source vendored path, scoped to: (a) build_native.py Conan-args patch, (b) vendoring the 13-package closure, (c) a build wrapper that never writes into the canonical vendor/catboost snapshot. ya make remains a dead end regardless."
  confidence: 90
error_log: null
```

