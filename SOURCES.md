# SOURCES — provenance and integrity of every external artifact

This is the single inventory of everything this repository pulls in from
outside itself, with the exact version, the exact upstream location, and the
integrity check that is enforced against it.

Scope note: this file covers **provenance and integrity**. It does not restate
licences — `inst/COPYRIGHTS` owns those, and the two files deliberately do not
duplicate each other.

Rule for every row below: the check is **fail-closed**. A missing or
mismatching artifact aborts the acquisition or the build; nothing is ever
silently re-downloaded, downgraded, or substituted with a system copy.

---

## 1. Compiled into `inst/libs/libcatboostr.so`

These are the artifacts whose bytes end up inside the shared object R loads.

### 1.1 Upstream CatBoost source snapshot

| field | value |
| :--- | :--- |
| what | the CatBoost C++ core and R glue |
| version | `v1.2.10` |
| upstream | `https://github.com/catboost/catboost.git` |
| pin | commit `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084` |
| lands in | `vendor/catboost/` (gitignored; acquired, not committed) |
| acquired by | `tools/vendor/acquire.sh` |
| verified by | `git rev-parse HEAD` compared to the pinned SHA, fail-closed (`tools/vendor/acquire.sh:40-44`) |

`vendor/catboost/` is a **read-only pin**. No file under it is ever edited.
Everything this fork changes lives outside it (`configure`, `cmake/`, `src/`)
and is injected into a disposable copy at build time.

### 1.2 openssl — ACCEPTED into the link, pinned, and CVE-tracked

| field | value |
| :--- | :--- |
| version | **3.5.7** (openssl 3.5 LTS series) |
| upstream | `https://github.com/openssl/openssl/releases/download/openssl-3.5.7/openssl-3.5.7.tar.gz` |
| sha256 | `a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8` |
| sha256 provenance | upstream's own published `openssl-3.5.7.tar.gz.sha256` alongside the release asset; independently recomputed on the downloaded file |
| lands in | `vendor/thirdparty/openssl-3.5.7.tar.gz` (gitignored; acquired, not committed) |
| acquired by | `tools/vendor/acquire-thirdparty.sh` |
| verified by | SHA256 at acquisition **and again on every build** (`configure`, `catboostr_verify_tarball`) |
| built as | static `libssl.a` + `libcrypto.a`, `-fPIC`, `no-shared no-tests no-docs no-zlib no-apps` |
| consumed by | `openssl::openssl`, defined in `cmake/openssl-target.cmake` |

**Why it is in the link at all.** It was proposed that
`catboost/private/libs/distributed` be gated out so openssl would drop from
the graph. That premise is false and the decision is closed
(`catboost-8z4.9`, superseded): `catboost/libs/train_lib/train_model.cpp:18-19`
hard-includes `<catboost/private/libs/distributed/master.h>` and
`.../worker.h`, and `:1099-1102` constructs `TMasterContext`
unconditionally. The coupling is source-level, so severing it would require
editing the read-only pin. openssl is therefore **accepted** into the R1 link
and pinned instead of removed.

> **STANDING OBLIGATION — openssl CVE tracking.**
>
> Because openssl 3.5.7 is linked **statically** into `libcatboostr.so`, users
> do not get openssl security fixes from their distribution. Every openssl CVE
> that affects 3.5.x is a CVE against this package until the pin is moved.
>
> * **Watch:** <https://openssl-library.org/news/vulnerabilities/> and the
>   `openssl-announce` list, for the **3.5 LTS** series.
> * **Cadence:** check on every release of this package, and within 7 days of
>   any openssl advisory rated MODERATE or higher.
> * **Action on a hit:** bump `OPENSSL_VERSION`/`OPENSSL_SHA256` in
>   `tools/vendor/acquire-thirdparty.sh` **and** in `configure` (§3b — the two
>   are deliberately duplicated so `configure` runs standalone from an
>   unpacked tarball), refresh this section, and ship. Do not wait for the
>   next feature release.
> * **Series choice:** 3.5 is an LTS series. Upstream's own
>   `conanfile.py:16` asks for `openssl/3.0.15`, which is both older than the
>   3.0 series' final patch level and on a series nearing end of life; this
>   package deliberately pins **newer** than upstream. That divergence is
>   intentional and is the reason `conan.lock` and this file do not agree on
>   the openssl version.

### 1.3 zlib — NOT vendored, NOT needed

Measured, not assumed:

* `find_package(ZLIB)` appears **nowhere** in the pinned tree (grep over all
  `CMakeLists*.txt` and `cmake/*.cmake`). CatBoost carries its own
  `contrib/libs/zlib` and compiles it in-tree.
* zlib appeared in the Conan graph only as a transitive requirement of the
  Conan `openssl` recipe (`docs/phase-0/catboost-8z4.1-report.md`: "zlib
  shared transitive dep both `openssl` requirement `tool_requires`").
  Configuring openssl with `no-zlib` removes that edge entirely.
* R itself does link zlib and exposes it — `$(R_HOME)/etc/Makeconf` has
  `LIBS = -lpcre2-8 -ldeflate -lzstd -llzma -lbz2 -lz -ltirpc ...` — so an
  R-supplied `-lz` **was** available had it been needed. It is not needed, so
  it is not used. Nothing here depends on R's zlib.

**Net: zero zlib rows in this inventory.**

---

## 2. Build tools that generate code compiled into the shared object

These are not linked, but they emit source that is. Treating them as
untrusted-but-unpinned would leave exactly the same hole as an unpinned
library.

| tool | version | upstream | sha256 |
| :--- | :--- | :--- | :--- |
| ragel | 6.10 | `https://www.colm.net/files/ragel/ragel-6.10.tar.gz` | `5f156edb65d20b856d638dd9ee2dfb43285914d9aa2b6ec779dac0270cd56c3f` |
| yasm | 1.3.0 | `https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz` | `3dce6601b495f5b3d45b59f7d2492a340ee7e84b5beca17e48f862502bd5603f` |

* acquired by `tools/vendor/acquire-thirdparty.sh` into `vendor/thirdparty/`
* SHA256 verified at acquisition **and again on every build** (`configure` §3b)
* built from source during `configure`, then placed at
  `${CMAKE_BINARY_DIR}/bin/{ragel,yasm}` — the one hardcoded path the pinned
  tree looks at (`vendor/catboost/cmake/common.cmake:26` and `:43`)
* **what they generate:** `ragel` compiles
  `util/datetime/parser.rl6` (`util/CMakeLists.linux-x86_64.txt:589`
  `target_ragel_lexers(yutil ...)`) and `yasm` assembles
  `util/system/context_x86.asm` (`:595` `target_yasm_source(yutil ...)`).
  `yutil` is linked unconditionally by `catboostr`, so both run on every
  build — there is no configuration in which they are optional.
* **not declared as `SystemRequirements`** on purpose: ragel 6.10 in
  particular is absent from current mainstream distribution repositories, so
  a `SystemRequirements` declaration would convert a ~7-second source build
  into an install that fails on most machines.

### 2.1 swig — deliberately absent

`vendor/catboost/conanfile.py:20` declares `tool_requires("swig/4.0.2")`, so
it belongs in this discussion. It is **not** acquired, pinned, or built,
because it is never invoked: the only `find_package(SWIG ...)` call sites in
the pinned tree are the ten
`catboost/spark/catboost4j-spark/core/src/native_impl/CMakeLists.*.txt:17`,
and `-DCATBOOST_COMPONENTS=R-package-fork` never configures `catboost/spark`.
If that component list ever changes, swig must be added here.

---

## 3. Conan — maintainer-side only, not in the install path

Conan resolved openssl, ragel, swig and yasm upstream
(`vendor/catboost/conanfile.py:16,19,20,21`) by contacting a remote at build
time with no integrity pin this repository could audit. **It has been removed
from the install path entirely.** `R CMD INSTALL` never invokes `conan`, never
reads a Conan cache, and never needs a Conan remote to be reachable.

`conan.lock` (repository root) is still committed and still maintained. Its
job is auditing, not building: it records what upstream's own `conanfile.py`
resolves to, so that drift between upstream's dependency intent and what this
package actually ships is visible in a diff rather than invisible.

| field | value |
| :--- | :--- |
| file | `conan.lock` (committed) |
| generated from | `vendor/catboost/conanfile.py` |
| generated/used by | `tools/vendor/conan-lock.sh` (`verify` / `update`), always with `--lockfile` |
| locked `requires` | `openssl/3.0.15`, `zlib/1.3.2` |
| locked `build_requires` | `ragel/6.10`, `swig/4.0.2`, `yasm/1.3.0`, `pcre/8.45`, `m4/1.4.19`, `gnu-config/cci.20210814`, `flex/2.6.4`, `bzip2/1.0.8`, `bison/3.8.2`, `automake/1.16.5`, `autoconf/2.71`, `zlib/1.3.2` |

Each entry carries Conan's own recipe revision hash (e.g.
`openssl/3.0.15#1232bb6e17f82566842fcfc34bb7259c`), which is what makes the
lock an integrity record and not just a version list.

Note the deliberate divergence already flagged in §1.2: the lock says
`openssl/3.0.15` because that is what **upstream** asks for; this package
links **3.5.7**. The lock is evidence about upstream, not a description of
what ships.

---

## 4. Test-oracle artifacts (not part of the shared object)

These never reach the compiler. They are pinned because the test suite's
verdicts are only meaningful if the reference implementation is fixed.

| artifact | version | source | integrity |
| :--- | :--- | :--- | :--- |
| CatBoost CLI binary | `v1.2.10`, `catboost-linux-x86_64-1.2.10` | GitHub Release asset `https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-linux-x86_64-1.2.10` | sha256 `478dc57f4c19de205b19b709fd4c6af93f79753dc462b6bb49d33c920dddec75` **and** exact size `287580072` bytes, both fail-closed (`tools/oracle/cli/acquire.sh:39-48`) |
| Python oracle env | `catboost==1.2.10` on Python `==3.14.6` | PyPI, via `uv` | `tools/oracle/uv.lock` — 19 locked packages carrying **192** `sha256:` artifact hashes; `tools/oracle/pyproject.toml` pins the direct requirement |

---

## 5. What a maintainer runs, in order

```sh
tools/vendor/acquire.sh              # pinned CatBoost snapshot   -> vendor/catboost/
tools/vendor/acquire-thirdparty.sh   # openssl, ragel, yasm       -> vendor/thirdparty/
tools/oracle/cli/acquire.sh          # pinned CLI oracle          -> tools/oracle/cli/bin/
tools/vendor/conan-lock.sh verify    # audit upstream drift (needs conan; optional)
```

All four are re-runnable and refuse to overwrite an artifact whose checksum
does not match. After they have run once, `R CMD INSTALL` needs no network.
