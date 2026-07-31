# catboost-8z4.32 — Research: how does `vendor/catboost` reach an end user / CRAN?

Date: 2026-07-31 · Repo `/home/dd/Gemini/catboost`, branch `phase-0`
Status: RESEARCH ONLY. Nothing implemented, nothing changed outside this file.

---

## 0. One-paragraph result

The current architecture — `vendor/catboost` external, gitignored, excluded by
`.Rbuildignore`, `configure` requiring it to pre-exist — was **never decided**.
Every spec section and every ticket that speaks to the question assumes the
**opposite**: that a pruned subset of the upstream C++ source physically ships
inside the CRAN source tarball, xgboost/lightgbm-style. The 16.078 MiB figure
that the spec's Phase-1a size gate, §7's top risk row and P1.14 (`catboost-8z4.24`)
are all built on **is a measurement of exactly that shipped-source tarball**.
P1.6's own ticket text even instructed the executor to "state clearly how the
pruned tree reaches the tarball" — the executor instead reinterpreted the prune
script as a build-input integrity tool and wrote the no-ship assumption into
`tools/vendor/prune.sh`'s header comment. The gap P1.6 §8 flags is therefore not
an unsolved design problem; it is **drift away from a design that already
existed**. Confidence: **92**.

---

## 1. What `catboost-8z4.24` (P1.14) actually says

Read via `bd show catboost-8z4.24 --json`. Verbatim, load-bearing excerpts:

> **Objective:** Quantify how much vendored source is never compiled, and draft
> the `cran-comments.md` justification for the current tarball size.

> The measured tarball is 16.078 MiB against CRAN policy's 10MB — ~1.7x, not the
> 3.2x an earlier draft of the spec claimed against a wrong 5 MiB figure. Not
> disqualifying, but worth reducing. Comparable packages prove the headroom:
> `xgboost` ships at 1.5M and `lightgbm` at 1.7M, both vendoring C++ ML cores,
> and they get there by amalgamating and stripping unused source rather than by
> linking system libraries. P1.2's directory gating is a ~20% cut; this ticket
> pursues the rest.

> The thin-package alternative is REFUTED (spec 6) — no `libcatboost` development
> package exists in Debian, Ubuntu, Fedora, Homebrew, conda-forge or vcpkg, so
> `SystemRequirements` would leave the package uninstallable on CRAN's own check
> machines. There is no cheaper packaging route to fall back on; the only lever
> is a smaller tarball.

> CRAN precedent for large vendored tarballs is real: `rcdklibs` 19M,
> `fastrmodels` 16M, `acss.data` 14M are live.

The ticket is coherent **only** under the reading "the vendored C++ source ships
inside the tarball". "The only lever is a smaller tarball" is meaningless if the
C++ source is not in the tarball at all — under the current architecture the
tarball is a few hundred KiB of R code and the size gate is trivially passed.
Confidence: **95**.

---

## 2. Where 16.078 MiB came from, and what it measured

Source: **`docs/phase-1/catboost-8z4.6-report.md`** (Phase-1 spike
`catboost-8z4.6`, "measure the prune set and CRAN tarball size"), §"Size
measurements":

| Quantity | Bytes | MiB |
|---|---|---|
| Full disposable-copy tree (`vendor/catboost` minus `.git`) | 912,454,928 | 870.185 |
| Pure compile + header closure (7819 files) | 78,682,869 | 75.045 |
| Final pruned tree, uncompressed (13,371 files) | 99,511,153 | 94.901 |
| **Final pruned tarball, compressed (`tar czf`)** | **16,858,704** | **16.078** |

Method, quoted from the report's step 11: *"Measured final pruned tree size
(`du -sb`) and built the CRAN-relevant compressed tarball (`tar czf`)."* The
machine-readable block records it as `pruned_tarball_bytes: 16858704`, and the
verdict field is keyed `cran_size_viable`.

So the measured object is **a `tar czf` of the 13,371-file pruned upstream C++
tree** — the source that must be compiled at install time. It was explicitly
labelled "the CRAN-relevant compressed tarball", i.e. the spike's whole purpose
was to answer "is the vendored source small enough to ship on CRAN". It was *not*
a measurement of a `vendor/` copy-out unrelated to the R tarball. Confidence: **93**
(the report never literally writes "this tree goes inside the R package
directory", but its stated objective, its `cran_size_viable` verdict key and the
spec's consumption of the number leave no other reading).

Caveat worth recording: the spike tarballed the **pruned CatBoost tree alone**,
not an actual `R CMD build` output of `catboostr` with that tree nested inside
`src/`. A real R tarball would add the R/man/inst/tests layers (small) and, more
importantly, would be subject to `R CMD build`'s own filtering and to the
portability checks discussed in §6. So 16.078 MiB is a good estimate of the
shipped size, not a produced artifact.

## 2b. The spike's prune set is NOT what `tools/vendor/prune.sh` produces

Two different, non-interchangeable prune sets exist in this repo:

| | spike (`catboost-8z4.6`) | committed `tools/vendor/prune.sh` (P1.6) |
|---|---|---|
| Method | compile-closure derived (`compile_commands.json` + `ninja -t deps` + reactive build-failure fixes) | delete 61 named top-level paths (44 CMake-gated test dirs + 17 never-referenced dirs + `.git`) |
| Files | 13,371 | 21,888 |
| Uncompressed | 94.9 MiB | 306 MiB |
| Reproducible? | **No** — never scripted or committed; existed only in a scratch dir | Yes, deterministic, manifest committed |
| Validated by a real build? | Yes (`ninja catboostr`, rc=0, real `libcatboostr.so`) | Not rebuilt; it is a superset of the gated set so it should build |

The 16.078 MiB number therefore has **no committed, reproducible producer**.
Anything that ships "the pruned tree" today would ship the 306 MiB / 21,888-file
version, which compresses to substantially more than 16 MiB. Confidence: **95**.

---

## 3. The current architecture, as built

* `.gitignore:11` → `/vendor/`; `.Rbuildignore:21-22` → `^vendor$`, `^vendor/`.
* `configure:117-125` — `VENDOR_SRC="${CATBOOSTR_VENDOR_SRC:-${PKG_ROOT}/vendor/catboost}"`,
  and if absent: print an error naming `tools/vendor/acquire.sh` and `exit 1`.
* `configure:188` — the same pattern for `THIRDPARTY_SRC` (`vendor/thirdparty`,
  openssl/ragel/yasm), also gitignored, also not shipped. **The install path
  therefore depends on TWO externally-acquired, unshipped directories, not one.**
* `docs/phase-1/P1.6-report.md` §4.1: a real `R CMD check --as-cran` on the
  actual tarball fails with
  `*** .../00_pkg_src/catboostr/vendor/catboost not found -- this fork requires the pinned vendor/catboost snapshot. ERROR: configuration failed`.
* §4.2: symlinking `vendor/` back in makes install succeed but produces
  `checking for portable file names ... WARNING` and `Found non-portable file
  paths ... ERROR`, because `R CMD check` statically scans every file physically
  present under the package directory.
* §4.3: the shipped fix is the `CATBOOSTR_VENDOR_SRC` / `CATBOOSTR_THIRDPARTY_SRC`
  env-var overrides, pointed at a checkout outside the package tree. P1.6 itself
  scores **60** confidence "on whether CRAN's own build farm reach same 0-ERROR
  state without equivalent env-var pointed pre-acquired `vendor/catboost` —
  CRAN's farm has no mechanism set it".
* §8, verbatim: *"How does a real end user (or CRAN's build farm) obtain
  `vendor/catboost` (~870 MiB) with zero network access at install time...? This
  was never solved by any prior ticket (P1.4's `configure` simply assumed
  `vendor/catboost` exists at `${PKG_ROOT}/vendor/catboost`) and is not solved
  here."*

Net effect today: `install.packages("catboostr")` from a CRAN mirror **cannot
work**. Only a developer who has run `tools/vendor/acquire.sh` (network,
~870 MiB) and `tools/vendor/acquire-thirdparty.sh` can install. That directly
contradicts the epic's own success criterion ("installs from a clean checkout
with zero network access during install") in the only sense that matters to an
end user.

---

## 4. Original design intent (spec), quoted

`docs/superpowers/specs/2026-07-30-catboostr-design.md`:

**§4.1, "Modes":**
> **`vendored`** (default; release, r-universe, CRAN): builds the pinned core
> with no network access at any point during installation.

**§4.1b, "CRAN size — CONDITIONALLY viable":**
> 3902 unique translation units. Pruned tarball **16.078 MiB** (16,858,704 bytes).
> ... The CRAN Repository Policy says source tarballs "should if possible not
> exceed **10MB**", and packages well above it are live on CRAN today
> (`rcdklibs` 19M, `fastrmodels` 16M, `acss.data` 14M). The real overshoot is
> ~1.7×, not 3.2×, and the mechanism for exceeding the figure is a justification
> in `cran-comments.md`, which policy explicitly contemplates for C++ and Rust
> packages.

**§5, gate 1a:**
> **Size gate. MEASURED 2026-07-31: 16.078 MiB pruned** ... `xgboost` (1.5M) and
> `lightgbm` (1.7M) vendor comparable C++ ML cores and stay small by
> **amalgamating and stripping unused source** ... A re-measurement after pruning
> plus a drafted `cran-comments.md` justification is the Phase 1 exit criterion.

**§6, Alternatives Considered, thin-package row:**
> CRAN policy states software is installed on its Debian check machines only when
> it is available from Debian repositories for 'testing', and that **bundling or
> vendoring sources is usually the faster path**. A `SystemRequirements:
> libcatboost` would therefore leave the package **uninstallable on CRAN's own
> check machines** — a harder failure than an oversized tarball, because it fails
> at build rather than at review.

**§7, first risk row:**
> **Vendored source tarball ~1.7× the CRAN policy figure** | MEASURED at 16.078 MiB
> pruned ... Requires a written justification at submission.

**§9.0, abandonment trigger:**
> Phase 1a pruned tarball exceeds 30 MB | Escalate to the user.

Every one of these sentences presupposes that the vendored C++ source is
**inside the distributed artifact**. A tarball that does not contain the core has
no size risk, needs no `cran-comments.md` justification, and cannot trip a 30 MB
abandonment trigger. Confidence: **95**.

---

## 5. Was "external and gitignored" ever deliberately chosen? — No.

I searched all 22 closed tickets (`bd list --status closed --json`) plus the spec
plus every report in `docs/phase-0/` and `docs/phase-1/` for any deliberate
decision that `vendor/catboost` stays outside the shipped artifact.

**Findings:**

1. **No ticket, spec section, or report ever states that decision.** The only
   two occurrences of the claim in the repo are *outputs* of P1.6, not inputs:
   * `tools/vendor/prune.sh:6-11` — "This is a BUILD-INPUT determinism/integrity
     tool, not a second exclusion mechanism for the R source tarball:
     vendor/catboost is NEVER part of the tarball".
   * `docs/phase-1/P1.6-report.md` §6 — the same claim, restated.

2. **P1.6's own ticket said the opposite.** Verbatim from
   `bd show catboost-8z4.14`:
   > `.Rbuildignore` must exclude `docs/`, `tools/`, ... and — **critically —
   > `vendor/` and `.git/`**. `vendor/catboost` is ~870 MiB; omitting it from the
   > denylist ships the entire pinned upstream monorepo. ... **NOTE: the shipped
   > source comes from the PRUNE SCRIPT's output, not from `vendor/` directly —
   > state clearly how the pruned tree reaches the tarball without `vendor/`
   > itself being included.**

   That is the whole design in one sentence: exclude the raw 870 MiB pin, ship
   the prune script's output. The executor answered a different question —
   it declared prune.sh an integrity tool, never wired its output into the
   tarball, and recorded the unanswered part honestly in §8 as an open gap. The
   `.Rbuildignore ^vendor/` line is correct and required under *both* designs;
   what is missing is the second half (a shipped `src/`-side copy of the pruned
   tree). Confidence: **93**.

3. **P1.4 (`catboost-8z4.12`) never addressed source acquisition.** Its ticket
   text covers deleting the download route, `CATBOOST_DYNLIB` disposition,
   `R CMD config` forwarding, generator selection, the disposable-copy rule.
   `${PKG_ROOT}/vendor/catboost` was inherited from the Phase-0
   `tools/vendor/acquire.sh` developer workflow and simply carried forward.
   P1.6 §8 states this directly: *"P1.4's `configure` simply assumed
   `vendor/catboost` exists"*.

4. **`/vendor/` in `.gitignore` is a Phase-0 developer-workflow artifact**
   (acquire.sh writes an 870 MiB working copy including its own `.git`; not
   committing that is obviously right). It was never a distribution decision.

**Conclusion: accidental side effect, not a considered decision.** The one
deliberate, evidenced decision in this area — P1.6's ticket instruction to ship
the prune script's output — was not executed. Confidence: **92**.

---

## 6. Does the xgboost / lightgbm precedent actually hold? — Yes for the shape, no for the mechanism

Web-sourced (labelled as such; repo evidence is §§1-5):

**lightgbm.** `build-cran-package.sh` at the repo root assembles a temporary
R-package tree, copies the C++ core into `R-package/src` —
`cp -R include "${TEMP_R_DIR}/src/"`, `cp -R src/* "${TEMP_R_DIR}/src/"` — plus a
hand-curated subset of Eigen (only `Cholesky Core Dense Eigenvalues Geometry
Householder Jacobi LU QR SVD`, explicitly "to keep the R-package small and avoid
redistributing code with licenses incompatible with LightGBM's license"), strips
`inst/`, `pkgdown/`, CLI-only sources and CRAN-forbidden `#pragma` lines with
`sed`, then runs `R CMD build`. The C++ source **is** physically in the CRAN
tarball. Critically: **lightgbm does not use CMake in the CRAN package at all.**
Its own docs say "Because CRAN packages typically do not assume the presence of
CMake, the R package uses an alternative method that is in the CRAN-supported
toolchain: Autoconf." Since v3.0.0 the "build the C++ lib first, then wrap it"
model was **removed**.

**xgboost.** Same shape: the git repo uses submodules and CMake, and the CRAN
tarball is a flattened, submodule-resolved copy of the C++ sources under
`src/`, compiled by standard `configure` + `Makevars`. Historically produced by
a `make Rpack` packaging step; now by CMake's R-package install target. Again the
C++ source is inside the tarball; again the *installing user* needs no CMake.

**What this means for catboostr:**

* The **"ship the core inside the tarball"** half of the precedent is real,
  standard, and directly applicable. P1.14's framing is sound on that point.
* The **"1.5M / 1.7M"** half is not transferable. Those numbers come from cores
  of a few hundred source files with no code generation. CatBoost's R-package
  build is 3,902 TUs and requires build-time code generation by **protoc,
  flatbuffers, Ragel, yasm and Python** (`__vcs_version__.c`, the linker version
  script), plus 4,627 machine-generated `CMakeLists.txt` files whose absence
  breaks configure (spike §"Structural findings" 1-4). Reaching xgboost/lightgbm
  size would mean doing what lightgbm did — **abandoning CMake for
  Autoconf+Makevars** — which for CatBoost additionally means reproducing five
  code generators inside an R `Makevars`. That is a multi-month rewrite of
  upstream's build, against a read-only pin that §8 forbids editing.
  Confidence: **85**.

---

## 7. Candidate mechanisms

### (a) Ship a pruned core inside the R source tarball — RECOMMENDED shape

The spec's original intent, P1.14's framing, and the xgboost/lightgbm precedent.
`configure` keeps its current CMake-driven model; `VENDOR_SRC` simply defaults to
an in-package path (e.g. `${PKG_ROOT}/src/catboost-core`) instead of
`${PKG_ROOT}/vendor/catboost`.

Work required, all currently missing:

1. **A reproducible producer for the 16 MiB set.** The spike's 13,371-file
   closure was never scripted. `tools/vendor/prune.sh` today emits 21,888 files /
   306 MiB — far too large. Someone must re-derive the compile closure as a
   committed script, and re-validate it with a real `ninja catboostr` build
   (the spike proved configure + dry-run are insufficient: the Ragel include
   chain only failed at real build time).
2. **Long-path remediation.** Measured directly from the committed
   `tools/vendor/PRUNED_MANIFEST.sha256` (21,888 paths, read-only analysis):

   | metric | value |
   |---|---|
   | non-ASCII filenames in the pruned tree | **0** |
   | paths with spaces | **0** |
   | paths > 100 chars, whole pruned tree | 515 |
   | paths > 100 chars, excluding `python-package` / `spark` / `canondata` | **41** (25 `contrib/libs/nvidia`, 12 `contrib/restricted/boost`, 4 other) |
   | same, once prefixed `catboostr/` (10 ch) | 174 |
   | same, once prefixed `catboostr/src/` (14 ch) | **378** |
   | same, with a 20-char prefix | 879 |
   | longest path | 197 ch (`catboost/python-package/ut/medium/canondata/...`) |

   Good news: the P1.6 ERROR was driven by `.git`, `.github`, `tutorials/` and a
   `.pptx` with spaces and non-ASCII — **none of which survive pruning**. Bad
   news: `R CMD check`'s non-portable-path threshold applies to the path *as it
   appears under the package directory*, so nesting depth is decisive. At a
   `catboostr/src/` prefix, 378 paths exceed 100 chars — a real ERROR that must
   be fixed by shortening the nesting (e.g. mount the core at `src/cb/`), by
   dropping the nvidia/CUDA and boost-math offenders if they are outside the
   no-CUDA compile closure, or by both. This is tractable, not fatal.
   Confidence: **88** on the numbers (computed from a committed manifest),
   **70** on "tractable" (untested).
3. **Re-run `R CMD check --as-cran`** against the new tarball. P1.6's 0-ERROR
   result was obtained with `vendor/` absent *and* env vars pointing outside the
   tree; every portability/licence/size check must be redone with ~13k upstream
   files physically present. Expect new NOTEs (installed size, subdirectories
   > 1 MB) and possibly new WARNINGs on upstream source (`#pragma` suppressions
   — the exact thing lightgbm strips with `sed`).
4. **`vendor/thirdparty` (openssl / ragel / yasm) has the identical problem** and
   is not covered by the 16.078 MiB figure at all. Either those sources ship too
   (adding several MiB and a GPL-2.0 Ragel redistribution question that
   `inst/COPYRIGHTS` currently sidesteps precisely because Ragel is *not*
   shipped), or `configure` must build without them. **This is a second,
   unquantified hole and it is not in any ticket.** Confidence: **90** that the
   hole is real (`configure:188-228` fails closed if `vendor/thirdparty` is
   absent).
5. `SystemRequirements` still needs CMake + Python3 + a C++20 compiler. That is
   permitted, but it is strictly worse than lightgbm's autoconf route and is a
   standing CRAN review risk.

Trade-offs: restores the epic's success criterion for real; keeps the verified
CMake build; ~16-20 MiB tarball needing a `cran-comments.md` justification
(precedented: `rcdklibs` 19M, `fastrmodels` 16M live on CRAN); adds the
open build-time risk (3,902 TUs on a 2-core CRAN runner — gate 1c /
`catboost-8z4.17` still has no measurement, and build time is the *other*
structural rejection cause).

### (b) Full lightgbm-style flattening (autoconf + `Makevars`, no CMake)

The only route to 1.5-2 MiB. Requires reproducing protoc, flatbuffers, Ragel,
yasm and Python codegen plus 23 whole-archive link groups inside R's own build
machinery. The spec already rejected the closest analogue ("R's own `Makevars`
compiling the glue and linking CMake-built static libraries" — §4.1, rejected
because whole-archive semantics for 28 `.global` archives are a runtime-failure
trap). Not viable in Phase 1. Record as a long-horizon option only.

### (c) Network fetch at `configure`- or `.onLoad`-time

What upstream CatBoost's own `configure` does (`configure:1793`, deleted by P1.4).
Disqualified twice over: it violates the epic's explicit "zero network access
during install", and it is the exact mechanism P1.4 was written to remove. A
`.onLoad`/user-invoked post-install fetch (the `keras3`/`tensorflow` pattern the
spec's §6 notes) is technically CRAN-acceptable but means the package does not
work after `install.packages()` alone — a product decision, not a packaging one.
Stated for completeness; **not recommended**.

### (d) Status quo, retargeted: r-universe / GitHub only, no CRAN

Keep `vendor/` external; document `tools/vendor/acquire.sh` as an install
prerequisite. This is what the repo *currently is*. It is not a CRAN package and
never becomes one. The spec's §9.0 already contemplates "Ship r-universe only" —
but only as a response to two structural CRAN rejections, not as a starting
position. Note that r-universe builds from a git checkout and would also need
`vendor/` present, so even this route needs a submodule or an acquire step in CI.

### (e) Split into a companion source package (`catboostr.core`)

The `BH` / `rcdklibs` model: a second CRAN package containing only the C++
source, with `catboostr` declaring `LinkingTo`/`Depends`. Does not reduce total
bytes and does not avoid the portability scan — it just moves both to a package
whose only content is upstream source. Adds a second submission and a version-
lockstep obligation. No advantage over (a) here, because CatBoost is not a
header-only library. Not recommended.

---

## 8. Recommendation

**Escalate to the user before any implementation.** Mechanism (a) is clearly the
right shape and it is what the spec always said — but adopting it:

* reverses a claim now written into a committed, executed artifact
  (`tools/vendor/prune.sh:6-11`) and into `docs/phase-1/P1.6-report.md` §6;
* invalidates P1.6's headline result (0 ERRORs under `--as-cran`), which must be
  re-earned with ~13k upstream files physically inside the package;
* requires a new, scripted, build-validated prune producer that does not exist
  (P1.14 `catboost-8z4.24` is the natural home, but its current scope is
  explicitly "QUANTIFIES and JUSTIFIES … does not run an open-ended reduction
  programme" — that scope limit must be lifted or a new ticket opened);
* surfaces an unticketed second hole: `vendor/thirdparty` (openssl, ragel, yasm)
  has exactly the same acquisition problem and is not covered by any measurement
  or any ticket;
* leaves a genuine, unresolved CRAN risk (build time for 3,902 TUs on a 2-core
  runner) that gate 1c has never measured.

If the user confirms mechanism (a), the minimum ticket set is:

1. Script and build-validate the 13,371-file compile-closure prune (supersedes /
   extends `tools/vendor/prune.sh`; re-measure the tarball).
2. Relocate the core into the package tree at a **short** path and fix the
   378 over-100-char paths; re-run `R CMD check --as-cran` to 0 ERRORs.
3. Resolve `vendor/thirdparty` the same way (ship pinned sources, or eliminate
   the dependency) — including the GPL-2.0 Ragel redistribution question, which
   `inst/COPYRIGHTS` currently answers only on the premise that Ragel is *not*
   shipped.
4. Point `configure`'s `VENDOR_SRC` / `THIRDPARTY_SRC` defaults at the in-package
   paths, keeping the env-var overrides for development.
5. Only then re-open P1.14's size-reduction/justification work.

If the user prefers (d), the epic's success criterion and the spec's §4.1
`vendored` mode, §5 gate 1a, §7 risk row and §9.0 trigger all need rewriting to
match, and P1.14 should be closed as moot.

---

## 9. Housekeeping

* `git -C vendor/catboost status --porcelain` — **empty** (never touched; this
  ticket performed no implementation).
* Files written by this ticket: `docs/phase-1/catboost-8z4.32-research.md` only.
* Read-only analysis of `tools/vendor/PRUNED_MANIFEST.sha256` was done by copying
  path strings into the scratch dir; no repo file was modified.

Web-sourced material in §6 is labelled as such; everything in §§1-5 and §7's
measurements is repo-sourced and quoted.
