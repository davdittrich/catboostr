# Phase 3 — final review fix wave 2

Scope: four residual items from the whole-branch review that were not covered by
wave 1 (`docs/phase-3/final-review-fix-wave-report.md`): a broken Rd unicode escape,
inconsistent guard brace-style in `catboost.load_pool`/`catboost.from_matrix`,
ticket-ID leakage into a user-facing error message, and recording the wave-2 plan.

Branch: `phase-3-pool-parity`.

---

## 1. `\U0001F522` rendered as a literal escape string in `catboost.Tokenizer.Rd` — FIXED

### Bug mechanism

`R/text_processing.R`'s `@param number_token` roxygen doc used the R-level Unicode
escape syntax `"\U0001F522"` inside a plain doc string. Roxygen2 copies doc text into
the `.Rd` file verbatim; it does not evaluate R string escapes. The generated
`man/catboost.Tokenizer.Rd` therefore contained the eight literal characters
`\U0001F522` instead of the intended 🔢 (U+1F522, KEYCAP DIGIT ZERO / "input symbol for
numbers") character, and `Rd2txt()`/`?catboost.Tokenizer` displayed the escape text
rather than the emoji that matches the vendor default.

### Fix mechanism

Replaced the escape sequence with the literal UTF-8 character in both the roxygen
source (`R/text_processing.R:50`) and the generated `man/catboost.Tokenizer.Rd:31`, so
Rd2txt renders the real glyph. No other `\Uxxxxxxxx`-style escapes exist elsewhere in
the package's roxygen docs (checked via search across `R/`).

### Verification

    Rscript -e 'library(tools); rd <- Rd_db("catboostr")[["catboost.Tokenizer.Rd"]]; Rd2txt(rd)'

Output line for `number_token` now reads:

    Default value: "🔢" (matches vendor default)

not the literal `\U0001F522` string. See full command output below.

## 2. Inconsistent guard brace-style around `feature_tags` rejection — FIXED

### Bug mechanism

`catboost-8z4.49` added the same `if (!is.null(feature_tags)) stop(...)` guard to both
`catboost.load_pool` and `catboost.from_matrix`, but with different brace styles: one
used `{ }`, the other a bare single-statement `if`. Both call `stop()` with the same
ticket-ID string appended (see item 3), so the two copies had drifted in two
independent ways.

### Fix mechanism

Both guards now use the same brace-delimited `if { ... }` form:

- `R/catboost.R:112-115` (`catboost.load_pool`)
- `R/catboost.R:199-201` (`catboost.from_matrix`)

matching the brace style used by the file's other multi-context guards (e.g. the
`pairs`/`data` type check immediately below in `catboost.load_pool`).

## 3. `catboost-8z4.49` ticket ID leaked into user-facing `stop()` messages — FIXED

### Bug mechanism

Both `feature_tags` guards raised errors of the form:

    stop("feature_tags is not currently supported by catboostr; tracked as catboost-8z4.49")

An internal issue-tracker ID in a user-facing error message is meaningless to an
`catboostr` user (the ID refers to a ticket in this repo's private `bd` database, not
anything resolvable from a CRAN install) and would go stale the moment the ticket is
closed or renumbered.

### Fix mechanism

The ticket ID was moved out of the message text and into a `# catboost-8z4.49` code
comment directly above each guard, so the traceability for future maintainers is
preserved without exposing it to end users:

    if (!is.null(feature_tags)) {
        # catboost-8z4.49
        stop("feature_tags is not currently supported by catboostr")
    }

Applied identically to both call sites (`R/catboost.R:112-115`, `R/catboost.R:199-201`).

## 4. Wave-2 fix plan recorded

Added `docs/superpowers/plans/2026-08-03-phase3-parity-debt.md`, the `planning-with-beads`
plan document covering this fix wave's scope and gate approvals.

---

## Verification

### Step 1 — full regression suite

    cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'

Tail of output:

    bestTest = 0.6477622102
    bestIteration = 9

    Custom logger is already specified. Specify more than one logger at same time is not thread safe.Training...
    Did not manage to build 5 units!
    [ FAIL 0 | WARN 0 | SKIP 1 | PASS 301 ]

`FAIL 0 | WARN 0 | SKIP 1 | PASS 301` — the single skip is the pre-existing
`test_caret_parameter_tuning.R:35:3` (`{caret} is not installed.`), unrelated to this
wave.

### Step 2 — Rd render check

    Rscript -e 'library(tools); rd <- Rd_db("catboostr")[["catboost.Tokenizer.Rd"]]; Rd2txt(rd)'

Relevant excerpt of the rendered text:

    number_token: Replacement token used when 'number_process_policy' is
              "Replace".

              Default value: "🔢" (matches vendor default)

Confirms the real 🔢 character renders, not the literal `\U0001F522` string.

### Step 3 — vendor tree untouched

    git -C vendor/catboost status --porcelain

Output: empty (no changes under `vendor/catboost/`).

## Not in scope for this wave

Everything already covered by wave 1
(`docs/phase-3/final-review-fix-wave-report.md`) and by tickets `catboost-8z4.48` /
`catboost-8z4.49` themselves — this wave only addresses the four residual review
findings listed above.
