# Plan: Phase 6 follow-ups (catboost-8z4.95-.99 + catboost-upw)

**Mechanism:** Execute 5 already-hermetic, already-validated beads tickets (filed after Phase 6's final whole-branch review) via subagent-driven-development, same process as Phase 6 itself. No new planning content needed per ticket — each is self-contained. This doc covers only what a single ticket can't: scope of `catboost-upw`, and execution order.

**Forbidden:** Non-API R interrupt hacks (`Rf_onintr()`, `R_interrupts_pending`) — CRAN `R CMD check` NOTE risk, a release blocker. No new abstractions beyond what each ticket already specifies (already ponytail-scoped at creation).

**Audit:** Each ticket's own DoD (full test suite green, no `vendor/` edits, behavior-preserving where stated). No new audit mechanism needed.

## Alternatives considered (catboost-upw only)

`catboost-upw` (Ctrl-C during a bridged run surfaces as an R error, not an interrupt) lists 3 options in its own body:
- **(a) accept and document at R level** — this IS catboost-8z4.96's exact scope (add the deviation to `catboost.train`'s user-facing docs). No new ticket needed; catboost-8z4.96's implementation closes catboost-upw too.
- **(b) R-level re-signal from the `.Call` wrapper** — rejected: requires designing a new interrupt-condition-signaling mechanism not yet specified anywhere; real engineering scope, not a follow-up cleanup item.
- **(c) non-API `Rf_onintr()` at the outermost frame** — rejected: risks a CRAN `R CMD check` NOTE (release blocker per spec), and the ticket's own text says "blocks nothing yet."

Selected: (a), via catboost-8z4.96. (b)/(c) left open in catboost-upw's body for a future ticket if real interrupt semantics are ever required — not created speculatively here (YAGNI).

## Execution order

All 6 items are independent (no code dependency between them), sequenced only to reduce file-churn collisions and rebuild cost:

1. `catboost-8z4.98` — duplicate-params-key detection (`R/catboost.R`, R-only)
2. `catboost-8z4.97` — helper-call ordering fix (`R/catboost.R`, R-only, same file as .98 — sequential, not parallel, per SDD's "never dispatch multiple implementers in parallel")
3. `catboost-8z4.96` — interrupt-deviation docs, resolves `catboost-upw` (`R/catboost.R`, R-only, same file — do after .97/.98 so line-number references in the dispatch are current). This ticket's own DoD (v2, re-validated) mandates step 5: `bd comment catboost-upw "..."` + `bd close catboost-upw` after committing the docs change -- the closure is a concrete, mechanized step inside .96's own hermetic body, not an unenforced side effect of this plan doc.
4. `catboost-8z4.99` — test fixture dedup (`tests/testthat/*`, independent of R/catboost.R edits above)
5. `catboost-8z4.95` — C++ `.Call`-site dedupe (`src/catboostr.cpp`) — LAST: the only ticket requiring `--preclean` (~15 min vendor rebuild); the other four only need a fast R-only `R CMD INSTALL .`

Each ticket: dispatch implementer -> task review -> fix loop (same SDD process as Phase 6). Full suite must stay green after each (`[ FAIL 0 | WARN 0 | SKIP 2 | PASS 816 ]` baseline, expect PASS count to grow only where a ticket's DoD adds a test — .97 and .98 each add one regression test).

`catboost-upw`'s closure is step 5 of `catboost-8z4.96`'s own DoD (mandatory `bd comment` + `bd close`), verified as part of that task's review, not left to a plan-level assertion.
