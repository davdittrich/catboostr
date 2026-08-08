# Active Plan
<!-- approved: 2026-08-08 -->
<!-- gate-iterations: 7 -->
<!-- user-approved: yes -->
<!-- status: in-progress -->

R1 first public release. Tickets (bd show <id> = full hermetic brief; no
separate plan doc):

Exec order:
1. catboost-8z4.110 -- fix .Rbuildignore-stripped test fixtures (both
   build-check jobs red)
2. catboost-8z4.111 -- fix hardcoded dev-machine paths in tests
   (darwin-prune-validation red)
3. catboost-8z4.107 -- generate docs/PARITY.md
4. catboost-8z4.108 -- release-readiness audit
5. catboost-8z4.106 -- README.md (blocked by .110, .111)
6. catboost-8z4.109 -- cut the tag + GitHub release (blocked by all above)

Worktree: .claude/worktrees/r1-release, branch worktree-r1-release,
branched from phase-0 tip aa5506e.

Plan review gate: APPROVED iteration 7 (Feasibility PASS, Completeness
PASS, Scope & Alignment PASS). Earlier iterations caught: a false "macOS
untested" README claim, an elided spec quote used to justify skipping
vignettes, and two real pre-existing CI bugs (.110, .111) that were
invisible to the local `Rscript testthat.R` command.
