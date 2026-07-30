# SDD ledger — plan: .beads/plans/active-plan.md

Phase 0 of catboost-8z4. Controller commits; implementers write files only.

BASE: d2aa27c (branch phase-0)
Dispatched 2026-07-30, parallel, sonnet:
  catboost-8z4.1 (build spike)   -> vendor/catboost/ + scratch
  catboost-8z4.2 (cpp11 probe)   -> external scratchpad only
  catboost-8z4.3 (python oracle) -> tools/oracle/, tests/fixtures/oracle/
  catboost-8z4.4 (cli oracle)    -> tools/oracle/cli/, tests/fixtures/oracle-cli/
Deviation from skill (recorded): implementers run no writing git commands;
controller commits. Reason: four concurrent agents in one worktree would
race on the git index. Write scopes verified disjoint by feasibility review.
catboost-8z4.3: implemented, committed 84d956d. Predictions bit-identical
  across fresh processes; .cbm bytes NOT stable (embedded timestamp) - parity
  tests must diff predictions, not model bytes. Task review dispatched.
catboost-8z4.4: complete (commit 9aea78b, review pending). Findings -> bd comment.
  SCOPE CHANGE: CLI has 15 modes not 8; 7 new CLI-only capabilities for parity matrix.
  CONSTRAINT: calc output ~10 sig digits, no --precision -> caps bit-exact CLI compare.
catboost-8z4.2: complete (probe external, no repo diff). Verdict cpp11_for_new.
  Findings -> bd comment. Cost: init.c table hand-synced per new cpp11 function.
catboost-8z4.1: fix round 1/5 - brief guard was over-restrictive (forbade ninja/conan
  entirely, blocking the measurement). Corrected: local venv allowed. Resumed implementer.
catboost-8z4.3: fix round 1/5 - train_dir unset leaks catboost_info/ to repo root;
  report self-cert inaccurate; wheel-list traceability. Resumed implementer.
  PARKED: reviewer's root-.gitignore finding - ruling: controller made those edits,
  not the implementer. No defect.
catboost-8z4.1: fix round 1 - implementer twice returned without waiting on its own
  build (attached a Monitor, then returned; a Monitor cannot re-invoke a completed
  agent). Two attempts on the same step = controller takes the wait. Build confirmed
  alive and progressing (3849/4265 targets), conan graph captured.
  KEY CORRECTION: the round-0 verdict "vendored build NOT tractable" was an artifact
  of the brief forbidding ninja/conan. With both pip-installed into a local venv the
  core builds. Verdict must be re-derived from the completed run, not from round 0.
catboost-8z4.2, catboost-8z4.4: task reviews dispatched (were closed prematurely
  before review; will reopen if findings land).
catboost-8z4.4: complete (commit 9aea78b, review APPROVED - reviewer independently
  re-ran --help and calc -?, reproduced both scope-change claims). Spec updated with
  the CLI precision ceiling. Minor deferred: smoke_params.json 'verbose' field is
  cosmetic-inconsistent with recorded cli_args.
catboost-8z4.2: complete (no repo diff, review APPROVED - reviewer rebuilt the probe
  from tarball and reproduced both return values with dynamicLookup FALSE). Spec gains
  4.7 recording the registration mechanism and the cpp_register() maintenance trap.
  Important finding folded into spec rather than left in the report.
catboost-8z4.1: review = Changes needed, 2 Critical.
  C1: link deps are 2 (openssl, zlib) NOT 4 - bzip2/pcre link into swig (build tool).
      Controller had propagated the wrong 4 into the spec; corrected in eb7da14.
  C2: "tractable" overstated - no end-to-end offline build existed.
  Controller then drove the offline build directly:
    run3: conan remote disabled -> Conan clear, 0 network. Blocked on NumPy (hnsw).
    run4: +numpy -> blocked on Cython (hnsw).
    run5: +cython -> configure_rc=0 OFFLINE, 0 network, 141s compile, link FAILED
          (TLS reloc / missing PIC) - operator error, flags omitted.
    run6: + -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCATBOOST_COMPONENTS=R-package
          -DHAVE_CUDA=no (the flags run1 used). In flight.
  CONFIRMED CONSTRAINT: CMakeUserPresets.json written into vendor snapshot on every
  configure; removed each time. Phase 1 must build from a disposable copy.
catboost-8z4.1: COMPLETE. Gate PASSED - end-to-end offline build demonstrated
  (configure_rc=0 build_rc=0, 174s, 4265/4265, 0 network, 32.87 MiB).
  Review had flagged 2 Critical; both addressed: link-dep count corrected to 2,
  tractability re-established by demonstration rather than downgraded claim.
catboost-8z4.5: fix round 1/5. Review = Changes needed, 2 Critical, both reproduced
  by re-running the generators:
  C1 submodule enum scan drops enum members (catboost.eval.EvalType/LabelMode/
     ScoreType) - same bug class the implementer fixed at top level only.
  C2 compute_diff.py substring rule (>=4 chars) marks 92/140 "covered" rows as
     false positives and silently drops them - functionally the forbidden filter.
  Consequence: 1439 gap count UNDER-counts; "all 375 hyperparams are gap rows" was
  wrong (355). Determinism verified perfect (all 5 generators byte-identical on
  re-run). Resumed implementer.
catboost-8z4.5: complete (commits 780c1fa, 6678bf4; re-review clean, both Criticals
  ADDRESSED, determinism byte-identical). FINAL: 1537 gaps, 1281 missing from spec.
  PARKED minors -> bd comment (stdlib leaks defaultdict/logger, unshared callable()
  fix in submodule scan). All fail in the over-report direction; none drops a gap.
catboost-8z4.3: complete (commits 84d956d, 37f0891).
PHASE 0 COMPLETE. All 5 tickets closed. Gate PASSED.
