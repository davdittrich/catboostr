# roxygen2 DLL-reload investigation (catboost-8z4.86) — implementation plan

Epic: catboost-8z4. Base branch: phase-0 (e0cdcbc, followup-triage merge).

## Global Constraints

- **Mechanism:** dev-tooling/build-environment investigation and fix/workaround, not a package runtime change.
- **Forbidden:** no touching vendor/catboost/ (read-only); no changes to R/catboost.R's actual functions or any shipped package behavior; do not re-adopt the prior (independently disproven) OpenSSL diagnosis without new evidence.
- **Audit:** investigation-first -- the ticket's own prior diagnosis (OpenSSL) was checked by the controller and found wrong (full rebuild succeeds; the real failure is a post-install `getDLLRegisteredRoutines.DLLInfo` error in roxygen2/pkgload's native-routine introspection step). The implementer must reproduce and trace this real failure, not the disproven one.
- **Verification method:** regenerate at least one real `.Rd` file via the fixed/documented path and diff it against the currently-correct, hand-edited version (from catboost-8z4.83) to confirm the fix/workaround actually works. No R package rebuild-suite gate needed beyond this (dev-tooling only, no shipped-code change expected).
- vendor/catboost/ untouched.

## Execution Order

1. catboost-8z4.86 — investigate and fix/document-workaround for roxygen2's post-install DLL-reload failure

## Tickets (verbatim, hermetic — see `bd show catboost-8z4.86` for full body)

- catboost-8z4.86
