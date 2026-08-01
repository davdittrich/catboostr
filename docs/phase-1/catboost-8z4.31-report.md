# catboost-8z4.31 report: CI check that the openssl/ragel/yasm pins agree

**Date:** 2026-08-01. **Status:** DONE.

## 1. What was created

- `tools/vendor/check-pin-sync.sh` (new, executable) — greps `configure`'s
  `OPENSSL_VERSION`/`OPENSSL_SHA256` (and the `ragel`/`yasm` equivalents)
  and `tools/vendor/acquire-thirdparty.sh`'s `PACKAGES` array entries for
  the same three names, and fails closed with an exact diagnostic if
  either the version or the SHA256 disagrees between the two files.
- `.github/workflows/ci.yml` — one new step in the `integrity` job:
  "Third-party pins agree between configure and acquire-thirdparty.sh",
  running the script above. No vendor acquisition needed for this check
  (pure static grep against two files already in the checkout), so it
  costs seconds, not minutes.

## 2. Proof it catches real drift (not just inspected)

```
$ bash tools/vendor/check-pin-sync.sh
OK: openssl 3.5.7 (a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8) agrees in both files.
OK: ragel 6.10 (5f156edb65d20b856d638dd9ee2dfb43285914d9aa2b6ec779dac0270cd56c3f) agrees in both files.
OK: yasm 1.3.0 (3dce6601b495f5b3d45b59f7d2492a340ee7e84b5beca17e48f862502bd5603f) agrees in both files.
*** All third-party pins agree between configure and acquire-thirdparty.sh.
$ echo $?
0
```

Perturbation (`OPENSSL_SHA256` in `configure` changed to a bogus value):

```
$ sed -i 's/OPENSSL_SHA256="a8c0d28a.../OPENSSL_SHA256="deadbeef.../' configure
$ bash tools/vendor/check-pin-sync.sh
FATAL: openssl SHA256 mismatch: configure=deadbeef00000000000000000000000000000000000000000000000000000 acquire-thirdparty.sh=a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8
OK: ragel ...
OK: yasm ...
*** Pin drift detected between configure and tools/vendor/acquire-thirdparty.sh.
$ echo $?
1
```

Reverted; `git diff --stat configure` shows zero diff after revert, confirming
the perturbation left no trace.

## 3. Output schema

```toon
task_id: catboost-8z4.31
success: true
data:
  check_script_path: tools/vendor/check-pin-sync.sh
  ci_wired: true
  drift_detected_in_test: true
report_path: docs/phase-1/catboost-8z4.31-report.md
vendor_clean: true
error_log: null
```

## 4. Definition of Done

- [x] A script exists that compares the two pin sources and fails on mismatch.
- [x] The check is proven to catch real, deliberately introduced drift (exit 0 -> 1 -> 0 across perturb/revert).
- [x] `git -C vendor/catboost status --porcelain` empty.
- [x] Report written with real command output.
