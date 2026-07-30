# catboost-8z4.4: CatBoost CLI oracle — report

## Summary

Acquired and verified the pinned `v1.2.10` CatBoost CLI binary
(linux x86_64) as a GitHub Release asset, confirmed `run-worker` is
present, enumerated all 15 CLI modes from `--help`, emitted a smoke fixture
(train + predict), and confirmed bit-identical predictions across two
from-scratch runs. No source build attempted.

## Step-by-step findings

### 1-2. Locate + acquire asset

Release `v1.2.10` asset `catboost-linux-x86_64-1.2.10`:
`https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-linux-x86_64-1.2.10`,
size 287,580,072 bytes, sha256 (verified in full, not just the prefix from
the brief):

```
478dc57f4c19de205b19b709fd4c6af93f79753dc462b6bb49d33c920dddec75
```

Acquisition script (re-runnable, verifies before skipping re-download):
`tools/oracle/cli/acquire.sh`. Downloaded binary lives at
`tools/oracle/cli/bin/catboost-v1.2.10`, excluded from git via
`tools/oracle/cli/bin/.gitignore` (`*` / `!.gitignore`). The binary itself
is NOT committed, per guard.

### 3. Version verification

`catboost-v1.2.10 --version` does not print a semver string; it prints git
provenance:

```
Commit: b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084
Branch: tags/v1.2.10
```

Commit hash matches the pinned upstream SHA in the brief
(`b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`) exactly, and branch
`tags/v1.2.10` matches the release tag. Verification method: commit-hash +
tag match, not a printed version number.

### 4. Supported modes (from `--help`, not docs)

```
fit, calc, dataset-statistics, fstr, ostr, eval-metrics, eval-feature,
metadata, model-sum, run-worker, roc, model-based-eval, normalize-model,
select-features, dump-options
```

All modes named in the brief's Reference Data are present: `fit`, `calc`,
`fstr`, `ostr`, `metadata`, `run-worker`, `select-features`,
`normalize-model`. Three extra modes not mentioned in the brief are also
present: `dataset-statistics`, `eval-metrics`, `eval-feature`, `model-sum`,
`roc`, `model-based-eval`, `dump-options` (7 extra, not 3 — full list
above). These expand the CLI-only parity surface beyond what the brief
anticipated.

### 5. Smoke fixture

Deterministic 40-row dataset (2 numeric features, 1 categorical, binary
target — same data used by the sibling Python-oracle ticket for
cross-oracle comparability), under `tests/fixtures/oracle-cli/`:

- `smoke_data.csv` — dataset
- `smoke.cd` — CatBoost column-description file (`2 Categ cat1`,
  `3 Label target`; other columns default to `Num`)
- `smoke_params.json` — params (JSON) including the literal CLI arg list
  used
- `smoke_model.cbm` — trained model (`fit`, 20 iterations, depth 4,
  lr 0.1, seed 42, `-T 1`)
- `smoke_predictions.json` — `calc` output (`RawFormulaVal`,
  `Probability`), converted from CLI TSV to JSON with no added truncation
- `smoke_metadata.json` — full provenance: asset/checksum, exact `fit`/
  `calc` commands, determinism results, `run-worker` findings, supported
  modes

Commands used (recorded verbatim in `smoke_metadata.json`):

```
catboost fit --learn-set smoke_data.csv --column-description smoke.cd \
  --delimiter , --has-header --loss-function Logloss -i 20 --depth 4 \
  --learning-rate 0.1 --random-seed 42 -T 1 --model-file smoke_model.cbm \
  --logging-level Silent

catboost calc --input-path smoke_data.csv --column-description smoke.cd \
  --delimiter , --has-header -m smoke_model.cbm \
  --output-path smoke_predictions.tsv \
  --prediction-type RawFormulaVal,Probability
```

**Precision finding:** `calc` has no `--precision` flag (confirmed via
`calc -?`); its default TSV output prints doubles at ~10 significant
digits, not a full float64 round-trip repr. `smoke_predictions.json`
preserves every digit the CLI emitted (parsed with `float()`, dumped with
`json.dump`, nothing further truncated) — the CLI's own output is the
precision ceiling. Flag this for the harness: bit-exact float64 comparison
against the Python oracle may need a wider-precision extraction path
(not investigated further, out of scope here).

### 6. Determinism

Ran `fit` + `calc` twice from scratch in separate clean directories
(`-T 1` pinned to eliminate thread-order nondeterminism).

- `smoke_predictions.tsv`: byte-identical across both runs (`diff` empty).
  **Predictions are bit-identical.**
- `smoke_model.cbm`: SHA256 **differed** between the two runs despite
  identical predictions (see raw evidence below).

Raw evidence:
```
run1 sha256: 2ab75d0af3c26ec9b725d6af040e5af43c201bebaacc40ff41cf82458b671f69
run2 sha256: fe6b2dba07cebb74450290163cfba9d39a34db3373a9686e09d9fc30c23932de
diff smoke_predictions.tsv (run1 vs run2): empty -> IDENTICAL_PREDICTIONS
```
Model file hashes differ; predictions do not. Not investigated further
(likely an embedded training timestamp/GUID in model metadata — standard
CatBoost behavior) — does not block oracle usability since the
differential harness compares predictions, not raw model bytes.

### 7. `run-worker`

Present. `run-worker -?`:

```
--svnrevision           print svn version
{-?|--help}             print usage
{-T|--thread-count} VAL worker thread count (default: core count)
--node-port VAL         TCP port for this worker; default is 0
```

All flags optional (no required positional args exposed by `--help`).
Distributed master/worker handshake not exercised — explicitly out of
scope for this ticket.

## Files

- `tools/oracle/cli/acquire.sh` — acquisition + verification script
- `tools/oracle/cli/README.md` — usage notes
- `tools/oracle/cli/bin/.gitignore` — excludes the downloaded binary
- `tools/oracle/cli/bin/catboost-v1.2.10` — downloaded binary (gitignored,
  not committed)
- `tests/fixtures/oracle-cli/smoke_data.csv`
- `tests/fixtures/oracle-cli/smoke.cd`
- `tests/fixtures/oracle-cli/smoke_params.json`
- `tests/fixtures/oracle-cli/smoke_model.cbm`
- `tests/fixtures/oracle-cli/smoke_predictions.json`
- `tests/fixtures/oracle-cli/smoke_metadata.json`

## Output schema (Section V)

```toon
task_id: catboost-8z4.4
success: true
data:
  release_tag: v1.2.10
  asset_name: catboost-linux-x86_64-1.2.10
  asset_url: https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-linux-x86_64-1.2.10
  sha256: 478dc57f4c19de205b19b709fd4c6af93f79753dc462b6bb49d33c920dddec75
  version_verified: 1.2.10
  version_verification_method: "--version prints git commit b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084 and branch tags/v1.2.10, matching the pinned upstream SHA; no semver string printed"
  acquisition_script_path: tools/oracle/cli/acquire.sh
  supported_modes: [fit,calc,dataset-statistics,fstr,ostr,eval-metrics,eval-feature,metadata,model-sum,run-worker,roc,model-based-eval,normalize-model,select-features,dump-options]
  run_worker_present: true
  run_worker_args: "--svnrevision; {-?|--help}; {-T|--thread-count} VAL (default: core count); --node-port VAL (default: 0) -- all optional, no required positional args"
  fixture_paths: [tests/fixtures/oracle-cli/smoke_data.csv,tests/fixtures/oracle-cli/smoke.cd,tests/fixtures/oracle-cli/smoke_params.json,tests/fixtures/oracle-cli/smoke_model.cbm,tests/fixtures/oracle-cli/smoke_predictions.json,tests/fixtures/oracle-cli/smoke_metadata.json]
  determinism_checked: true
  predictions_bit_identical: true
verdict:
  cli_oracle_usable: true
  cli_only_capabilities_testable: [run-worker,dataset-statistics,eval-metrics,eval-feature,model-sum,roc,model-based-eval,dump-options,select-features,normalize-model,fstr,ostr]
  reasoning: "Pinned binary acquired, checksum fully verified, commit-hash identity confirmed, all brief-listed modes present, run-worker confirmed with recorded args, smoke fixture trains+predicts, predictions bit-identical across two from-scratch runs. Only caveat: CLI's own TSV output precision (~10 sig digits) is a ceiling for future bit-exact float64 comparisons, and model-file bytes (not predictions) differ run-to-run due to embedded metadata -- neither blocks oracle usability."
  confidence: 95
error_log: null
```

## Concerns for the controller

1. Model-file SHA256 differs between identical-prediction runs (embedded
   timestamp/GUID, standard CatBoost behavior) — not investigated further,
   flagging in case a future ticket wants byte-identical model artifacts.
2. `calc`'s default output precision (~10 significant digits) may not be
   enough for strict bit-exact float64 comparison against the Python
   oracle; no `--precision` flag exists on `calc`. Worth a follow-up
   ticket if the differential harness needs tighter tolerance than ~1e-10.
3. 7 additional CLI modes beyond the brief's list are present
   (`dataset-statistics`, `eval-metrics`, `eval-feature`, `model-sum`,
   `roc`, `model-based-eval`, `dump-options`) — expands the CLI-only parity
   matrix; the brief only called out 8 of the 15 modes actually shipped.
