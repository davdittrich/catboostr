# CatBoost CLI oracle

Pinned CatBoost CLI binary (`v1.2.10`, linux x86_64), used as the reference
oracle for capabilities the Python package does not expose (e.g.
`run-worker` / distributed training).

## Acquire

```
tools/oracle/cli/acquire.sh
```

Downloads `catboost-linux-x86_64-1.2.10` from the pinned GitHub release into
`tools/oracle/cli/bin/` (gitignored, not committed), verifies size and
SHA256 against the values hardcoded in the script, `chmod +x`s it. Re-running
is a no-op if the file already matches.

- SHA256: `478dc57f4c19de205b19b709fd4c6af93f79753dc462b6bb49d33c920dddec75`
- Size: `287580072` bytes
- Source: `https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-linux-x86_64-1.2.10`

## Verify

```
tools/oracle/cli/bin/catboost-v1.2.10 --version
tools/oracle/cli/bin/catboost-v1.2.10 --help
```

`--version` prints git commit info (no semver string); confirm the commit
hash matches `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084` and branch
`tags/v1.2.10`.

Smoke fixture (dataset, column description, params, trained model, raw
predictions) lives under `tests/fixtures/oracle-cli/`; see
`smoke_metadata.json` there for exact commands and determinism results.

Do NOT build this binary from source — released binaries only.
