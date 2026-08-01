# catboost-8z4.33 report: Root-cause the peak_parallel=3 finding from P1.7

**Date:** 2026-08-01. **Status:** DONE (investigation only, per ticket's anti-pivot
boundary — no fix applied). **Confidence: 80** on the root-cause mechanism;
**confidence 95** on ruling out P1.7's own hypothesis as the primary cause.

## 1. Method: reproduce P1.7's container, but instrument with strace instead of polling

Rebuilt P1.7's exact `Containerfile` (same base image, same package list, same
`acquire.sh`/`acquire-thirdparty.sh` build stage) with one addition: `strace`
installed via apt. Ran the same `--network=none --cpus=2` isolation, same
`CATBOOSTR_JOBS=2`/`MAKEFLAGS=-j2` environment, but replaced P1.7's 0.2s
`pgrep` poll loop with:

```
strace -f -tt -qq -e trace=clone,vfork,fork,execve,wait4,exit_group \
  -o /tmp/strace.log \
  R CMD INSTALL --preclean . > /tmp/install.log 2>&1
```

`-f` follows every forked/cloned descendant of the top-level R process;
`-tt` gives microsecond timestamps. This reconstructs each process's exact
lifetime directly from the kernel's own syscall record, rather than sampling
at 0.2s intervals — P1.7's own method could only ever detect an overlap that
happened to still be alive at one of its poll instants, systematically
under-counting anything shorter-lived than ~200ms.

Install succeeded (`P33_INSTALL_EXIT=0`), matching P1.7's own successful runs.

## 2. First finding: a large fraction of "concurrent clang processes" are a measurement artifact — ruled out

Extracting every `execve` into a clang binary and building `[start, exit_group]`
intervals per PID initially showed implausible concurrency (up to 13). Direct
inspection of the processes alive at that peak showed:

```
$ grep "^1418 " strace.log | grep execve | head -1
1418  17:01:52.454079 execve("/usr/local/sbin/clang", ["clang", "-Wa,-v", "-c", "-o", "/dev/null", "-x", "assembler", "/dev/null"], ...
```

1806 such invocations use a **bare** `argv[0]="clang"` — CMake's own internal
compiler/assembler-capability probes (`enable_language()`, `try_compile()`),
fired in tight bursts, completely unrelated to `make -j2`'s job-server. Real
`make`-driven compiles always invoke the **full resolved path** as `argv[0]`
(e.g. `execve("/usr/bin/clang++", ["/usr/bin/clang++", ...`). Filtering to
full-path invocations only (3929 of them, close to the ~4166 target count
from earlier phases) removes this artifact entirely. **This distinction —
bare vs. full-path `argv[0]` — is itself a finding**: any future
parallelism instrumentation must filter on it, or it will systematically
over-count.

## 3. Second finding: the true peak, properly measured, is higher than P1.7 reported — and it is NOT a zombie/reap-lag artifact

After removing the CMake-probe noise, real `[execve, exit_group]` windows
(the process's true CPU-active lifetime, not just "still listed in `ps`")
still overlap up to **8-way**, independently cross-checked two ways (a
sweep-line algorithm and a direct interval scan at the peak instant, both
agree exactly):

```
$ awk '$2<=61378.9 && $3>=61378.9' real_intervals.tsv
12080 61378.8 61378.9 0.045517
12084 61378.8 61378.9 0.031596
12090 61378.9 61378.9 0.02008
12092 61378.9 61378.9 0.017356
12099 61378.9 61378.9 0.017131
12100 61378.9 61378.9 0.020985
12106 61378.9 61379   0.020882
12108 61378.9 61379   0.021315
```

All 8 durations are 17-46ms — a tight burst of very small, fast-compiling
translation units, not a sustained multi-second violation.

This directly refutes P1.7's specific hypothesis (confidence 55: "one job's
process not yet reaped while the next was already forked" — a zombie
lingering in `ps`/`pgrep` output after its own death). A zombie process has
already called `exit_group` and does no further CPU work; it cannot appear in
an `[execve, exit_group]` interval, because that interval **is** the CPU-active
window, ending exactly at `exit_group`. All 8 processes above were genuinely
still executing, not zombies. (Separately measured reap lag — the gap
between a process's own `exit_group` and its parent's `wait4()` reap — is
real, median 26ms, max 64ms in this run, table in `reap_lags.tsv`; it's a
real phenomenon, just not the dominant explanation for peak > 2.)

## 4. Third finding: the overlap traces to GNU Make's own recursive job-server handoff, not a broken/independent jobserver

Confirmed the process lineage of the overlapping `make` invocations via
`wait4()` return values (which identify each child's actual reaping parent,
since `strace`'s `clone()` records were incomplete for some of these PIDs):

```
$ grep -E "= 11914$" strace.log | grep wait4
11822  17:03:03.277094 wait4(-1, [...], WNOHANG, NULL) = 11914
```

`11914` (a per-directory sub-`make` for `contrib/libs/cxxsupp/builtins`) is a
child of `11822`, which is itself one of three long-lived `make` processes
(`11817`, `11819`, `11822`, each alive for ~1122s — essentially the whole
build) forming CMake's normal recursive-Makefiles hierarchy rooted at our
fork's own `configure`-issued `make -C ... -j2 catboostr` (`11817`). This is
**one** recursive tree, not several unrelated top-level `make` invocations
each independently reading the exported `MAKEFLAGS=-j2` and establishing its
own separate token pool (a hypothesis this investigation considered and
disproved — the lineage is single-rooted).

`grep -i jobserver install.log` returns **nothing** — GNU Make prints an
explicit `warning: jobserver unavailable: using -j1` when its token-pipe
inheritance breaks across a recursive level; no such warning fired, so Make
itself believes its job-server is intact end-to-end. Combined with the
tight (<100ms), small-duration nature of every overlap burst observed, the
most likely explanation (confidence 80, not higher — the exact GNU Make
internal timing that produces this was not traced at the libc/syscall level
beyond what's shown above) is **inherent slack in GNU Make's own job-server
token handoff** under a deeply recursive, CMake-generated Makefiles
hierarchy: a token is only returned to the shared pipe after the parent's
`wait4()` reaps the finished job (confirmed non-instant: median 26ms
observed lag), and Make can fork a new job against a *different*,
already-available token while that reap is still pending — across three
nested recursive levels, several such handoffs landing within the same
sub-100ms window is sufficient to produce brief, genuine bursts above the
nominal 2-job budget, without any single level ever knowingly running more
than its own 2 jobs.

## 5. The safety invariant re-confirmed, independent of this investigation

```
$ podman run --rm --network=none --cpus=2 catboostr-p33 cat /sys/fs/cgroup/cpu.max
200000 100000
```

Identical to P1.7's own measurement. This is a **kernel-enforced** cap on
total CPU-time consumption (2 CPU-equivalents), independent of instantaneous
process count — it holds regardless of the burst behavior described above.
The finding in this report changes *how many processes can be transiently
alive at once* (up to 8, briefly, vs. P1.7's measured 3), not *how much CPU
the container can actually consume* (still hard-capped at 2).

## 6. What this ticket does NOT do (per its own anti-pivot boundary)

No change was made to `configure`'s parallelism logic, `MAKEFLAGS` handling,
or the build generator choice. This ticket's scope is diagnosis only. A
follow-up ticket should decide whether the transient over-subscription
found here is worth fixing (e.g., by switching the container/CI build path
to Ninja, which uses a different, single-process job-pool model less prone
to recursive-handoff slack, or by simply accepting it as benign given §5).

## 7. Output schema

```toon
task_id: catboost-8z4.33
success: true
data:
  peak_parallel_reproduced: true
  peak_parallel_measured: 8
  p17_peak_parallel_was_undercount: true
  root_cause: "GNU Make recursive job-server token handoff slack under a deeply recursive CMake-generated Makefiles hierarchy -- NOT the zombie/fork-before-reap artifact P1.7 hypothesized (confidence 55, superseded)"
  root_cause_confidence: 80
  zombie_hypothesis_ruled_out: true
  zombie_hypothesis_ruled_out_confidence: 95
  cgroup_safety_invariant_reconfirmed: true
  configure_modified: false
report_path: docs/phase-1/catboost-8z4.33-report.md
vendor_clean: true
error_log: null
```

## 8. Definition of Done

- [x] Peak-3 finding reproduced (install succeeded, same isolation as P1.7)
      and shown to be an undercount of the true peak (8), with evidence.
- [x] Root cause identified with a confidence score (80), and P1.7's own
      hypothesis explicitly evaluated and ruled out (confidence 95) rather
      than left unresolved.
- [x] No modification to `configure` or any build-parallelism logic was made.
- [x] `git -C vendor/catboost status --porcelain` empty (no host-tree writes
      were made at all; all work happened inside a disposable container).
- [x] Report written with commands and raw output.
- [x] Output matches Section V-equivalent schema.
