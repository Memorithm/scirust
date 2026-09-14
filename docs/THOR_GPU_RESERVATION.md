# Thor GPU reservation audit and regression contract

## Scope and findings

Audited `sciagent-thor-gate.yml` at SciRust commit
`e1237827f73268bf43b2b971674a697cd186d95e` (workflow blob
`704f258236a64c3363bdeacd765805f643c355c8`). This is a focused follow-up to
PR #1418, not a complete workspace or physical-hardware certification.

1. The initial availability step released the GPU lock before later workflow
   steps. Five hardware test/diagnostic steps then acquired the lock without
   checking whether a non-cooperating external workload had started. The two
   benchmark steps checked occupancy but failed immediately instead of retrying
   when a workload had appeared between steps.
2. `ldconfig -p | grep -q` ran under `pipefail`. When the reader exits early after
   a match, a sufficiently large producer can receive SIGPIPE, falsely taking
   the `runtime-unexposed` branch. The predicate now consumes all output without
   `grep -q`. Producer errors still propagate.
3. CUDA Clippy ran after the hardware availability wait. It is now before that
   wait so GPU contention alone cannot prevent that lint step from being reached
   after successful preceding setup/format steps.

## Implemented reservation

All seven CUDA test/diagnostic/benchmark command sites now invoke:

```bash
python3 scripts/thor_gpu_guard.py -- COMMAND ARGUMENTS...
```

The guard checks current compute occupancy, attempts a non-blocking advisory
lock, and checks occupancy again under that lock immediately before replacing
itself with the foreground command. It retains the opened lock description
across `exec`, so the command holds the reservation through its lifetime.
Commands and arguments are passed directly, without shell evaluation. The
command's own exit code and normal signal handling are preserved.

A workload appearing during acquisition causes immediate unlock and retry.
Polling never holds the lock. The reservation deadline uses `time.monotonic`;
occupancy queries have their own bounded timeout capped by the remaining budget.
It does not create or truncate the lock target. Missing targets, regular files,
and symlinks are rejected; the default is the existing `/dev/nvidia0` character
device shared with the current physical Thor workflows.

| Outcome | Exit behavior |
| --- | --- |
| Reservation acquired | Execute the command; preserve its exit behavior |
| Reservation deadline exhausted | 75, `reason=resource-unavailable`, no command started |
| Probe/open/lock/exec error | 1, `reason=guard-error`, no command started |
| Invalid CLI configuration | 2 |

No exit status is rewritten into a successful test. A command can itself return
75, so classify an infrastructure deferral only with the explicit guard message
`execution_started=false`, never from an exit code alone. This change does not
extend the owner's one-PR resource exception for #1418 to any other PR.

The initial workflow's existing deferral policy for absent/unqueryable CUDA and
recognized detached production training is unchanged. A workflow success is
therefore not sufficient evidence that hardware tests ran: inspect step outcomes
and retained reports. Once the initial preflight permits execution, each guarded
command uses a fresh strict occupancy check.

## Validation

```bash
python3 scripts/test_thor_gpu_guard.py -v
```

The CPU-only suite tests contention, arrivals between observation and locking,
bounded retries, late observations, failed probes, cancellation during recheck,
lock-target validation, actual POSIX lock contention, inherited locking across
`exec`, preservation of a nonzero child exit code, and all seven workflow call
sites. It also runs the actual updated runtime-detection predicate with a large
synthetic producer, no CUDA match, and a failing producer.

These tests do not invoke CUDA or NVIDIA hardware. The separate GitHub-hosted
`Thor GPU guard regressions` workflow can run without the Thor runner. The Thor
workflow also runs the suite before waiting for hardware. Required exact-head
checks still apply before merge; this document is not an acceptance waiver.

## Limitations and follow-up boundaries

- An advisory lock only coordinates processes using the **same inode**. A process
  which ignores it can still start after the final occupancy observation. This
  guard is not an OS-enforced GPU isolation mechanism or a continuous
  contamination monitor. Performance evidence still requires its existing
  workload-isolation and metrology checks.
- Only `sciagent-thor-gate.yml` is migrated here. The stricter M33 continuous-idle,
  cooldown and contamination protocol is deliberately not replaced by this
  instantaneous reservation helper. Do not nest this guard around code which
  independently reacquires the same lock: that would self-contend.
- NNIS was the competing workload in the historical #1418 logs. The helper has no
  SciRust package dependency and is a candidate for reuse by NNIS and
  FLAT-ATTENTION, but no downstream integration or hardware qualification is
  claimed. Any port must preserve each repository's own bootstrap, locking and
  evidence contracts, and record the exact source/target revisions.
- No model, kernel, numerical tolerance, benchmark threshold, performance claim,
  ML maturity score or public tensor contract changes in this slice.
