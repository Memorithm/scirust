# SciRust vs Optuna vs Rustuna — optimizer overhead microbenchmark v1

Date: 2026-09-19

## Purpose

Measure the current scalar/reference SciRust TPE before SIMD, multivariate TPE
or Streaming-TPE work. The benchmark is deliberately a cheap deterministic
objective so optimizer/runtime overhead remains visible.

This protocol does **not** establish universal optimizer quality. It establishes
an overhead baseline and supplies one controlled sample-efficiency signal.

## Compared engines

- SciRust `scirust-opt-tpe` from the tested Git revision.
- Optuna release `5.0.0`.
- Rustuna release `0.1.0`.

The attached Optuna source snapshot reporting `5.1.0.dev` remains the
algorithmic/differential oracle used by SciRust tests; the public releases above
are used for same-machine executable comparison.

## Search space and objective

For dimension `d`, all parameters are continuous:

`x_i in [-5, 5]`.

The deterministic target is:

`target(i) = ((i mod 5) - 2) * 0.5`.

The minimized objective is:

`sum_i (x_i - target(i))^2`.

No model training, sleep, I/O or deliberate objective cost is present.

## Sampler controls

The first comparison uses the common independent/univariate regime:

- startup trials: 10;
- TPE independent / `multivariate=false`;
- minimize direction;
- deterministic seed;
- in-memory study/storage;
- sequential execution;
- continuous parameters only.

SciRust additionally pins the audited defaults currently represented by its
reference implementation: 24 EI candidates, prior weight 1, magic clip enabled,
endpoint influence disabled and constant-liar support enabled. Constant-liar
does not alter this sequential run.

## Timed regions

`proposal_ns` includes all work required to obtain concrete parameter values:

- SciRust: `Study::ask` (which returns a complete Candidate);
- Optuna/Rustuna: `Study.ask` plus all `trial.suggest_float` calls.

`tell_ns` measures the terminal result commit only.

`total_ns` includes proposal, objective arithmetic, tell and loop overhead.

Each configuration is launched in a fresh process. Linux `VmRSS` is recorded
before and after the measured loop and `VmHWM` records process peak RSS.

## Matrix

Initial dimensions:

- 1
- 5
- 10

Initial trial counts:

- 100
- 1,000
- 5,000

Use five deterministic seeds per cell:

- 23063
- 23064
- 23065
- 23066
- 23067

The 5,000-trial cells may be omitted from the first exploratory pass if a
reference implementation becomes prohibitively slow. Any omission must be
reported rather than silently discarded.

## Environment capture

Record at minimum:

- kernel and architecture;
- CPU model/core count;
- Rust compiler/toolchain;
- Python version;
- exact Optuna/Rustuna package versions;
- exact SciRust Git SHA.

CPU-frequency policy and competing workload can influence microbenchmarks.
Headline conclusions require repeated runs under a controlled governor/load
rather than a single opportunistic measurement.

## Reporting

For each cell report median across process-level repeats for:

- proposal ns/trial;
- tell ns/trial;
- total ns/trial;
- peak RSS;
- best objective reached.

Raw process records are retained as CSV. Do not infer speedup from results
executed on different hosts.

The next optimization step is chosen from measured cost concentration. SIMD or
Streaming-TPE is not considered validated merely because it exists.
