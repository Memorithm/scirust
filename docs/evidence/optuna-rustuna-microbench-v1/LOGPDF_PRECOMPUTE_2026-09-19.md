# TPE truncated-normal normalization precompute — controlled Thor evidence

Date: 2026-09-19  
Evidence class: controlled microbenchmark, five deterministic seeds  
Branch: `research/optuna-surpass-parzen-cache`  
Tested implementation head before evidence commits: `d1c7490e65a29e8245e826717c241103f8630b51`

## Change under test

The reference TPE previously recomputed each truncated-normal kernel's
normalization mass for every acquisition candidate. With 24 EI candidates, an
above model containing approximately N kernels therefore repeated a term that
depends on the kernel but not on the candidate roughly 24 times.

The optimized path:

- precomputes each numerical component's truncation normalization once per model;
- precomputes the candidate-independent log factor;
- evaluates continuous log-density with a streaming log-sum-exp and no temporary
  terms vector;
- retains the stable log-domain computation;
- retains Optuna density oracle tests;
- retains exact cached/reference sampling-sequence tests.

The same branch also contains incremental StudyEvent history synchronization and
cached chronological/value-ordered parameter observations. Those earlier cache
steps were benchmarked separately and were approximately neutral in isolation;
the large gain below appears only after the log-density normalization change.

## Controlled environment

- NVIDIA Thor CPU, aarch64, 14 cores;
- CPU0 pinned with `taskset -c 0`;
- CPU0 governor explicitly set to `performance`;
- reported CPU0 frequency: 2,601,000 kHz at campaign start/end;
- one fresh process per engine/configuration;
- five seeds: 23063..23067;
- 1000 trials per process;
- independent/univariate, single-objective TPE;
- Optuna 5.0.0;
- Rustuna 0.1.0;
- SciRust `scirust-opt-tpe`.

The governor was restored to `schedutil` after the campaign.

## Controlled 1000-trial medians

| dimensions | SciRust proposal | Rustuna proposal | SciRust / Rustuna | Optuna proposal | Optuna / SciRust |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.221 ms/trial | 0.166 ms/trial | 1.329x | 2.974 ms/trial | 13.651x |
| 5 | 1.089 ms/trial | 0.771 ms/trial | 1.407x | 16.432 ms/trial | 15.093x |
| 10 | 2.169 ms/trial | 1.594 ms/trial | 1.364x | 39.099 ms/trial | 18.178x |

Paired SciRust/Rustuna proposal-time ratios were stable under the fixed governor:

- 1D: 1.272x .. 1.358x;
- 5D: 1.318x .. 1.430x;
- 10D: 1.361x .. 1.371x.

Thus this benchmark does **not** support a claim that SciRust has surpassed
Rustuna on univariate TPE proposal latency. It does support that the remaining
gap is now relatively narrow and reproducible.

## Process peak RSS medians

| dimensions | SciRust | Rustuna | Optuna |
|---:|---:|---:|---:|
| 1 | 2.50 MiB | 15.13 MiB | ~49.0 MiB |
| 5 | 2.88 MiB | 15.88 MiB | ~49.9 MiB |
| 10 | 3.25 MiB | 17.00 MiB | ~51.3 MiB |

These are end-to-end process metrics. Python interpreter/import footprint is
therefore included for the Python bindings.

## Before/after exploratory checkpoint

Using the same seed 23063 and the same cells as the earlier retained benchmark,
proposal latency improved by:

| dimensions | trials | old reference | normalization-precompute path | speedup |
|---:|---:|---:|---:|---:|
| 1 | 100 | 0.343 ms | 0.041 ms | 8.271x |
| 1 | 1000 | 2.225 ms | 0.234 ms | 9.509x |
| 5 | 100 | 1.539 ms | 0.205 ms | 7.505x |
| 5 | 1000 | 10.972 ms | 1.093 ms | 10.043x |
| 10 | 100 | 3.057 ms | 0.417 ms | 7.338x |
| 10 | 1000 | 22.133 ms | 2.180 ms | 10.151x |

This before/after table is exploratory because the original checkpoint ran under
`schedutil`, although CPU frequency was observed at the same 2.601 GHz in the
fast run. The fixed-governor cross-engine table above is the stronger evidence.

## Correctness gates

At the tested implementation:

- `scirust-opt-core`: 19 tests pass;
- `scirust-opt-tpe`: 15 tests pass;
- Optuna numerical density oracles remain green;
- cached numerical and categorical Parzen models reproduce the reference
  sampling sequence for 64 same-seed draws;
- `cargo clippy ... -D warnings` passes.

## Next measured target

The controlled gap to Rustuna is 1.33x–1.41x. The next branch removes the
per-candidate temporary/log-sum-exp overhead by accumulating mixture density
directly in probability space, with a stable log-domain fallback and explicit
fast-vs-stable numerical tests.

A further SIMD/batched-candidate implementation is justified only after that
scalar change is measured.
