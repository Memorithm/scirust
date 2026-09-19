# Exploratory optimizer microbenchmark — NVIDIA Thor

Date: 2026-09-19  
Evidence class: exploratory, repeated process-level measurements  
Protocol: `docs/research/SCIRUST_OPTUNA_RUSTUNA_MICROBENCH_V1.md`

## Environment

- Linux 6.8.12-tegra
- aarch64
- NVIDIA Thor CPU, 14 cores, max reported frequency 2.601 GHz
- Python 3.12.3
- Optuna 5.0.0
- Rustuna 0.1.0
- SciRust post-cache benchmark code head: `d1c7490e65a29e8245e826717c241103f8630b51`
- measured processes pinned to CPU 0 with `taskset -c 0`
- five-seed post-cache campaign ran with CPU 0 governor set to `performance`
  for each engine campaign and restored afterward

Raw post-cache process records are in
`thor-five-seed-post-cache.csv`. The older
`thor-seed23063-exploratory.csv` records the pre-cache reference slice.

## Pre-cache exploratory signal

The first single-seed run exposed history reconstruction as the dominant SciRust
problem. Proposal time per trial was:

| dimensions | trials | SciRust | Optuna 5.0.0 | Rustuna 0.1.0 |
|---:|---:|---:|---:|---:|
| 1 | 100 | 0.343 ms | 0.791 ms | 0.0317 ms |
| 1 | 1000 | 2.225 ms | 2.994 ms | 0.163 ms |
| 5 | 100 | 1.539 ms | 3.479 ms | 0.131 ms |
| 5 | 1000 | 10.972 ms | 15.565 ms | 0.771 ms |
| 10 | 100 | 3.057 ms | 6.625 ms | 0.256 ms |
| 10 | 1000 | 22.133 ms | 33.690 ms | 1.585 ms |

That run motivated event-watermark history state plus cached parameter
observations and Parzen construction.

## Post-cache five-seed medians

The following values are medians across seeds 23063–23067. All three engines in
these tables used the same host, CPU affinity, objective, dimensions, trial
count and performance-governor policy.

### Proposal time per trial

| dimensions | trials | SciRust | Optuna 5.0.0 | Rustuna 0.1.0 | Optuna / SciRust | SciRust / Rustuna |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.0412 ms | 0.766 ms | 0.0310 ms | 18.57x | 1.33x |
| 5 | 100 | 0.208 ms | 3.301 ms | 0.132 ms | 15.84x | 1.58x |
| 10 | 100 | 0.441 ms | 7.172 ms | 0.257 ms | 16.27x | 1.71x |
| 1 | 1000 | 0.219 ms | 2.974 ms | 0.165 ms | 13.57x | 1.33x |
| 5 | 1000 | 1.089 ms | 16.432 ms | 0.768 ms | 15.09x | 1.42x |
| 10 | 1000 | 2.164 ms | 39.099 ms | 1.556 ms | 18.07x | 1.39x |

The result is a major change from the initial reference implementation:
SciRust is now in the same order of magnitude as Rustuna rather than roughly
14x slower at 1000 trials. Rustuna remains faster in every currently measured
post-cache cell, by about 1.33x–1.71x.

The pre-cache and post-cache files were not produced under an identical
documented governor policy, so the approximately order-of-magnitude difference
between them is an exploratory engineering signal, not a controlled attribution
of the full speedup to one code change.

### Tell time per trial

At 1000 trials the median SciRust terminal-commit cost is only about
0.166–0.226 microseconds/trial across 1–10 dimensions. Rustuna is about
1.49–3.29 microseconds and Optuna about 85.6–218.6 microseconds in the same
campaign. The optimization bottleneck is therefore proposal generation, not
`tell`, for this workload.

### Peak process RSS at 1000 trials

| dimensions | SciRust | Optuna | Rustuna |
|---:|---:|---:|---:|
| 1 | 2.50 MiB | 49.00 MiB | 15.13 MiB |
| 5 | 2.88 MiB | 49.94 MiB | 15.88 MiB |
| 10 | 3.25 MiB | 51.29 MiB | 17.00 MiB |

This is end-to-end process RSS. Python interpreter/import footprint is included
for the Python Optuna and Rustuna bindings, so these numbers are not a pure
sampler-allocation comparison.

## Interpretation and next target

The event watermark and observation caches removed the dominant repeated
history reconstruction. The remaining Rustuna gap has two visible components:

1. At 100 trials the gap grows with dimension, indicating fixed/per-parameter
   proposal overhead.
2. At 1000 trials SciRust remains about 1.33x–1.42x slower, indicating remaining
   per-observation density work.

The current numerical Parzen path still evaluates candidate log densities
candidate-by-candidate and performs repeated truncated-normal sampling-bound
CDF work. The next optimization should therefore target batched numerical
candidate scoring and reusable/precomputed truncation terms before introducing
architecture-specific SIMD. SIMD should accelerate a batched kernel, not mask an
avoidable scalar layout.

## Limitations

These measurements remain exploratory rather than confirmatory:

- five seeds are repeated, but no confidence interval or preregistered acceptance
  decision has yet been attached;
- 5000-trial cells have not yet been completed;
- this is a continuous independent/univariate TPE workload only;
- best-objective values are descriptive and must not be used to rank sample
  efficiency from this microbenchmark;
- Optuna 5.1.0.dev remains the differential/oracle source snapshot, whereas the
  executable comparison uses released Optuna 5.0.0 and Rustuna 0.1.0.
