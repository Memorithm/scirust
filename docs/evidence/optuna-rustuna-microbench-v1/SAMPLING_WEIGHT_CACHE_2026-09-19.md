# TPE sampling-CDF and chronological-weight cache — Thor evidence

Date: 2026-09-19  
Evidence class: controlled microbenchmark, five deterministic seeds  
Implementation head benchmarked: `eaabcbfdd5c2826d845eabe08848b9edd3d86c58`

## Changes

This slice removes two pieces of repeated numerical/history work that remained
after batched EI candidate scoring:

1. truncated-normal lower CDF and CDF mass are computed once per Parzen
   component and reused for every sample drawn from that component;
2. numerical cached-model construction no longer performs a separate
   chronological pass to materialize a trial-sized `weight_by_trial` vector.
   The audited chronological weight formula is evaluated by index while the
   final component arrays are emitted.

The original truncated-normal sampler remains a test oracle.

## Correctness

- `scirust-opt-tpe`: 20/20 tests pass;
- Clippy `-D warnings`: pass;
- 128 same-seed draws for continuous, log-continuous and integer distributions
  are exactly identical to the old path that recomputes both CDF bounds;
- indexed chronological weights are bit-identical to `default_weights()` for
  history sizes 1..199;
- all pre-existing Optuna density and cached/reference tests remain green;
- all 15 measured 1000-trial cells retain exactly the same `best` bitstring as
  the batched-density master baseline.

## Environment

Same controlled Thor protocol as the retained benchmark evidence:

- NVIDIA Thor aarch64;
- one process per cell;
- CPU0 pinned with `taskset -c 0`;
- CPU0 governor `performance`;
- five seeds: 23063..23067;
- 1000 trials;
- dimensions 1, 5 and 10.

Raw records: `thor-sampling-weight-cache-scirust.csv`.

## 1000-trial proposal medians

| dimensions | master after #1481 | this slice | improvement | Rustuna 0.1.0 | SciRust / Rustuna |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2081 ms | 0.2010 ms | 3.41% | 0.1654 ms | 1.215x |
| 5 | 1.0070 ms | 0.9876 ms | 1.92% | 0.7679 ms | 1.286x |
| 10 | 2.0780 ms | 1.9809 ms | 4.67% | 1.5563 ms | 1.273x |

Every paired seed improved in every measured dimension:

- 1D new/master ratios: 0.937, 0.972, 0.931, 0.986, 0.973;
- 5D: 0.993, 0.995, 0.913, 0.963, 0.995;
- 10D: 0.992, 0.951, 0.926, 0.984, 0.962.

The remaining proposal-latency gap to Rustuna is now approximately 21%–29% in
this continuous independent-TPE workload.

Median process peak RSS remains 2.50 / 2.88 / 3.25 MiB for 1/5/10 dimensions.

## Interpretation

The result validates reuse of component sampling bounds and eliminating the
redundant weight scan. The larger remaining cost is now model-construction
scratch traffic:

- `selected_sorted` is allocated and populated for each below/above model;
- `sigma_by_trial` is allocated to `trial_count` for each numerical model;
- six component arrays are newly allocated for every ephemeral numerical
  Parzen model;
- below/above trial masks are recreated for every proposal.

The next optimization should therefore reuse build scratch/buffers before
introducing architecture-specific SVE/NEON kernels.
