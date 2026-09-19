# Reusable Parzen build scratch — Thor evidence

Date: 2026-09-19  
Evidence class: controlled microbenchmark, five deterministic seeds  
Implementation head benchmarked: `a54ec2107107d5622276bb5244de78093adbb5a3`

## Change

Numerical cached-model construction now reuses:

- the value-ordered `selected_sorted` buffer;
- dense `sigma_by_trial` storage without refilling unchanged slots;
- the proposal-level `below_mask` and `above_mask` bit vectors.

The same scratch is reused sequentially for below and above model construction.
No optimizer equation or RNG step was changed.

## Correctness

- `scirust-opt-tpe`: 20/20 tests pass;
- Clippy `-D warnings`: pass;
- all existing Optuna density, cached/reference, chronological-weight and
  sampling-sequence tests remain green;
- all 15 measured 1000-trial cells keep the same best-objective bitstring as
  the #1482 baseline.

## Controlled 1000-trial medians

| dims | #1482 baseline | reusable scratch | improvement | Rustuna | SciRust / Rustuna |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2010 ms | 0.1922 ms | 4.38% | 0.1654 ms | 1.162x |
| 5 | 0.9876 ms | 0.9458 ms | 4.23% | 0.7679 ms | 1.232x |
| 10 | 1.9809 ms | 1.9428 ms | 1.92% | 1.5563 ms | 1.248x |

Four of five paired seeds improve in every dimension. Median process peak RSS
remains 2.50 / 2.88 / 3.25 MiB.

Raw records: `thor-parzen-scratch-scirust.csv`.

## Next target

Each below/above numerical model still allocates fresh component arrays for
weights, means, sigmas, density factors and sampling CDF state. Reusing those
component capacities is the next scalar/layout target. After that, the
component-major batched density loop is suitable for AArch64 SIMD/SVE.
