# TPE batched candidate-density scoring — controlled Thor evidence

Date: 2026-09-19  
Evidence class: controlled microbenchmark, five deterministic seeds  
Implementation head benchmarked: `821d5666d22962c171901036de029e7943301f60`

## Change under test

The previous optimized TPE already cached history and precomputed
candidate-independent truncated-normal log factors, but acquisition still scored
the 24 EI candidates one at a time. Each candidate therefore walked the same
Parzen component arrays again.

This branch adds:

- precomputed linear component factors alongside the stable log factors;
- a direct probability-density fast path with stable log-domain fallback;
- numerical candidate scoring in batches;
- component-major traversal: each Parzen component is loaded once and applied
  to all candidates;
- scalar density methods retained under tests as differential oracles.

The batched path preserves candidate sampling order. The stable fallback remains
authoritative if direct probability accumulation underflows or becomes
non-finite.

## Correctness gates

At the benchmarked implementation:

- `scirust-opt-tpe`: 18 tests pass;
- Optuna density oracle tests remain green;
- direct density is checked against the stable log-domain implementation;
- an explicit underflow case exercises the stable fallback;
- batched candidate log-density matches scalar log-density;
- the selected acquisition candidate matches the scalar reference;
- `cargo clippy ... -D warnings` passes.

Across all 30 benchmark cells (100 and 1000 trials, 1/5/10 dimensions, five
seeds), the recorded best-objective bitstring is identical to the pre-batch
SciRust baseline.

## Controlled environment

Same environment and protocol as
`docs/research/SCIRUST_OPTUNA_RUSTUNA_MICROBENCH_V1.md`:

- NVIDIA Thor aarch64 CPU;
- CPU0 pinned with `taskset -c 0`;
- CPU0 governor set to `performance` during the campaign;
- five process-level seeds 23063..23067;
- independent/univariate continuous TPE;
- 24 EI candidates;
- one fresh process per cell.

Raw SciRust records are in `thor-batched-density-scirust.csv`.
The comparison baseline is `thor-five-seed-post-cache.csv`.

## 1000-trial medians

| dimensions | pre-batch SciRust | batched SciRust | improvement | Rustuna 0.1.0 | SciRust / Rustuna |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2193 ms | 0.2081 ms | 5.08% | 0.1654 ms | 1.258x |
| 5 | 1.0889 ms | 1.0070 ms | 7.52% | 0.7679 ms | 1.311x |
| 10 | 2.1635 ms | 2.0780 ms | 3.95% | 1.5563 ms | 1.335x |

Every one of the five paired seeds improved at 1000 trials in every measured
dimension. New/pre-batch proposal-time ratios were:

- 1D: 0.880, 0.949, 0.988, 0.944, 0.940;
- 5D: 0.926, 0.925, 0.982, 0.931, 0.917;
- 10D: 0.961, 0.953, 0.990, 0.930, 0.936.

The remaining Rustuna gap is therefore approximately 26%–34% in this workload.

## 100-trial note

The short-history cells are noisier:

- 1D median improves by about 4.9%;
- 10D median improves by about 2.1%;
- 5D median regresses by about 11.3%, with large seed-to-seed variability.

Because the entire 100-trial process is very short, this mixed result is not used
as the primary acceptance signal. The 1000-trial paired results are both larger
in absolute duration and directionally consistent across all five seeds.

## Interpretation

Batching is beneficial but does not close the Rustuna gap. The next measured
targets are now narrower:

1. truncated-normal sampling still recomputes lower/upper CDF bounds for every
   sampled candidate even though those bounds depend only on the selected
   mixture component;
2. Parzen model construction still allocates and scans temporary masks/sigma
   maps on each parameter proposal;
3. the component-major batch loop is now the correct place to introduce
   AArch64 SIMD/SVE once scalar/precomputation opportunities are exhausted.

The next step is therefore to precompute truncation sampling bounds and measure
that change before adding architecture-specific SIMD.
