# Exploratory optimizer microbenchmark — NVIDIA Thor — seed 23063

Date: 2026-09-19
Evidence class: exploratory, single seed
Protocol: `docs/research/SCIRUST_OPTUNA_RUSTUNA_MICROBENCH_V1.md`

## Environment

- Linux 6.8.12-tegra
- aarch64
- NVIDIA Thor CPU, 14 cores, max reported frequency 2.601 GHz
- Python 3.12.3
- Optuna 5.0.0
- Rustuna 0.1.0
- SciRust benchmark branch head used for this exploratory run
- all measured processes pinned to CPU 0 with `taskset -c 0`

## Result summary

Proposal time per trial:

| dimensions | trials | SciRust | Optuna 5.0.0 | Rustuna 0.1.0 |
|---:|---:|---:|---:|---:|
| 1 | 100 | 0.343 ms | 0.791 ms | 0.0317 ms |
| 1 | 1000 | 2.225 ms | 2.994 ms | 0.163 ms |
| 5 | 100 | 1.539 ms | 3.479 ms | 0.131 ms |
| 5 | 1000 | 10.972 ms | 15.565 ms | 0.771 ms |
| 10 | 100 | 3.057 ms | 6.625 ms | 0.256 ms |
| 10 | 1000 | 22.133 ms | 33.690 ms | 1.585 ms |

On this single exploratory run, SciRust proposal generation is faster than
Optuna Python in every measured cell, but materially slower than Rustuna. At
1000 trials the SciRust/Rustuna proposal-time ratio is approximately 13.7x,
14.2x and 14.0x for 1, 5 and 10 dimensions respectively.

Peak process RSS:

| dimensions | trials | SciRust | Optuna | Rustuna |
|---:|---:|---:|---:|---:|
| 1 | 1000 | 2.4 MiB | 49.0 MiB | 15.1 MiB |
| 5 | 1000 | 2.6 MiB | 49.9 MiB | 15.9 MiB |
| 10 | 1000 | 2.9 MiB | 51.3 MiB | 17.0 MiB |

The process-RSS comparison includes Python interpreter/import footprint for the
Python bindings, so it is an end-to-end process metric rather than a pure
sampler-allocation metric.

## Interpretation

The growth from 100 to 1000 trials is the most actionable signal. SciRust's
proposal cost increases much more than would be explained by the fixed 24
candidate count alone. The current reference implementation rebuilds ranked
history and Parzen observation/model state for every proposal. That behavior is
therefore the first optimization target.

SIMD is not the first response to this result. The next implementation step is
to remove history-wide reconstruction through incremental sampler state/event
watermarks. SIMD remains important after the algorithmic history cost is
reduced.

## Limitations

This is one seed and therefore not confirmatory evidence. The preregistered five
seeds and 5000-trial cells remain to be run after the first incremental-state
optimization so the before/after comparison can be performed with the same
protocol. Best-objective values in this file are descriptive only and must not
be used to rank sample efficiency from a single seed.
