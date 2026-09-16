# SciRust rustc AutoDiff / Enzyme probe

This directory is an **experimental, standalone Cargo workspace**. It is not a
member of the root SciRust workspace and therefore cannot change the framework's
Rust 1.89 MSRV or normal stable CI graph.

Its purpose is narrow: compile derivatives with rustc's experimental AutoDiff
support and use Enzyme as an independent correctness/performance oracle for
SciRust differentiation engines, including MorphoDiff.

## Requirements

Rust's AutoDiff support is nightly-only and requires a compiler/toolchain built
with the AutoDiff/Enzyme backend available. The local Cargo configuration passes
`-Zautodiff=Enable`, as required by rustc's unstable AutoDiff workflow.

Run from this directory with a compatible nightly toolchain:

```bash
cargo +nightly test
```

If the installed nightly reports that AutoDiff/Enzyme is unavailable, that is a
toolchain capability failure, not a failure of SciRust's production AutoDiff or
MorphoDiff. Production SciRust remains independent of this experiment.

## Current probes

The library probe differentiates the Rosenbrock function with respect to `x`
using:

1. rustc AutoDiff / Enzyme forward mode;
2. SciRust's native forward-mode `Dual` implementation;
3. an analytic derivative oracle.

The F32 path exists specifically so Enzyme and MorphoDiff can be checked with the
same scalar type. A correctness failure aborts the benchmark before timing.

## Executable MorphoDiff vs Enzyme harness

Run:

```bash
cargo +nightly run --release --bin morphodiff_vs_enzyme
```

Optional iteration controls:

```bash
MORPHODIFF_BENCH_ITERS=1000000 \
MORPHODIFF_TRANSFORM_ITERS=10000 \
cargo +nightly run --release --bin morphodiff_vs_enzyme
```

The executable builds the Rosenbrock function as canonical Tensor IR, transforms
it with `MorphoDiff::grad`, validates the resulting derivative against the same
analytic oracle used for Enzyme, and reports:

- source, transformed and generated Tensor-IR node counts;
- Enzyme compiled derivative runtime;
- MorphoDiff transformation time;
- MorphoDiff execution time through the deterministic Core2 CPU reference path.

The Core2 runtime is an interpreter/reference oracle. Its execution number is
**not** a valid final speed comparison against Enzyme's compiled derivative. The
harness prints `runtime_speed_verdict=not-comparable-until-compiled-backend` to
make accidental benchmark claims difficult. The next benchmark stage will lower
MorphoDiff derivatives to a compiled CPU backend before comparing runtime ratios.

## Benchmark policy

MorphoDiff transforms canonical SciRust Tensor IR, while Enzyme transforms
compiler IR. The benchmark track therefore records two classes of metrics
separately:

1. transformation/compilation cost (AD transform, optimization and lowering);
2. execution cost of the already-compiled derivative.

A workload is timed only after its gradients agree with the selected correctness
oracles. The canonical protocol, required metadata and benchmark ladder are
documented in `../../docs/MORPHODIFF.md`.

The next fixtures after the scalar Rosenbrock probe are elementwise tensor chains,
MatMul/BatchMatMul, then an attention microkernel. FLAT-ATTENTION enters the suite
after those smaller fixtures establish reproducible parity.
