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

## Current probe

The first probe differentiates the Rosenbrock function with respect to `x` using:

1. rustc AutoDiff / Enzyme forward mode;
2. SciRust's native forward-mode `Dual` implementation;
3. a known analytic derivative at `(x, y) = (3, 1)`.

The probe is intentionally a comparison harness, not a production backend.

## MorphoDiff comparison track

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
