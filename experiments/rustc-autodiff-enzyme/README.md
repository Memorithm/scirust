# SciRust MorphoDiff / rustc AutoDiff / Enzyme probes

This directory is an **experimental, standalone Cargo workspace**. It is not a
member of the root SciRust workspace and therefore cannot change the framework's
Rust 1.89 MSRV or normal stable CI graph.

Its purpose is narrow: use rustc's experimental AutoDiff/Enzyme support as an
independent correctness and performance oracle for MorphoDiff while keeping the
production SciRust dependency and toolchain graph stable.

## Requirements

Rust's AutoDiff support is nightly-only. Install the official `enzyme` component
on the nightly toolchain:

```bash
rustup toolchain install nightly --profile minimal
rustup +nightly component add enzyme
```

Do **not** enable the AutoDiff compiler pass globally for this workspace. The
current distributed Enzyme pass is invoked only for the isolated native oracle:

```bash
RUSTFLAGS="-Zautodiff=Enable" \
cargo +nightly test --release -p morphodiff-enzyme-native
```

Keeping that flag scoped is part of the benchmark contract: the MorphoDiff lane
must not be compiled under Enzyme's LLVM pass. If the installed nightly does not
provide a compatible Enzyme component, that is an experimental toolchain
capability failure, not a failure of production MorphoDiff.

## Why the benchmark is split

The Enzyme lane and the MorphoDiff lane intentionally live in separate
executables. A combined fat-LTO executable caused the current rustc-distributed
Enzyme pass to abort inside its LLVM type analysis when it was asked to process
the much larger SciRust Tensor/Runtime/GPU dependency graph. The minimal Enzyme
crate avoids turning that upstream implementation limitation into benchmark
noise and gives Enzyme the small native workload it is supposed to differentiate.

Both lanes run on the same CI runner, use F32, evaluate the same Rosenbrock
partial derivative `df/dx`, and must independently pass the same analytic
correctness oracle before timing.

## Validate the two lanes

MorphoDiff reference code, without the Enzyme compiler pass:

```bash
cargo +nightly test --release -p scirust-autodiff-enzyme-probe
```

Enzyme native oracle, with the pass scoped to that crate:

```bash
RUSTFLAGS="-Zautodiff=Enable" \
cargo +nightly test --release -p morphodiff-enzyme-native
```

## Enzyme native lane

```bash
RUSTFLAGS="-Zautodiff=Enable" \
MORPHODIFF_BENCH_ITERS=100000 \
cargo +nightly run --release -p morphodiff-enzyme-native
```

This crate has no SciRust dependencies. It reports Enzyme's compiled native
F32 derivative runtime after validating six Rosenbrock points against the
analytic derivative.

## MorphoDiff reference lanes

```bash
MORPHODIFF_BENCH_ITERS=100000 \
MORPHODIFF_TRANSFORM_ITERS=2000 \
cargo +nightly run --release --bin morphodiff-reference
```

This executable builds Rosenbrock as canonical Tensor IR and checks MorphoDiff in
two independent execution paths:

1. `MorphoDiff::grad` through `Core2ReferenceSession`, including the explicit
   `OnesLike` gradient seed;
2. `MorphoDiff::vjp` with an explicit cotangent of one through
   `ReferenceJitSession<CpuComputeAdapter>`, which performs optimization,
   logical lowering, reference-kernel generation and backend preparation once
   before repeated execution.

It reports source/transformed/generated node counts, prepared kernel/dispatch
counts, MorphoDiff transformation time and both reference execution timings.

Neither MorphoDiff execution number is presented as a native-code speed verdict
against Enzyme. The executable prints
`runtime_speed_verdict=not-comparable-until-native-codegen` explicitly. A direct
runtime ratio becomes meaningful only after a native MorphoDiff CPU codegen lane
is available.

## Benchmark policy

MorphoDiff transforms canonical SciRust Tensor IR, whereas Enzyme transforms
compiler/LLVM IR. Measurements therefore remain separated into:

1. differentiation/transformation and compilation/preparation cost;
2. execution cost of an already prepared or compiled derivative;
3. structural metrics such as generated Tensor-IR nodes and prepared kernels.

Correctness gates always precede timing. The canonical protocol and benchmark
ladder are documented in `../../docs/MORPHODIFF.md`.

The next fixtures after scalar Rosenbrock are elementwise tensor chains,
MatMul/BatchMatMul and an attention microkernel. FLAT-ATTENTION enters the suite
after those smaller fixtures establish reproducible parity.
