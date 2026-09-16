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
executables. A combined fat-LTO executable caused the rustc-distributed Enzyme
pass to abort inside LLVM type analysis when it was asked to process the much
larger SciRust Tensor/Runtime/GPU dependency graph. The minimal Enzyme crate
avoids turning that upstream implementation limitation into benchmark noise and
gives Enzyme the native workload it is supposed to differentiate.

The lanes share explicit workload contracts and must pass analytic correctness
oracles before timing. Two contracts are currently exercised:

1. scalar Rosenbrock `df/dx` in F32;
2. 16x16 matrix objective `L(A,B) = sum(A @ B)`, differentiated with respect to
   `A` while `B` remains constant.

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
MORPHODIFF_MATMUL_ITERS=2000 \
cargo +nightly run --release -p morphodiff-enzyme-native
```

The native crate has no SciRust dependencies. It reports Enzyme's compiled F32
Rosenbrock derivative and the reverse-mode MatMul objective gradient. The MatMul
lane uses `Duplicated` for `A`, `Const` for `B` and `Active` for the scalar
result. Because duplicated reverse shadows accumulate by contract, the benchmark
zeroes the preallocated `dA` shadow before each differentiated call and names the
reported metric accordingly.

## MorphoDiff scalar reference lanes

```bash
MORPHODIFF_BENCH_ITERS=100000 \
MORPHODIFF_TRANSFORM_ITERS=2000 \
cargo +nightly run --release --bin morphodiff-reference
```

This executable builds Rosenbrock as canonical Tensor IR and checks MorphoDiff in
two independent execution paths:

1. `MorphoDiff::grad` through `Core2ReferenceSession`;
2. `MorphoDiff::vjp` with an explicit cotangent of one through
   `ReferenceJitSession<CpuComputeAdapter>`.

It reports source/transformed/generated node counts, prepared kernel/dispatch
counts, transformation time and both reference execution timings.

## MorphoDiff MatMul reference lane

```bash
MORPHODIFF_MATMUL_ITERS=2000 \
MORPHODIFF_MATMUL_TRANSFORM_ITERS=1000 \
cargo +nightly run --release --bin morphodiff-matmul-reference
```

This fixture uses the exact same mathematical workload as the Enzyme matrix
lane: F32 16x16 `sum(A @ B)` with gradients requested only for `A`. MorphoDiff's
all-one cotangent for the matrix output is mathematically equivalent to the
scalar sum objective. The transformed graph is optimized and prepared once by
`ReferenceJitSession<CpuComputeAdapter>` before repeated execution.

The fixture records source/transformed/generated Tensor-IR nodes, compiled
Reference kernel count, dispatch count, AD transform time and prepared execution
time. It validates every element of `dA` against the analytic row-repeated column
sum of `B` before timing.

## Interpreting runtime numbers

The current Enzyme MatMul lane is native LLVM-generated code. The current
MorphoDiff MatMul lane is a prepared Reference CPU lane: transformation,
optimization, lowering and kernel preparation are outside the timed execution,
but it is not yet dedicated native MorphoDiff CPU code generation.

For that reason the CI may display both timings, but no winner or speed ratio is
claimed. The MorphoDiff executable prints `native_speed_verdict=not-yet-comparable`.
A direct runtime performance contest starts only after a native CPU codegen lane
uses the same workload, data, derivative semantics and measurement boundary.

## Benchmark policy

MorphoDiff transforms canonical SciRust Tensor IR, whereas Enzyme transforms
compiler/LLVM IR. Measurements therefore remain separated into:

1. differentiation/transformation and compilation/preparation cost;
2. execution cost of an already prepared or compiled derivative;
3. structural metrics such as generated Tensor-IR nodes and prepared kernels.

Correctness gates always precede timing. The canonical protocol and benchmark
ladder are documented in `../../docs/MORPHODIFF.md`.

After MatMul/BatchMatMul parity, the next research fixtures are an attention
microkernel and then FLAT-ATTENTION differentiable kernels.
