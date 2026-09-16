# MorphoDiff

MorphoDiff is SciRust's backend-neutral differential program transformer.

It transforms canonical Tensor IR into explicit differential programs before the
normal optimization, representation planning and backend lowering stages. The
first implementation exposes forward JVP, reverse VJP, seeded gradient and
value-and-gradient transforms through `scirust_tensor_ir::MorphoDiff`.

## Position in the SciRust stack

```text
Rust / SciRust frontend
        |
        v
Canonical Tensor IR
        |
        v
+-------------------+
|    MorphoDiff     |
| JVP / VJP / grad  |
+-------------------+
        |
        v
Differential Tensor IR
        |
        +--> graph optimization
        +--> representation planning
        +--> CPU / SIMD
        +--> WGPU
        +--> CUDA
```

MorphoDiff is not an Enzyme fork. Enzyme operates on LLVM/MLIR and remains an
external independent oracle. MorphoDiff owns its transformation semantics at the
SciRust Tensor IR level, where tensor shapes, representations and higher-level
operations are still explicit.

## Initial API

```rust
use scirust_tensor_ir::{MorphoDiff, MorphoDiffMode};

let program = MorphoDiff::transform(&graph, output, &[x], MorphoDiffMode::ReverseGrad)?;
```

A `MorphoDiffProgram` contains:

- the transformed graph;
- the primal output node;
- derivative outputs;
- explicit tangent/cotangent seed inputs when required;
- a structural `MorphoDiffReport`.

## Correctness policy

No performance claim is accepted unless the differentiated program first passes
cross-oracle validation. Depending on the case, the comparison set should
contain at least two of:

1. analytic derivative;
2. central finite differences with a documented step size;
3. SciRust scalar/tape AutoDiff;
4. rustc `std::autodiff` / Enzyme;
5. MorphoDiff.

Tolerance, dtype, shape, compiler revision and backend must be recorded with the
result.

## MorphoDiff versus Enzyme: benchmark protocol

Enzyme and MorphoDiff operate at different compiler levels. A fair comparison
therefore separates transformation cost from derivative execution cost.

### A. Transformation / compilation metrics

Record independently:

- frontend-to-IR time;
- AD transformation time;
- post-AD optimization time;
- backend lowering/code-generation time;
- total clean build time;
- incremental rebuild time;
- generated IR node/instruction count;
- generated binary or kernel size where available.

For MorphoDiff, `MorphoDiffReport` supplies source/transformed/generated Tensor
IR node counts. Enzyme-side LLVM metrics are collected by the experimental
nightly harness and compiler tooling rather than mixed into stable SciRust CI.

### B. Runtime metrics

After both derivative programs have been compiled for the same target, compare:

- primal latency;
- derivative latency;
- derivative/primal slowdown ratio;
- throughput;
- peak resident memory;
- peak accelerator memory where applicable;
- temporary allocation volume;
- warm versus cold execution;
- deterministic output/gradient reproducibility.

Wall-clock measurements must use identical inputs, target ISA/device, thread
count, affinity policy, optimization level and warm-up policy.

### C. Accuracy metrics

For every timed workload also record:

- maximum absolute error;
- maximum relative error;
- L2 error;
- NaN/Inf count;
- gradient agreement against the selected oracle(s).

A faster result with an invalid gradient is a failed result, not a benchmark win.

## Benchmark ladder

The benchmark suite should grow in this order:

1. scalar Rosenbrock (existing rustc/Enzyme probe);
2. elementwise vector chains;
3. dense matrix multiplication;
4. batched matrix multiplication;
5. reshape/transpose/broadcast/reduce pipelines;
6. small MLP block;
7. attention microkernel;
8. FLAT-ATTENTION differentiable kernels;
9. representative SciRust solver/research workloads.

Each step is kept only after correctness parity has been established.

## Toolchain isolation

Stable SciRust remains on its declared MSRV. The rustc/Enzyme comparison stays
inside `experiments/rustc-autodiff-enzyme`, which is an independent Cargo
workspace requiring a compatible nightly compiler with the AutoDiff backend.
MorphoDiff itself must not require nightly Rust.

## Near-term roadmap

1. Establish the named MorphoDiff facade and structural reports.
2. Keep all current Tensor IR JVP/VJP rules green under stable and `no_std` CI.
3. Add explicit rule coverage/introspection.
4. Add higher-order transforms (JVP-of-VJP / Hessian-vector products).
5. Integrate post-AD graph optimization and representation-aware planning.
6. Add reproducible MorphoDiff-vs-Enzyme benchmark fixtures.
7. Exercise an attention microkernel and then FLAT-ATTENTION.
