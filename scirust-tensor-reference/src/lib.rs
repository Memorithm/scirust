//! Deterministic Reference-format kernel generation and serial CPU
//! interpretation for SciRust.
//!
//! # Where this sits
//!
//! ```text
//! scirust-tensor-ir
//!         |
//! scirust-tensor-compile   (Graph -> ExecutionPlan -> MemoryPlan -> LoweredPlan)
//!         |
//! scirust-tensor-reference (this crate: LoweredPlan -> artefacts -> CPU execution)
//!         |
//! scirust-compute          (KernelModule, KernelFormat::Reference)
//! ```
//!
//! # The four objects this crate owns
//!
//! | Type | Role |
//! |---|---|
//! | [`ReferenceKernelArtifact`] | one logical kernel in the canonical binary format: encode, decode, wrap as a [`scirust_compute::KernelModule`] |
//! | [`PreparedReferenceKernel`] | that artefact decoded, validated and converted into the form the CPU executes |
//! | [`ReferenceInterpreter`] | serial execution of **one individual** kernel over `f32` slices |
//! | `CpuComputeAdapter` (in `scirust-gpu`) | downstream physical integration; this crate does not own graph-session scheduling or buffer orchestration |
//!
//! ```text
//! LoweredPlan
//!   -> ReferenceKernelGenerator -> ReferenceKernelArtifact
//!   -> ReferenceKernelArtifact::to_kernel_module -> KernelModule{ format: Reference }
//!   -> PreparedReferenceKernel::from_kernel_module
//!   -> ReferenceInterpreter::execute(&prepared, &[&[f32]], &mut [f32])
//! ```
//!
//! # What this crate does not do
//!
//! * **No plan execution.** [`ReferenceInterpreter`] runs one kernel. It does
//!   not walk a `scirust_tensor_compile::LoweredPlan`, resolve inter-kernel
//!   dependencies, consult a memory plan, allocate slots, bind external inputs
//!   or constants, or order instructions. Those responsibilities live in the
//!   downstream tensor runtime.
//! * **No backend buffer ownership, no device ownership, no physical allocation.**
//!   Operands and results are plain `&[f32]` / `&mut [f32]` the caller owns.
//!   This crate does not itself allocate a `scirust_compute::BufferBinding` or
//!   select a `MemorySpace` or device.
//! * **No graph-session or `ComputeBackend` orchestration is owned here.**
//!   Downstream runtime/adapter code may prepare and execute Reference modules;
//!   this crate owns only their deterministic format, validation and serial
//!   per-kernel interpreter semantics.
//! * **No WGSL and no PTX.** Those are separate target-specific generators.
//!
//! # `Exp` and `Log` are not executable
//!
//! [`PreparedReferenceKernel`] **rejects** the `Exp` and `Log` opcodes with
//! [`ReferenceExecutionError::DeterministicMathUnavailable`], at preparation
//! time. `f32::exp` and `f32::ln` delegate to the platform's math library;
//! neither Rust nor IEEE 754 requires a correctly-rounded result for
//! transcendental functions, and implementations differ in practice across
//! operating systems, architectures and versions. Using them would contradict
//! the determinism this format exists to provide, so the artefact format keeps
//! both opcodes — a generator may emit them — while this interpreter refuses
//! them outright rather than returning a number it cannot reproduce.
//!
//! # Numeric guarantees, and their limits
//!
//! Element-wise opcodes run serially, in increasing index order, one scalar
//! `f32` operation per element, with no reassociation, no `mul_add`, no explicit
//! FMA, no explicit SIMD, no Rayon, no fast-math and no widening to `f64`.
//! Matrix products use row-major layouts and a fixed increasing-`K` scalar
//! accumulation order under the same no-`mul_add`, no explicit FMA and no-`f64`
//! policy. These are Reference semantics, not an optimized GEMM claim.
//!
//! This crate does **not** claim a compiled `f32` operator is bit-identical on
//! every conforming architecture; that depends on target and toolchain, not on
//! this code. The verified guarantee is narrower: golden-bit tests over the
//! platforms this project's CI covers.
//!
//! NaN payloads are preserved bit-for-bit **only** where a value is returned or
//! copied without arithmetic — the NaN branch of `Relu`, [`ReferenceOpcode::ShapeCopy`]
//! and [`ReferenceOpcode::Permute`]. For arithmetic operations, including
//! matrix products, a NaN operand produces arithmetic governed by Rust/target
//! floating-point semantics; no payload-preservation claim is made.
//!
//! `Scale`'s factor, by contrast, is carried bit-exact end to end: it travels as
//! a raw `u32` bit pattern, so `-0.0`, `+0.0`, `+inf`, `-inf` and every NaN
//! payload reach the multiplication unchanged, with no canonicalisation.
//!
//! # Deterministic invariants of generation
//!
//! * [`ReferenceKernelGenerator::generate`] preserves the exact order of
//!   `LoweredPlan::kernels`, and independently re-verifies that each
//!   [`scirust_tensor_compile::LogicalKernelId`] equals its position.
//! * An artefact's entry-point name is derived *only* from its
//!   `LogicalKernelId` (`scirust_reference_kernel_<id>`) — it is not stored in
//!   the encoded bytes, so there is exactly one name for a given kernel and no
//!   second source of truth that could disagree with it.
//!   [`PreparedReferenceKernel::from_kernel_module`] re-derives it and rejects a
//!   module whose `entry_point` disagrees.
//! * Two generations of the same `LoweredPlan` produce identical artefacts,
//!   identical entry-point names, identical encoded bytes and identical
//!   `KernelModule`s, in identical order — independent of memory addresses,
//!   thread count, or any hash-map iteration order (this crate uses none).
//!
//! # Supported and rejected operations
//!
//! The artefact format covers every kernel family
//! `scirust_tensor_compile::KernelLowerer` produces in this phase: element-wise
//! unary operations, element-wise binary operations, shape-copy/reduction/
//! permutation families, rank-2 `MatMul` and identical-prefix `BatchMatMul`.
//! The Reference CPU interpreter executes the supported F32 families, including
//! both matrix-product families, **except `Exp` and `Log`**, as explained above.
//!
//! Matrix-product lowering must already have established `[M,K] @ [K,N] ->
//! [M,N]` for `MatMul` or an identical batch prefix for `BatchMatMul`; generation
//! and preparation retain those layouts and reject malformed artefacts rather
//! than silently adapting them. Any kernel family this crate does not recognise
//! is rejected with [`ReferenceGenerationError::UnsupportedKernelFamily`]. There
//! is no silent fallback, no no-op substitution and no partial artefact.
//!
//! Reference v1.1 generates exactly `DType::F32`. Every other `DType` SciRust
//! declares has a *reserved, stable wire tag*, so a decoder can always
//! distinguish a tag it has never heard of from a `DType` it recognises but does
//! not support — but generating, decoding or executing a kernel of any other
//! type is rejected. No type is ever silently narrowed to `F32`.
//!
//! # Safety and failure discipline
//!
//! The crate forbids `unsafe`. No preparation or execution path contains
//! `panic!`, `unwrap()`, `expect()` or unchecked indexing: every buffer, shape
//! and stride access goes through `get`/`get_mut`, every index arithmetic step
//! is checked, and a broken internal invariant surfaces as a typed error rather
//! than an abort.
//!
//! [`ReferenceInterpreter::execute`] validates an invocation completely before
//! writing anything, so a rejected call leaves the caller's output buffer
//! bit-for-bit unchanged.

#![forbid(unsafe_code)]

mod artifact;
mod codec;
mod error;
mod format;
mod generator;
mod interpreter;
mod invocation;

pub use artifact::{ReferenceAttributes, ReferenceKernelArtifact, ReferenceTensorLayout};
pub use error::{ReferenceDecodeError, ReferenceExecutionError, ReferenceGenerationError};
pub use format::{
    REFERENCE_FORMAT_VERSION, REFERENCE_MAGIC, REFERENCE_MAX_RANK, ReferenceFormatVersion,
    ReferenceOpcode,
};
pub use generator::{ReferenceKernelGenerator, ReferenceKernelSet};
pub use interpreter::{PreparedReferenceKernel, ReferenceInterpreter};

#[cfg(test)]
mod tests;
