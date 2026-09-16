//! Optimized CPU execution for Reference matrix-product kernels.
//!
//! This module deliberately does **not** replace `cpu_reference`: the existing
//! scalar interpreter remains the deterministic oracle. The SIMD adapter routes
//! only `MatMul` / `BatchMatMul` here, while every other Reference opcode keeps
//! the established oracle implementation.
//!
//! The binding and failure-atomicity contract matches the scalar CPU Reference
//! path: validate all bindings, copy all operands out of the byte buffers,
//! compute into temporary F32 storage, then write the result only after the
//! whole kernel succeeds. The optimized part is therefore the GEMM itself, not
//! looser buffer semantics.

use core::fmt;

use scirust_compute::{BufferAccess, BufferBinding, KernelFormat, KernelModule, LaunchConfig};
use scirust_simd::gemm::sgemm_tiled;
use scirust_simd::matrix::view::{MatrixView, MatrixViewMut};
use scirust_tensor_reference::{
    PreparedReferenceKernel, ReferenceKernelArtifact, ReferenceOpcode,
};

use crate::compute_adapter::CpuBuffer;

const F32_BYTES: usize = 4;
const CANONICAL_GRID: [u32; 3] = [1, 1, 1];
const CANONICAL_BLOCK: [u32; 3] = [1, 1, 1];
const CANONICAL_SHARED_MEMORY_BYTES: u32 = 0;

/// Matrix dimensions extracted once, when the Reference module is compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpuSimdMatrixSpec {
    pub batches: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CpuSimdReferenceError {
    InvalidEncoding(String),
    UnsupportedMatrixLayout,
    DimensionOverflow,
    InvalidLaunchConfig,
    InvalidBindingCount { expected: usize, actual: usize },
    DuplicateBindingSlot { slot: u32 },
    UnexpectedBindingSlot { slot: u32 },
    MissingBindingSlot { slot: u32 },
    InvalidOperandAccess { slot: u32, access: BufferAccess },
    InvalidOutputAccess { slot: u32, access: BufferAccess },
    MisalignedRange { slot: u32 },
    BindingLengthMismatch {
        slot: u32,
        expected: usize,
        actual: usize,
    },
    BufferRangeOverflow { slot: u32 },
    BufferOutOfBounds { slot: u32 },
    BufferBorrowConflict { slot: u32 },
    InternalShapeMismatch,
}

impl fmt::Display for CpuSimdReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEncoding(error) => write!(f, "invalid Reference encoding: {error}"),
            Self::UnsupportedMatrixLayout => write!(f, "unsupported Reference matrix layout"),
            Self::DimensionOverflow => write!(f, "matrix dimension or byte range overflows usize"),
            Self::InvalidLaunchConfig => write!(
                f,
                "SIMD Reference kernels require grid {CANONICAL_GRID:?}, block {CANONICAL_BLOCK:?} and zero shared bytes"
            ),
            Self::InvalidBindingCount { expected, actual } => write!(
                f,
                "SIMD matrix kernel needs exactly {expected} bindings but received {actual}"
            ),
            Self::DuplicateBindingSlot { slot } => write!(f, "binding slot {slot} is duplicated"),
            Self::UnexpectedBindingSlot { slot } => write!(f, "binding slot {slot} is outside 0..=2"),
            Self::MissingBindingSlot { slot } => write!(f, "binding slot {slot} is missing"),
            Self::InvalidOperandAccess { slot, access } => write!(
                f,
                "operand slot {slot} declares {access:?}; SIMD matrix operands must be ReadOnly"
            ),
            Self::InvalidOutputAccess { slot, access } => write!(
                f,
                "output slot {slot} declares {access:?}; SIMD matrix output must be WriteOnly"
            ),
            Self::MisalignedRange { slot } => write!(f, "slot {slot} is not aligned to F32 bytes"),
            Self::BindingLengthMismatch {
                slot,
                expected,
                actual,
            } => write!(
                f,
                "slot {slot} must bind {expected} bytes but binds {actual}"
            ),
            Self::BufferRangeOverflow { slot } => write!(f, "slot {slot} range overflows usize"),
            Self::BufferOutOfBounds { slot } => write!(f, "slot {slot} range exceeds its buffer"),
            Self::BufferBorrowConflict { slot } => write!(f, "slot {slot} buffer is already borrowed"),
            Self::InternalShapeMismatch => write!(f, "prepared matrix shape disagrees with Reference lengths"),
        }
    }
}

/// Extract a checked row-major matrix-product shape from one Reference module.
/// Non-matrix opcodes return `Ok(None)` and remain on the scalar Reference path.
pub(crate) fn matrix_spec_from_module(
    module: &KernelModule,
) -> Result<Option<CpuSimdMatrixSpec>, CpuSimdReferenceError> {
    if module.format != KernelFormat::Reference {
        return Ok(None);
    }
    let artifact = ReferenceKernelArtifact::decode(&module.code)
        .map_err(|error| CpuSimdReferenceError::InvalidEncoding(error.to_string()))?;
    let opcode = artifact.opcode();
    if !matches!(opcode, ReferenceOpcode::MatMul | ReferenceOpcode::BatchMatMul) {
        return Ok(None);
    }

    let lhs = artifact
        .operands()
        .first()
        .ok_or(CpuSimdReferenceError::UnsupportedMatrixLayout)?;
    let rhs = artifact
        .operands()
        .get(1)
        .ok_or(CpuSimdReferenceError::UnsupportedMatrixLayout)?;
    let lhs_dims = checked_dims(lhs.dims())?;
    let rhs_dims = checked_dims(rhs.dims())?;
    let out_dims = checked_dims(artifact.result().dims())?;

    let rank = lhs_dims.len();
    let batched = opcode == ReferenceOpcode::BatchMatMul;
    if (!batched && (rank != 2 || rhs_dims.len() != 2 || out_dims.len() != 2))
        || (batched && (rank < 3 || rhs_dims.len() != rank || out_dims.len() != rank))
    {
        return Err(CpuSimdReferenceError::UnsupportedMatrixLayout);
    }

    let matrix_axis = rank - 2;
    let prefixes_match = !batched
        || (lhs_dims[..matrix_axis] == rhs_dims[..matrix_axis]
            && lhs_dims[..matrix_axis] == out_dims[..matrix_axis]);
    let contract = lhs_dims[matrix_axis + 1] == rhs_dims[matrix_axis]
        && out_dims[matrix_axis] == lhs_dims[matrix_axis]
        && out_dims[matrix_axis + 1] == rhs_dims[matrix_axis + 1];
    if !prefixes_match || !contract {
        return Err(CpuSimdReferenceError::UnsupportedMatrixLayout);
    }

    let batches = if batched {
        checked_product(&lhs_dims[..matrix_axis])?
    } else {
        1
    };

    Ok(Some(CpuSimdMatrixSpec {
        batches,
        m: lhs_dims[matrix_axis],
        k: lhs_dims[matrix_axis + 1],
        n: rhs_dims[matrix_axis + 1],
    }))
}

/// Execute a prepared matrix product using SciRust's tiled SIMD SGEMM.
pub(crate) fn launch_matrix_kernel(
    prepared: &PreparedReferenceKernel,
    spec: CpuSimdMatrixSpec,
    config: LaunchConfig,
    bindings: &[BufferBinding<'_, CpuBuffer>],
) -> Result<(), CpuSimdReferenceError> {
    if config.grid != CANONICAL_GRID
        || config.block != CANONICAL_BLOCK
        || config.shared_memory_bytes != CANONICAL_SHARED_MEMORY_BYTES
    {
        return Err(CpuSimdReferenceError::InvalidLaunchConfig);
    }

    let operand_lengths = prepared.operand_lengths();
    if operand_lengths.len() != 2 {
        return Err(CpuSimdReferenceError::InternalShapeMismatch);
    }
    if bindings.len() != 3 {
        return Err(CpuSimdReferenceError::InvalidBindingCount {
            expected: 3,
            actual: bindings.len(),
        });
    }

    let resolved = resolve_bindings(bindings)?;
    let lhs_bytes = checked_bytes(operand_lengths[0])?;
    let rhs_bytes = checked_bytes(operand_lengths[1])?;
    let out_bytes = checked_bytes(prepared.result_length())?;
    validate_binding(resolved[0], lhs_bytes, true)?;
    validate_binding(resolved[1], rhs_bytes, true)?;
    validate_binding(resolved[2], out_bytes, false)?;

    // Copy both operands before the first output write. This also makes
    // overlapping input/output bindings safe and preserves failure atomicity.
    let lhs = read_f32(resolved[0])?;
    let rhs = read_f32(resolved[1])?;
    let mut output = vec![0.0f32; prepared.result_length()];

    let lhs_matrix = checked_product(&[spec.m, spec.k])?;
    let rhs_matrix = checked_product(&[spec.k, spec.n])?;
    let out_matrix = checked_product(&[spec.m, spec.n])?;
    if checked_product(&[spec.batches, lhs_matrix])? != lhs.len()
        || checked_product(&[spec.batches, rhs_matrix])? != rhs.len()
        || checked_product(&[spec.batches, out_matrix])? != output.len()
    {
        return Err(CpuSimdReferenceError::InternalShapeMismatch);
    }

    for batch in 0..spec.batches {
        let lhs_start = batch
            .checked_mul(lhs_matrix)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;
        let rhs_start = batch
            .checked_mul(rhs_matrix)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;
        let out_start = batch
            .checked_mul(out_matrix)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;
        let lhs_end = lhs_start
            .checked_add(lhs_matrix)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;
        let rhs_end = rhs_start
            .checked_add(rhs_matrix)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;
        let out_end = out_start
            .checked_add(out_matrix)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;

        let lhs_batch = lhs
            .get(lhs_start..lhs_end)
            .ok_or(CpuSimdReferenceError::InternalShapeMismatch)?;
        let rhs_batch = rhs
            .get(rhs_start..rhs_end)
            .ok_or(CpuSimdReferenceError::InternalShapeMismatch)?;
        let out_batch = output
            .get_mut(out_start..out_end)
            .ok_or(CpuSimdReferenceError::InternalShapeMismatch)?;

        sgemm_tiled(
            1.0,
            MatrixView::new(lhs_batch, spec.m, spec.k),
            MatrixView::new(rhs_batch, spec.k, spec.n),
            0.0,
            MatrixViewMut::new(out_batch, spec.m, spec.n),
        );
    }

    write_f32(resolved[2], &output)
}

fn resolve_bindings<'a>(
    bindings: &'a [BufferBinding<'a, CpuBuffer>],
) -> Result<[&'a BufferBinding<'a, CpuBuffer>; 3], CpuSimdReferenceError> {
    let mut slots: [Option<&BufferBinding<'_, CpuBuffer>>; 3] = [None, None, None];
    for binding in bindings {
        let index = usize::try_from(binding.slot)
            .map_err(|_| CpuSimdReferenceError::UnexpectedBindingSlot { slot: binding.slot })?;
        let slot = slots
            .get_mut(index)
            .ok_or(CpuSimdReferenceError::UnexpectedBindingSlot { slot: binding.slot })?;
        if slot.is_some() {
            return Err(CpuSimdReferenceError::DuplicateBindingSlot { slot: binding.slot });
        }
        *slot = Some(binding);
    }
    Ok([
        slots[0].ok_or(CpuSimdReferenceError::MissingBindingSlot { slot: 0 })?,
        slots[1].ok_or(CpuSimdReferenceError::MissingBindingSlot { slot: 1 })?,
        slots[2].ok_or(CpuSimdReferenceError::MissingBindingSlot { slot: 2 })?,
    ])
}

fn validate_binding(
    binding: &BufferBinding<'_, CpuBuffer>,
    expected_bytes: usize,
    operand: bool,
) -> Result<(), CpuSimdReferenceError> {
    if operand {
        if binding.access != BufferAccess::ReadOnly {
            return Err(CpuSimdReferenceError::InvalidOperandAccess {
                slot: binding.slot,
                access: binding.access,
            });
        }
    } else if binding.access != BufferAccess::WriteOnly {
        return Err(CpuSimdReferenceError::InvalidOutputAccess {
            slot: binding.slot,
            access: binding.access,
        });
    }
    if binding.offset_bytes % F32_BYTES != 0 || binding.length_bytes % F32_BYTES != 0 {
        return Err(CpuSimdReferenceError::MisalignedRange { slot: binding.slot });
    }
    if binding.length_bytes != expected_bytes {
        return Err(CpuSimdReferenceError::BindingLengthMismatch {
            slot: binding.slot,
            expected: expected_bytes,
            actual: binding.length_bytes,
        });
    }
    let end = binding
        .offset_bytes
        .checked_add(binding.length_bytes)
        .ok_or(CpuSimdReferenceError::BufferRangeOverflow { slot: binding.slot })?;
    if end > binding.buffer.len() {
        return Err(CpuSimdReferenceError::BufferOutOfBounds { slot: binding.slot });
    }
    Ok(())
}

fn read_f32(binding: &BufferBinding<'_, CpuBuffer>) -> Result<Vec<f32>, CpuSimdReferenceError> {
    let end = binding
        .offset_bytes
        .checked_add(binding.length_bytes)
        .ok_or(CpuSimdReferenceError::BufferRangeOverflow { slot: binding.slot })?;
    let bytes = binding
        .buffer
        .bytes
        .try_borrow()
        .map_err(|_| CpuSimdReferenceError::BufferBorrowConflict { slot: binding.slot })?;
    let range = bytes
        .get(binding.offset_bytes..end)
        .ok_or(CpuSimdReferenceError::BufferOutOfBounds { slot: binding.slot })?;
    let mut values = Vec::with_capacity(range.len() / F32_BYTES);
    for chunk in range.chunks_exact(F32_BYTES) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn write_f32(
    binding: &BufferBinding<'_, CpuBuffer>,
    values: &[f32],
) -> Result<(), CpuSimdReferenceError> {
    let end = binding
        .offset_bytes
        .checked_add(binding.length_bytes)
        .ok_or(CpuSimdReferenceError::BufferRangeOverflow { slot: binding.slot })?;
    let mut bytes = binding
        .buffer
        .bytes
        .try_borrow_mut()
        .map_err(|_| CpuSimdReferenceError::BufferBorrowConflict { slot: binding.slot })?;
    let range = bytes
        .get_mut(binding.offset_bytes..end)
        .ok_or(CpuSimdReferenceError::BufferOutOfBounds { slot: binding.slot })?;
    if range.len() != values.len() * F32_BYTES {
        return Err(CpuSimdReferenceError::InternalShapeMismatch);
    }
    for (chunk, value) in range.chunks_exact_mut(F32_BYTES).zip(values) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn checked_bytes(elements: usize) -> Result<usize, CpuSimdReferenceError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or(CpuSimdReferenceError::DimensionOverflow)
}

fn checked_product(dims: &[usize]) -> Result<usize, CpuSimdReferenceError> {
    let mut product = 1usize;
    for &dimension in dims {
        product = product
            .checked_mul(dimension)
            .ok_or(CpuSimdReferenceError::DimensionOverflow)?;
    }
    Ok(product)
}

fn checked_dims(dims: &[u64]) -> Result<Vec<usize>, CpuSimdReferenceError> {
    dims.iter()
        .map(|&value| usize::try_from(value).map_err(|_| CpuSimdReferenceError::DimensionOverflow))
        .collect()
}
