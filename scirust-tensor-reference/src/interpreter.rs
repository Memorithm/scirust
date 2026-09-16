//! Serial CPU interpretation of one individual Reference kernel.
//!
//! # Numeric contract
//!
//! For every element-wise opcode (`Relu`, `Scale`, `Add`, `Sub`, `Mul`, `Div`)
//! this interpreter guarantees:
//!
//! * serial execution, one thread, no Rayon and no explicit SIMD;
//! * increasing element order, from index `0` upward;
//! * exactly one scalar operation per output element;
//! * no reassociation — no output element is the result of more than one
//!   arithmetic operation, so there is no accumulation order to specify;
//! * no `mul_add`, no explicit FMA;
//! * no fast-math, no reordering hints;
//! * no conversion to `f64`;
//! * plain Rust `f32` operators (`+`, `-`, `*`, `/`).
//!
//! Matrix products are also serial and row-major. Each output element accumulates
//! `lhs * rhs` terms in strictly increasing `K` order using separate multiply and
//! add expressions; this implementation does not call `mul_add` and never
//! promotes the accumulator to `f64`. Batched products traverse the flattened
//! batch prefix in increasing row-major order.
//!
//! That contract fixes *what this crate does*. It deliberately does **not**
//! claim that a compiled `f32` operator is bit-identical on every conforming
//! architecture — that is a property of the target and toolchain, not something
//! this crate can promise on their behalf. What is actually verified is
//! narrower and checked: golden-bit tests over the platforms this project's CI
//! covers.
//!
//! # NaN handling
//!
//! Bit-exact NaN preservation is guaranteed **only** where a value is returned
//! or copied without arithmetic:
//!
//! * the NaN branch of `Relu`, which returns its input untouched;
//! * `ShapeCopy`, which copies elements verbatim;
//! * `Permute`, which copies elements verbatim to permuted positions.
//!
//! For arithmetic operations, including matrix products, a NaN operand can yield
//! a NaN result, but **the payload of that NaN is not specified** by this crate:
//! IEEE 754 leaves the propagated payload to the implementation. Tests assert
//! `is_nan()` for such cases and never a specific payload bit pattern.
//!
//! # Scope
//!
//! One kernel at a time. This module does not execute a
//! `scirust_tensor_compile::LoweredPlan`, does not resolve inter-kernel
//! dependencies, does not consult a memory plan, does not allocate slots, does
//! not bind external inputs or constants, and does not order instructions — all
//! of that belongs to a later plan runtime. It also touches no backend buffer,
//! no device and no `scirust_compute::ComputeBackend`; see the crate
//! documentation for the CPU adapter bridge.

use scirust_compute::{DType, KernelFormat, KernelModule};
use scirust_tensor_compile::LogicalKernelId;

use crate::artifact::{ReferenceAttributes, ReferenceKernelArtifact};
use crate::error::ReferenceExecutionError;
use crate::format::{REFERENCE_MAX_RANK, ReferenceOpcode};
use crate::invocation;

/// Upper bound on rank, as a `usize`, matching [`REFERENCE_MAX_RANK`].
const MAX_RANK: usize = REFERENCE_MAX_RANK as usize;

/// Attribute payload converted once, at preparation, into the exact form the
/// executor needs.
///
/// `MatrixProduct` is derived from tensor layouts in the Reference artefact; it
/// is runtime preparation metadata, not an additional wire-format attribute.
#[derive(Debug, Clone)]
pub(crate) enum PreparedAttributes {
    None,
    Scale {
        factor: f32,
    },
    Permute {
        input_dims: Vec<usize>,
        output_dims: Vec<usize>,
        input_strides: Vec<usize>,
        output_strides: Vec<usize>,
        permutation: Vec<usize>,
    },
    MatrixProduct {
        batches: usize,
        m: usize,
        k: usize,
        n: usize,
    },
}

/// A Reference artefact decoded, validated and converted into the form the CPU
/// interpreter executes.
///
/// Holds no pointer, no borrowed data, no device handle and no backend buffer:
/// it is `'static` and independent of any invocation. All fields are private,
/// and the only ways to build one are [`Self::from_artifact`] and
/// [`Self::from_kernel_module`], both of which reject anything they cannot
/// execute.
///
/// `Exp` and `Log` are rejected **here**, at preparation, rather than at
/// execution: a caller learns that a kernel is not executable before it ever
/// assembles operand buffers for it.
#[derive(Debug, Clone)]
pub struct PreparedReferenceKernel {
    kernel_id: LogicalKernelId,
    opcode: ReferenceOpcode,
    dtype: DType,
    operand_lengths: Vec<usize>,
    result_length: usize,
    prepared_attributes: PreparedAttributes,
}

impl PreparedReferenceKernel {
    /// Prepares a decoded artefact for CPU execution.
    ///
    /// Rejects a non-`F32` element type, `Exp`, `Log`, and any attribute payload
    /// inconsistent with the opcode. Converts every `u64` dimension and element
    /// count to `usize` through a checked conversion, precomputes `Permute`
    /// strides, and extracts checked matrix-product dimensions once.
    pub fn from_artifact(
        artifact: &ReferenceKernelArtifact,
    ) -> Result<Self, ReferenceExecutionError> {
        let opcode = artifact.opcode();

        // Reject what this interpreter cannot evaluate, before any other work.
        match opcode
        {
            ReferenceOpcode::Exp | ReferenceOpcode::Log =>
            {
                return Err(ReferenceExecutionError::DeterministicMathUnavailable { opcode });
            },
            ReferenceOpcode::ReluGrad
            | ReferenceOpcode::BroadcastTo
            | ReferenceOpcode::ReduceSumTo =>
            {
                return Err(ReferenceExecutionError::UnsupportedOpcode { opcode });
            },
            ReferenceOpcode::Relu
            | ReferenceOpcode::ZerosLike
            | ReferenceOpcode::OnesLike
            | ReferenceOpcode::Scale
            | ReferenceOpcode::Add
            | ReferenceOpcode::Sub
            | ReferenceOpcode::Mul
            | ReferenceOpcode::Div
            | ReferenceOpcode::ShapeCopy
            | ReferenceOpcode::Permute
            | ReferenceOpcode::MatMul
            | ReferenceOpcode::BatchMatMul =>
            {},
        }

        let dtype = artifact.dtype();
        if dtype != DType::F32
        {
            return Err(ReferenceExecutionError::UnsupportedDType { dtype });
        }

        let mut operand_lengths = Vec::with_capacity(artifact.operands().len());
        for layout in artifact.operands()
        {
            operand_lengths.push(checked_usize(layout.elements())?);
        }
        let result_length = checked_usize(artifact.result().elements())?;

        let prepared_attributes = match (opcode, artifact.attributes())
        {
            (ReferenceOpcode::Scale, ReferenceAttributes::Scale { factor_bits }) =>
            {
                PreparedAttributes::Scale {
                    factor: f32::from_bits(*factor_bits),
                }
            },
            (ReferenceOpcode::Permute, ReferenceAttributes::Permute { axes }) =>
            {
                let operand = artifact
                    .operands()
                    .first()
                    .ok_or(ReferenceExecutionError::AttributeMismatch { opcode })?;

                let input_dims = checked_dims(operand.dims())?;
                let output_dims = checked_dims(artifact.result().dims())?;

                if axes.len() > MAX_RANK
                    || input_dims.len() > MAX_RANK
                    || output_dims.len() > MAX_RANK
                {
                    return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
                }

                let mut permutation = Vec::with_capacity(axes.len());
                for &axis in axes
                {
                    permutation.push(usize::from(axis));
                }

                let input_strides = row_major_strides(&input_dims)?;
                let output_strides = row_major_strides(&output_dims)?;

                PreparedAttributes::Permute {
                    input_dims,
                    output_dims,
                    input_strides,
                    output_strides,
                    permutation,
                }
            },
            (
                ReferenceOpcode::MatMul | ReferenceOpcode::BatchMatMul,
                ReferenceAttributes::None,
            ) => prepare_matrix_product(artifact, opcode)?,
            (
                ReferenceOpcode::Relu
                | ReferenceOpcode::ZerosLike
                | ReferenceOpcode::OnesLike
                | ReferenceOpcode::Add
                | ReferenceOpcode::Sub
                | ReferenceOpcode::Mul
                | ReferenceOpcode::Div
                | ReferenceOpcode::ShapeCopy,
                ReferenceAttributes::None,
            ) => PreparedAttributes::None,
            _ =>
            {
                return Err(ReferenceExecutionError::AttributeMismatch { opcode });
            },
        };

        Ok(Self {
            kernel_id: artifact.kernel_id(),
            opcode,
            dtype,
            operand_lengths,
            result_length,
            prepared_attributes,
        })
    }

    /// Prepares a [`KernelModule`] for CPU execution.
    pub fn from_kernel_module(module: &KernelModule) -> Result<Self, ReferenceExecutionError> {
        if module.format != KernelFormat::Reference
        {
            return Err(ReferenceExecutionError::WrongKernelFormat {
                found: module.format,
            });
        }

        let artifact = ReferenceKernelArtifact::decode(&module.code)
            .map_err(ReferenceExecutionError::InvalidEncoding)?;

        let expected = artifact.entry_point();
        if module.entry_point != expected
        {
            return Err(ReferenceExecutionError::EntryPointMismatch {
                expected,
                found: module.entry_point.clone(),
            });
        }

        Self::from_artifact(&artifact)
    }

    /// The canonical [`LogicalKernelId`] this kernel was generated from.
    pub const fn kernel_id(&self) -> LogicalKernelId {
        self.kernel_id
    }

    pub const fn opcode(&self) -> ReferenceOpcode {
        self.opcode
    }

    /// Element type of every operand and of the result. Always `F32` for a
    /// kernel this crate agreed to prepare.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Element count each operand slice must hold, in operand order.
    pub fn operand_lengths(&self) -> &[usize] {
        &self.operand_lengths
    }

    /// Element count the output slice must hold.
    pub const fn result_length(&self) -> usize {
        self.result_length
    }

    pub(crate) const fn attributes(&self) -> &PreparedAttributes {
        &self.prepared_attributes
    }
}

/// Serial CPU interpreter for one prepared Reference kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferenceInterpreter;

impl ReferenceInterpreter {
    pub const fn new() -> Self {
        Self
    }

    /// Executes `prepared` over `operands`, writing into `output`.
    ///
    /// Every invocation is validated in full before the first write, so a
    /// rejected call leaves `output` bit-for-bit unchanged.
    pub fn execute(
        &self,
        prepared: &PreparedReferenceKernel,
        operands: &[&[f32]],
        output: &mut [f32],
    ) -> Result<(), ReferenceExecutionError> {
        invocation::validate(prepared, operands, output)?;

        let opcode = prepared.opcode();

        match opcode
        {
            ReferenceOpcode::Relu =>
            {
                let operand = first_operand(operands)?;
                for (target, &value) in output.iter_mut().zip(operand.iter())
                {
                    *target = relu(value);
                }
            },
            ReferenceOpcode::ZerosLike =>
            {
                let _ = first_operand(operands)?;
                output.fill(0.0);
            },
            ReferenceOpcode::OnesLike =>
            {
                let _ = first_operand(operands)?;
                output.fill(1.0);
            },
            ReferenceOpcode::Scale =>
            {
                let operand = first_operand(operands)?;
                let PreparedAttributes::Scale { factor } = prepared.attributes()
                else
                {
                    return Err(ReferenceExecutionError::AttributeMismatch { opcode });
                };
                let factor = *factor;
                for (target, &value) in output.iter_mut().zip(operand.iter())
                {
                    *target = value * factor;
                }
            },
            ReferenceOpcode::Add =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs + rhs;
                }
            },
            ReferenceOpcode::Sub =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs - rhs;
                }
            },
            ReferenceOpcode::Mul =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs * rhs;
                }
            },
            ReferenceOpcode::Div =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs / rhs;
                }
            },
            ReferenceOpcode::MatMul | ReferenceOpcode::BatchMatMul =>
            {
                execute_matrix_product(prepared.attributes(), operands, output)?;
            },
            ReferenceOpcode::ShapeCopy =>
            {
                let operand = first_operand(operands)?;
                if operand.len() != output.len()
                {
                    return Err(ReferenceExecutionError::OutputLengthMismatch {
                        expected: operand.len(),
                        actual: output.len(),
                    });
                }
                output.copy_from_slice(operand);
            },
            ReferenceOpcode::Permute =>
            {
                let operand = first_operand(operands)?;
                let PreparedAttributes::Permute {
                    input_dims,
                    output_dims,
                    input_strides,
                    output_strides,
                    permutation,
                } = prepared.attributes()
                else
                {
                    return Err(ReferenceExecutionError::AttributeMismatch { opcode });
                };

                execute_permute(
                    input_dims,
                    output_dims,
                    input_strides,
                    output_strides,
                    permutation,
                    operand,
                    output,
                )?;
            },
            ReferenceOpcode::Exp | ReferenceOpcode::Log =>
            {
                return Err(ReferenceExecutionError::DeterministicMathUnavailable { opcode });
            },
            ReferenceOpcode::ReluGrad
            | ReferenceOpcode::BroadcastTo
            | ReferenceOpcode::ReduceSumTo =>
            {
                return Err(ReferenceExecutionError::UnsupportedOpcode { opcode });
            },
        }

        Ok(())
    }
}

fn prepare_matrix_product(
    artifact: &ReferenceKernelArtifact,
    opcode: ReferenceOpcode,
) -> Result<PreparedAttributes, ReferenceExecutionError> {
    let lhs = artifact
        .operands()
        .first()
        .ok_or(ReferenceExecutionError::AttributeMismatch { opcode })?;
    let rhs = artifact
        .operands()
        .get(1)
        .ok_or(ReferenceExecutionError::AttributeMismatch { opcode })?;
    let lhs_dims = checked_dims(lhs.dims())?;
    let rhs_dims = checked_dims(rhs.dims())?;
    let output_dims = checked_dims(artifact.result().dims())?;

    let rank = lhs_dims.len();
    let batched = opcode == ReferenceOpcode::BatchMatMul;
    if (!batched && (rank != 2 || rhs_dims.len() != 2 || output_dims.len() != 2))
        || (batched && (rank < 3 || rhs_dims.len() != rank || output_dims.len() != rank))
    {
        return Err(ReferenceExecutionError::AttributeMismatch { opcode });
    }

    let matrix_axis = rank - 2;
    let prefixes_match = !batched
        || (lhs_dims[..matrix_axis] == rhs_dims[..matrix_axis]
            && lhs_dims[..matrix_axis] == output_dims[..matrix_axis]);
    let matrix_contract = lhs_dims.get(matrix_axis + 1) == rhs_dims.get(matrix_axis)
        && output_dims.get(matrix_axis) == lhs_dims.get(matrix_axis)
        && output_dims.get(matrix_axis + 1) == rhs_dims.get(matrix_axis + 1);
    if !prefixes_match || !matrix_contract
    {
        return Err(ReferenceExecutionError::AttributeMismatch { opcode });
    }

    let batches = if batched
    {
        checked_product(&lhs_dims[..matrix_axis])?
    }
    else
    {
        1
    };

    Ok(PreparedAttributes::MatrixProduct {
        batches,
        m: lhs_dims[matrix_axis],
        k: lhs_dims[matrix_axis + 1],
        n: rhs_dims[matrix_axis + 1],
    })
}

fn execute_matrix_product(
    attributes: &PreparedAttributes,
    operands: &[&[f32]],
    output: &mut [f32],
) -> Result<(), ReferenceExecutionError> {
    let (lhs, rhs) = binary_operands(operands)?;
    let PreparedAttributes::MatrixProduct { batches, m, k, n } = attributes
    else
    {
        return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
    };
    let (batches, m, k, n) = (*batches, *m, *k, *n);
    let lhs_matrix = m
        .checked_mul(k)
        .ok_or(ReferenceExecutionError::IndexOverflow)?;
    let rhs_matrix = k
        .checked_mul(n)
        .ok_or(ReferenceExecutionError::IndexOverflow)?;
    let out_matrix = m
        .checked_mul(n)
        .ok_or(ReferenceExecutionError::IndexOverflow)?;

    for batch in 0..batches
    {
        let lhs_base = batch
            .checked_mul(lhs_matrix)
            .ok_or(ReferenceExecutionError::IndexOverflow)?;
        let rhs_base = batch
            .checked_mul(rhs_matrix)
            .ok_or(ReferenceExecutionError::IndexOverflow)?;
        let out_base = batch
            .checked_mul(out_matrix)
            .ok_or(ReferenceExecutionError::IndexOverflow)?;

        for row in 0..m
        {
            let lhs_row = lhs_base
                .checked_add(
                    row.checked_mul(k)
                        .ok_or(ReferenceExecutionError::IndexOverflow)?,
                )
                .ok_or(ReferenceExecutionError::IndexOverflow)?;
            let out_row = out_base
                .checked_add(
                    row.checked_mul(n)
                        .ok_or(ReferenceExecutionError::IndexOverflow)?,
                )
                .ok_or(ReferenceExecutionError::IndexOverflow)?;

            for col in 0..n
            {
                let mut sum = 0.0f32;
                for inner in 0..k
                {
                    let lhs_index = lhs_row
                        .checked_add(inner)
                        .ok_or(ReferenceExecutionError::IndexOverflow)?;
                    let rhs_index = rhs_base
                        .checked_add(
                            inner
                                .checked_mul(n)
                                .ok_or(ReferenceExecutionError::IndexOverflow)?,
                        )
                        .and_then(|base| base.checked_add(col))
                        .ok_or(ReferenceExecutionError::IndexOverflow)?;
                    let lhs_value = *lhs
                        .get(lhs_index)
                        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
                    let rhs_value = *rhs
                        .get(rhs_index)
                        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
                    sum += lhs_value * rhs_value;
                }
                let output_index = out_row
                    .checked_add(col)
                    .ok_or(ReferenceExecutionError::IndexOverflow)?;
                let target = output
                    .get_mut(output_index)
                    .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
                *target = sum;
            }
        }
    }

    Ok(())
}

/// Rectified linear unit, with fully specified behaviour on every `f32` class.
#[inline]
fn relu(x: f32) -> f32 {
    if x.is_nan() || x > 0.0 { x } else { 0.0 }
}

fn first_operand<'a>(operands: &[&'a [f32]]) -> Result<&'a [f32], ReferenceExecutionError> {
    operands
        .first()
        .copied()
        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)
}

fn binary_operands<'a>(
    operands: &[&'a [f32]],
) -> Result<(&'a [f32], &'a [f32]), ReferenceExecutionError> {
    let left = operands
        .first()
        .copied()
        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
    let right = operands
        .get(1)
        .copied()
        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;

    Ok((left, right))
}

fn row_major_strides(dims: &[usize]) -> Result<Vec<usize>, ReferenceExecutionError> {
    let mut strides = vec![0usize; dims.len()];
    let mut accumulator = 1usize;

    for axis in (0..dims.len()).rev()
    {
        let stride = strides
            .get_mut(axis)
            .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
        *stride = accumulator;

        let dim = *dims
            .get(axis)
            .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
        accumulator = accumulator
            .checked_mul(dim)
            .ok_or(ReferenceExecutionError::IndexOverflow)?;
    }

    Ok(strides)
}

fn execute_permute(
    input_dims: &[usize],
    output_dims: &[usize],
    input_strides: &[usize],
    output_strides: &[usize],
    permutation: &[usize],
    operand: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceExecutionError> {
    if output.is_empty()
    {
        return Ok(());
    }

    if permutation.len() != output_dims.len()
        || input_dims.len() != output_dims.len()
        || input_strides.len() != input_dims.len()
        || output_strides.len() != output_dims.len()
        || output_dims.len() > MAX_RANK
    {
        return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
    }

    let mut output_coordinates = [0usize; MAX_RANK];

    for (output_index, target) in output.iter_mut().enumerate()
    {
        for (axis, (&stride, &dim)) in output_strides.iter().zip(output_dims.iter()).enumerate()
        {
            if stride == 0 || dim == 0
            {
                return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
            }

            let coordinate = (output_index / stride) % dim;
            let slot = output_coordinates
                .get_mut(axis)
                .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
            *slot = coordinate;
        }

        let mut source_index = 0usize;
        for (axis, &input_axis) in permutation.iter().enumerate()
        {
            let coordinate = *output_coordinates
                .get(axis)
                .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
            let input_stride = *input_strides
                .get(input_axis)
                .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;

            let term = coordinate
                .checked_mul(input_stride)
                .ok_or(ReferenceExecutionError::IndexOverflow)?;
            source_index = source_index
                .checked_add(term)
                .ok_or(ReferenceExecutionError::IndexOverflow)?;
        }

        let value = *operand
            .get(source_index)
            .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
        *target = value;
    }

    Ok(())
}

fn checked_product(dims: &[usize]) -> Result<usize, ReferenceExecutionError> {
    let mut product = 1usize;
    for &dimension in dims
    {
        product = product
            .checked_mul(dimension)
            .ok_or(ReferenceExecutionError::IndexOverflow)?;
    }
    Ok(product)
}

/// Checked `u64 -> usize` conversion for one dimension or element count.
fn checked_usize(value: u64) -> Result<usize, ReferenceExecutionError> {
    usize::try_from(value)
        .map_err(|_| ReferenceExecutionError::DimensionConversionOverflow { dimension: value })
}

/// Checked `u64 -> usize` conversion for a whole dimension list.
fn checked_dims(dims: &[u64]) -> Result<Vec<usize>, ReferenceExecutionError> {
    let mut converted = Vec::with_capacity(dims.len());
    for &dim in dims
    {
        converted.push(checked_usize(dim)?);
    }

    Ok(converted)
}
