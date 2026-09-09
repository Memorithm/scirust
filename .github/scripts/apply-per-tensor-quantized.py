from pathlib import Path
import textwrap


def clean(value: str) -> str:
    value = textwrap.dedent(value).strip("\n")
    return "\n".join(line[1:] if line.startswith("|") else line for line in value.splitlines())


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    old = clean(old)
    new = clean(new)
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}\n--- needle ---\n{old}")
    file.write_text(text.replace(old, new, 1))


representation = "scirust-tensor-ir/src/representation.rs"

replace_once(
    representation,
    """
|//! [`PrimitiveRepresentation::Quantized`] and [`PrimitiveRepresentation::Sparse`]
|//! are retained as declaration skeletons only. Their typed components and
|//! declaration invariants can be explored and interned internally, but they are
|//! not valid logical-tensor representations until their layout and reconstruction
|//! geometry are explicitly defined. In particular, no storage saving is claimed
|//! merely from the sizes of codes/scales or indices/values.
|//!
|//! Codebook layouts, block geometry, packed/sub-bit payloads, sparse formats,
|//! cost models and backend-specific materialization remain intentionally out of
|//! scope.
""",
    """
|//! [`PrimitiveRepresentation::Quantized`] and [`PrimitiveRepresentation::Sparse`]
|//! remain declaration skeletons. [`PrimitiveRepresentation::QuantizedPerTensor`]
|//! is the first quantized family with complete reconstruction geometry: integer
|//! codes have exactly the logical tensor shape and one scalar floating scale is
|//! shared by the complete tensor. Reconstruction is `logical = code * scale`, so
|//! the affine zero-point is fixed to zero and occupies no physical storage.
|//!
|//! Codebook layouts, block geometry, non-zero affine zero-points, packed/sub-bit
|//! payloads, sparse formats, cost models and backend-specific materialization remain
|//! intentionally out of scope.
""",
)

replace_once(
    representation,
    """
|    Quantized {
|        /// Discrete code payload with an integer dtype.
|        codes: RepresentationComponent,
|        /// Continuous dequantization factors with a floating dtype.
|        scales: RepresentationComponent,
|    },
|    /// Sparse storage skeleton: positions of the nonzeros plus their numeric
""",
    """
|    Quantized {
|        /// Discrete code payload with an integer dtype.
|        codes: RepresentationComponent,
|        /// Continuous dequantization factors with a floating dtype.
|        scales: RepresentationComponent,
|    },
|    /// Per-tensor scaled quantization with an implicit zero-point of zero.
|    ///
|    /// `codes` must have exactly the logical tensor shape and `scale` must be a
|    /// scalar floating tensor. The reconstruction contract is
|    /// `logical = codes * scale`. This family defines geometry only; it does not
|    /// claim a kernel, packed/sub-bit storage, or an automatic quantization policy.
|    QuantizedPerTensor {
|        /// Integer code payload with the complete logical tensor shape.
|        codes: RepresentationComponent,
|        /// Scalar floating scale shared by every code.
|        scale: RepresentationComponent,
|    },
|    /// Sparse storage skeleton: positions of the nonzeros plus their numeric
""",
)

replace_once(
    representation,
    """
|    pub const fn quantized(
|        codes: RepresentationComponent,
|        scales: RepresentationComponent,
|    ) -> Self {
|        Self::Quantized { codes, scales }
|    }
|
|    /// Construct a sparse representation from indices and values.
""",
    """
|    pub const fn quantized(
|        codes: RepresentationComponent,
|        scales: RepresentationComponent,
|    ) -> Self {
|        Self::Quantized { codes, scales }
|    }
|
|    /// Construct per-tensor scaled quantization with an implicit zero-point of zero.
|    pub const fn quantized_per_tensor(
|        codes: RepresentationComponent,
|        scale: RepresentationComponent,
|    ) -> Self {
|        Self::QuantizedPerTensor { codes, scale }
|    }
|
|    /// Construct a sparse representation from indices and values.
""",
)

replace_once(
    representation,
    """
|            Self::Factorized { .. } | Self::Quantized { .. } | Self::Sparse { .. } => None,
""",
    """
|            Self::Factorized { .. }
|            | Self::Quantized { .. }
|            | Self::QuantizedPerTensor { .. }
|            | Self::Sparse { .. } => None,
""",
)

replace_once(
    representation,
    """
|            Self::Quantized { codes, scales } =>
|            {
|                components[0] = Some(codes);
|                components[1] = Some(scales);
|            },
|            Self::Sparse { indices, values } =>
""",
    """
|            Self::Quantized { codes, scales } =>
|            {
|                components[0] = Some(codes);
|                components[1] = Some(scales);
|            },
|            Self::QuantizedPerTensor { codes, scale } =>
|            {
|                components[0] = Some(codes);
|                components[1] = Some(scale);
|            },
|            Self::Sparse { indices, values } =>
""",
)

replace_once(
    representation,
    """
|            Self::Quantized { .. } => Err(RepresentationError::QuantizedLayoutUndefined),
|            Self::Sparse { .. } => Err(RepresentationError::SparseLayoutUndefined),
""",
    """
|            Self::Quantized { .. } => Err(RepresentationError::QuantizedLayoutUndefined),
|            Self::QuantizedPerTensor { codes, scale } =>
|            {
|                if codes.tensor_type().shape == logical.shape
|                    && scale.tensor_type().shape.dims().is_empty()
|                {
|                    Ok(())
|                }
|                else
|                {
|                    Err(RepresentationError::QuantizedPerTensorIncompatibleShapes {
|                        codes: codes.tensor_type().shape.clone(),
|                        scale: scale.tensor_type().shape.clone(),
|                        logical: logical.shape.clone(),
|                    })
|                }
|            },
|            Self::Sparse { .. } => Err(RepresentationError::SparseLayoutUndefined),
""",
)

replace_once(
    representation,
    """
|            Self::Quantized { codes, scales } =>
|            {
|                let codes_dtype = codes.tensor_type().dtype;
|                let scales_dtype = scales.tensor_type().dtype;
|
|                if !is_integer_dtype(codes_dtype) || !is_float_dtype(scales_dtype)
|                {
|                    Err(RepresentationError::QuantizedInvalidComponentDTypes {
|                        codes: codes_dtype,
|                        scales: scales_dtype,
|                    })
|                }
|                else
|                {
|                    Ok(())
|                }
|            },
|            Self::Sparse { indices, values } =>
""",
    """
|            Self::Quantized { codes, scales } =>
|            {
|                let codes_dtype = codes.tensor_type().dtype;
|                let scales_dtype = scales.tensor_type().dtype;
|
|                if !is_integer_dtype(codes_dtype) || !is_float_dtype(scales_dtype)
|                {
|                    Err(RepresentationError::QuantizedInvalidComponentDTypes {
|                        codes: codes_dtype,
|                        scales: scales_dtype,
|                    })
|                }
|                else
|                {
|                    Ok(())
|                }
|            },
|            Self::QuantizedPerTensor { codes, scale } =>
|            {
|                let codes_dtype = codes.tensor_type().dtype;
|                let scale_dtype = scale.tensor_type().dtype;
|
|                if !is_integer_dtype(codes_dtype) || !is_float_dtype(scale_dtype)
|                {
|                    Err(RepresentationError::QuantizedInvalidComponentDTypes {
|                        codes: codes_dtype,
|                        scales: scale_dtype,
|                    })
|                }
|                else
|                {
|                    Ok(())
|                }
|            },
|            Self::Sparse { indices, values } =>
""",
)

replace_once(
    representation,
    """
|    QuantizedInvalidComponentDTypes {
|        /// Dtype carried by the codes component.
|        codes: DType,
|        /// Dtype carried by the scales component.
|        scales: DType,
|    },
|    /// Sparse component dtypes violate the family contract.
""",
    """
|    QuantizedInvalidComponentDTypes {
|        /// Dtype carried by the codes component.
|        codes: DType,
|        /// Dtype carried by the scales component.
|        scales: DType,
|    },
|    /// Per-tensor quantized geometry does not reconstruct the logical tensor.
|    QuantizedPerTensorIncompatibleShapes {
|        /// Shape carried by the integer codes.
|        codes: Shape,
|        /// Shape carried by the shared scale; must be scalar.
|        scale: Shape,
|        /// Logical tensor shape being represented.
|        logical: Shape,
|    },
|    /// Sparse component dtypes violate the family contract.
""",
)

replace_once(
    representation,
    """
|            Self::QuantizedInvalidComponentDTypes { codes, scales } =>
|            {
|                write!(
|                    formatter,
|                    "quantized codes dtype {codes:?} must be integer-valued and scales dtype {scales:?} floating-point"
|                )
|            },
|            Self::SparseInvalidComponentDTypes { indices, values } =>
""",
    """
|            Self::QuantizedInvalidComponentDTypes { codes, scales } =>
|            {
|                write!(
|                    formatter,
|                    "quantized codes dtype {codes:?} must be integer-valued and scales dtype {scales:?} floating-point"
|                )
|            },
|            Self::QuantizedPerTensorIncompatibleShapes {
|                codes,
|                scale,
|                logical,
|            } =>
|            {
|                write!(
|                    formatter,
|                    "per-tensor quantized codes shape {codes:?} must equal logical shape {logical:?} and scale shape {scale:?} must be scalar"
|                )
|            },
|            Self::SparseInvalidComponentDTypes { indices, values } =>
""",
)

replace_once(
    representation,
    """
|    pub fn declare_factorized(
|        &mut self,
|        left_type: TensorType,
|        left_representation: RepresentationId,
|        right_type: TensorType,
|        right_representation: RepresentationId,
|    ) -> Result<RepresentationId, RepresentationError> {
|        let left = self.component(left_type, left_representation)?;
|        let right = self.component(right_type, right_representation)?;
|
|        self.declare(PrimitiveRepresentation::factorized(left, right))
|    }
|
|    /// Declare and intern one representation.
""",
    """
|    pub fn declare_factorized(
|        &mut self,
|        left_type: TensorType,
|        left_representation: RepresentationId,
|        right_type: TensorType,
|        right_representation: RepresentationId,
|    ) -> Result<RepresentationId, RepresentationError> {
|        let left = self.component(left_type, left_representation)?;
|        let right = self.component(right_type, right_representation)?;
|
|        self.declare(PrimitiveRepresentation::factorized(left, right))
|    }
|
|    /// Declare and intern per-tensor scaled quantization.
|    ///
|    /// The integer `codes` tensor must later match the complete logical tensor
|    /// shape when bound, while `scale` must be a scalar floating tensor. Both
|    /// components are resolved against this plan, preserving declaration-order
|    /// and graph-independent component validation.
|    pub fn declare_quantized_per_tensor(
|        &mut self,
|        codes_type: TensorType,
|        codes_representation: RepresentationId,
|        scale_type: TensorType,
|        scale_representation: RepresentationId,
|    ) -> Result<RepresentationId, RepresentationError> {
|        let codes = self.component(codes_type, codes_representation)?;
|        let scale = self.component(scale_type, scale_representation)?;
|
|        self.declare(PrimitiveRepresentation::quantized_per_tensor(codes, scale))
|    }
|
|    /// Declare and intern one representation.
""",
)

replace_once(
    representation,
    """
|    /// Quantized and sparse declaration skeletons deliberately return a typed
|    /// error here: component byte counts alone do not prove that those
|    /// components reconstruct the requested logical tensor. All successful
|    /// accounting uses checked integer arithmetic and never floating-point.
""",
    """
|    /// The legacy quantized and sparse declaration skeletons deliberately return
|    /// typed errors here because component byte counts alone do not prove logical
|    /// reconstruction. `QuantizedPerTensor` has complete geometry and therefore
|    /// sums the exact recursive storage of its codes and scalar scale. All
|    /// successful accounting uses checked integer arithmetic and never floating-point.
""",
)

replace_once(
    representation,
    """
|            PrimitiveRepresentation::Quantized { codes, scales } =>
|            {
|                let codes_bits = self.storage_bits(codes.representation(), codes.tensor_type())?;
|                let scales_bits =
|                    self.storage_bits(scales.representation(), scales.tensor_type())?;
|
|                codes_bits
|                    .get()
|                    .checked_add(scales_bits.get())
|                    .map(StorageBits::new)
|                    .ok_or(RepresentationError::StorageSizeOverflow)
|            },
|            PrimitiveRepresentation::Sparse { indices, values } =>
""",
    """
|            PrimitiveRepresentation::Quantized { codes, scales } =>
|            {
|                let codes_bits = self.storage_bits(codes.representation(), codes.tensor_type())?;
|                let scales_bits =
|                    self.storage_bits(scales.representation(), scales.tensor_type())?;
|
|                codes_bits
|                    .get()
|                    .checked_add(scales_bits.get())
|                    .map(StorageBits::new)
|                    .ok_or(RepresentationError::StorageSizeOverflow)
|            },
|            PrimitiveRepresentation::QuantizedPerTensor { codes, scale } =>
|            {
|                let codes_bits = self.storage_bits(codes.representation(), codes.tensor_type())?;
|                let scale_bits = self.storage_bits(scale.representation(), scale.tensor_type())?;
|
|                codes_bits
|                    .get()
|                    .checked_add(scale_bits.get())
|                    .map(StorageBits::new)
|                    .ok_or(RepresentationError::StorageSizeOverflow)
|            },
|            PrimitiveRepresentation::Sparse { indices, values } =>
""",
)

public_tests = Path("scirust-tensor-ir/tests/representation.rs")
public_tests.write_text(
    public_tests.read_text().rstrip()
    + "\n\n"
    + clean(
        """
|#[test]
|fn per_tensor_quantized_representation_binds_and_accounts_exactly() {
|    let mut graph = Graph::new();
|    let weight = graph
|        .add_input("weight", tensor_type(DType::F32, &[8, 8]))
|        .unwrap();
|    let bias = graph
|        .add_input("bias", tensor_type(DType::F32, &[8]))
|        .unwrap();
|    graph.set_outputs(vec![weight, bias]).unwrap();
|
|    let mut plan = RepresentationPlan::dense(&graph).unwrap();
|    let dense_u8 = plan.declare_dense(DType::U8).unwrap();
|    let dense_f16 = plan.declare_dense(DType::F16).unwrap();
|
|    let quantized = plan
|        .declare_quantized_per_tensor(
|            tensor_type(DType::U8, &[8, 8]),
|            dense_u8,
|            TensorType::new(DType::F16, Shape::scalar()),
|            dense_f16,
|        )
|        .unwrap();
|
|    plan.assign(&graph, weight, quantized).unwrap();
|
|    // 64 U8 codes + one F16 scale = 64*8 + 16 physical bits.
|    assert_eq!(
|        plan.node_storage_bits(&graph, weight),
|        Ok(StorageBits::new(528))
|    );
|    // Bias remains dense F32[8] = 256 bits.
|    assert_eq!(
|        plan.total_storage_bits(&graph),
|        Ok(StorageBits::new(528 + 256))
|    );
|
|    let before = plan.assignments().to_vec();
|    let incompatible = plan
|        .declare_quantized_per_tensor(
|            tensor_type(DType::U8, &[7, 8]),
|            dense_u8,
|            TensorType::new(DType::F16, Shape::scalar()),
|            dense_f16,
|        )
|        .unwrap();
|
|    assert_eq!(
|        plan.replan(
|            &graph,
|            &[
|                Rebinding {
|                    node: bias,
|                    representation: plan.assignment(bias).unwrap(),
|                },
|                Rebinding {
|                    node: weight,
|                    representation: incompatible,
|                },
|            ],
|        ),
|        Err(RepresentationError::QuantizedPerTensorIncompatibleShapes {
|            codes: Shape::new(vec![7, 8]),
|            scale: Shape::scalar(),
|            logical: Shape::new(vec![8, 8]),
|        })
|    );
|    assert_eq!(plan.assignments(), &before[..]);
|}
"""
    )
    + "\n"
)

intent = "scirust-attention-intent/src/lib.rs"

replace_once(
    intent,
    """
|//! Representation-awareness is deliberately one-way: `TensorType` describes
|//! *logical* tensors; a `RepresentationPlan` is a side-table. Dense is the
|//! only currently executable physical path; quantized and sparse are honest
|//! declaration skeletons and fail explicitly when an execution intent is
|//! requested. Unsupported cases fail; they never fall back silently.
""",
    """
|//! Representation-awareness is deliberately one-way: `TensorType` describes
|//! *logical* tensors; a `RepresentationPlan` is a side-table. Dense is the
|//! only currently executable physical path. Bindable per-tensor quantization
|//! can be described by an intent but remains explicitly non-executable; legacy
|//! quantized and sparse skeletons still fail because they lack reconstruction
|//! geometry. Unsupported cases never fall back silently.
""",
)

replace_once(
    intent,
    """
|    /// Quantized codes + scales contract — representable but not executable
|    /// through dense attention.
|    QuantizedSkeleton,
|    /// Sparse indices + values contract — likewise representable only.
""",
    """
|    /// Legacy quantized codes + scales skeleton without reconstruction geometry.
|    QuantizedSkeleton,
|    /// Per-tensor integer codes plus one scalar scale. Representable and exactly
|    /// accountable, but no attention kernel is mapped to it in this slice.
|    QuantizedPerTensor,
|    /// Sparse indices + values contract — likewise representable only.
""",
)

replace_once(
    intent,
    """
|    /// Today this means: `value_dim == head_dim` and every bound variant is
|    /// `Dense { storage_dtype }`. Quantized and sparse succeed in plan
|    /// declaration but not here.
|    #[must_use]
|    pub const fn is_executable(&self) -> bool {
|        matches!(
|            self.representation.query_variant,
|            RepresentationVariant::Dense { .. }
|        )
|    }
""",
    """
|    /// Today this means: `value_dim == head_dim` and every bound variant is
|    /// dense in the logical dtype. Representable quantized intents remain false.
|    #[must_use]
|    pub const fn is_executable(&self) -> bool {
|        matches!(
|            self.representation.query_variant,
|            RepresentationVariant::Dense { storage_dtype } if storage_dtype == self.logical_dtype
|        ) && matches!(
|            self.representation.key_variant,
|            RepresentationVariant::Dense { storage_dtype } if storage_dtype == self.logical_dtype
|        ) && matches!(
|            self.representation.value_variant,
|            RepresentationVariant::Dense { storage_dtype } if storage_dtype == self.logical_dtype
|        )
|    }
""",
)

replace_once(
    intent,
    """
|    match (q_variant, k_variant, v_variant)
|    {
|        _ if dense_dtype_matches(q_variant, logical_dtype)
|            && dense_dtype_matches(k_variant, logical_dtype)
|            && dense_dtype_matches(v_variant, logical_dtype) =>
|        {},
|        _ =>
|        {
|            let failing = if !matches!(q_variant, RepresentationVariant::Dense { .. })
|            {
|                (TensorRole::Query, q_variant)
|            }
|            else if !matches!(k_variant, RepresentationVariant::Dense { .. })
|            {
|                (TensorRole::Key, k_variant)
|            }
|            else
|            {
|                (TensorRole::Value, v_variant)
|            };
|            return Err(IntentError::UnsupportedRepresentation {
|                role: failing.0,
|                variant: failing.1,
|            });
|        },
|    }
""",
    """
|    fn describable(variant: RepresentationVariant, logical: DType) -> bool {
|        dense_dtype_matches(variant, logical)
|            || matches!(variant, RepresentationVariant::QuantizedPerTensor)
|    }
|    if !describable(q_variant, logical_dtype)
|        || !describable(k_variant, logical_dtype)
|        || !describable(v_variant, logical_dtype)
|    {
|        let failing = if !describable(q_variant, logical_dtype)
|        {
|            (TensorRole::Query, q_variant)
|        }
|        else if !describable(k_variant, logical_dtype)
|        {
|            (TensorRole::Key, k_variant)
|        }
|        else
|        {
|            (TensorRole::Value, v_variant)
|        };
|        return Err(IntentError::UnsupportedRepresentation {
|            role: failing.0,
|            variant: failing.1,
|        });
|    }
""",
)

replace_once(
    intent,
    """
|        Prim::Quantized { .. } => Ok(RepresentationVariant::QuantizedSkeleton),
|        Prim::Sparse { .. } => Ok(RepresentationVariant::SparseSkeleton),
""",
    """
|        Prim::Quantized { .. } => Ok(RepresentationVariant::QuantizedSkeleton),
|        Prim::QuantizedPerTensor { .. } => Ok(RepresentationVariant::QuantizedPerTensor),
|        Prim::Sparse { .. } => Ok(RepresentationVariant::SparseSkeleton),
""",
)

replace_once(
    intent,
    """
|    #[test]
|    fn rejects_non_dense_bindings_honestly() {
""",
    """
|    #[test]
|    fn derives_non_executable_intent_for_per_tensor_quantized_bindings() {
|        let (graph, q, k, v, mut plan) = graph_fixture(1, 2, 4, 8);
|        let dense_f32 = plan.assignment(q).expect("f32");
|        let dense_u8 = plan.declare_dense(DType::U8).expect("u8");
|        let logical_shape = Shape::new([1usize, 2, 4, 8]);
|        let quantized = plan
|            .declare_quantized_per_tensor(
|                TensorType::new(DType::U8, logical_shape),
|                dense_u8,
|                TensorType::new(DType::F32, Shape::scalar()),
|                dense_f32,
|            )
|            .expect("quantized");
|
|        plan.replan(
|            &graph,
|            &[
|                scirust_tensor_ir::Rebinding {
|                    node: q,
|                    representation: quantized,
|                },
|                scirust_tensor_ir::Rebinding {
|                    node: k,
|                    representation: quantized,
|                },
|                scirust_tensor_ir::Rebinding {
|                    node: v,
|                    representation: quantized,
|                },
|            ],
|        )
|        .expect("bind quantized");
|
|        let intent = derive_attention_intent(&graph, &plan, q, k, v, true)
|            .expect("quantized intent remains describable");
|        assert!(!intent.is_executable());
|        assert_eq!(
|            intent.representation.query_variant,
|            RepresentationVariant::QuantizedPerTensor
|        );
|        // Per tensor: 64 U8 codes (512 bits) + one F32 scale (32 bits).
|        assert_eq!(
|            intent.representation.total_storage_bits,
|            StorageBits::new(3 * 544)
|        );
|    }
|
|    #[test]
|    fn rejects_non_dense_bindings_honestly() {
""",
)
