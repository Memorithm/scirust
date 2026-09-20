//! Versioned canonical byte identities for Tensor IR graphs and representation plans.
//!
//! Canonical bytes are the authoritative identity surface. Hashes or digests may
//! be computed by callers for indexing, provenance or attestation, but a digest
//! must never replace structural validation or byte equality.
//!
//! Three domains are intentionally distinct:
//!
//! - [`canonical_graph_bytes`] includes user-facing input names and therefore
//!   follows the complete structural [`Graph`] record;
//! - [`canonical_representation_anchor_bytes`] ignores input display names and
//!   follows the compatibility semantics used by [`RepresentationPlan`];
//! - [`canonical_representation_plan_bytes`] binds that representation anchor to
//!   the complete declaration table and canonical node assignments.
//!
//! The encoding uses explicit variant tags, little-endian fixed-width integers,
//! and length-prefixed sequences. Future variants require an explicit encoding
//! update; unsupported future [`DType`] variants fail closed.

use alloc::vec::Vec;
use core::fmt;

use scirust_compute::{DType, Shape};

use crate::{
    Graph, GraphError, Operation, PrimitiveRepresentation, RepresentationComponent,
    RepresentationError, RepresentationPlan, Scalar, TensorType,
};

/// Canonical byte-domain tag for complete graph structural identity.
pub const GRAPH_STRUCTURAL_IDENTITY_V1: &str = "scirust.tensor-ir.graph-structural.v1";

/// Canonical byte-domain tag for representation-plan graph compatibility.
pub const REPRESENTATION_ANCHOR_IDENTITY_V1: &str =
    "scirust.tensor-ir.representation-anchor.v1";

/// Canonical byte-domain tag for graph-bound representation-plan identity.
pub const REPRESENTATION_PLAN_IDENTITY_V1: &str =
    "scirust.tensor-ir.representation-plan.v1";

/// Failure to produce one canonical Tensor IR identity.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CanonicalIdentityError {
    /// The graph is structurally invalid.
    Graph(GraphError),
    /// The representation plan is incompatible with the supplied graph.
    Representation(RepresentationError),
    /// A host-sized value cannot be represented by the stable u64 wire width.
    LengthOverflow {
        /// Field whose length/value exceeded the canonical wire width.
        field: &'static str,
        /// Host value that could not be represented.
        value: usize,
    },
    /// A future scalar type has no explicit v1 identity tag yet.
    UnsupportedDType {
        /// Scalar type requiring a new versioned encoding decision.
        dtype: DType,
    },
}

impl fmt::Display for CanonicalIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "invalid graph for canonical identity: {error}"),
            Self::Representation(error) => {
                write!(
                    formatter,
                    "invalid representation plan for canonical identity: {error}"
                )
            }
            Self::LengthOverflow { field, value } => {
                write!(
                    formatter,
                    "canonical identity field {field} cannot encode host value {value}"
                )
            }
            Self::UnsupportedDType { dtype } => {
                write!(
                    formatter,
                    "dtype {dtype:?} has no canonical Tensor IR v1 identity tag"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CanonicalIdentityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Representation(error) => Some(error),
            Self::LengthOverflow { .. } | Self::UnsupportedDType { .. } => None,
        }
    }
}

impl From<GraphError> for CanonicalIdentityError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<RepresentationError> for CanonicalIdentityError {
    fn from(value: RepresentationError) -> Self {
        Self::Representation(value)
    }
}

/// Encode a complete structurally valid graph into versioned canonical bytes.
///
/// Input names are included. Constants contribute their stable [`crate::ConstantId`]
/// but not external tensor payload bytes. This function is an identity encoder,
/// not a semantic-validity attestation; use [`crate::validate_semantics`] when
/// shape/dtype semantics also need qualification.
///
/// # Errors
///
/// Returns [`CanonicalIdentityError::Graph`] for an invalid graph,
/// [`CanonicalIdentityError::UnsupportedDType`] for a future unassigned dtype,
/// or [`CanonicalIdentityError::LengthOverflow`] if a host-sized length cannot
/// fit the stable u64 encoding.
///
/// # Examples
///
/// Identical graphs produce identical bytes:
///
/// ```
/// use scirust_tensor_ir::{
///     canonical_graph_bytes, DType, Graph, Shape, TensorType,
/// };
/// fn graph() -> Graph {
///     let mut graph = Graph::new();
///     let x = graph
///         .add_input("x", TensorType::new(DType::F32, Shape::new([2usize])))
///         .unwrap();
///     graph.set_outputs(vec![x]).unwrap();
///     graph
/// }
/// assert_eq!(
///     canonical_graph_bytes(&graph()).unwrap(),
///     canonical_graph_bytes(&graph()).unwrap()
/// );
/// ```
///
/// Structural graph identity includes input names:
///
/// ```
/// use scirust_tensor_ir::{
///     canonical_graph_bytes, DType, Graph, Shape, TensorType,
/// };
/// fn named(name: &str) -> Graph {
///     let mut graph = Graph::new();
///     let x = graph
///         .add_input(name, TensorType::new(DType::F32, Shape::new([1usize])))
///         .unwrap();
///     graph.set_outputs(vec![x]).unwrap();
///     graph
/// }
/// assert_ne!(
///     canonical_graph_bytes(&named("lhs")).unwrap(),
///     canonical_graph_bytes(&named("renamed")).unwrap()
/// );
/// ```
pub fn canonical_graph_bytes(graph: &Graph) -> Result<Vec<u8>, CanonicalIdentityError> {
    graph.validate()?;
    encode_graph(graph, GraphEncodingMode::Structural, GRAPH_STRUCTURAL_IDENTITY_V1)
}

/// Encode the graph identity used by representation-plan compatibility.
///
/// This encoding intentionally ignores only [`Operation::Input`] display names.
/// Canonical node order, all other operation variants and attributes, input
/// edges, tensor types and the ordered graph output set remain authoritative.
///
/// # Errors
///
/// Returns [`CanonicalIdentityError`] under the same structural/encoding
/// conditions as [`canonical_graph_bytes`].
///
/// # Examples
///
/// Input renaming does not change the representation anchor:
///
/// ```
/// use scirust_tensor_ir::{
///     canonical_representation_anchor_bytes, DType, Graph, Shape, TensorType,
/// };
/// fn named(name: &str) -> Graph {
///     let mut graph = Graph::new();
///     let x = graph
///         .add_input(name, TensorType::new(DType::F32, Shape::new([1usize])))
///         .unwrap();
///     graph.set_outputs(vec![x]).unwrap();
///     graph
/// }
/// assert_eq!(
///     canonical_representation_anchor_bytes(&named("lhs")).unwrap(),
///     canonical_representation_anchor_bytes(&named("renamed")).unwrap()
/// );
/// ```
///
/// A logical shape change does change the anchor:
///
/// ```
/// use scirust_tensor_ir::{
///     canonical_representation_anchor_bytes, DType, Graph, Shape, TensorType,
/// };
/// fn shaped(n: usize) -> Graph {
///     let mut graph = Graph::new();
///     let x = graph
///         .add_input("x", TensorType::new(DType::F32, Shape::new([n])))
///         .unwrap();
///     graph.set_outputs(vec![x]).unwrap();
///     graph
/// }
/// assert_ne!(
///     canonical_representation_anchor_bytes(&shaped(1)).unwrap(),
///     canonical_representation_anchor_bytes(&shaped(2)).unwrap()
/// );
/// ```
pub fn canonical_representation_anchor_bytes(
    graph: &Graph,
) -> Result<Vec<u8>, CanonicalIdentityError> {
    graph.validate()?;
    encode_graph(
        graph,
        GraphEncodingMode::RepresentationAnchor,
        REPRESENTATION_ANCHOR_IDENTITY_V1,
    )
}

/// Encode one compatible representation plan and its graph anchor.
///
/// The complete representation declaration table and every canonical node
/// assignment are encoded in declaration/node order. Derived quantities such as
/// total storage bits are not duplicated in the identity because they are
/// deterministically recomputable from the encoded plan and graph.
///
/// Input display-name changes are intentionally ignored, matching
/// [`RepresentationPlan::ensure_compatible_with`].
///
/// # Errors
///
/// Returns [`CanonicalIdentityError::Graph`] if the presented graph is
/// structurally invalid, [`CanonicalIdentityError::Representation`] if the plan
/// is incompatible with it, or a stable encoding error for unsupported dtypes or
/// impossible host-sized lengths.
///
/// # Examples
///
/// Compatible input renaming preserves plan identity:
///
/// ```
/// use scirust_tensor_ir::{
///     canonical_representation_plan_bytes, DType, Graph, RepresentationPlan,
///     Shape, TensorType,
/// };
/// fn named(name: &str) -> Graph {
///     let mut graph = Graph::new();
///     let x = graph
///         .add_input(name, TensorType::new(DType::F32, Shape::new([1usize])))
///         .unwrap();
///     graph.set_outputs(vec![x]).unwrap();
///     graph
/// }
/// let source = named("source");
/// let renamed = named("display-only");
/// let plan = RepresentationPlan::dense(&source).unwrap();
/// assert_eq!(
///     canonical_representation_plan_bytes(&plan, &source).unwrap(),
///     canonical_representation_plan_bytes(&plan, &renamed).unwrap()
/// );
/// ```
///
/// An added representation declaration changes plan identity even before a node
/// is rebound to it:
///
/// ```
/// use scirust_tensor_ir::{
///     canonical_representation_plan_bytes, DType, Graph, RepresentationPlan,
///     Shape, TensorType,
/// };
/// let mut graph = Graph::new();
/// let x = graph
///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
///     .unwrap();
/// graph.set_outputs(vec![x]).unwrap();
/// let base = RepresentationPlan::dense(&graph).unwrap();
/// let mut extended = base.clone();
/// extended.declare_dense(DType::F16).unwrap();
/// assert_ne!(
///     canonical_representation_plan_bytes(&base, &graph).unwrap(),
///     canonical_representation_plan_bytes(&extended, &graph).unwrap()
/// );
/// ```
pub fn canonical_representation_plan_bytes(
    plan: &RepresentationPlan,
    graph: &Graph,
) -> Result<Vec<u8>, CanonicalIdentityError> {
    graph.validate()?;
    plan.ensure_compatible_with(graph)?;

    let anchor = encode_graph(
        graph,
        GraphEncodingMode::RepresentationAnchor,
        REPRESENTATION_ANCHOR_IDENTITY_V1,
    )?;

    let mut encoder = Encoder::default();
    encoder.bytes("identity-domain", REPRESENTATION_PLAN_IDENTITY_V1.as_bytes())?;
    encoder.bytes("representation-anchor", &anchor)?;

    encoder.len("representation-count", plan.representations().len())?;
    for representation in plan.representations() {
        encode_representation(&mut encoder, representation)?;
    }

    encoder.len("assignment-count", plan.assignments().len())?;
    for assignment in plan.assignments() {
        encoder.u32(assignment.get());
    }

    Ok(encoder.finish())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphEncodingMode {
    Structural,
    RepresentationAnchor,
}

fn encode_graph(
    graph: &Graph,
    mode: GraphEncodingMode,
    domain: &str,
) -> Result<Vec<u8>, CanonicalIdentityError> {
    let mut encoder = Encoder::default();
    encoder.bytes("identity-domain", domain.as_bytes())?;
    encoder.len("node-count", graph.nodes().len())?;

    for node in graph.nodes() {
        encode_operation(&mut encoder, &node.operation, mode)?;
        encoder.len("node-input-count", node.inputs.len())?;
        for input in &node.inputs {
            encoder.u32(input.get());
        }
        encode_tensor_type(&mut encoder, &node.output)?;
    }

    encoder.len("graph-output-count", graph.outputs().len())?;
    for output in graph.outputs() {
        encoder.u32(output.get());
    }

    Ok(encoder.finish())
}

fn encode_tensor_type(
    encoder: &mut Encoder,
    tensor_type: &TensorType,
) -> Result<(), CanonicalIdentityError> {
    encode_dtype(encoder, tensor_type.dtype)?;
    encode_shape(encoder, &tensor_type.shape)
}

fn encode_shape(encoder: &mut Encoder, shape: &Shape) -> Result<(), CanonicalIdentityError> {
    encoder.len("shape-rank", shape.dims().len())?;
    for &dimension in shape.dims() {
        encoder.usize("shape-dimension", dimension)?;
    }
    Ok(())
}

fn encode_scalar(encoder: &mut Encoder, scalar: Scalar) -> Result<(), CanonicalIdentityError> {
    encode_dtype(encoder, scalar.dtype())?;
    encoder.u64(scalar.bits());
    Ok(())
}

fn encode_dtype(encoder: &mut Encoder, dtype: DType) -> Result<(), CanonicalIdentityError> {
    let tag = match dtype {
        DType::Bool => 0,
        DType::U8 => 1,
        DType::I8 => 2,
        DType::U16 => 3,
        DType::I16 => 4,
        DType::F16 => 5,
        DType::Bf16 => 6,
        DType::U32 => 7,
        DType::I32 => 8,
        DType::F32 => 9,
        DType::U64 => 10,
        DType::I64 => 11,
        DType::F64 => 12,
        _ => return Err(CanonicalIdentityError::UnsupportedDType { dtype }),
    };
    encoder.byte(tag);
    Ok(())
}

fn encode_operation(
    encoder: &mut Encoder,
    operation: &Operation,
    mode: GraphEncodingMode,
) -> Result<(), CanonicalIdentityError> {
    match operation {
        Operation::Input { name } => {
            encoder.byte(0);
            if mode == GraphEncodingMode::Structural {
                encoder.bytes("input-name", name.as_bytes())?;
            }
        }
        Operation::Constant { id } => {
            encoder.byte(1);
            encoder.u64(id.get());
        }
        Operation::Add => encoder.byte(2),
        Operation::Sub => encoder.byte(3),
        Operation::Mul => encoder.byte(4),
        Operation::Div => encoder.byte(5),
        Operation::Scale { factor } => {
            encoder.byte(6);
            encode_scalar(encoder, *factor)?;
        }
        Operation::Relu => encoder.byte(7),
        Operation::Exp => encoder.byte(8),
        Operation::Log => encoder.byte(9),
        Operation::ReluGrad => encoder.byte(10),
        Operation::ZerosLike => encoder.byte(11),
        Operation::OnesLike => encoder.byte(12),
        Operation::MatMul => encoder.byte(13),
        Operation::BatchMatMul => encoder.byte(14),
        Operation::Reshape { shape } => {
            encoder.byte(15);
            encode_shape(encoder, shape)?;
        }
        Operation::Transpose { permutation } => {
            encoder.byte(16);
            encoder.len("transpose-rank", permutation.len())?;
            for &axis in permutation {
                encoder.usize("transpose-axis", axis)?;
            }
        }
        Operation::BroadcastTo { shape } => {
            encoder.byte(17);
            encode_shape(encoder, shape)?;
        }
        Operation::ReduceSumTo { shape } => {
            encoder.byte(18);
            encode_shape(encoder, shape)?;
        }
        Operation::StopGradient => encoder.byte(19),
        Operation::Checkpoint => encoder.byte(20),
    }
    Ok(())
}

fn encode_component(
    encoder: &mut Encoder,
    component: &RepresentationComponent,
) -> Result<(), CanonicalIdentityError> {
    encode_tensor_type(encoder, component.tensor_type())?;
    encoder.u32(component.representation().get());
    Ok(())
}

fn encode_representation(
    encoder: &mut Encoder,
    representation: &PrimitiveRepresentation,
) -> Result<(), CanonicalIdentityError> {
    match representation {
        PrimitiveRepresentation::Dense { storage_dtype } => {
            encoder.byte(0);
            encode_dtype(encoder, *storage_dtype)?;
        }
        PrimitiveRepresentation::Factorized { left, right } => {
            encoder.byte(1);
            encode_component(encoder, left)?;
            encode_component(encoder, right)?;
        }
        PrimitiveRepresentation::Quantized { codes, scales } => {
            encoder.byte(2);
            encode_component(encoder, codes)?;
            encode_component(encoder, scales)?;
        }
        PrimitiveRepresentation::QuantizedPerTensor { codes, scale } => {
            encoder.byte(3);
            encode_component(encoder, codes)?;
            encode_component(encoder, scale)?;
        }
        PrimitiveRepresentation::Sparse { indices, values } => {
            encoder.byte(4);
            encode_component(encoder, indices)?;
            encode_component(encoder, values)?;
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn usize(
        &mut self,
        field: &'static str,
        value: usize,
    ) -> Result<(), CanonicalIdentityError> {
        let value = u64::try_from(value)
            .map_err(|_| CanonicalIdentityError::LengthOverflow { field, value })?;
        self.u64(value);
        Ok(())
    }

    fn len(
        &mut self,
        field: &'static str,
        value: usize,
    ) -> Result<(), CanonicalIdentityError> {
        self.usize(field, value)
    }

    fn bytes(
        &mut self,
        field: &'static str,
        value: &[u8],
    ) -> Result<(), CanonicalIdentityError> {
        self.len(field, value.len())?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::String, vec};

    use super::*;
    use crate::{ConstantId, Rebinding, RepresentationId};

    fn input_graph(name: &str, width: usize) -> Graph {
        let mut graph = Graph::new();
        let input = graph
            .add_input(
                String::from(name),
                TensorType::new(DType::F32, Shape::new([width])),
            )
            .unwrap();
        graph.set_outputs(vec![input]).unwrap();
        graph
    }

    #[test]
    fn structural_and_representation_anchor_domains_differ_on_input_names() {
        let left = input_graph("left", 2);
        let renamed = input_graph("renamed", 2);

        assert_ne!(
            canonical_graph_bytes(&left).unwrap(),
            canonical_graph_bytes(&renamed).unwrap()
        );
        assert_eq!(
            canonical_representation_anchor_bytes(&left).unwrap(),
            canonical_representation_anchor_bytes(&renamed).unwrap()
        );
    }

    #[test]
    fn operation_attributes_are_bit_exact_identity_inputs() {
        let graph_with = |factor: f32| {
            let mut graph = Graph::new();
            let input = graph
                .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
                .unwrap();
            let scaled = graph
                .add_node(
                    Operation::Scale {
                        factor: Scalar::f32(factor),
                    },
                    vec![input],
                    TensorType::new(DType::F32, Shape::new([1usize])),
                )
                .unwrap();
            graph.set_outputs(vec![scaled]).unwrap();
            graph
        };

        assert_ne!(
            canonical_graph_bytes(&graph_with(0.0)).unwrap(),
            canonical_graph_bytes(&graph_with(-0.0)).unwrap()
        );
    }

    #[test]
    fn constant_identity_binds_identifier_not_external_payload() {
        let graph_with = |id| {
            let mut graph = Graph::new();
            let constant = graph
                .add_constant(
                    ConstantId::new(id),
                    TensorType::new(DType::F32, Shape::new([1usize])),
                )
                .unwrap();
            graph.set_outputs(vec![constant]).unwrap();
            graph
        };

        assert_ne!(
            canonical_graph_bytes(&graph_with(1)).unwrap(),
            canonical_graph_bytes(&graph_with(2)).unwrap()
        );
    }

    #[test]
    fn representation_plan_identity_includes_declarations_and_assignments() {
        let graph = input_graph("x", 4);
        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let dense_before = canonical_representation_plan_bytes(&plan, &graph).unwrap();

        let codes = plan.declare_dense(DType::U8).unwrap();
        let scale = plan.declare_dense(DType::F32).unwrap();
        let quantized = plan
            .declare_quantized_per_tensor(
                TensorType::new(DType::U8, Shape::new([4usize])),
                codes,
                TensorType::new(DType::F32, Shape::scalar()),
                scale,
            )
            .unwrap();
        let declarations_only = canonical_representation_plan_bytes(&plan, &graph).unwrap();
        assert_ne!(dense_before, declarations_only);

        plan.replan(
            &graph,
            &[Rebinding {
                node: crate::NodeId::new(0),
                representation: quantized,
            }],
        )
        .unwrap();
        let rebound = canonical_representation_plan_bytes(&plan, &graph).unwrap();
        assert_ne!(declarations_only, rebound);
    }

    #[test]
    fn representation_plan_identity_rejects_incompatible_graph() {
        let graph = input_graph("x", 2);
        let plan = RepresentationPlan::dense(&graph).unwrap();
        let incompatible = input_graph("x", 3);

        assert!(matches!(
            canonical_representation_plan_bytes(&plan, &incompatible),
            Err(CanonicalIdentityError::Representation(
                RepresentationError::GraphNodeTypeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn operation_tag_changes_are_visible() {
        let graph_for = |operation: Operation| {
            let mut graph = Graph::new();
            let lhs = graph
                .add_input("lhs", TensorType::new(DType::F32, Shape::new([1usize])))
                .unwrap();
            let rhs = graph
                .add_input("rhs", TensorType::new(DType::F32, Shape::new([1usize])))
                .unwrap();
            let output = graph
                .add_node(
                    operation,
                    vec![lhs, rhs],
                    TensorType::new(DType::F32, Shape::new([1usize])),
                )
                .unwrap();
            graph.set_outputs(vec![output]).unwrap();
            graph
        };

        assert_ne!(
            canonical_graph_bytes(&graph_for(Operation::Add)).unwrap(),
            canonical_graph_bytes(&graph_for(Operation::Mul)).unwrap()
        );
    }

    #[test]
    fn explicit_domain_tags_prevent_cross_domain_aliases() {
        let graph = input_graph("x", 1);
        let structural = canonical_graph_bytes(&graph).unwrap();
        let anchor = canonical_representation_anchor_bytes(&graph).unwrap();
        assert_ne!(structural, anchor);
    }

    #[test]
    fn representation_ids_are_encoded_in_declaration_order() {
        let graph = input_graph("x", 1);
        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let first = plan.declare_dense(DType::U8).unwrap();
        let second = plan.declare_dense(DType::F16).unwrap();
        assert_eq!(first, RepresentationId::new(1));
        assert_eq!(second, RepresentationId::new(2));

        let bytes = canonical_representation_plan_bytes(&plan, &graph).unwrap();
        assert!(!bytes.is_empty());
    }
}
