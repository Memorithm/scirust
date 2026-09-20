//! Exact, policy-free cost aggregation for canonical Tensor IR plans.
//!
//! This module only aggregates quantities that another SciRust contract already
//! defines exactly. It does not estimate latency, bandwidth, energy, free RAM,
//! storage capacity, or backend availability.
//!
//! Unknown is a first-class state: absence of a physical accounting scope,
//! resident materialization, or transfer list is never interpreted as zero.
//! This is the intended hand-off boundary to policy systems such as ElasticXxx.

use core::fmt;

use scirust_compute::TransferRequest;

use crate::{
    Graph, GraphError, MaterializationClass, Operation, PhysicalAccountingError,
    PhysicalAccountingScope, RepresentationError, RepresentationPlan, SemanticError, StorageBits,
    validate_semantics,
};

/// One exact cost dimension whose value may not be available at this planning layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactCost<T> {
    /// No exact value was supplied or derivable at this layer.
    Unknown,
    /// Exact value under the referenced contract.
    Known(T),
}

/// Exact resident-memory cost for one named materialization class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentCost {
    /// Materialization class whose resident extent was requested.
    pub class: MaterializationClass,
    /// Exact resident physical extent.
    pub bits: StorageBits,
}

/// Backend-neutral exact cost vector for one representation plan.
///
/// representation_storage_bits is always known from the plan. Serialized and
/// resident physical costs are only known when a PhysicalAccountingScope is
/// explicitly supplied. Transfer bytes are only known when the caller supplies
/// an explicit complete transfer slice; Some(empty) therefore means exact zero,
/// while None means unknown.
///
/// checkpoint_markers counts canonical Operation::Checkpoint nodes. It is a
/// structural recomputation marker count, not an estimate of recomputation time
/// or FLOPs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactCostVector {
    /// Exact storage implied by the representation plan.
    pub representation_storage_bits: StorageBits,
    /// Exact serialized physical extent when a physical scope was supplied.
    pub serialized_bits: ExactCost<StorageBits>,
    /// Exact resident extent for the requested materialization class, if any.
    pub resident: ExactCost<ResidentCost>,
    /// Exact sum of declared transfer byte lengths, or unknown.
    pub transfer_bytes: ExactCost<u64>,
    /// Exact number of checkpoint/recomputation markers in the canonical graph.
    pub checkpoint_markers: u64,
}

/// Failure while constructing an ExactCostVector.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CostVectorError {
    /// The graph is structurally invalid.
    Graph(GraphError),
    /// The graph is structurally valid but semantically invalid.
    Semantic(SemanticError),
    /// The representation plan is incompatible with the graph or overflows accounting.
    Representation(RepresentationError),
    /// Physical segment ownership/materialization accounting failed.
    Physical(PhysicalAccountingError),
    /// A resident class was requested without a physical accounting scope.
    ResidentClassWithoutPhysicalScope,
    /// A host-sized transfer byte count did not fit the stable u64 aggregate.
    TransferLengthOverflow {
        /// Transfer request index.
        index: usize,
        /// Host byte length that did not fit.
        byte_len: usize,
    },
    /// Summing exact transfer lengths overflowed u64.
    TransferTotalOverflow,
    /// Counting checkpoint markers overflowed u64.
    CheckpointCountOverflow,
}

impl fmt::Display for CostVectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "invalid graph for cost accounting: {error}"),
            Self::Semantic(error) => {
                write!(formatter, "invalid graph semantics for cost accounting: {error}")
            }
            Self::Representation(error) => {
                write!(formatter, "invalid representation cost accounting: {error}")
            }
            Self::Physical(error) => write!(formatter, "invalid physical cost accounting: {error}"),
            Self::ResidentClassWithoutPhysicalScope => formatter.write_str(
                "resident materialization class requires an explicit physical accounting scope",
            ),
            Self::TransferLengthOverflow { index, byte_len } => write!(
                formatter,
                "transfer {index} byte length {byte_len} does not fit canonical u64 accounting"
            ),
            Self::TransferTotalOverflow => {
                formatter.write_str("exact transfer-byte aggregation overflowed u64")
            }
            Self::CheckpointCountOverflow => {
                formatter.write_str("checkpoint marker count overflowed u64")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CostVectorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Semantic(error) => Some(error),
            Self::Representation(error) => Some(error),
            Self::Physical(error) => Some(error),
            Self::ResidentClassWithoutPhysicalScope
            | Self::TransferLengthOverflow { .. }
            | Self::TransferTotalOverflow
            | Self::CheckpointCountOverflow => None,
        }
    }
}

impl From<GraphError> for CostVectorError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<SemanticError> for CostVectorError {
    fn from(value: SemanticError) -> Self {
        Self::Semantic(value)
    }
}

impl From<RepresentationError> for CostVectorError {
    fn from(value: RepresentationError) -> Self {
        Self::Representation(value)
    }
}

impl From<PhysicalAccountingError> for CostVectorError {
    fn from(value: PhysicalAccountingError) -> Self {
        Self::Physical(value)
    }
}

/// Aggregate exact costs for one canonical graph + representation plan.
///
/// physical_scope and resident_class are deliberately optional because the
/// canonical representation plan can exist before a concrete serialized/resident
/// layout is known. When a physical scope is supplied, its ownership/reference
/// invariants are validated before serialized or resident totals are exposed.
///
/// transfers = None means no exact transfer programme has been supplied.
/// transfers = Some(empty) means the caller supplied a complete transfer
/// programme containing exactly zero transfers.
///
/// The function does not prove that a caller-supplied physical scope or transfer
/// programme was generated from this plan; provenance/binding is a separate
/// evidence-layer responsibility.
///
/// # Errors
///
/// Fails closed on invalid graph/semantics, incompatible representation plans,
/// invalid physical scopes/materializations, or checked integer overflow.
///
/// # Examples
///
/// A representation-only plan exposes unknown physical/transfer dimensions.
///
/// ```
/// use scirust_tensor_ir::{
///     exact_cost_vector, DType, ExactCost, Graph, RepresentationPlan, Shape, TensorType,
/// };
///
/// let mut graph = Graph::new();
/// let x = graph
///     .add_input("x", TensorType::new(DType::F32, Shape::new([2usize, 2])))
///     .unwrap();
/// graph.set_outputs(vec![x]).unwrap();
/// let plan = RepresentationPlan::dense(&graph).unwrap();
/// let costs = exact_cost_vector(&graph, &plan, None, None, None).unwrap();
/// assert_eq!(costs.representation_storage_bits.get(), 128);
/// assert_eq!(costs.serialized_bits, ExactCost::Unknown);
/// assert_eq!(costs.resident, ExactCost::Unknown);
/// assert_eq!(costs.transfer_bytes, ExactCost::Unknown);
/// ```
///
/// An explicit empty transfer programme is exact zero, not unknown.
///
/// ```
/// use scirust_tensor_ir::{
///     exact_cost_vector, DType, ExactCost, Graph, Operation, RepresentationPlan, Shape,
///     TensorType,
/// };
///
/// let mut graph = Graph::new();
/// let x = graph
///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
///     .unwrap();
/// let checkpoint = graph
///     .add_node(
///         Operation::Checkpoint,
///         vec![x],
///         TensorType::new(DType::F32, Shape::new([1usize])),
///     )
///     .unwrap();
/// graph.set_outputs(vec![checkpoint]).unwrap();
/// let plan = RepresentationPlan::dense(&graph).unwrap();
/// let transfers = [];
/// let costs = exact_cost_vector(&graph, &plan, None, None, Some(&transfers)).unwrap();
/// assert_eq!(costs.transfer_bytes, ExactCost::Known(0));
/// assert_eq!(costs.checkpoint_markers, 1);
/// ```
pub fn exact_cost_vector(
    graph: &Graph,
    plan: &RepresentationPlan,
    physical_scope: Option<&PhysicalAccountingScope>,
    resident_class: Option<MaterializationClass>,
    transfers: Option<&[TransferRequest]>,
) -> Result<ExactCostVector, CostVectorError> {
    graph.validate()?;
    validate_semantics(graph)?;
    plan.ensure_compatible_with(graph)?;

    let representation_storage_bits = plan.total_storage_bits(graph)?;

    let (serialized_bits, resident) = match (physical_scope, resident_class) {
        (None, None) => (ExactCost::Unknown, ExactCost::Unknown),
        (None, Some(_)) => return Err(CostVectorError::ResidentClassWithoutPhysicalScope),
        (Some(scope), None) => {
            scope.validate()?;
            (
                ExactCost::Known(scope.serialized_bits()?),
                ExactCost::Unknown,
            )
        }
        (Some(scope), Some(class)) => {
            scope.validate()?;
            (
                ExactCost::Known(scope.serialized_bits()?),
                ExactCost::Known(ResidentCost {
                    class,
                    bits: scope.resident_bits(class)?,
                }),
            )
        }
    };

    let transfer_bytes = match transfers {
        None => ExactCost::Unknown,
        Some(transfers) => {
            let mut total = 0u64;
            for (index, transfer) in transfers.iter().enumerate() {
                let bytes = u64::try_from(transfer.byte_len()).map_err(|_| {
                    CostVectorError::TransferLengthOverflow {
                        index,
                        byte_len: transfer.byte_len(),
                    }
                })?;
                total = total
                    .checked_add(bytes)
                    .ok_or(CostVectorError::TransferTotalOverflow)?;
            }
            ExactCost::Known(total)
        }
    };

    let checkpoint_count = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.operation, Operation::Checkpoint))
        .count();
    let checkpoint_markers =
        u64::try_from(checkpoint_count).map_err(|_| CostVectorError::CheckpointCountOverflow)?;

    Ok(ExactCostVector {
        representation_storage_bits,
        serialized_bits,
        resident,
        transfer_bytes,
        checkpoint_markers,
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use scirust_compute::{
        DeviceId, DeviceKind, TensorResidency, TransferMode, TransferRequest,
    };

    use super::*;
    use crate::{
        ContentIdentity, LayoutIdentity, PhysicalSegment, PhysicalSegmentId,
        PhysicalSegmentRole, ReconstructionRole, ResidentMaterialization, SegmentLifetime,
        TensorType,
    };

    fn graph() -> (Graph, RepresentationPlan) {
        let mut graph = Graph::new();
        let x = graph
            .add_input(
                "x",
                TensorType::new(crate::DType::F32, crate::Shape::new([2usize, 2])),
            )
            .unwrap();
        graph.set_outputs(vec![x]).unwrap();
        let plan = RepresentationPlan::dense(&graph).unwrap();
        (graph, plan)
    }

    #[test]
    fn missing_exact_inputs_remain_unknown() {
        let (graph, plan) = graph();
        let costs = exact_cost_vector(&graph, &plan, None, None, None).unwrap();
        assert_eq!(costs.representation_storage_bits, StorageBits::new(128));
        assert_eq!(costs.serialized_bits, ExactCost::Unknown);
        assert_eq!(costs.resident, ExactCost::Unknown);
        assert_eq!(costs.transfer_bytes, ExactCost::Unknown);
    }

    #[test]
    fn explicit_empty_transfer_set_is_known_zero() {
        let (graph, plan) = graph();
        let costs = exact_cost_vector(&graph, &plan, None, None, Some(&[])).unwrap();
        assert_eq!(costs.transfer_bytes, ExactCost::Known(0));
    }

    #[test]
    fn physical_scope_and_transfer_requests_aggregate_exactly() {
        let (graph, plan) = graph();
        let class = MaterializationClass::new(7);
        let segment = PhysicalSegment::new(
            PhysicalSegmentId::new(1),
            PhysicalSegmentRole::Payload,
            ContentIdentity::new(11),
            LayoutIdentity::new(12),
            StorageBits::new(128),
            StorageBits::new(128),
            8,
            SegmentLifetime::GraphStatic,
            ReconstructionRole::Stored,
            vec![ResidentMaterialization::new(class, StorageBits::new(256), 8).unwrap()],
        )
        .unwrap();
        let mut scope = PhysicalAccountingScope::new();
        scope.own(segment);

        let transfers = [
            TransferRequest::new(
                TensorResidency::Host,
                TensorResidency::Device(DeviceId::new(DeviceKind::Wgpu, 0)),
                crate::DType::F32,
                16,
                TransferMode::Synchronous,
            )
            .unwrap(),
            TransferRequest::new(
                TensorResidency::Device(DeviceId::new(DeviceKind::Wgpu, 0)),
                TensorResidency::Host,
                crate::DType::F32,
                8,
                TransferMode::Synchronous,
            )
            .unwrap(),
        ];

        let costs =
            exact_cost_vector(&graph, &plan, Some(&scope), Some(class), Some(&transfers)).unwrap();

        assert_eq!(
            costs.serialized_bits,
            ExactCost::Known(StorageBits::new(128))
        );
        assert_eq!(
            costs.resident,
            ExactCost::Known(ResidentCost {
                class,
                bits: StorageBits::new(256),
            })
        );
        assert_eq!(costs.transfer_bytes, ExactCost::Known(24));
    }

    #[test]
    fn resident_class_without_scope_fails_closed() {
        let (graph, plan) = graph();
        assert_eq!(
            exact_cost_vector(
                &graph,
                &plan,
                None,
                Some(MaterializationClass::new(1)),
                None,
            ),
            Err(CostVectorError::ResidentClassWithoutPhysicalScope)
        );
    }

    #[test]
    fn checkpoint_count_is_structural_not_a_time_estimate() {
        let mut graph = Graph::new();
        let ty = TensorType::new(crate::DType::F32, crate::Shape::new([1usize]));
        let x = graph.add_input("x", ty.clone()).unwrap();
        let a = graph
            .add_node(Operation::Checkpoint, vec![x], ty.clone())
            .unwrap();
        let b = graph.add_node(Operation::Checkpoint, vec![a], ty).unwrap();
        graph.set_outputs(vec![b]).unwrap();
        let plan = RepresentationPlan::dense(&graph).unwrap();

        let costs = exact_cost_vector(&graph, &plan, None, None, None).unwrap();
        assert_eq!(costs.checkpoint_markers, 2);
    }
}
