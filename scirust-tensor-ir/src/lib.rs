//! Canonical backend-neutral tensor graph IR for SciRust.
//!
//! This crate contains graph structure, tensor metadata and pure graph
//! transformations. It does not embed tensor storage, executable kernels,
//! devices, streams, or backend state.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod autodiff;
mod canonical_identity;
mod eligibility;
mod error;
mod graph;
mod ids;
mod morphodiff;
mod operation;
mod optimize;
mod physical;
mod representation;
mod shard;
mod transition;
mod verify;
mod vmap;

/// Build a reusable first-order linearization graph. The returned [`JvpGraph`]
/// exposes explicit tangent inputs, so it can be executed repeatedly for
/// different tangent vectors without rebuilding the primal transform.
pub use autodiff::jvp as linearize;
pub use autodiff::{AutodiffError, GradGraph, JvpGraph, VjpGraph, grad, jvp, value_and_grad, vjp};
pub use canonical_identity::{
    CanonicalIdentityError, GRAPH_STRUCTURAL_IDENTITY_V1, REPRESENTATION_ANCHOR_IDENTITY_V1,
    REPRESENTATION_PLAN_IDENTITY_V1, canonical_graph_bytes,
    canonical_representation_anchor_bytes, canonical_representation_plan_bytes,
};
pub use eligibility::{
    EligibilityBatch, EligibilityError, EligibilityEvidence, EligibilityReport, EligibilityState,
};
pub use error::GraphError;
pub use graph::{Graph, Node, TensorType};
pub use ids::{ConstantId, NodeId};
pub use morphodiff::{
    MORPHODIFF_ENGINE, MorphoDiff, MorphoDiffMode, MorphoDiffProgram, MorphoDiffReport,
};
pub use operation::{Operation, Scalar};
pub use optimize::{
    OptimizationConfig, OptimizationError, OptimizationStats, OptimizedGraph, optimize_graph,
};
pub use physical::{
    ContentIdentity, EffectiveBitsRate, LayoutIdentity, MaterializationClass,
    PhysicalAccountingError, PhysicalAccountingScope, PhysicalSegment, PhysicalSegmentId,
    PhysicalSegmentReference, PhysicalSegmentRole, ReconstructionRole, ResidentMaterialization,
    SegmentLifetime, SegmentUse,
};
pub use representation::{
    PrimitiveRepresentation, Rebinding, RepresentationComponent, RepresentationError,
    RepresentationId, RepresentationPlan, StorageBits,
};
pub use shard::{
    AxisShard, DeviceMesh, MeshAxis, PartitionSpec, RankShard, ShardError, ShardMapGraph,
    ShardPlan, ShardPolicy, plan_sharding, shard_map,
};
pub use transition::{PreparedReplan, PreparedReplanError};
pub use verify::{SemanticError, validate_semantics};
pub use vmap::{VmapError, VmapGraph, vmap};

pub use scirust_compute::{DType, Shape};
