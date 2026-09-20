//! Two-phase preparation and commit for representation-plan transitions.
//!
//! This module deliberately contains no selection policy. A caller such as
//! ElasticXxx, Forge, an attention planner, or a research bench may propose a
//! [`Rebinding`] set, inspect its exact storage consequence, verify it using its
//! own invariants/evidence, and commit only if the source plan is still current.
//! Dropping a prepared transition performs no mutation and therefore provides
//! the rollback/no-op boundary for rejected candidates.

use core::fmt;

use crate::{
    Graph, Rebinding, RepresentationError, RepresentationId, RepresentationPlan, StorageBits,
};

/// Validated, non-mutating representation transition prepared from one exact plan state.
///
/// The token retains both the source and projected plans. [`Self::commit`] is
/// fail-closed: if the target plan changed after preparation, the token is
/// rejected as stale instead of overwriting newer decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedReplan {
    base: RepresentationPlan,
    projected: RepresentationPlan,
    before_storage_bits: StorageBits,
    after_storage_bits: StorageBits,
}

/// Failure while preparing or committing a two-phase representation transition.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreparedReplanError {
    /// The underlying representation plan or graph contract was invalid.
    Representation(RepresentationError),
    /// The target plan changed after this transition was prepared.
    StalePlan,
}

impl fmt::Display for PreparedReplanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Representation(error) =>
            {
                write!(formatter, "representation transition failed: {error}")
            },
            Self::StalePlan => write!(formatter, "prepared representation transition is stale"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PreparedReplanError {}

impl From<RepresentationError> for PreparedReplanError {
    fn from(value: RepresentationError) -> Self {
        Self::Representation(value)
    }
}

impl PreparedReplan {
    /// Prepare and validate a representation transition without mutating `plan`.
    ///
    /// Validation is exactly the same as [`RepresentationPlan::replan`]. The
    /// returned token also records exact storage totals before and after the
    /// projected transition.
    ///
    /// # Errors
    ///
    /// Returns [`PreparedReplanError::Representation`] when the graph is no
    /// longer compatible with the plan, a node/representation is unknown, a
    /// representation cannot encode the logical tensor, or exact storage
    /// accounting overflows.
    ///
    /// # Examples
    ///
    /// An empty proposal is a validated no-op:
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, Graph, PreparedReplan, RepresentationPlan, Shape, TensorType,
    /// };
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([2usize, 2])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let prepared = PreparedReplan::prepare(&plan, &graph, &[]).unwrap();
    /// assert_eq!(prepared.before_storage_bits(), prepared.after_storage_bits());
    /// assert_eq!(
    ///     plan.assignment(x),
    ///     Some(prepared.projected_assignments()[0])
    /// );
    /// ```
    ///
    /// Invalid representation identifiers are rejected before any mutation:
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, Graph, PreparedReplan, PreparedReplanError, Rebinding, RepresentationError,
    ///     RepresentationId, RepresentationPlan, Shape, TensorType,
    /// };
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let error = PreparedReplan::prepare(
    ///     &plan,
    ///     &graph,
    ///     &[Rebinding {
    ///         node: x,
    ///         representation: RepresentationId::new(999),
    ///     }],
    /// )
    /// .unwrap_err();
    /// assert!(matches!(
    ///     error,
    ///     PreparedReplanError::Representation(
    ///         RepresentationError::InvalidRepresentationId { .. }
    ///     )
    /// ));
    /// ```
    pub fn prepare(
        plan: &RepresentationPlan,
        graph: &Graph,
        rebinding: &[Rebinding],
    ) -> Result<Self, PreparedReplanError> {
        let before_storage_bits = plan.total_storage_bits(graph)?;
        let mut projected = plan.clone();
        projected.replan(graph, rebinding)?;
        let after_storage_bits = projected.total_storage_bits(graph)?;

        Ok(Self {
            base: plan.clone(),
            projected,
            before_storage_bits,
            after_storage_bits,
        })
    }

    /// Exact storage total of the source plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_tensor_ir::{DType, Graph, PreparedReplan, RepresentationPlan, Shape, TensorType};
    /// let mut graph = Graph::new();
    /// let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let prepared = PreparedReplan::prepare(&plan, &graph, &[]).unwrap();
    /// assert_eq!(prepared.before_storage_bits().get(), 32);
    /// ```
    #[must_use]
    pub const fn before_storage_bits(&self) -> StorageBits {
        self.before_storage_bits
    }

    /// Exact storage total after the projected transition.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_tensor_ir::{DType, Graph, PreparedReplan, RepresentationPlan, Shape, TensorType};
    /// let mut graph = Graph::new();
    /// let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let prepared = PreparedReplan::prepare(&plan, &graph, &[]).unwrap();
    /// assert_eq!(prepared.after_storage_bits(), prepared.before_storage_bits());
    /// ```
    #[must_use]
    pub const fn after_storage_bits(&self) -> StorageBits {
        self.after_storage_bits
    }

    /// Projected node assignments in canonical node order.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_tensor_ir::{DType, Graph, PreparedReplan, RepresentationPlan, Shape, TensorType};
    /// let mut graph = Graph::new();
    /// let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let prepared = PreparedReplan::prepare(&plan, &graph, &[]).unwrap();
    /// assert_eq!(prepared.projected_assignments(), plan.assignments());
    /// ```
    #[must_use]
    pub fn projected_assignments(&self) -> &[RepresentationId] {
        self.projected.assignments()
    }

    /// Commit this prepared transition if the target plan is still unchanged.
    ///
    /// No partial mutation occurs. If `plan` differs from the exact source plan
    /// captured during preparation, the transition is rejected as stale. This
    /// includes representation declarations as well as assignment changes.
    ///
    /// # Errors
    ///
    /// Returns [`PreparedReplanError::StalePlan`] if `plan` changed since
    /// preparation, or [`PreparedReplanError::Representation`] if `graph` no
    /// longer matches the canonical graph anchor.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, Graph, PreparedReplan, RepresentationPlan, Shape, TensorType,
    /// };
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let mut plan = RepresentationPlan::dense(&graph).unwrap();
    /// let prepared = PreparedReplan::prepare(&plan, &graph, &[]).unwrap();
    /// prepared.commit(&mut plan, &graph).unwrap();
    /// ```
    ///
    /// A plan modified after preparation is not overwritten:
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, Graph, PreparedReplan, PreparedReplanError, RepresentationPlan, Shape,
    ///     TensorType,
    /// };
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let mut plan = RepresentationPlan::dense(&graph).unwrap();
    /// let prepared = PreparedReplan::prepare(&plan, &graph, &[]).unwrap();
    /// plan.declare_dense(DType::F16).unwrap();
    /// assert_eq!(
    ///     prepared.commit(&mut plan, &graph),
    ///     Err(PreparedReplanError::StalePlan)
    /// );
    /// ```
    pub fn commit(
        self,
        plan: &mut RepresentationPlan,
        graph: &Graph,
    ) -> Result<(), PreparedReplanError> {
        plan.ensure_compatible_with(graph)?;
        self.projected.ensure_compatible_with(graph)?;
        if *plan != self.base
        {
            return Err(PreparedReplanError::StalePlan);
        }

        *plan = self.projected;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use scirust_compute::{DType, Shape};

    use super::*;
    use crate::TensorType;

    fn matrix_graph() -> (Graph, crate::NodeId) {
        let mut graph = Graph::new();
        let node = graph
            .add_input("x", TensorType::new(DType::F32, Shape::new([2usize, 2])))
            .unwrap();
        graph.set_outputs(vec![node]).unwrap();
        (graph, node)
    }

    fn quantized_representation(plan: &mut RepresentationPlan) -> RepresentationId {
        let dense_u8 = plan.declare_dense(DType::U8).unwrap();
        let dense_f32 = plan.declare_dense(DType::F32).unwrap();
        plan.declare_quantized_per_tensor(
            TensorType::new(DType::U8, Shape::new([2usize, 2])),
            dense_u8,
            TensorType::new(DType::F32, Shape::scalar()),
            dense_f32,
        )
        .unwrap()
    }

    #[test]
    fn prepare_is_non_mutating_and_reports_exact_storage_delta() {
        let (graph, node) = matrix_graph();
        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let quantized = quantized_representation(&mut plan);
        let before_assignments = plan.assignments().to_vec();

        let prepared = PreparedReplan::prepare(
            &plan,
            &graph,
            &[Rebinding {
                node,
                representation: quantized,
            }],
        )
        .unwrap();

        assert_eq!(prepared.before_storage_bits(), StorageBits::new(128));
        assert_eq!(prepared.after_storage_bits(), StorageBits::new(64));
        assert_eq!(plan.assignments(), before_assignments);
        assert_eq!(prepared.projected_assignments(), &[quantized]);
    }

    #[test]
    fn commit_applies_projected_plan_atomically() {
        let (graph, node) = matrix_graph();
        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let quantized = quantized_representation(&mut plan);
        let prepared = PreparedReplan::prepare(
            &plan,
            &graph,
            &[Rebinding {
                node,
                representation: quantized,
            }],
        )
        .unwrap();

        prepared.commit(&mut plan, &graph).unwrap();

        assert_eq!(plan.assignment(node), Some(quantized));
        assert_eq!(plan.total_storage_bits(&graph), Ok(StorageBits::new(64)));
    }

    #[test]
    fn commit_rejects_stale_plan_without_overwrite() {
        let (graph, node) = matrix_graph();
        let mut plan = RepresentationPlan::dense(&graph).unwrap();
        let quantized = quantized_representation(&mut plan);
        let prepared = PreparedReplan::prepare(
            &plan,
            &graph,
            &[Rebinding {
                node,
                representation: quantized,
            }],
        )
        .unwrap();

        let extra = plan.declare_dense(DType::F16).unwrap();
        let assignments_before_commit = plan.assignments().to_vec();
        assert_eq!(
            prepared.commit(&mut plan, &graph),
            Err(PreparedReplanError::StalePlan)
        );
        assert_eq!(plan.assignments(), assignments_before_commit);
        assert_eq!(
            plan.representation(extra),
            Some(&crate::PrimitiveRepresentation::dense(DType::F16))
        );
    }
}
