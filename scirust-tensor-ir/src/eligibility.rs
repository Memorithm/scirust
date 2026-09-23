//! Immutable eligibility evidence sidecar for representation candidates.
//!
//! This module carries externally-qualified eligibility without teaching the
//! canonical Tensor IR how that qualification was computed. Boolean, F2,
//! Zhegalkin, max-plus, benchmark, or policy engines remain outside this crate.
//!
//! The sidecar is bound to canonical representation-plan bytes. Every candidate
//! rebinding is mechanically validated before its evidence is accepted, and a
//! later plan change makes the batch stale. Unknown eligibility remains distinct
//! from explicit ineligibility.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::{
    CanonicalIdentityError, Graph, NodeId, PreparedReplan, PreparedReplanError, Rebinding,
    RepresentationId, RepresentationPlan, canonical_representation_plan_bytes,
};

/// External qualification state of one representation candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EligibilityState {
    /// Evidence is missing or insufficient; the candidate is not eligible yet.
    Unknown,
    /// External qualification explicitly admitted the candidate.
    Eligible,
    /// External qualification explicitly rejected the candidate.
    Ineligible,
}

/// Opaque evidence binding for one validated representation candidate.
///
/// evidence_domain identifies the external semantic/qualification contract.
/// evidence_identity is an opaque, non-empty identity owned by that external
/// system. Tensor IR does not interpret it and does not claim it is a signature,
/// digest, scientific verdict, or benchmark result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibilityEvidence {
    /// Representation candidate covered by this evidence.
    pub candidate: Rebinding,
    /// External eligibility result.
    pub state: EligibilityState,
    /// Non-empty domain/schema identifier of the external qualifier.
    pub evidence_domain: String,
    /// Non-empty opaque identity of the qualifying evidence.
    pub evidence_identity: Vec<u8>,
}

/// Immutable eligibility snapshot anchored to one canonical representation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibilityBatch {
    plan_identity: Vec<u8>,
    entries: Vec<EligibilityEvidence>,
}

/// Categorized eligibility evidence, retaining provenance for every candidate.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EligibilityReport {
    /// Candidates explicitly admitted by external evidence.
    pub eligible: Vec<EligibilityEvidence>,
    /// Candidates with missing/insufficient evidence.
    pub unknown: Vec<EligibilityEvidence>,
    /// Candidates explicitly rejected by external evidence.
    pub ineligible: Vec<EligibilityEvidence>,
}

/// Failure while constructing or reading an eligibility sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EligibilityError {
    /// Canonical graph/plan identity could not be produced.
    Canonical(CanonicalIdentityError),
    /// Candidate rebinding is not mechanically valid for the anchored plan.
    InvalidCandidate {
        /// Entry index inside the submitted batch.
        index: usize,
        /// Underlying prepared-transition validation error.
        source: PreparedReplanError,
    },
    /// Evidence domain is blank.
    EmptyEvidenceDomain {
        /// Entry index inside the submitted batch.
        index: usize,
    },
    /// Evidence identity is empty.
    EmptyEvidenceIdentity {
        /// Entry index inside the submitted batch.
        index: usize,
    },
    /// The exact same node/representation candidate was submitted twice.
    DuplicateCandidate {
        /// Candidate node.
        node: NodeId,
        /// Candidate representation.
        representation: RepresentationId,
    },
    /// The current representation plan no longer matches the evidence snapshot.
    StalePlanEvidence,
}

impl fmt::Display for EligibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonical(error) => write!(formatter, "eligibility anchor failed: {error}"),
            Self::InvalidCandidate { index, source } => write!(
                formatter,
                "eligibility candidate {index} is not valid for the anchored plan: {source}"
            ),
            Self::EmptyEvidenceDomain { index } => {
                write!(formatter, "eligibility entry {index} has an empty evidence domain")
            }
            Self::EmptyEvidenceIdentity { index } => {
                write!(formatter, "eligibility entry {index} has an empty evidence identity")
            }
            Self::DuplicateCandidate {
                node,
                representation,
            } => write!(
                formatter,
                "eligibility candidate node {} representation {} occurs more than once",
                node.get(),
                representation.get()
            ),
            Self::StalePlanEvidence => formatter.write_str(
                "eligibility evidence was produced for a different representation-plan identity",
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EligibilityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Canonical(error) => Some(error),
            Self::InvalidCandidate { source, .. } => Some(source),
            Self::EmptyEvidenceDomain { .. }
            | Self::EmptyEvidenceIdentity { .. }
            | Self::DuplicateCandidate { .. }
            | Self::StalePlanEvidence => None,
        }
    }
}

impl From<CanonicalIdentityError> for EligibilityError {
    fn from(value: CanonicalIdentityError) -> Self {
        Self::Canonical(value)
    }
}

impl EligibilityBatch {
    /// Validate and freeze external evidence for representation candidates.
    ///
    /// Construction binds the batch to canonical representation-plan bytes and
    /// mechanically validates every candidate through PreparedReplan. This does
    /// not validate the truth of the external eligibility claim; it only proves
    /// that the claim names a valid candidate under the current plan and carries
    /// non-empty provenance identifiers.
    ///
    /// # Errors
    ///
    /// Fails closed on invalid/stale graph-plan identity, invalid candidate
    /// rebindings, blank evidence metadata, or duplicate candidates.
    ///
    /// # Examples
    ///
    /// A valid candidate can carry explicit eligibility evidence.
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, EligibilityBatch, EligibilityEvidence, EligibilityState, Graph, Rebinding,
    ///     RepresentationPlan, Shape, TensorType,
    /// };
    ///
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let evidence = EligibilityEvidence {
    ///     candidate: Rebinding {
    ///         node: x,
    ///         representation: plan.assignment(x).unwrap(),
    ///     },
    ///     state: EligibilityState::Eligible,
    ///     evidence_domain: "elastic.boolean-survivor.v1".into(),
    ///     evidence_identity: b"evidence-1".to_vec(),
    /// };
    /// assert!(EligibilityBatch::new(&graph, &plan, vec![evidence]).is_ok());
    /// ```
    ///
    /// Empty external evidence identity is rejected rather than interpreted as
    /// unknown evidence.
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, EligibilityBatch, EligibilityError, EligibilityEvidence, EligibilityState,
    ///     Graph, Rebinding, RepresentationPlan, Shape, TensorType,
    /// };
    ///
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let evidence = EligibilityEvidence {
    ///     candidate: Rebinding {
    ///         node: x,
    ///         representation: plan.assignment(x).unwrap(),
    ///     },
    ///     state: EligibilityState::Unknown,
    ///     evidence_domain: "tdi.boolean-carrier.v1".into(),
    ///     evidence_identity: Vec::new(),
    /// };
    /// assert!(matches!(
    ///     EligibilityBatch::new(&graph, &plan, vec![evidence]),
    ///     Err(EligibilityError::EmptyEvidenceIdentity { .. })
    /// ));
    /// ```
    pub fn new(
        graph: &Graph,
        plan: &RepresentationPlan,
        entries: Vec<EligibilityEvidence>,
    ) -> Result<Self, EligibilityError> {
        let plan_identity = canonical_representation_plan_bytes(plan, graph)?;
        let mut seen = BTreeSet::new();

        for (index, entry) in entries.iter().enumerate() {
            if entry.evidence_domain.trim().is_empty() {
                return Err(EligibilityError::EmptyEvidenceDomain { index });
            }
            if entry.evidence_identity.is_empty() {
                return Err(EligibilityError::EmptyEvidenceIdentity { index });
            }

            let key = (entry.candidate.node, entry.candidate.representation);
            if !seen.insert(key) {
                return Err(EligibilityError::DuplicateCandidate {
                    node: entry.candidate.node,
                    representation: entry.candidate.representation,
                });
            }

            PreparedReplan::prepare(plan, graph, &[entry.candidate]).map_err(|source| {
                EligibilityError::InvalidCandidate { index, source }
            })?;
        }

        Ok(Self {
            plan_identity,
            entries,
        })
    }

    /// Revalidate the plan anchor and categorize the retained evidence.
    ///
    /// The report preserves complete evidence entries rather than reducing
    /// Unknown to Ineligible or discarding provenance. A stale plan is rejected
    /// before any category is returned.
    ///
    /// # Errors
    ///
    /// Returns StalePlanEvidence if the current plan identity differs from the
    /// construction snapshot, or propagates canonical identity errors.
    ///
    /// # Examples
    ///
    /// Unknown and ineligible states remain separate.
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, EligibilityBatch, EligibilityEvidence, EligibilityState, Graph, Rebinding,
    ///     RepresentationPlan, Shape, TensorType,
    /// };
    ///
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let plan = RepresentationPlan::dense(&graph).unwrap();
    /// let candidate = Rebinding {
    ///     node: x,
    ///     representation: plan.assignment(x).unwrap(),
    /// };
    /// let batch = EligibilityBatch::new(
    ///     &graph,
    ///     &plan,
    ///     vec![EligibilityEvidence {
    ///         candidate,
    ///         state: EligibilityState::Unknown,
    ///         evidence_domain: "flat.maa.v1".into(),
    ///         evidence_identity: b"route-7".to_vec(),
    ///     }],
    /// )
    /// .unwrap();
    /// let report = batch.report(&graph, &plan).unwrap();
    /// assert_eq!(report.unknown.len(), 1);
    /// assert!(report.eligible.is_empty());
    /// assert!(report.ineligible.is_empty());
    /// ```
    ///
    /// Mutating the representation plan invalidates the evidence snapshot.
    ///
    /// ```
    /// use scirust_tensor_ir::{
    ///     DType, EligibilityBatch, EligibilityError, EligibilityEvidence, EligibilityState,
    ///     Graph, Rebinding, RepresentationPlan, Shape, TensorType,
    /// };
    ///
    /// let mut graph = Graph::new();
    /// let x = graph
    ///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
    ///     .unwrap();
    /// graph.set_outputs(vec![x]).unwrap();
    /// let mut plan = RepresentationPlan::dense(&graph).unwrap();
    /// let candidate = Rebinding {
    ///     node: x,
    ///     representation: plan.assignment(x).unwrap(),
    /// };
    /// let batch = EligibilityBatch::new(
    ///     &graph,
    ///     &plan,
    ///     vec![EligibilityEvidence {
    ///         candidate,
    ///         state: EligibilityState::Eligible,
    ///         evidence_domain: "elastic.v1".into(),
    ///         evidence_identity: b"evidence".to_vec(),
    ///     }],
    /// )
    /// .unwrap();
    /// plan.declare_dense(DType::F16).unwrap();
    /// assert_eq!(
    ///     batch.report(&graph, &plan),
    ///     Err(EligibilityError::StalePlanEvidence)
    /// );
    /// ```
    pub fn report(
        &self,
        graph: &Graph,
        plan: &RepresentationPlan,
    ) -> Result<EligibilityReport, EligibilityError> {
        let current = canonical_representation_plan_bytes(plan, graph)?;
        if current != self.plan_identity {
            return Err(EligibilityError::StalePlanEvidence);
        }

        let mut report = EligibilityReport::default();
        for entry in &self.entries {
            match entry.state {
                EligibilityState::Unknown => report.unknown.push(entry.clone()),
                EligibilityState::Eligible => report.eligible.push(entry.clone()),
                EligibilityState::Ineligible => report.ineligible.push(entry.clone()),
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{DType, Shape, TensorType};

    fn graph_plan() -> (Graph, NodeId, RepresentationPlan) {
        let mut graph = Graph::new();
        let x = graph
            .add_input("x", TensorType::new(DType::F32, Shape::new([2usize])))
            .unwrap();
        graph.set_outputs(vec![x]).unwrap();
        let plan = RepresentationPlan::dense(&graph).unwrap();
        (graph, x, plan)
    }

    fn evidence(
        node: NodeId,
        representation: RepresentationId,
        state: EligibilityState,
        identity: &[u8],
    ) -> EligibilityEvidence {
        EligibilityEvidence {
            candidate: Rebinding {
                node,
                representation,
            },
            state,
            evidence_domain: "test.qualifier.v1".into(),
            evidence_identity: identity.to_vec(),
        }
    }

    #[test]
    fn duplicate_candidate_is_rejected() {
        let (graph, node, plan) = graph_plan();
        let representation = plan.assignment(node).unwrap();
        let batch = EligibilityBatch::new(
            &graph,
            &plan,
            vec![
                evidence(node, representation, EligibilityState::Eligible, b"e1"),
                evidence(
                    node,
                    plan.representations()
                        .iter()
                        .position(|r| r == plan.representation(representation).unwrap())
                        .map(|i| RepresentationId::new(i as u32))
                        .unwrap(),
                    EligibilityState::Unknown,
                    b"e2",
                ),
            ],
        );
        assert!(matches!(
            batch,
            Err(EligibilityError::DuplicateCandidate { .. })
        ));
    }

    #[test]
    fn distinct_candidates_can_have_distinct_eligibility_states() {
        let (graph, node, mut plan) = graph_plan();
        let dense = plan.assignment(node).unwrap();
        let f32_again = plan.declare_dense(DType::F32).unwrap();
        assert_eq!(dense, f32_again);

        let batch = EligibilityBatch::new(
            &graph,
            &plan,
            vec![evidence(node, dense, EligibilityState::Ineligible, b"reject")],
        )
        .unwrap();
        let report = batch.report(&graph, &plan).unwrap();
        assert_eq!(report.ineligible.len(), 1);
        assert_eq!(report.ineligible[0].evidence_identity, b"reject");
    }

    #[test]
    fn different_valid_representation_candidate_is_preserved() {
        let (graph, node, mut plan) = graph_plan();
        let dense_u8 = plan.declare_dense(DType::U8).unwrap();
        let dense_f32 = plan.assignment(node).unwrap();
        let quantized = plan
            .declare_quantized_per_tensor(
                TensorType::new(DType::U8, Shape::new([2usize])),
                dense_u8,
                TensorType::new(DType::F32, Shape::scalar()),
                dense_f32,
            )
            .unwrap();

        let batch = EligibilityBatch::new(
            &graph,
            &plan,
            vec![
                evidence(node, dense_f32, EligibilityState::Eligible, b"dense"),
                evidence(node, quantized, EligibilityState::Unknown, b"quantized"),
            ],
        )
        .unwrap();
        let report = batch.report(&graph, &plan).unwrap();
        assert_eq!(report.eligible.len(), 1);
        assert_eq!(report.unknown.len(), 1);
        assert_eq!(report.eligible[0].candidate.representation, dense_f32);
        assert_eq!(report.unknown[0].candidate.representation, quantized);
    }

    #[test]
    fn stale_plan_is_rejected_before_reporting() {
        let (graph, node, mut plan) = graph_plan();
        let representation = plan.assignment(node).unwrap();
        let batch = EligibilityBatch::new(
            &graph,
            &plan,
            vec![evidence(
                node,
                representation,
                EligibilityState::Eligible,
                b"evidence",
            )],
        )
        .unwrap();

        plan.declare_dense(DType::F16).unwrap();
        assert_eq!(
            batch.report(&graph, &plan),
            Err(EligibilityError::StalePlanEvidence)
        );
    }

    #[test]
    fn invalid_candidate_is_rejected_before_evidence_can_be_used() {
        let (graph, node, plan) = graph_plan();
        let entry = evidence(
            node,
            RepresentationId::new(999),
            EligibilityState::Eligible,
            b"evidence",
        );
        assert!(matches!(
            EligibilityBatch::new(&graph, &plan, vec![entry]),
            Err(EligibilityError::InvalidCandidate { .. })
        ));
    }

    #[test]
    fn blank_metadata_fails_closed() {
        let (graph, node, plan) = graph_plan();
        let representation = plan.assignment(node).unwrap();

        let mut blank_domain =
            evidence(node, representation, EligibilityState::Unknown, b"identity");
        blank_domain.evidence_domain = " ".into();
        assert_eq!(
            EligibilityBatch::new(&graph, &plan, vec![blank_domain]),
            Err(EligibilityError::EmptyEvidenceDomain { index: 0 })
        );

        assert_eq!(
            EligibilityBatch::new(
                &graph,
                &plan,
                vec![evidence(
                    node,
                    representation,
                    EligibilityState::Unknown,
                    b"",
                )],
            ),
            Err(EligibilityError::EmptyEvidenceIdentity { index: 0 })
        );
    }
}
