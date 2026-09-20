//! Verify-before-measure bridge for Tensor IR search candidates.
//!
//! This crate sits deliberately outside both canonical Tensor IR and
//! `scirust-opt-core`. Tensor IR owns semantic candidates and atomic commit;
//! the optimizer owns trials/objectives; external systems such as Forge may own
//! proposal/search/selection policy.
//!
//! The typestate chain is:
//!
//! ```text
//! TensorSearchCandidate
//!   -> VerificationOutcome
//!      -> FailedVerification                (never measurable)
//!      -> VerifiedTensorCandidate
//!           -> MeasurementPermit
//!                -> MeasuredTensorCandidate
//!                     -> SearchRecord
//!                          -> commit only when Survivor
//! ```
//!
//! Canonical Tensor IR bytes are carried as the semantic candidate identity.
//! This crate does not choose or require a hash algorithm. Environment,
//! verification and search evidence remain separate from logical identity.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

use scirust_opt_core::{TrialId, TrialOutcome};
use scirust_tensor_ir::{
    CanonicalIdentityError, Graph, PreparedReplan, PreparedReplanError, Rebinding,
    RepresentationError, RepresentationPlan, StorageBits, canonical_representation_plan_bytes,
};

/// Prepared Tensor IR candidate associated with one optimization trial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSearchCandidate {
    /// SciRust optimization trial that proposed this candidate.
    pub trial: TrialId,
    /// Canonical bytes of the plan before the proposal.
    pub base_plan_identity: Vec<u8>,
    /// Canonical bytes of the projected candidate plan.
    pub candidate_plan_identity: Vec<u8>,
    /// Ordered representation rebindings proposed for this candidate.
    pub rebindings: Vec<Rebinding>,
    /// Exact representation storage before the proposal.
    pub before_storage_bits: StorageBits,
    /// Exact representation storage after the proposal.
    pub after_storage_bits: StorageBits,
    prepared: PreparedReplan,
}

/// Independent verification evidence attached to a Tensor IR candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationEvidence {
    /// Stable identity of the verifier/oracle contract.
    pub verifier_identity: String,
    /// Opaque, non-empty identity of executed verification evidence.
    pub evidence_identity: Vec<u8>,
    /// Stable machine-oriented reason for the verification disposition.
    pub reason_code: String,
}

/// Candidate that passed independent verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTensorCandidate {
    /// Prepared semantic candidate.
    pub candidate: TensorSearchCandidate,
    /// Independent verification evidence.
    pub verification: VerificationEvidence,
}

/// Candidate that failed independent verification.
///
/// This type intentionally has no measurement-permit constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedVerification {
    /// Prepared semantic candidate.
    pub candidate: TensorSearchCandidate,
    /// Negative verification evidence retained for provenance.
    pub verification: VerificationEvidence,
}

/// Typed result of independent candidate verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationOutcome {
    /// Candidate passed and may proceed to the measurement-permit boundary.
    Passed(VerifiedTensorCandidate),
    /// Candidate failed and cannot acquire performance merit.
    Failed(FailedVerification),
}

/// Permission to measure one already-verified candidate in one bound environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementPermit {
    /// Verified candidate covered by this permit.
    pub verified: VerifiedTensorCandidate,
    /// Opaque, non-empty identity of the measurement environment.
    pub environment_identity: Vec<u8>,
}

/// Finite objective measurements for one verified candidate/environment pair.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredTensorCandidate {
    /// Permit binding candidate, verification and environment.
    pub permit: MeasurementPermit,
    /// Objective values in the optimizer's configured objective order.
    pub objectives: Vec<f64>,
}

/// Forge/search-local disposition after successful verification and measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDisposition {
    /// Candidate did not survive the declared search/selection stage.
    Rejected,
    /// Candidate survived the declared search/selection stage.
    Survivor,
}

/// Search disposition with retained prerequisite evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchRecord {
    /// Measured candidate covered by the search disposition.
    pub measured: MeasuredTensorCandidate,
    /// Search-local disposition; not a scientific verdict.
    pub disposition: SearchDisposition,
    /// Opaque, non-empty identity of executed search/selection evidence.
    pub search_evidence_identity: Vec<u8>,
    /// Stable machine-oriented reason code.
    pub reason_code: String,
}

/// Result of applying a search record to a mutable representation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitResult {
    /// Candidate was rejected by search; the plan was not mutated.
    NotCommitted,
    /// Candidate survived and its prepared transition was committed.
    Committed,
}

/// Failure at the Tensor IR / optimization search boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchBridgeError {
    /// Canonical plan identity could not be produced.
    Canonical(CanonicalIdentityError),
    /// Prepared transition validation or stale-plan commit failed.
    Prepared(PreparedReplanError),
    /// Projecting the candidate plan failed.
    Representation(RepresentationError),
    /// Verifier identity is blank.
    EmptyVerifierIdentity,
    /// Verification evidence identity is empty.
    EmptyVerificationEvidenceIdentity,
    /// Verification reason code is blank.
    EmptyVerificationReason,
    /// Measurement environment identity is empty.
    EmptyEnvironmentIdentity,
    /// A successful measurement supplied no objectives.
    EmptyObjectives,
    /// One objective value was NaN or infinite.
    NonFiniteObjective {
        /// Objective index.
        index: usize,
    },
    /// Search evidence identity is empty.
    EmptySearchEvidenceIdentity,
    /// Search reason code is blank.
    EmptySearchReason,
}

impl fmt::Display for SearchBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonical(error) => write!(formatter, "canonical candidate identity failed: {error}"),
            Self::Prepared(error) => write!(formatter, "prepared candidate transition failed: {error}"),
            Self::Representation(error) => write!(formatter, "candidate projection failed: {error}"),
            Self::EmptyVerifierIdentity => formatter.write_str("verifier identity must be non-empty"),
            Self::EmptyVerificationEvidenceIdentity => {
                formatter.write_str("verification evidence identity must be non-empty")
            }
            Self::EmptyVerificationReason => {
                formatter.write_str("verification reason code must be non-empty")
            }
            Self::EmptyEnvironmentIdentity => {
                formatter.write_str("measurement environment identity must be non-empty")
            }
            Self::EmptyObjectives => formatter.write_str("measurement must contain at least one objective"),
            Self::NonFiniteObjective { index } => {
                write!(formatter, "objective {index} is not finite")
            }
            Self::EmptySearchEvidenceIdentity => {
                formatter.write_str("search evidence identity must be non-empty")
            }
            Self::EmptySearchReason => formatter.write_str("search reason code must be non-empty"),
        }
    }
}

impl std::error::Error for SearchBridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Canonical(error) => Some(error),
            Self::Prepared(error) => Some(error),
            Self::Representation(error) => Some(error),
            Self::EmptyVerifierIdentity
            | Self::EmptyVerificationEvidenceIdentity
            | Self::EmptyVerificationReason
            | Self::EmptyEnvironmentIdentity
            | Self::EmptyObjectives
            | Self::NonFiniteObjective { .. }
            | Self::EmptySearchEvidenceIdentity
            | Self::EmptySearchReason => None,
        }
    }
}

impl From<CanonicalIdentityError> for SearchBridgeError {
    fn from(value: CanonicalIdentityError) -> Self {
        Self::Canonical(value)
    }
}

impl From<PreparedReplanError> for SearchBridgeError {
    fn from(value: PreparedReplanError) -> Self {
        Self::Prepared(value)
    }
}

impl From<RepresentationError> for SearchBridgeError {
    fn from(value: RepresentationError) -> Self {
        Self::Representation(value)
    }
}

/// Prepare one canonical Tensor IR representation candidate for a search trial.
///
/// The trial identifier is envelope metadata and is not part of canonical Tensor
/// IR identity. The candidate identity is computed from the projected
/// representation plan. No verification, measurement or execution occurs.
///
/// # Errors
///
/// Fails closed on incompatible graphs/plans, invalid rebindings or canonical
/// encoding failure.
///
/// # Examples
///
/// ```
/// use scirust_opt_core::TrialId;
/// use scirust_tensor_ir::{DType, Graph, Rebinding, RepresentationPlan, Shape, TensorType};
/// use scirust_tensor_search_bridge::prepare_candidate;
///
/// let mut graph = Graph::new();
/// let x = graph
///     .add_input("x", TensorType::new(DType::F32, Shape::new([2usize])))
///     .unwrap();
/// graph.set_outputs(vec![x]).unwrap();
/// let plan = RepresentationPlan::dense(&graph).unwrap();
/// let candidate = prepare_candidate(
///     TrialId::new(7),
///     &graph,
///     &plan,
///     vec![Rebinding {
///         node: x,
///         representation: plan.assignment(x).unwrap(),
///     }],
/// )
/// .unwrap();
/// assert_eq!(candidate.trial, TrialId::new(7));
/// assert_eq!(candidate.before_storage_bits, candidate.after_storage_bits);
/// ```
///
/// A representation unknown to the plan is rejected before verification.
///
/// ```
/// use scirust_opt_core::TrialId;
/// use scirust_tensor_ir::{
///     DType, Graph, Rebinding, RepresentationId, RepresentationPlan, Shape, TensorType,
/// };
/// use scirust_tensor_search_bridge::prepare_candidate;
///
/// let mut graph = Graph::new();
/// let x = graph
///     .add_input("x", TensorType::new(DType::F32, Shape::new([1usize])))
///     .unwrap();
/// graph.set_outputs(vec![x]).unwrap();
/// let plan = RepresentationPlan::dense(&graph).unwrap();
/// assert!(prepare_candidate(
///     TrialId::new(0),
///     &graph,
///     &plan,
///     vec![Rebinding {
///         node: x,
///         representation: RepresentationId::new(999),
///     }],
/// )
/// .is_err());
/// ```
pub fn prepare_candidate(
    trial: TrialId,
    graph: &Graph,
    plan: &RepresentationPlan,
    rebindings: Vec<Rebinding>,
) -> Result<TensorSearchCandidate, SearchBridgeError> {
    let base_plan_identity = canonical_representation_plan_bytes(plan, graph)?;
    let prepared = PreparedReplan::prepare(plan, graph, &rebindings)?;
    let before_storage_bits = prepared.before_storage_bits();
    let after_storage_bits = prepared.after_storage_bits();

    let mut projected = plan.clone();
    projected.replan(graph, &rebindings)?;
    let candidate_plan_identity = canonical_representation_plan_bytes(&projected, graph)?;

    Ok(TensorSearchCandidate {
        trial,
        base_plan_identity,
        candidate_plan_identity,
        rebindings,
        before_storage_bits,
        after_storage_bits,
        prepared,
    })
}

/// Record independent correctness verification for a prepared candidate.
///
/// Failed verification is retained as [`VerificationOutcome::Failed`] and has no
/// API path to [`permit_measurement`]. This enforces verify-before-measure at
/// the type boundary rather than by convention.
///
/// # Errors
///
/// Rejects blank verifier identity, empty evidence identity or blank reason code.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::{prepare_candidate, record_verification, VerificationOutcome};
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// let outcome = record_verification(
///     candidate,
///     "reference-oracle-v1",
///     b"verification:42".to_vec(),
///     true,
///     "bit-exact",
/// )
/// .unwrap();
/// assert!(matches!(outcome, VerificationOutcome::Passed(_)));
/// ```
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::{prepare_candidate, record_verification, VerificationOutcome};
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// let outcome = record_verification(
///     candidate,
///     "reference-oracle-v1",
///     b"verification:bad".to_vec(),
///     false,
///     "oracle-mismatch",
/// )
/// .unwrap();
/// assert!(matches!(outcome, VerificationOutcome::Failed(_)));
/// ```
pub fn record_verification(
    candidate: TensorSearchCandidate,
    verifier_identity: impl Into<String>,
    evidence_identity: Vec<u8>,
    passed: bool,
    reason_code: impl Into<String>,
) -> Result<VerificationOutcome, SearchBridgeError> {
    let verifier_identity = verifier_identity.into();
    if verifier_identity.trim().is_empty() {
        return Err(SearchBridgeError::EmptyVerifierIdentity);
    }
    if evidence_identity.is_empty() {
        return Err(SearchBridgeError::EmptyVerificationEvidenceIdentity);
    }
    let reason_code = reason_code.into();
    if reason_code.trim().is_empty() {
        return Err(SearchBridgeError::EmptyVerificationReason);
    }
    let verification = VerificationEvidence {
        verifier_identity,
        evidence_identity,
        reason_code,
    };
    if passed {
        Ok(VerificationOutcome::Passed(VerifiedTensorCandidate {
            candidate,
            verification,
        }))
    } else {
        Ok(VerificationOutcome::Failed(FailedVerification {
            candidate,
            verification,
        }))
    }
}

/// Bind a verified candidate to the environment in which objectives may be measured.
///
/// # Errors
///
/// Rejects an empty environment identity.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::{prepare_candidate, record_verification, permit_measurement, VerificationOutcome};
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"ok".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// let permit = permit_measurement(verified, b"cpu:x86_64;rust=1.89".to_vec()).unwrap();
/// assert!(!permit.environment_identity.is_empty());
/// ```
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::{prepare_candidate, record_verification, permit_measurement, VerificationOutcome};
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"ok".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// assert!(permit_measurement(verified, Vec::new()).is_err());
/// ```
pub fn permit_measurement(
    verified: VerifiedTensorCandidate,
    environment_identity: Vec<u8>,
) -> Result<MeasurementPermit, SearchBridgeError> {
    if environment_identity.is_empty() {
        return Err(SearchBridgeError::EmptyEnvironmentIdentity);
    }
    Ok(MeasurementPermit {
        verified,
        environment_identity,
    })
}

/// Record finite objective measurements under an existing permit.
///
/// # Errors
///
/// Requires at least one objective and rejects every NaN/infinite value.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// let measured = record_measurements(permit, vec![12.5, 1024.0]).unwrap();
/// assert_eq!(measured.objectives, vec![12.5, 1024.0]);
/// ```
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// assert!(record_measurements(permit, vec![f64::NAN]).is_err());
/// ```
pub fn record_measurements(
    permit: MeasurementPermit,
    objectives: Vec<f64>,
) -> Result<MeasuredTensorCandidate, SearchBridgeError> {
    if objectives.is_empty() {
        return Err(SearchBridgeError::EmptyObjectives);
    }
    for (index, value) in objectives.iter().enumerate() {
        if !value.is_finite() {
            return Err(SearchBridgeError::NonFiniteObjective { index });
        }
    }
    Ok(MeasuredTensorCandidate { permit, objectives })
}

/// Bind a search-local rejection/survival decision to its executed evidence.
///
/// This disposition is deliberately not a scientific conclusion.
///
/// # Errors
///
/// Rejects empty search evidence identity or blank reason code.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// # let measured = record_measurements(permit, vec![1.0]).unwrap();
/// let record = record_search_disposition(
///     measured,
///     SearchDisposition::Survivor,
///     b"search:1".to_vec(),
///     "pareto-survivor",
/// )
/// .unwrap();
/// assert_eq!(record.disposition, SearchDisposition::Survivor);
/// ```
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// # let measured = record_measurements(permit, vec![1.0]).unwrap();
/// assert!(record_search_disposition(
///     measured,
///     SearchDisposition::Rejected,
///     Vec::new(),
///     "rejected",
/// )
/// .is_err());
/// ```
pub fn record_search_disposition(
    measured: MeasuredTensorCandidate,
    disposition: SearchDisposition,
    search_evidence_identity: Vec<u8>,
    reason_code: impl Into<String>,
) -> Result<SearchRecord, SearchBridgeError> {
    if search_evidence_identity.is_empty() {
        return Err(SearchBridgeError::EmptySearchEvidenceIdentity);
    }
    let reason_code = reason_code.into();
    if reason_code.trim().is_empty() {
        return Err(SearchBridgeError::EmptySearchReason);
    }
    Ok(SearchRecord {
        measured,
        disposition,
        search_evidence_identity,
        reason_code,
    })
}

/// Convert a measured search record to the terminal SciRust optimizer outcome.
///
/// Verification failure has a separate [`failed_verification_outcome`] path and
/// therefore can never fabricate objective values.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::{TrialId, TrialOutcome};
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// # let measured = record_measurements(permit, vec![3.0]).unwrap();
/// # let record = record_search_disposition(measured, SearchDisposition::Rejected, b"s".to_vec(), "not-selected").unwrap();
/// assert_eq!(optimizer_outcome(&record), TrialOutcome::Complete(vec![3.0]));
/// ```
///
/// ```
/// # use scirust_opt_core::{TrialId, TrialOutcome};
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// # let measured = record_measurements(permit, vec![1.0, 2.0]).unwrap();
/// # let record = record_search_disposition(measured, SearchDisposition::Survivor, b"s".to_vec(), "selected").unwrap();
/// assert_eq!(optimizer_outcome(&record), TrialOutcome::Complete(vec![1.0, 2.0]));
/// ```
#[must_use]
pub fn optimizer_outcome(record: &SearchRecord) -> TrialOutcome {
    TrialOutcome::Complete(record.measured.objectives.clone())
}

/// Map an incorrect candidate to a failed optimizer trial without objectives.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::{TrialId, TrialOutcome};
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Failed(failed) = record_verification(candidate, "oracle", b"bad".to_vec(), false, "mismatch").unwrap() else { unreachable!() };
/// assert_eq!(failed_verification_outcome(&failed), TrialOutcome::Failed);
/// ```
///
/// ```
/// # use scirust_opt_core::{TrialId, TrialOutcome};
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(9), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Failed(failed) = record_verification(candidate, "oracle", b"bad".to_vec(), false, "mismatch").unwrap() else { unreachable!() };
/// assert_eq!(failed.candidate.trial, TrialId::new(9));
/// assert_eq!(failed_verification_outcome(&failed), TrialOutcome::Failed);
/// ```
#[must_use]
pub const fn failed_verification_outcome(_failed: &FailedVerification) -> TrialOutcome {
    TrialOutcome::Failed
}

/// Commit a search record only when its disposition is Survivor.
///
/// Rejected records leave the plan untouched. Survivor commits delegate to
/// Tensor IR's stale-plan checked [`PreparedReplan::commit`].
///
/// # Errors
///
/// A survivor fails if the mutable plan changed since candidate preparation or
/// the graph no longer matches its canonical anchor.
///
/// # Examples
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let mut plan = RepresentationPlan::dense(&graph).unwrap();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// # let measured = record_measurements(permit, vec![1.0]).unwrap();
/// # let record = record_search_disposition(measured, SearchDisposition::Survivor, b"s".to_vec(), "selected").unwrap();
/// assert_eq!(
///     commit_search_record(record, &mut plan, &graph).unwrap(),
///     CommitResult::Committed
/// );
/// ```
///
/// ```
/// # use scirust_opt_core::TrialId;
/// # use scirust_tensor_ir::{DType, Graph, RepresentationPlan, Shape, TensorType};
/// # use scirust_tensor_search_bridge::*;
/// # let mut graph = Graph::new();
/// # let x = graph.add_input("x", TensorType::new(DType::F32, Shape::new([1usize]))).unwrap();
/// # graph.set_outputs(vec![x]).unwrap();
/// # let mut plan = RepresentationPlan::dense(&graph).unwrap();
/// # let before = plan.clone();
/// # let candidate = prepare_candidate(TrialId::new(0), &graph, &plan, vec![]).unwrap();
/// # let VerificationOutcome::Passed(verified) = record_verification(candidate, "oracle", b"v".to_vec(), true, "verified").unwrap() else { unreachable!() };
/// # let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
/// # let measured = record_measurements(permit, vec![1.0]).unwrap();
/// # let record = record_search_disposition(measured, SearchDisposition::Rejected, b"s".to_vec(), "not-selected").unwrap();
/// assert_eq!(
///     commit_search_record(record, &mut plan, &graph).unwrap(),
///     CommitResult::NotCommitted
/// );
/// assert_eq!(plan, before);
/// ```
pub fn commit_search_record(
    record: SearchRecord,
    plan: &mut RepresentationPlan,
    graph: &Graph,
) -> Result<CommitResult, SearchBridgeError> {
    match record.disposition {
        SearchDisposition::Rejected => Ok(CommitResult::NotCommitted),
        SearchDisposition::Survivor => {
            record
                .measured
                .permit
                .verified
                .candidate
                .prepared
                .commit(plan, graph)?;
            Ok(CommitResult::Committed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_ir::{DType, RepresentationId, Shape, TensorType};

    fn fixture() -> (Graph, scirust_tensor_ir::NodeId, RepresentationPlan) {
        let mut graph = Graph::new();
        let node = graph
            .add_input("x", TensorType::new(DType::F32, Shape::new([2usize])))
            .unwrap();
        graph.set_outputs(vec![node]).unwrap();
        let plan = RepresentationPlan::dense(&graph).unwrap();
        (graph, node, plan)
    }

    #[test]
    fn failed_verification_has_no_measurement_permit_path() {
        let (graph, _node, plan) = fixture();
        let candidate = prepare_candidate(TrialId::new(1), &graph, &plan, vec![]).unwrap();
        let outcome = record_verification(
            candidate,
            "reference-oracle-v1",
            b"verification:1".to_vec(),
            false,
            "mismatch",
        )
        .unwrap();
        let VerificationOutcome::Failed(failed) = outcome else {
            panic!("must retain negative verification");
        };
        assert_eq!(failed_verification_outcome(&failed), TrialOutcome::Failed);
    }

    #[test]
    fn verified_measurement_maps_to_optimizer_complete() {
        let (graph, _node, plan) = fixture();
        let candidate = prepare_candidate(TrialId::new(2), &graph, &plan, vec![]).unwrap();
        let VerificationOutcome::Passed(verified) = record_verification(
            candidate,
            "reference-oracle-v1",
            b"verification:2".to_vec(),
            true,
            "bit-exact",
        )
        .unwrap()
        else {
            panic!("candidate should pass");
        };
        let permit = permit_measurement(verified, b"env:cpu".to_vec()).unwrap();
        let measured = record_measurements(permit, vec![10.0, 20.0]).unwrap();
        let record = record_search_disposition(
            measured,
            SearchDisposition::Rejected,
            b"search:2".to_vec(),
            "dominated",
        )
        .unwrap();
        assert_eq!(
            optimizer_outcome(&record),
            TrialOutcome::Complete(vec![10.0, 20.0])
        );
    }

    #[test]
    fn survivor_commits_projected_representation_only_after_measurement() {
        let (graph, node, mut plan) = fixture();
        let dense_f32 = plan.assignment(node).unwrap();
        let dense_u8 = plan.declare_dense(DType::U8).unwrap();
        let quantized = plan
            .declare_quantized_per_tensor(
                TensorType::new(DType::U8, Shape::new([2usize])),
                dense_u8,
                TensorType::new(DType::F32, Shape::scalar()),
                dense_f32,
            )
            .unwrap();

        let candidate = prepare_candidate(
            TrialId::new(3),
            &graph,
            &plan,
            vec![Rebinding {
                node,
                representation: quantized,
            }],
        )
        .unwrap();
        assert_ne!(candidate.base_plan_identity, candidate.candidate_plan_identity);

        let VerificationOutcome::Passed(verified) = record_verification(
            candidate,
            "reference-oracle-v1",
            b"verification:3".to_vec(),
            true,
            "equivalent",
        )
        .unwrap()
        else {
            panic!("candidate should pass");
        };
        let measured =
            record_measurements(permit_measurement(verified, b"env".to_vec()).unwrap(), vec![1.0])
                .unwrap();
        let record = record_search_disposition(
            measured,
            SearchDisposition::Survivor,
            b"search:3".to_vec(),
            "selected",
        )
        .unwrap();

        assert_eq!(
            commit_search_record(record, &mut plan, &graph).unwrap(),
            CommitResult::Committed
        );
        assert_eq!(plan.assignment(node), Some(quantized));
    }

    #[test]
    fn stale_survivor_commit_fails_closed() {
        let (graph, node, mut plan) = fixture();
        let candidate = prepare_candidate(TrialId::new(4), &graph, &plan, vec![]).unwrap();
        let VerificationOutcome::Passed(verified) = record_verification(
            candidate,
            "oracle",
            b"verification:4".to_vec(),
            true,
            "ok",
        )
        .unwrap()
        else {
            panic!("candidate should pass");
        };
        let measured =
            record_measurements(permit_measurement(verified, b"env".to_vec()).unwrap(), vec![1.0])
                .unwrap();
        let record = record_search_disposition(
            measured,
            SearchDisposition::Survivor,
            b"search:4".to_vec(),
            "selected",
        )
        .unwrap();

        let extra = plan.declare_dense(DType::F16).unwrap();
        assert_eq!(extra, RepresentationId::new(1));
        assert!(commit_search_record(record, &mut plan, &graph).is_err());
        assert_eq!(plan.assignment(node), Some(RepresentationId::new(0)));
    }

    #[test]
    fn nonfinite_objectives_never_reach_optimizer_outcome() {
        let (graph, _node, plan) = fixture();
        let candidate = prepare_candidate(TrialId::new(5), &graph, &plan, vec![]).unwrap();
        let VerificationOutcome::Passed(verified) =
            record_verification(candidate, "oracle", b"v".to_vec(), true, "ok").unwrap()
        else {
            panic!("candidate should pass");
        };
        let permit = permit_measurement(verified, b"env".to_vec()).unwrap();
        assert_eq!(
            record_measurements(permit, vec![f64::INFINITY]),
            Err(SearchBridgeError::NonFiniteObjective { index: 0 })
        );
    }
}
