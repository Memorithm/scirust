//! Policy-free algebra graduation gate for prospective Tensor IR semantics.
//!
//! This module does **not** add Boolean, F2, tropical/max-plus, or Zhegalkin
//! operators to the canonical [`crate::Operation`] enum. It provides the
//! machine-enforced evidence gate that must be satisfied before a future,
//! separately reviewed canonical semantic extension can be considered.
//!
//! The gate is intentionally backend-neutral. Benchmark policy, Forge search
//! state, TDI verdicts, FLAT runtime policy, and hardware measurements remain
//! outside canonical Tensor IR identity.

use alloc::{string::String, vec::Vec};
use core::fmt;

/// Versioned identity of the algebra-graduation evidence contract.
pub const ALGEBRA_GRADUATION_CONTRACT_V1: &str =
    "scirust.tensor-ir.algebra-graduation/v1";

/// Prospective algebra family requesting canonical Tensor IR review.
///
/// These values identify research candidates only. Their presence here does
/// not make them canonical tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlgebraCandidate {
    /// Boolean algebra over explicit Boolean values.
    Boolean,
    /// Arithmetic over the field F2.
    F2,
    /// Max-plus/tropical arithmetic.
    TropicalMaxPlus,
    /// Zhegalkin / algebraic-normal-form Boolean semantics.
    Zhegalkin,
}

/// One mandatory qualification requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraGraduationRequirement {
    /// Operator meaning and edge cases are frozen under a versioned contract.
    FrozenOperatorSemantics,
    /// Shape and dtype rules are frozen and executable.
    ShapeAndTypeRules,
    /// A deterministic reference interpreter exists.
    DeterministicReferenceInterpreter,
    /// Negative controls exercise invalid and non-equivalent cases.
    NegativeControls,
    /// Differential/oracle evidence is retained when applicable.
    DifferentialOrOracleEvidence,
    /// Candidate semantics remain separate from FLAT stable-softmax semantics.
    StableSoftmaxSeparation,
}

/// State of one evidence claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraEvidenceState {
    /// Evidence is absent or insufficient. This never counts as a pass.
    Unknown,
    /// The requirement passed under the retained evidence identity.
    Passed,
    /// The requirement failed. Negative evidence is retained.
    Failed,
}

/// Evidence for one required graduation property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraEvidenceClaim {
    /// Qualification state.
    pub state: AlgebraEvidenceState,
    /// Non-empty versioned evidence-domain identifier.
    pub evidence_domain: String,
    /// Non-empty opaque identity of the retained evidence.
    pub evidence_identity: Vec<u8>,
    /// Non-empty machine-readable reason/status code.
    pub reason_code: String,
}

/// Differential/oracle evidence may be explicitly inapplicable, but never
/// silently omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DifferentialEvidence {
    /// Differential/oracle comparison is applicable and carries a normal claim.
    Applicable(AlgebraEvidenceClaim),
    /// A versioned external contract determined that comparison is not meaningful.
    NotApplicable {
        /// Non-empty versioned evidence-domain identifier.
        evidence_domain: String,
        /// Non-empty opaque identity supporting inapplicability.
        evidence_identity: Vec<u8>,
        /// Non-empty reason code.
        reason_code: String,
    },
}

/// Complete evidence record for one prospective algebra semantics version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraGraduationRecord {
    /// Contract version. Must equal [`ALGEBRA_GRADUATION_CONTRACT_V1`].
    pub contract: String,
    /// Research algebra family.
    pub candidate: AlgebraCandidate,
    /// Non-empty frozen semantic version/identity.
    pub semantics_version: String,
    /// Frozen operator semantics evidence.
    pub operator_semantics: AlgebraEvidenceClaim,
    /// Shape and dtype rule evidence.
    pub shape_type_rules: AlgebraEvidenceClaim,
    /// Deterministic reference interpreter evidence.
    pub reference_interpreter: AlgebraEvidenceClaim,
    /// Negative-control evidence.
    pub negative_controls: AlgebraEvidenceClaim,
    /// Differential/oracle evidence or explicit justified inapplicability.
    pub differential_oracle: DifferentialEvidence,
    /// Evidence that the candidate is distinct from FLAT stable-softmax semantics.
    pub stable_softmax_separation: AlgebraEvidenceClaim,
}

/// Opaque proof that every IR-E5 requirement passed.
///
/// Creating this value directly is impossible outside this module. A future
/// canonical-operation extension can require this token rather than trusting
/// prose or a collection of booleans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraduatedAlgebra {
    /// Graduated research algebra family.
    pub candidate: AlgebraCandidate,
    /// Frozen semantic version covered by the evidence.
    pub semantics_version: String,
    _private: (),
}

/// Result of evaluating one complete graduation record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraGraduationReport {
    /// Requirements that are not proven passed.
    pub blocking_requirements: Vec<AlgebraGraduationRequirement>,
    /// Present only when every requirement passes.
    pub token: Option<GraduatedAlgebra>,
}

/// Malformed evidence is an error rather than a failed scientific result.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlgebraGraduationError {
    /// The record does not use the current graduation contract identity.
    WrongContract,
    /// The frozen semantic version is blank.
    EmptySemanticsVersion,
    /// A required evidence domain is blank.
    EmptyEvidenceDomain {
        /// Requirement whose evidence was malformed.
        requirement: AlgebraGraduationRequirement,
    },
    /// A required evidence identity is empty.
    EmptyEvidenceIdentity {
        /// Requirement whose evidence was malformed.
        requirement: AlgebraGraduationRequirement,
    },
    /// A required reason/status code is blank.
    EmptyReasonCode {
        /// Requirement whose evidence was malformed.
        requirement: AlgebraGraduationRequirement,
    },
}

impl fmt::Display for AlgebraGraduationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongContract => formatter.write_str("wrong algebra-graduation contract identity"),
            Self::EmptySemanticsVersion => {
                formatter.write_str("algebra semantic version must not be blank")
            }
            Self::EmptyEvidenceDomain { requirement } => {
                write!(formatter, "{requirement:?} has an empty evidence domain")
            }
            Self::EmptyEvidenceIdentity { requirement } => {
                write!(formatter, "{requirement:?} has an empty evidence identity")
            }
            Self::EmptyReasonCode { requirement } => {
                write!(formatter, "{requirement:?} has an empty reason code")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AlgebraGraduationError {}

/// Evaluate the IR-E5 machine graduation gate.
///
/// This function validates provenance fields first, then blocks every requirement
/// whose evidence is [`AlgebraEvidenceState::Unknown`] or
/// [`AlgebraEvidenceState::Failed`]. Differential evidence can be explicitly
/// [`DifferentialEvidence::NotApplicable`] only when that inapplicability itself
/// has non-empty retained evidence.
///
/// A successful result creates a [`GraduatedAlgebra`] token but does not modify
/// [`crate::Operation`], representation identity, execution intent, or runtime
/// policy. Canonical admission remains a separate code-review change.
///
/// # Errors
///
/// Returns [`AlgebraGraduationError`] for malformed or unversioned evidence.
///
/// # Examples
///
/// ```
/// use scirust_tensor_ir::{
///     ALGEBRA_GRADUATION_CONTRACT_V1, AlgebraCandidate, AlgebraEvidenceClaim,
///     AlgebraEvidenceState, AlgebraGraduationRecord, DifferentialEvidence,
///     evaluate_algebra_graduation,
/// };
///
/// fn pass(domain: &str, id: &[u8]) -> AlgebraEvidenceClaim {
///     AlgebraEvidenceClaim {
///         state: AlgebraEvidenceState::Passed,
///         evidence_domain: domain.into(),
///         evidence_identity: id.to_vec(),
///         reason_code: "passed".into(),
///     }
/// }
///
/// let record = AlgebraGraduationRecord {
///     contract: ALGEBRA_GRADUATION_CONTRACT_V1.into(),
///     candidate: AlgebraCandidate::F2,
///     semantics_version: "f2.v1".into(),
///     operator_semantics: pass("semantics.v1", b"s"),
///     shape_type_rules: pass("shape.v1", b"t"),
///     reference_interpreter: pass("reference.v1", b"r"),
///     negative_controls: pass("negative.v1", b"n"),
///     differential_oracle: DifferentialEvidence::Applicable(pass("oracle.v1", b"o")),
///     stable_softmax_separation: pass("flat-separation.v1", b"f"),
/// };
/// let report = evaluate_algebra_graduation(&record).unwrap();
/// assert!(report.blocking_requirements.is_empty());
/// assert_eq!(report.token.unwrap().candidate, AlgebraCandidate::F2);
/// ```
///
/// Unknown evidence remains blocking rather than being treated as false or zero:
///
/// ```
/// use scirust_tensor_ir::{
///     ALGEBRA_GRADUATION_CONTRACT_V1, AlgebraCandidate, AlgebraEvidenceClaim,
///     AlgebraEvidenceState, AlgebraGraduationRecord, AlgebraGraduationRequirement,
///     DifferentialEvidence, evaluate_algebra_graduation,
/// };
///
/// let claim = |state, id: &[u8]| AlgebraEvidenceClaim {
///     state,
///     evidence_domain: "test.v1".into(),
///     evidence_identity: id.to_vec(),
///     reason_code: "retained".into(),
/// };
/// let record = AlgebraGraduationRecord {
///     contract: ALGEBRA_GRADUATION_CONTRACT_V1.into(),
///     candidate: AlgebraCandidate::Boolean,
///     semantics_version: "bool.v1".into(),
///     operator_semantics: claim(AlgebraEvidenceState::Passed, b"s"),
///     shape_type_rules: claim(AlgebraEvidenceState::Passed, b"t"),
///     reference_interpreter: claim(AlgebraEvidenceState::Unknown, b"r"),
///     negative_controls: claim(AlgebraEvidenceState::Passed, b"n"),
///     differential_oracle: DifferentialEvidence::Applicable(
///         claim(AlgebraEvidenceState::Passed, b"o")
///     ),
///     stable_softmax_separation: claim(AlgebraEvidenceState::Passed, b"f"),
/// };
/// let report = evaluate_algebra_graduation(&record).unwrap();
/// assert_eq!(
///     report.blocking_requirements,
///     vec![AlgebraGraduationRequirement::DeterministicReferenceInterpreter]
/// );
/// assert!(report.token.is_none());
/// ```
pub fn evaluate_algebra_graduation(
    record: &AlgebraGraduationRecord,
) -> Result<AlgebraGraduationReport, AlgebraGraduationError> {
    if record.contract != ALGEBRA_GRADUATION_CONTRACT_V1 {
        return Err(AlgebraGraduationError::WrongContract);
    }
    if record.semantics_version.trim().is_empty() {
        return Err(AlgebraGraduationError::EmptySemanticsVersion);
    }

    let checks = [
        (AlgebraGraduationRequirement::FrozenOperatorSemantics, &record.operator_semantics),
        (AlgebraGraduationRequirement::ShapeAndTypeRules, &record.shape_type_rules),
        (
            AlgebraGraduationRequirement::DeterministicReferenceInterpreter,
            &record.reference_interpreter,
        ),
        (AlgebraGraduationRequirement::NegativeControls, &record.negative_controls),
        (
            AlgebraGraduationRequirement::StableSoftmaxSeparation,
            &record.stable_softmax_separation,
        ),
    ];

    let mut blocking_requirements = Vec::new();
    for (requirement, claim) in checks {
        validate_claim(requirement, claim)?;
        if claim.state != AlgebraEvidenceState::Passed {
            blocking_requirements.push(requirement);
        }
    }

    match &record.differential_oracle {
        DifferentialEvidence::Applicable(claim) => {
            let requirement = AlgebraGraduationRequirement::DifferentialOrOracleEvidence;
            validate_claim(requirement, claim)?;
            if claim.state != AlgebraEvidenceState::Passed {
                blocking_requirements.push(requirement);
            }
        }
        DifferentialEvidence::NotApplicable {
            evidence_domain,
            evidence_identity,
            reason_code,
        } => validate_metadata(
            AlgebraGraduationRequirement::DifferentialOrOracleEvidence,
            evidence_domain,
            evidence_identity,
            reason_code,
        )?,
    }

    let token = if blocking_requirements.is_empty() {
        Some(GraduatedAlgebra {
            candidate: record.candidate,
            semantics_version: record.semantics_version.clone(),
            _private: (),
        })
    } else {
        None
    };

    Ok(AlgebraGraduationReport {
        blocking_requirements,
        token,
    })
}

fn validate_claim(
    requirement: AlgebraGraduationRequirement,
    claim: &AlgebraEvidenceClaim,
) -> Result<(), AlgebraGraduationError> {
    validate_metadata(
        requirement,
        &claim.evidence_domain,
        &claim.evidence_identity,
        &claim.reason_code,
    )
}

fn validate_metadata(
    requirement: AlgebraGraduationRequirement,
    evidence_domain: &str,
    evidence_identity: &[u8],
    reason_code: &str,
) -> Result<(), AlgebraGraduationError> {
    if evidence_domain.trim().is_empty() {
        return Err(AlgebraGraduationError::EmptyEvidenceDomain { requirement });
    }
    if evidence_identity.is_empty() {
        return Err(AlgebraGraduationError::EmptyEvidenceIdentity { requirement });
    }
    if reason_code.trim().is_empty() {
        return Err(AlgebraGraduationError::EmptyReasonCode { requirement });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(state: AlgebraEvidenceState, tag: u8) -> AlgebraEvidenceClaim {
        AlgebraEvidenceClaim {
            state,
            evidence_domain: "test.evidence.v1".into(),
            evidence_identity: alloc::vec![tag],
            reason_code: "retained".into(),
        }
    }

    fn complete_record() -> AlgebraGraduationRecord {
        AlgebraGraduationRecord {
            contract: ALGEBRA_GRADUATION_CONTRACT_V1.into(),
            candidate: AlgebraCandidate::Zhegalkin,
            semantics_version: "zhegalkin.v1".into(),
            operator_semantics: claim(AlgebraEvidenceState::Passed, 1),
            shape_type_rules: claim(AlgebraEvidenceState::Passed, 2),
            reference_interpreter: claim(AlgebraEvidenceState::Passed, 3),
            negative_controls: claim(AlgebraEvidenceState::Passed, 4),
            differential_oracle: DifferentialEvidence::Applicable(claim(
                AlgebraEvidenceState::Passed,
                5,
            )),
            stable_softmax_separation: claim(AlgebraEvidenceState::Passed, 6),
        }
    }

    #[test]
    fn complete_evidence_mints_private_graduation_token() {
        let report = evaluate_algebra_graduation(&complete_record()).unwrap();
        assert!(report.blocking_requirements.is_empty());
        let token = report.token.unwrap();
        assert_eq!(token.candidate, AlgebraCandidate::Zhegalkin);
        assert_eq!(token.semantics_version, "zhegalkin.v1");
    }

    #[test]
    fn unknown_and_failed_requirements_are_retained_as_blockers() {
        let mut record = complete_record();
        record.operator_semantics.state = AlgebraEvidenceState::Unknown;
        record.negative_controls.state = AlgebraEvidenceState::Failed;
        let report = evaluate_algebra_graduation(&record).unwrap();
        assert_eq!(
            report.blocking_requirements,
            alloc::vec![
                AlgebraGraduationRequirement::FrozenOperatorSemantics,
                AlgebraGraduationRequirement::NegativeControls,
            ]
        );
        assert!(report.token.is_none());
    }

    #[test]
    fn stable_softmax_separation_is_mandatory() {
        let mut record = complete_record();
        record.stable_softmax_separation.state = AlgebraEvidenceState::Unknown;
        let report = evaluate_algebra_graduation(&record).unwrap();
        assert_eq!(
            report.blocking_requirements,
            alloc::vec![AlgebraGraduationRequirement::StableSoftmaxSeparation]
        );
        assert!(report.token.is_none());
    }

    #[test]
    fn justified_differential_inapplicability_is_explicitly_retained() {
        let mut record = complete_record();
        record.differential_oracle = DifferentialEvidence::NotApplicable {
            evidence_domain: "oracle-applicability.v1".into(),
            evidence_identity: alloc::vec![9],
            reason_code: "no-common-codomain".into(),
        };
        let report = evaluate_algebra_graduation(&record).unwrap();
        assert!(report.blocking_requirements.is_empty());
        assert!(report.token.is_some());
    }

    #[test]
    fn empty_inapplicability_evidence_fails_closed() {
        let mut record = complete_record();
        record.differential_oracle = DifferentialEvidence::NotApplicable {
            evidence_domain: "oracle-applicability.v1".into(),
            evidence_identity: Vec::new(),
            reason_code: "no-common-codomain".into(),
        };
        assert_eq!(
            evaluate_algebra_graduation(&record),
            Err(AlgebraGraduationError::EmptyEvidenceIdentity {
                requirement: AlgebraGraduationRequirement::DifferentialOrOracleEvidence
            })
        );
    }

    #[test]
    fn malformed_metadata_is_not_relabelled_as_negative_evidence() {
        let mut record = complete_record();
        record.reference_interpreter.evidence_domain = " ".into();
        assert_eq!(
            evaluate_algebra_graduation(&record),
            Err(AlgebraGraduationError::EmptyEvidenceDomain {
                requirement: AlgebraGraduationRequirement::DeterministicReferenceInterpreter
            })
        );
    }

    #[test]
    fn canonical_operation_has_no_research_algebra_variant() {
        let operation = crate::Operation::Add;
        assert_eq!(operation.expected_arity(), 2);
    }
}
