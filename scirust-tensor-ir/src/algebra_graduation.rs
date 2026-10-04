//! Evidence gate for graduating research algebras into canonical Tensor IR.
//!
//! This module deliberately does not add algebraic operations to crate::Operation.
//! It only defines the minimum evidence envelope that must be complete before a
//! Boolean/F2/tropical/Zhegalkin proposal is eligible for canonical-operation review.

use core::fmt;

/// Versioned schema for algebra-graduation evidence.
pub const ALGEBRA_GRADUATION_SCHEMA_V1: &str = "scirust.tensor-ir.algebra-graduation/v1";

/// Research algebra families currently covered by the graduation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraFamily {
    /// Boolean algebra with explicitly frozen truth semantics.
    Boolean,
    /// Arithmetic over the field with two elements.
    F2,
    /// Tropical max-plus algebra.
    TropicalMaxPlus,
    /// Zhegalkin / algebraic-normal-form semantics.
    Zhegalkin,
}

/// Explicit status of differential/oracle evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentialOracleEvidence<'a> {
    /// A differential or independent oracle was required and verified.
    Verified {
        /// Stable evidence identifier.
        evidence_id: &'a str,
    },
    /// A differential oracle is not applicable, with an explicit reviewed reason.
    NotApplicable {
        /// Stable rationale identifier; an empty value is rejected.
        rationale_id: &'a str,
    },
}

/// Complete evidence required before an algebra may be reviewed for canonical IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlgebraGraduationEvidence<'a> {
    /// Must equal ALGEBRA_GRADUATION_SCHEMA_V1.
    pub schema: &'a str,
    /// Candidate algebra family.
    pub family: AlgebraFamily,
    /// Exact 40-hex source revision containing the frozen candidate semantics.
    pub source_revision: &'a str,
    /// Stable identifier for the frozen operator-semantics specification.
    pub operator_semantics_id: &'a str,
    /// Stable identifier for shape and type rules.
    pub shape_type_rules_id: &'a str,
    /// Stable identifier for the deterministic reference interpreter.
    pub reference_interpreter_id: &'a str,
    /// Stable identifier for negative/control evidence.
    pub negative_controls_id: &'a str,
    /// Differential/oracle evidence, or an explicit reviewed non-applicability reason.
    pub differential_oracle: DifferentialOracleEvidence<'a>,
    /// Evidence that the proposal remains separate from FLAT's stable softmax path.
    pub flat_softmax_separation_id: &'a str,
}

/// A successful gate result.
///
/// This is intentionally weaker than acceptance into crate::Operation: a
/// successful gate only permits a separate canonical-IR review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraGraduationVerdict {
    /// All mandatory evidence fields are present and syntactically valid.
    EligibleForCanonicalOperationReview,
}

impl AlgebraGraduationEvidence<'_> {
    /// Validate the machine-enforced algebra graduation criteria.
    ///
    /// The gate requires frozen semantics, shape/type rules, a deterministic
    /// reference interpreter, negative controls, differential/oracle disposition,
    /// and explicit separation from FLAT's stable softmax path. It does not modify
    /// a graph, representation plan, or canonical operation identity.
    ///
    /// # Errors
    ///
    /// Returns AlgebraGraduationError when the schema, source revision or any
    /// mandatory evidence identity is missing or malformed.
    ///
    /// # Examples
    ///
    /// ~~~rust
    /// use scirust_tensor_ir::{
    ///     ALGEBRA_GRADUATION_SCHEMA_V1, AlgebraFamily, AlgebraGraduationEvidence,
    ///     AlgebraGraduationVerdict, DifferentialOracleEvidence,
    /// };
    ///
    /// let evidence = AlgebraGraduationEvidence {
    ///     schema: ALGEBRA_GRADUATION_SCHEMA_V1,
    ///     family: AlgebraFamily::F2,
    ///     source_revision: "0123456789abcdef0123456789abcdef01234567",
    ///     operator_semantics_id: "f2-ops-v1",
    ///     shape_type_rules_id: "f2-shapes-v1",
    ///     reference_interpreter_id: "f2-reference-v1",
    ///     negative_controls_id: "f2-negative-controls-v1",
    ///     differential_oracle: DifferentialOracleEvidence::Verified {
    ///         evidence_id: "f2-oracle-v1",
    ///     },
    ///     flat_softmax_separation_id: "flat-softmax-separation-v1",
    /// };
    ///
    /// assert_eq!(
    ///     evidence.evaluate()?,
    ///     AlgebraGraduationVerdict::EligibleForCanonicalOperationReview
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ~~~
    pub fn evaluate(&self) -> Result<AlgebraGraduationVerdict, AlgebraGraduationError> {
        if self.schema != ALGEBRA_GRADUATION_SCHEMA_V1
        {
            return Err(AlgebraGraduationError::WrongSchema);
        }
        if !is_git_sha(self.source_revision)
        {
            return Err(AlgebraGraduationError::InvalidSourceRevision);
        }
        require_id(self.operator_semantics_id, "operator_semantics_id")?;
        require_id(self.shape_type_rules_id, "shape_type_rules_id")?;
        require_id(self.reference_interpreter_id, "reference_interpreter_id")?;
        require_id(self.negative_controls_id, "negative_controls_id")?;
        require_id(
            self.flat_softmax_separation_id,
            "flat_softmax_separation_id",
        )?;

        match self.differential_oracle
        {
            DifferentialOracleEvidence::Verified { evidence_id } =>
            {
                require_id(evidence_id, "differential_oracle.evidence_id")?;
            },
            DifferentialOracleEvidence::NotApplicable { rationale_id } =>
            {
                require_id(rationale_id, "differential_oracle.rationale_id")?;
            },
        }

        Ok(AlgebraGraduationVerdict::EligibleForCanonicalOperationReview)
    }
}

/// Validation errors from the algebra graduation gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraGraduationError {
    /// Evidence was encoded under an unsupported schema.
    WrongSchema,
    /// The source revision is not exactly 40 lowercase hexadecimal characters.
    InvalidSourceRevision,
    /// A required evidence identity was empty.
    MissingEvidence(&'static str),
}

impl fmt::Display for AlgebraGraduationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::WrongSchema => formatter.write_str("unsupported algebra-graduation schema"),
            Self::InvalidSourceRevision => formatter
                .write_str("algebra source revision must be 40 lowercase hexadecimal characters"),
            Self::MissingEvidence(field) =>
            {
                write!(formatter, "missing algebra-graduation evidence: {field}")
            },
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AlgebraGraduationError {}

fn require_id(value: &str, field: &'static str) -> Result<(), AlgebraGraduationError> {
    if value.trim().is_empty()
    {
        return Err(AlgebraGraduationError::MissingEvidence(field));
    }
    Ok(())
}

fn is_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete() -> AlgebraGraduationEvidence<'static> {
        AlgebraGraduationEvidence {
            schema: ALGEBRA_GRADUATION_SCHEMA_V1,
            family: AlgebraFamily::Zhegalkin,
            source_revision: "0123456789abcdef0123456789abcdef01234567",
            operator_semantics_id: "zhegalkin-semantics-v1",
            shape_type_rules_id: "zhegalkin-shapes-v1",
            reference_interpreter_id: "zhegalkin-reference-v1",
            negative_controls_id: "zhegalkin-negative-controls-v1",
            differential_oracle: DifferentialOracleEvidence::Verified {
                evidence_id: "zhegalkin-f2-differential-v1",
            },
            flat_softmax_separation_id: "flat-softmax-separation-v1",
        }
    }

    #[test]
    fn complete_evidence_is_only_eligible_for_review() {
        assert_eq!(
            complete().evaluate(),
            Ok(AlgebraGraduationVerdict::EligibleForCanonicalOperationReview)
        );
    }

    #[test]
    fn missing_required_evidence_fails_closed() {
        let mut evidence = complete();
        evidence.reference_interpreter_id = "";
        assert_eq!(
            evidence.evaluate(),
            Err(AlgebraGraduationError::MissingEvidence(
                "reference_interpreter_id"
            ))
        );
    }

    #[test]
    fn explicit_oracle_non_applicability_requires_a_reason() {
        let mut evidence = complete();
        evidence.differential_oracle =
            DifferentialOracleEvidence::NotApplicable { rationale_id: "" };
        assert_eq!(
            evidence.evaluate(),
            Err(AlgebraGraduationError::MissingEvidence(
                "differential_oracle.rationale_id"
            ))
        );

        evidence.differential_oracle = DifferentialOracleEvidence::NotApplicable {
            rationale_id: "reviewed-not-applicable-v1",
        };
        assert!(evidence.evaluate().is_ok());
    }

    #[test]
    fn malformed_source_revision_is_rejected() {
        let mut evidence = complete();
        evidence.source_revision = "NOT-A-GIT-SHA";
        assert_eq!(
            evidence.evaluate(),
            Err(AlgebraGraduationError::InvalidSourceRevision)
        );
    }
}
