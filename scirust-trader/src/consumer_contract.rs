//! Consumer-neutral contracts for trading research requests and results.
//!
//! This module is deliberately independent from MCP, Replikans, a particular
//! LLM, or a specific runtime. It defines the auditable envelope shared by any
//! caller of SciRust trading-research capabilities: agents, backtesters, paper
//! traders, CLIs, services, libraries, and human-facing tools.
//!
//! The contract carries explicit participant identity, request/result IDs,
//! provenance, declared assumptions, evidence references, and deterministic
//! fingerprints. It does not execute orders and does not select a strategy.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Current schema version for both request and result envelopes.
pub const TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Stable operation identifiers for research capabilities already exposed by
/// `scirust-trader`. Consumers may use other namespaced operation IDs without
/// requiring a schema change.
pub mod operations {
    pub const PURGED_CV: &str = "scirust.trader.research.purged_cv.v1";
    pub const DEFLATED_SHARPE: &str = "scirust.trader.research.deflated_sharpe.v1";
    pub const CSCV_PBO: &str = "scirust.trader.research.cscv_pbo.v1";
    pub const COST_STRESS: &str = "scirust.trader.research.cost_stress.v1";
    pub const RL_PLAN: &str = "scirust.trader.research.rl_plan.v1";
    pub const COMPARE_CANDIDATES: &str = "scirust.trader.research.compare_candidates.v1";
    pub const MANIFEST_FINGERPRINT: &str = "scirust.trader.research.manifest_fingerprint.v1";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticipantKind {
    Agent,
    Backtester,
    PaperTrader,
    Cli,
    HumanTool,
    Service,
    Library,
    Other,
}

/// Identity of a request consumer or result producer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractParticipant {
    pub participant_id: String,
    pub kind: ParticipantKind,
    /// Concrete implementation/model/runtime identifier, declared by the caller.
    pub implementation: String,
}

impl ContractParticipant {
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        require_text("participant.participant_id", &self.participant_id)?;
        require_text("participant.implementation", &self.implementation)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    MarketData,
    DerivedDataset,
    Code,
    Configuration,
    Model,
    ExternalEvidence,
    Other,
}

/// One immutable provenance dependency used to construct a request or result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRef {
    pub provenance_id: String,
    pub kind: ProvenanceKind,
    pub uri: String,
    /// Caller-declared immutable content/version fingerprint.
    pub fingerprint: String,
    pub observed_at_ms: Option<i64>,
}

impl ProvenanceRef {
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        require_text("provenance.provenance_id", &self.provenance_id)?;
        require_text("provenance.uri", &self.uri)?;
        require_text("provenance.fingerprint", &self.fingerprint)?;
        Ok(())
    }
}

/// Evidence emitted by a completed research operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub evidence_id: String,
    pub uri: String,
    pub fingerprint: String,
    pub media_type: Option<String>,
}

impl EvidenceRef {
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        require_text("evidence.evidence_id", &self.evidence_id)?;
        require_text("evidence.uri", &self.uri)?;
        require_text("evidence.fingerprint", &self.fingerprint)?;
        if let Some(media_type) = &self.media_type
        {
            require_text("evidence.media_type", media_type)?;
        }
        Ok(())
    }
}

/// Transport-neutral request envelope. `payload` is operation-specific, while
/// the outer envelope stays stable across Rust API, MCP, CLI, or other adapters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradingResearchRequest {
    pub schema_version: u32,
    pub request_id: String,
    pub experiment_id: Option<String>,
    pub consumer: ContractParticipant,
    pub submitted_at_ms: i64,
    pub operation: String,
    pub payload: serde_json::Value,
    pub provenance: Vec<ProvenanceRef>,
    pub declared_assumptions: BTreeMap<String, serde_json::Value>,
}

impl TradingResearchRequest {
    pub const CURRENT_SCHEMA_VERSION: u32 = TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION;

    pub fn validate(&self) -> Result<(), ContractValidationError> {
        require_schema(self.schema_version)?;
        require_text("request_id", &self.request_id)?;
        if let Some(experiment_id) = &self.experiment_id
        {
            require_text("experiment_id", experiment_id)?;
        }
        self.consumer.validate()?;
        require_text("operation", &self.operation)?;
        if !self.payload.is_object()
        {
            return Err(ContractValidationError::new(
                "payload",
                "must be a JSON object",
            ));
        }
        validate_provenance(&self.provenance)?;
        if self
            .declared_assumptions
            .keys()
            .any(|key| key.trim().is_empty())
        {
            return Err(ContractValidationError::new(
                "declared_assumptions",
                "assumption names must be non-empty",
            ));
        }
        Ok(())
    }

    /// Deterministic SHA-256 fingerprint of the fully declared request.
    pub fn fingerprint(&self) -> Result<String, ContractValidationError> {
        self.validate()?;
        fingerprint("scirust.trading-research.request.v1", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchOutcome {
    Succeeded,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractIssueSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractIssue {
    pub severity: ContractIssueSeverity,
    pub code: String,
    pub message: String,
}

impl ContractIssue {
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        require_text("issue.code", &self.code)?;
        require_text("issue.message", &self.message)?;
        Ok(())
    }
}

/// Transport-neutral result envelope linked cryptographically to its request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradingResearchResult {
    pub schema_version: u32,
    pub result_id: String,
    pub request_id: String,
    pub request_fingerprint: String,
    pub experiment_id: Option<String>,
    pub producer: ContractParticipant,
    pub produced_at_ms: i64,
    pub operation: String,
    pub outcome: ResearchOutcome,
    pub output: Option<serde_json::Value>,
    pub evidence: Vec<EvidenceRef>,
    pub provenance: Vec<ProvenanceRef>,
    pub issues: Vec<ContractIssue>,
}

impl TradingResearchResult {
    pub const CURRENT_SCHEMA_VERSION: u32 = TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION;

    pub fn validate(&self) -> Result<(), ContractValidationError> {
        require_schema(self.schema_version)?;
        require_text("result_id", &self.result_id)?;
        require_text("request_id", &self.request_id)?;
        require_sha256("request_fingerprint", &self.request_fingerprint)?;
        if let Some(experiment_id) = &self.experiment_id
        {
            require_text("experiment_id", experiment_id)?;
        }
        self.producer.validate()?;
        require_text("operation", &self.operation)?;
        if let Some(output) = &self.output
        {
            if !output.is_object()
            {
                return Err(ContractValidationError::new(
                    "output",
                    "must be a JSON object when present",
                ));
            }
        }
        validate_evidence(&self.evidence)?;
        validate_provenance(&self.provenance)?;
        for issue in &self.issues
        {
            issue.validate()?;
        }

        let has_error = self
            .issues
            .iter()
            .any(|issue| issue.severity == ContractIssueSeverity::Error);
        match self.outcome
        {
            ResearchOutcome::Succeeded =>
            {
                if self.output.is_none()
                {
                    return Err(ContractValidationError::new(
                        "output",
                        "successful results require an output object",
                    ));
                }
                if has_error
                {
                    return Err(ContractValidationError::new(
                        "issues",
                        "successful results cannot contain error issues",
                    ));
                }
            },
            ResearchOutcome::Rejected | ResearchOutcome::Failed =>
            {
                if !has_error
                {
                    return Err(ContractValidationError::new(
                        "issues",
                        "rejected or failed results require an error issue",
                    ));
                }
            },
        }
        Ok(())
    }

    /// Deterministic SHA-256 fingerprint of the complete result envelope.
    pub fn fingerprint(&self) -> Result<String, ContractValidationError> {
        self.validate()?;
        fingerprint("scirust.trading-research.result.v1", self)
    }

    /// Verify that this result belongs to exactly the supplied request.
    pub fn matches_request(
        &self,
        request: &TradingResearchRequest,
    ) -> Result<(), ContractValidationError> {
        self.validate()?;
        request.validate()?;
        if self.request_id != request.request_id
        {
            return Err(ContractValidationError::new(
                "request_id",
                "does not match request",
            ));
        }
        if self.request_fingerprint != request.fingerprint()?
        {
            return Err(ContractValidationError::new(
                "request_fingerprint",
                "does not match request",
            ));
        }
        if self.experiment_id != request.experiment_id
        {
            return Err(ContractValidationError::new(
                "experiment_id",
                "does not match request",
            ));
        }
        if self.operation != request.operation
        {
            return Err(ContractValidationError::new(
                "operation",
                "does not match request",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContractValidationError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl ContractValidationError {
    const fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl std::fmt::Display for ContractValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid trading research contract {}: {}",
            self.field, self.reason
        )
    }
}

impl std::error::Error for ContractValidationError {}

fn require_schema(schema_version: u32) -> Result<(), ContractValidationError> {
    if schema_version != TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION
    {
        return Err(ContractValidationError::new(
            "schema_version",
            "unsupported schema version",
        ));
    }
    Ok(())
}

fn require_text(field: &'static str, value: &str) -> Result<(), ContractValidationError> {
    if value.trim().is_empty()
    {
        return Err(ContractValidationError::new(field, "must be non-empty"));
    }
    Ok(())
}

fn require_sha256(field: &'static str, value: &str) -> Result<(), ContractValidationError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ContractValidationError::new(
            field,
            "must be a 64-character hexadecimal SHA-256 fingerprint",
        ));
    }
    Ok(())
}

fn validate_provenance(provenance: &[ProvenanceRef]) -> Result<(), ContractValidationError> {
    let mut ids = BTreeSet::new();
    for item in provenance
    {
        item.validate()?;
        if !ids.insert(item.provenance_id.as_str())
        {
            return Err(ContractValidationError::new(
                "provenance.provenance_id",
                "must be unique within the envelope",
            ));
        }
    }
    Ok(())
}

fn validate_evidence(evidence: &[EvidenceRef]) -> Result<(), ContractValidationError> {
    let mut ids = BTreeSet::new();
    for item in evidence
    {
        item.validate()?;
        if !ids.insert(item.evidence_id.as_str())
        {
            return Err(ContractValidationError::new(
                "evidence.evidence_id",
                "must be unique within the result",
            ));
        }
    }
    Ok(())
}

fn fingerprint<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> Result<String, ContractValidationError> {
    let bytes = serde_json::to_vec(value).map_err(|_| {
        ContractValidationError::new("serialization", "could not serialize envelope")
    })?;
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn participant(id: &str, kind: ParticipantKind) -> ContractParticipant {
        ContractParticipant {
            participant_id: id.into(),
            kind,
            implementation: "test-implementation@1".into(),
        }
    }

    fn provenance(id: &str) -> ProvenanceRef {
        ProvenanceRef {
            provenance_id: id.into(),
            kind: ProvenanceKind::DerivedDataset,
            uri: format!("artifact://{id}"),
            fingerprint: format!("fingerprint-{id}"),
            observed_at_ms: Some(100),
        }
    }

    fn request() -> TradingResearchRequest {
        TradingResearchRequest {
            schema_version: TradingResearchRequest::CURRENT_SCHEMA_VERSION,
            request_id: "req-1".into(),
            experiment_id: Some("exp-1".into()),
            consumer: participant("backtester-1", ParticipantKind::Backtester),
            submitted_at_ms: 123,
            operation: operations::COST_STRESS.into(),
            payload: json!({"cost_grid_bps": [0.0, 5.0, 10.0]}),
            provenance: vec![provenance("dataset-1")],
            declared_assumptions: BTreeMap::from([("fees".into(), json!("explicit"))]),
        }
    }

    #[test]
    fn request_fingerprint_is_repeatable_and_sensitive() {
        let original = request();
        let first = original.fingerprint().unwrap();
        assert_eq!(first, original.fingerprint().unwrap());

        let mut changed = original;
        changed.consumer.kind = ParticipantKind::Agent;
        assert_ne!(first, changed.fingerprint().unwrap());
    }

    #[test]
    fn contract_is_not_tied_to_one_consumer_kind() {
        for kind in [
            ParticipantKind::Agent,
            ParticipantKind::Backtester,
            ParticipantKind::PaperTrader,
            ParticipantKind::Cli,
            ParticipantKind::HumanTool,
            ParticipantKind::Service,
            ParticipantKind::Library,
            ParticipantKind::Other,
        ]
        {
            let mut candidate = request();
            candidate.consumer = participant("consumer", kind);
            assert!(candidate.validate().is_ok());
        }
    }

    #[test]
    fn duplicate_provenance_is_rejected() {
        let mut candidate = request();
        candidate.provenance.push(provenance("dataset-1"));
        assert_eq!(
            candidate.validate().unwrap_err().field,
            "provenance.provenance_id"
        );
    }

    #[test]
    fn successful_result_is_bound_to_exact_request() {
        let request = request();
        let result = TradingResearchResult {
            schema_version: TradingResearchResult::CURRENT_SCHEMA_VERSION,
            result_id: "result-1".into(),
            request_id: request.request_id.clone(),
            request_fingerprint: request.fingerprint().unwrap(),
            experiment_id: request.experiment_id.clone(),
            producer: participant("scirust-trader", ParticipantKind::Library),
            produced_at_ms: 200,
            operation: request.operation.clone(),
            outcome: ResearchOutcome::Succeeded,
            output: Some(json!({"report": {"ok": true}})),
            evidence: vec![EvidenceRef {
                evidence_id: "report".into(),
                uri: "artifact://report.json".into(),
                fingerprint: "sha256:abc".into(),
                media_type: Some("application/json".into()),
            }],
            provenance: request.provenance.clone(),
            issues: vec![],
        };
        assert!(result.matches_request(&request).is_ok());

        let mut changed_request = request.clone();
        changed_request.payload = json!({"cost_grid_bps": [0.0, 50.0]});
        assert_eq!(
            result.matches_request(&changed_request).unwrap_err().field,
            "request_fingerprint"
        );
    }

    #[test]
    fn failed_result_requires_explicit_error_issue() {
        let request = request();
        let result = TradingResearchResult {
            schema_version: TradingResearchResult::CURRENT_SCHEMA_VERSION,
            result_id: "result-2".into(),
            request_id: request.request_id.clone(),
            request_fingerprint: request.fingerprint().unwrap(),
            experiment_id: request.experiment_id.clone(),
            producer: participant("scirust-trader", ParticipantKind::Library),
            produced_at_ms: 200,
            operation: request.operation.clone(),
            outcome: ResearchOutcome::Failed,
            output: None,
            evidence: vec![],
            provenance: vec![],
            issues: vec![],
        };
        assert_eq!(result.validate().unwrap_err().field, "issues");
    }

    #[test]
    fn request_payload_must_be_structured_object() {
        let mut candidate = request();
        candidate.payload = json!([1, 2, 3]);
        assert_eq!(candidate.validate().unwrap_err().field, "payload");
    }
}
