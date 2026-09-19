//! Transport-neutral mission contracts for agent-driven trading.
//!
//! A mission expresses objectives and hard constraints. Profit targets are
//! optimization targets, never guarantees. Financial execution remains subject
//! to an independent deterministic authorization/risk boundary.

use serde::{Deserialize, Serialize};

pub const TRADING_MISSION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionMode { Research, Backtest, Paper, Testnet, Shadow, Live }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionState { Proposed, Counterproposed, Approved, Active, Paused, Completed, Aborted }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfitObjective {
    pub target_net_profit: f64,
    pub horizon_ms: u64,
    pub preferred_net_profit_per_round_trip: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskBudget {
    pub max_drawdown_fraction: f64,
    pub max_daily_loss: f64,
    pub max_gross_exposure: f64,
    pub max_leverage: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPermissions {
    pub may_submit: bool,
    pub may_cancel: bool,
    pub may_replace: bool,
    pub may_close_positions: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionRequest {
    pub schema_version: u32,
    pub mission_id: String,
    pub mode: MissionMode,
    pub capital: f64,
    pub objective: ProfitObjective,
    pub risk: RiskBudget,
    pub execution: ExecutionPermissions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeasibilityAssessment {
    pub mission_id: String,
    pub feasible_as_requested: bool,
    pub reasons: Vec<String>,
    pub requires_counterproposal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterProposal {
    pub mission_id: String,
    pub rationale: Vec<String>,
    pub proposed: MissionRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionContract {
    pub request: MissionRequest,
    pub state: MissionState,
    pub approved_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionRevision {
    pub previous_mission_id: String,
    pub revised: MissionRequest,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionValidationError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for MissionValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid trading mission {}: {}", self.field, self.reason)
    }
}
impl std::error::Error for MissionValidationError {}

impl MissionRequest {
    pub fn validate(&self) -> Result<(), MissionValidationError> {
        if self.schema_version != TRADING_MISSION_SCHEMA_VERSION { return err("schema_version", "unsupported schema version"); }
        if self.mission_id.trim().is_empty() { return err("mission_id", "must be non-empty"); }
        finite_positive("capital", self.capital)?;
        finite("objective.target_net_profit", self.objective.target_net_profit)?;
        if self.objective.horizon_ms == 0 { return err("objective.horizon_ms", "must be positive"); }
        if let Some(value) = self.objective.preferred_net_profit_per_round_trip { finite("objective.preferred_net_profit_per_round_trip", value)?; }
        fraction("risk.max_drawdown_fraction", self.risk.max_drawdown_fraction)?;
        finite_nonnegative("risk.max_daily_loss", self.risk.max_daily_loss)?;
        finite_positive("risk.max_gross_exposure", self.risk.max_gross_exposure)?;
        finite_positive("risk.max_leverage", self.risk.max_leverage)?;
        Ok(())
    }
}

impl MissionContract {
    pub fn validate(&self) -> Result<(), MissionValidationError> {
        self.request.validate()?;
        if !matches!(self.state, MissionState::Approved | MissionState::Active | MissionState::Paused | MissionState::Completed | MissionState::Aborted) {
            return err("state", "a mission contract must have passed explicit approval");
        }
        if self.request.mode == MissionMode::Live && !self.request.execution.may_submit {
            return err("execution.may_submit", "live mode requires explicit submit permission");
        }
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), MissionValidationError> {
    if !value.is_finite() { return err(field, "must be finite"); } Ok(())
}
fn finite_positive(field: &'static str, value: f64) -> Result<(), MissionValidationError> {
    finite(field, value)?; if value <= 0.0 { return err(field, "must be positive"); } Ok(())
}
fn finite_nonnegative(field: &'static str, value: f64) -> Result<(), MissionValidationError> {
    finite(field, value)?; if value < 0.0 { return err(field, "must be non-negative"); } Ok(())
}
fn fraction(field: &'static str, value: f64) -> Result<(), MissionValidationError> {
    finite(field, value)?; if !(0.0..=1.0).contains(&value) { return err(field, "must be in [0, 1]"); } Ok(())
}
fn err<T>(field: &'static str, reason: &'static str) -> Result<T, MissionValidationError> {
    Err(MissionValidationError { field, reason })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: MissionMode) -> MissionRequest {
        MissionRequest {
            schema_version: TRADING_MISSION_SCHEMA_VERSION,
            mission_id: "mission-1".into(),
            mode,
            capital: 1_000.0,
            objective: ProfitObjective { target_net_profit: 100.0, horizon_ms: 86_400_000, preferred_net_profit_per_round_trip: Some(1.0) },
            risk: RiskBudget { max_drawdown_fraction: 0.05, max_daily_loss: 25.0, max_gross_exposure: 1_000.0, max_leverage: 1.0 },
            execution: ExecutionPermissions { may_submit: false, may_cancel: true, may_replace: true, may_close_positions: true },
        }
    }

    #[test]
    fn finite_bounded_mission_is_valid() { assert!(request(MissionMode::Paper).validate().is_ok()); }

    #[test]
    fn nonfinite_values_fail_closed() {
        let mut r = request(MissionMode::Paper);
        r.objective.target_net_profit = f64::NAN;
        assert_eq!(r.validate().unwrap_err().field, "objective.target_net_profit");
    }

    #[test]
    fn live_contract_requires_explicit_submit_permission() {
        let r = request(MissionMode::Live);
        let c = MissionContract { request: r, state: MissionState::Approved, approved_at_ms: 1 };
        assert_eq!(c.validate().unwrap_err().field, "execution.may_submit");
    }

    #[test]
    fn proposed_request_is_not_an_approved_contract() {
        let c = MissionContract { request: request(MissionMode::Paper), state: MissionState::Proposed, approved_at_ms: 0 };
        assert_eq!(c.validate().unwrap_err().field, "state");
    }
}
