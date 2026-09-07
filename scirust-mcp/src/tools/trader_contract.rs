//! MCP transport adapter for the consumer-neutral trading research contract.
//!
//! The contract itself lives in `scirust-trader`; MCP only parses, validates,
//! links, and fingerprints envelopes. No trading action is executed here.

use crate::registry::McpTool;
use serde_json::{Value, json};
use scirust_trader::consumer_contract::{TradingResearchRequest, TradingResearchResult};

pub fn trader_contract_tools() -> Vec<McpTool> {
    vec![validate_request_tool(), validate_result_tool()]
}

fn validate_request_tool() -> McpTool {
    McpTool {
        name: "trader_contract_validate_request".to_string(),
        description: "Validate and fingerprint a versioned consumer-neutral trading-research request envelope. The caller identity may represent an agent, backtester, paper trader, CLI, service, library, or human-facing tool. No research computation or order execution is performed.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["request"],
            "properties": {
                "request": {
                    "type": "object",
                    "description": "TradingResearchRequest v1 with explicit consumer, operation, payload, provenance and declared assumptions"
                }
            }
        }),
        handler: Box::new(|args| {
            let request: TradingResearchRequest = parse_required(&args, "request")?;
            let fingerprint = request.fingerprint().map_err(|error| error.to_string())?;
            Ok(json!({
                "schema_version": request.schema_version,
                "request": request,
                "fingerprint": fingerprint
            }))
        }),
    }
}

fn validate_result_tool() -> McpTool {
    McpTool {
        name: "trader_contract_validate_result".to_string(),
        description: "Validate a trading-research result envelope, verify that it is bound to the exact supplied request fingerprint, and return the result fingerprint. No hidden defaults, strategy selection, or order execution is performed.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["request", "result"],
            "properties": {
                "request": {"type": "object", "description": "TradingResearchRequest v1"},
                "result": {"type": "object", "description": "TradingResearchResult v1"}
            }
        }),
        handler: Box::new(|args| {
            let request: TradingResearchRequest = parse_required(&args, "request")?;
            let result: TradingResearchResult = parse_required(&args, "result")?;
            result
                .matches_request(&request)
                .map_err(|error| error.to_string())?;
            let fingerprint = result.fingerprint().map_err(|error| error.to_string())?;
            Ok(json!({
                "schema_version": result.schema_version,
                "result": result,
                "fingerprint": fingerprint,
                "request_match": true
            }))
        }),
    }
}

fn parse_required<T: serde::de::DeserializeOwned>(args: &Value, key: &str) -> Result<T, String> {
    serde_json::from_value(
        args.get(key)
            .cloned()
            .ok_or_else(|| format!("missing `{key}`"))?,
    )
    .map_err(|error| format!("invalid `{key}`: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_trader::consumer_contract::{
        ContractParticipant, ParticipantKind, ResearchOutcome, TradingResearchResult,
        TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION,
    };
    use std::collections::BTreeMap;

    fn request(kind: ParticipantKind) -> TradingResearchRequest {
        TradingResearchRequest {
            schema_version: TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION,
            request_id: "req-mcp-1".into(),
            experiment_id: Some("exp-mcp-1".into()),
            consumer: ContractParticipant {
                participant_id: "consumer-1".into(),
                kind,
                implementation: "consumer-test@1".into(),
            },
            submitted_at_ms: 100,
            operation: "scirust.trader.research.test.v1".into(),
            payload: json!({"sample": 1}),
            provenance: vec![],
            declared_assumptions: BTreeMap::new(),
        }
    }

    #[test]
    fn request_tool_accepts_multiple_consumer_kinds() {
        let tool = validate_request_tool();
        for kind in [ParticipantKind::Agent, ParticipantKind::Backtester, ParticipantKind::Cli] {
            let value = (tool.handler)(json!({"request": request(kind)})).unwrap();
            assert_eq!(value["schema_version"], 1);
            assert_eq!(value["fingerprint"].as_str().unwrap().len(), 64);
        }
    }

    #[test]
    fn result_tool_rejects_request_mismatch() {
        let request = request(ParticipantKind::Backtester);
        let result = TradingResearchResult {
            schema_version: TRADING_RESEARCH_CONTRACT_SCHEMA_VERSION,
            result_id: "result-mcp-1".into(),
            request_id: request.request_id.clone(),
            request_fingerprint: request.fingerprint().unwrap(),
            experiment_id: request.experiment_id.clone(),
            producer: ContractParticipant {
                participant_id: "scirust-trader".into(),
                kind: ParticipantKind::Library,
                implementation: "scirust-trader@0.1.0".into(),
            },
            produced_at_ms: 200,
            operation: request.operation.clone(),
            outcome: ResearchOutcome::Succeeded,
            output: Some(json!({"report": {"ok": true}})),
            evidence: vec![],
            provenance: vec![],
            issues: vec![],
        };

        let mut different_request = request.clone();
        different_request.payload = json!({"sample": 2});
        assert!((validate_result_tool().handler)(json!({
            "request": different_request,
            "result": result
        }))
        .is_err());
    }
}
