# Trading Consumer Contract v1

SciRust's trading-research layer is a shared project capability. Replikans is one possible consumer, not the architectural owner of the contract.

## Boundary

The contract is defined in `scirust-trader::consumer_contract` and is transport-neutral. Rust callers use it directly. MCP, CLI, backtest runtimes, paper-trading runtimes, services, and future agent systems adapt to the same envelope rather than defining parallel schemas.

The v1 contract intentionally covers research requests and results only. It does not authorize or execute real-money orders.

## Participants

A request declares a `ContractParticipant` consumer. Supported participant kinds include agent, backtester, paper trader, CLI, human-facing tool, service, library, and other. A result separately declares its producer.

No participant kind receives different scientific semantics. The same operation ID, payload, provenance, assumptions, and evidence contract applies to all consumers.

## Request integrity

`TradingResearchRequest` carries:

- schema version;
- request and optional experiment identifiers;
- consumer identity and implementation identifier;
- explicit operation ID;
- structured JSON object payload;
- provenance references;
- declared assumptions.

The full envelope is SHA-256 fingerprinted with a domain tag. Changing the consumer, operation, payload, provenance, or assumptions changes the fingerprint.

## Result integrity

`TradingResearchResult` carries:

- schema version and result ID;
- originating request ID and exact request fingerprint;
- optional experiment ID;
- producer identity;
- operation ID and outcome;
- structured output;
- evidence and provenance references;
- explicit issues.

A result can be checked with `matches_request`. This verifies request ID, request fingerprint, experiment ID, and operation rather than trusting transport context.

## MCP adapter

`scirust-mcp` exposes two transport helpers:

- `trader_contract_validate_request`;
- `trader_contract_validate_result`.

They validate and fingerprint envelopes only. The authoritative types and rules remain in `scirust-trader` so other consumers do not depend on MCP.

## Extension rule

New research operations use namespaced versioned operation IDs. Adding an operation does not require changing the outer contract schema unless the envelope semantics themselves change.
