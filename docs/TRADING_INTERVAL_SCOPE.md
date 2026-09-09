# Trading interval estimates: scope and failure contract

`scirust_trader::certify::try_certify` returns a typed `CertificationError` for
empty or dimensionally inconsistent features, nonfinite values, negative or
nonfinite radius, missing/malformed layers, non-scalar output and nonfinite
arithmetic. It never stops at a malformed layer and returns the previous layer
as though the entire model had been evaluated.

`TradingAgent::try_process` validates configuration and OHLCV geometry before
inference/narration, propagates interval errors, and rejects nonfinite prediction
or attribution. The agent expects `lookback + 3 == model.input_dim`. The existing
`certify` and `process` signatures remain compatibility wrappers for validated
callers; invalid input now panics explicitly. External boundaries must use the
fallible methods; `trader_certified_predict` does so.

The exported weight format is bias followed by row-major `(in, out)` linear
weights, with ReLU between layers and one scalar output. Public replacement of
`PricePredictor.net` with arbitrary modules is not qualified by this contract.
The fingerprint field is caller-supplied provenance, not independently verified
training identity or authenticity.

The numerical kernel is unchanged for valid inputs. It uses f32 operations
without outward/directed rounding. Therefore these estimates are NOT a formal
floating-point enclosure, a probabilistic confidence interval, a bound on actual
market returns or a guarantee about an LLM's prose. `llm_consistent` is only a
heuristic text check, not authorization, automatic blocking or an alert system.

The MCP demonstration constructs a seeded untrained MLP, not a loaded trained
forecast model. Its result now states `model_status: untrained_seeded_demo`,
the model-input perturbation scope, and false formal/market guarantee flags.
Existing `certified_interval` field names remain for API compatibility.

Validation commands:

```bash
cargo test -p scirust-trader --locked
cargo test -p scirust-mcp --locked
```

Tests cover malformed first/later layers, dimension mismatch, nonfinite values,
arithmetic overflow, scalar-output rejection, bad agent configuration, existing
forward-pass parity, and an independent affine corner oracle. These synthetic
tests do not establish market performance, full IBP formal soundness, full audit
F15 closure (scanner/proof metadata remain separate), or an ML maturity upgrade.
