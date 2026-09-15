# Checked trading annualisation migration

## Scope

This is the annualisation follow-up to #1386 (one part of historical finding F16),
implemented from `master` at `f31d4029302e79cfde7ebeff0375431324a93e75`.
It reuses `performance_convention::try_crypto_periods_per_year` and
`PerformanceConvention`; it introduces no second parser or trading engine.

## Behavior changes

| Surface | New behavior |
| --- | --- |
| `backtest::try_run_backtest` | Returns `Result<BacktestReport, PerformanceConventionError>`; validates the interval before simulation state is allocated or any strategy callback is invoked. |
| `backtest::run_backtest` | Preserves its signature, delegates to the checked implementation, and explicitly panics on an invalid interval. |
| `metrics::periods_per_year` | Preserves its signature, delegates to the shared checked parser, and explicitly panics on an invalid interval instead of inventing daily data. |
| MCP `ToolRegistry::call` | Rejects invalid supplied intervals with `invalid_arguments` before dispatch. |

This is a deliberate behavioral tightening for invalid native Rust inputs, not
full backward compatibility. Callers handling untrusted configuration should use
`try_run_backtest` and `try_crypto_periods_per_year`, or the production MCP
registry. Native scanner, walk-forward and optimizer signatures are unchanged;
they still inherit the compatibility backtest's panic behavior for invalid
intervals. A future fallible migration of those APIs remains separate work.

The MCP guard covers root `interval` in `trader_backtest`, `trader_metrics`,
`trader_walkforward`, `trader_monte_carlo`, `trader_portfolio_construct`,
`trader_regime`, and `trader_optimize`. It also checks each market's interval in
`trader_scan_opportunities` and `trader_dashboard`. Omitted intervals retain the
existing defaults. An explicitly supplied null, wrong type, malformed magnitude
or unknown unit cannot use a default. The fixed mock market identity/cadence
checks are unchanged. Opaque research-envelope metadata is not interpreted as
market configuration.

The compatibility helper also stops clamping annualisation to at least one.
Under the declared 365-day convention, `730d` means `365 / 730 = 0.5` periods per
year. For returns +10% and -10%, sample variance is 0.02, so annualised volatility
at that frequency is `sqrt(0.02 * 0.5) = 0.1`. The integration test uses this
independently derived oracle rather than copying the implementation.

## Native Rust usage

```rust
use scirust_trader::backtest::{BacktestConfig, try_run_backtest};
use scirust_trader::strategy::Momentum;

let config = BacktestConfig {
    interval: "1h".into(),
    ..Default::default()
};
let report = try_run_backtest(&Momentum::default(), &[], &config)?;
assert_eq!(report.performance.periods_per_year, 8760.0);
# Ok::<(), scirust_trader::performance_convention::PerformanceConventionError>(())
```

## Boundaries and remaining work

- Intervals use the existing case-insensitive fixed-duration grammar: seconds,
  minutes, hours, days and weeks in a 365-day, 24/7 year. This is not an
  exchange-specific calendar parser; uppercase `M` is not a calendar-month unit.
- The interval is declared metadata. This change does not infer or validate
  candle spacing, missing bars, session calendars or timestamp alignment.
- No general validation of capital, sizing, equity arrays, return arrays or
  finite metric outputs is added. Existing numerical and accounting semantics
  outside interval resolution are unchanged.
- Backtest reports still use zero per-period Sharpe reference and Sortino target.
  Migrating report/MCP APIs to separately supplied nonzero references remains
  independent work; F16 as a whole is not declared closed.
- Direct Rust invocation of an extracted raw MCP handler bypasses the registry
  guard. The advertised rejection guarantee is for `ToolRegistry::call`.
- The shared checked backtest is reusable by agents and downstream applications,
  but adoption by another repository is not claimed without a tested integration.
- No wallet authorization, signing, exchange-order, custody or funded execution
  path is enabled or changed.

## Qualification

The patch adds 16 test functions (backtest, metrics, boundary guard and production
MCP integration) and five Rustdoc examples. Tests cover early rejection, malformed
and out-of-range intervals, all nine MCP entry surfaces, nested error locations,
omitted defaults, supported aliases, subannual frequencies, and valid-path
agreement.

Local Cargo success is not claimed: the editing environment has no Rust tools
and cannot resolve GitHub for a local workspace checkout. Required evidence is
GitHub Actions on the exact final PR head, including the existing repository
formatting and documentation gates. Targeted commands from the repository root:

```bash
cargo test -p scirust-trader --locked
cargo test -p scirust-trader --doc --locked
cargo test -p scirust-mcp --test trader_annualisation_contract --locked
cargo test -p scirust-mcp --test trader_mcp_contract --locked
```

These commands are validation instructions, not a statement that they ran in the
editing environment. Merge requires all applicable exact-head CI checks to pass.
