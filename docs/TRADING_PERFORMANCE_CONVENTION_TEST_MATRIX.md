# Trading performance convention qualification matrix

The migration keeps the legacy report constructor intact while adding an explicit convention-aware construction path.

| Property | Evidence |
| --- | --- |
| Sharpe and Sortino references are independent | `performance_report_contract.rs`, `performance_report_target_oracle.rs` |
| Non-finite references are rejected | `performance_report_validation.rs` |
| Zero-reference behavior remains compatible | `performance_report_legacy_equivalence.rs`, `backtest_performance_convention_contract.rs` |
| Formula wiring is checked against an independent numeric oracle | `performance_report_oracle.rs` |
| Existing `PerformanceConvention` parser/validation remains authoritative | `performance_convention_contract.rs` |

The next implementation step is to make the backtest report path call `performance_report::from_curve_with_convention` directly. This is deliberately separated from the additive report-builder change so CI can prove the new calculation surface before changing a production caller.

A later slice will validate that OHLCV timestamps are coherent with the declared fixed-duration interval. Missing complete bars must be distinguishable from mislabelled or partially shifted cadence; this document does not claim that validation exists yet.
