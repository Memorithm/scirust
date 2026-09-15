# Trading next slice

This branch tracks the next verified trading-metrics correction after #1437.

## Implemented migration surface

`PerformanceConvention` already distinguishes the constant per-period Sharpe reference return from the per-period Sortino minimum acceptable return. `performance_report::from_curve_with_convention` now carries that distinction into a complete `PerformanceReport` without duplicating metric formulas: it delegates to the existing metric primitives.

The legacy `PerformanceReport::from_curve` remains untouched in this slice so downstream Rust callers are not broken before they can migrate. The new builder is additive and explicitly tested so changing only the Sortino target cannot change Sharpe, and changing only the Sharpe reference cannot change Sortino.

## Remaining migration

The next patch should switch the backtest report construction to the convention-aware builder, then expose separately validated optional Sharpe/Sortino inputs at the MCP boundary. Once consumers have migrated, the legacy single-reference constructor can be deprecated rather than silently changing its historical meaning.

A separate follow-up must validate that declared interval metadata is coherent with actual OHLCV timestamp cadence. Missing complete bars may be allowed under an explicit policy, but duplicate, reversed, or off-grid timestamps must not be certified as a regular series.

No funded execution, wallet authorization, signing, custody, or exchange-order path is part of this slice.
