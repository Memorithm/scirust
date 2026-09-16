//! Convention-aware performance reporting for backtests.
//!
//! This is the checked migration surface between the historical backtester and
//! the explicit performance convention introduced for Sharpe/Sortino semantics.
//! Simulation, fills and accounting remain owned by [`crate::backtest`].

use crate::backtest::{BacktestConfig, BacktestReport, try_run_backtest};
use crate::performance_convention::{PerformanceConvention, PerformanceConventionError};
use crate::performance_report::from_curve_with_convention;
use crate::strategy::Strategy;

/// Run the existing checked backtest and rebuild only its performance report
/// using the supplied independent Sharpe reference and Sortino target.
///
/// The supplied convention must describe the same annualisation as
/// `cfg.interval`; this prevents report metadata from disagreeing with the
/// simulated bar interval. The strategy, fills, trades and equity curve are
/// otherwise exactly those produced by [`try_run_backtest`].
pub fn try_run_backtest_with_convention(
    strategy: &dyn Strategy,
    candles: &[crate::market::Candle],
    cfg: &BacktestConfig,
    convention: PerformanceConvention,
) -> Result<BacktestReport, PerformanceConventionError> {
    let interval_convention = PerformanceConvention::for_crypto_interval(&cfg.interval, 0.0, 0.0)?;
    if interval_convention.periods_per_year() != convention.periods_per_year() {
        return Err(PerformanceConventionError::InvalidPeriodsPerYear(
            convention.periods_per_year(),
        ));
    }

    let mut report = try_run_backtest(strategy, candles, cfg)?;
    let pnls: Vec<f32> = report.trades.iter().map(|trade| trade.net_pnl).collect();
    report.performance = from_curve_with_convention(&report.equity_curve, &pnls, convention);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::Momentum;

    #[test]
    fn explicit_targets_do_not_change_simulation_outputs() {
        let cfg = BacktestConfig::default();
        let strategy = Momentum::default();
        let legacy = try_run_backtest(&strategy, &[], &cfg).unwrap();
        let convention = PerformanceConvention::for_crypto_interval("1h", 0.001, 0.002).unwrap();
        let explicit = try_run_backtest_with_convention(&strategy, &[], &cfg, convention).unwrap();
        assert_eq!(legacy.equity_curve, explicit.equity_curve);
        assert_eq!(legacy.trades, explicit.trades);
        assert_eq!(legacy.final_equity, explicit.final_equity);
        assert_eq!(explicit.performance.periods_per_year, 8760.0);
    }

    #[test]
    fn convention_must_match_declared_interval_annualisation() {
        let cfg = BacktestConfig::default();
        let convention = PerformanceConvention::for_crypto_interval("1d", 0.0, 0.0).unwrap();
        assert!(try_run_backtest_with_convention(&Momentum::default(), &[], &cfg, convention).is_err());
    }
}
