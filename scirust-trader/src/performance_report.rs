//! Checked construction helpers for performance reports.
//!
//! This module is the migration surface from legacy single-reference reports to
//! the explicit [`PerformanceConvention`] contract. It delegates every
//! statistic to [`crate::metrics`]; no metric formula is duplicated here.

use crate::metrics::{
    PerformanceReport, cagr, calmar, conditional_value_at_risk, kelly_continuous, max_drawdown,
    returns_from_equity, sharpe, sortino, total_return, trade_stats, ulcer_index, value_at_risk,
    volatility,
};
use crate::performance_convention::PerformanceConvention;

/// Build a performance report using independent Sharpe and Sortino references.
///
/// # Examples
///
/// ```
/// use scirust_trader::performance_convention::PerformanceConvention;
/// use scirust_trader::performance_report::from_curve_with_convention;
/// let convention = PerformanceConvention::new(365.0, 0.0, 0.001)?;
/// let report = from_curve_with_convention(&[100.0, 99.0, 101.0], &[], convention);
/// assert_eq!(report.periods_per_year, 365.0);
/// # Ok::<(), scirust_trader::performance_convention::PerformanceConventionError>(())
/// ```
pub fn from_curve_with_convention(
    equity: &[f32],
    trade_pnls: &[f32],
    convention: PerformanceConvention,
) -> PerformanceReport {
    let returns = returns_from_equity(equity);
    let periods_per_year = convention.periods_per_year();
    let (max_drawdown_value, _, _) = max_drawdown(equity);

    PerformanceReport {
        periods_per_year,
        total_return: total_return(equity),
        cagr: cagr(equity, periods_per_year),
        volatility: volatility(&returns, periods_per_year),
        sharpe: sharpe(
            &returns,
            convention.sharpe_reference_return_per_period(),
            periods_per_year,
        ),
        sortino: sortino(
            &returns,
            convention.sortino_target_per_period(),
            periods_per_year,
        ),
        calmar: calmar(equity, periods_per_year),
        max_drawdown: max_drawdown_value,
        ulcer_index: ulcer_index(equity),
        var_95: value_at_risk(&returns, 0.05),
        cvar_95: conditional_value_at_risk(&returns, 0.05),
        kelly_continuous: kelly_continuous(&returns),
        trades: trade_stats(trade_pnls),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_only_sortino_target_leaves_sharpe_unchanged() {
        let equity = [100.0, 102.0, 100.98, 104.0094];
        let a = PerformanceConvention::new(365.0, 0.0, 0.0).unwrap();
        let b = PerformanceConvention::new(365.0, 0.0, 0.01).unwrap();
        let report_a = from_curve_with_convention(&equity, &[], a);
        let report_b = from_curve_with_convention(&equity, &[], b);

        assert_eq!(report_a.sharpe, report_b.sharpe);
        assert_ne!(report_a.sortino, report_b.sortino);
    }

    #[test]
    fn changing_only_sharpe_reference_leaves_sortino_unchanged() {
        let equity = [100.0, 102.0, 100.98, 104.0094];
        let a = PerformanceConvention::new(365.0, 0.0, 0.0).unwrap();
        let b = PerformanceConvention::new(365.0, 0.005, 0.0).unwrap();
        let report_a = from_curve_with_convention(&equity, &[], a);
        let report_b = from_curve_with_convention(&equity, &[], b);

        assert_ne!(report_a.sharpe, report_b.sharpe);
        assert_eq!(report_a.sortino, report_b.sortino);
    }
}
