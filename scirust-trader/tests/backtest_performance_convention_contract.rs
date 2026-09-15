//! Contract for the next backtest migration step.
//!
//! The backtester already constructs a `PerformanceConvention`, but its report
//! path still forwards only the Sharpe scalar to the legacy constructor.  This
//! test keeps the intended zero-reference compatibility behavior explicit while
//! the convention-aware report builder is adopted by the backtest implementation.

use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn zero_reference_convention_matches_legacy_backtest_semantics() {
    let equity = [100.0, 101.0, 99.0, 103.0];
    let convention = PerformanceConvention::for_crypto_interval("1h", 0.0, 0.0).unwrap();
    let report = from_curve_with_convention(&equity, &[], convention);
    let legacy = scirust_trader::metrics::PerformanceReport::from_curve(
        &equity,
        &[],
        convention.periods_per_year(),
        0.0,
    );

    assert_eq!(report.periods_per_year, legacy.periods_per_year);
    assert_eq!(report.sharpe, legacy.sharpe);
    assert_eq!(report.sortino, legacy.sortino);
    assert_eq!(report.total_return, legacy.total_return);
}
