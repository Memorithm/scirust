use scirust_trader::metrics::PerformanceReport;
use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn zero_reference_convention_is_fieldwise_legacy_equivalent() {
    let equity = [10_000.0, 10_100.0, 10_050.0, 10_300.0, 10_250.0, 10_500.0];
    let pnls = [100.0, -50.0, 250.0, -50.0, 250.0];
    let convention = PerformanceConvention::new(365.0, 0.0, 0.0).unwrap();
    let checked = from_curve_with_convention(&equity, &pnls, convention);
    let legacy = PerformanceReport::from_curve(&equity, &pnls, 365.0, 0.0);

    assert_eq!(checked.periods_per_year, legacy.periods_per_year);
    assert_eq!(checked.total_return, legacy.total_return);
    assert_eq!(checked.cagr, legacy.cagr);
    assert_eq!(checked.volatility, legacy.volatility);
    assert_eq!(checked.sharpe, legacy.sharpe);
    assert_eq!(checked.sortino, legacy.sortino);
    assert_eq!(checked.calmar, legacy.calmar);
    assert_eq!(checked.max_drawdown, legacy.max_drawdown);
    assert_eq!(checked.ulcer_index, legacy.ulcer_index);
    assert_eq!(checked.var_95, legacy.var_95);
    assert_eq!(checked.cvar_95, legacy.cvar_95);
    assert_eq!(checked.kelly_continuous, legacy.kelly_continuous);
    assert_eq!(checked.trades, legacy.trades);
}
