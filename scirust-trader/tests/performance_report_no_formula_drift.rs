use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn references_do_not_change_reference_independent_metrics() {
    let equity = [100.0, 105.0, 103.0, 107.0, 101.0];
    let base = from_curve_with_convention(
        &equity,
        &[],
        PerformanceConvention::new(365.0, 0.0, 0.0).unwrap(),
    );
    let shifted = from_curve_with_convention(
        &equity,
        &[],
        PerformanceConvention::new(365.0, 0.003, 0.004).unwrap(),
    );

    assert_eq!(base.total_return, shifted.total_return);
    assert_eq!(base.cagr, shifted.cagr);
    assert_eq!(base.volatility, shifted.volatility);
    assert_eq!(base.calmar, shifted.calmar);
    assert_eq!(base.max_drawdown, shifted.max_drawdown);
    assert_eq!(base.ulcer_index, shifted.ulcer_index);
    assert_eq!(base.var_95, shifted.var_95);
    assert_eq!(base.cvar_95, shifted.cvar_95);
    assert_eq!(base.kelly_continuous, shifted.kelly_continuous);
}
