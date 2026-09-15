use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn nonzero_sortino_target_changes_only_sortino_when_sharpe_reference_is_fixed() {
    let equity = [100.0, 102.0, 100.98, 104.0094];
    let zero_target = PerformanceConvention::new(365.0, 0.002, 0.0).unwrap();
    let positive_target = PerformanceConvention::new(365.0, 0.002, 0.005).unwrap();

    let a = from_curve_with_convention(&equity, &[], zero_target);
    let b = from_curve_with_convention(&equity, &[], positive_target);

    assert_eq!(a.sharpe, b.sharpe);
    assert_ne!(a.sortino, b.sortino);
    assert_eq!(a.volatility, b.volatility);
    assert_eq!(a.max_drawdown, b.max_drawdown);
}
