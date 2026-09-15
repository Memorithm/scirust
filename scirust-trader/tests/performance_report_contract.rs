use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn report_uses_distinct_sharpe_and_sortino_inputs() {
    let equity = [100.0_f32, 102.0, 100.98, 104.0094];
    let base = PerformanceConvention::new(365.0, 0.0, 0.0).unwrap();
    let sortino_only = PerformanceConvention::new(365.0, 0.0, 0.01).unwrap();
    let sharpe_only = PerformanceConvention::new(365.0, 0.005, 0.0).unwrap();

    let a = from_curve_with_convention(&equity, &[], base);
    let b = from_curve_with_convention(&equity, &[], sortino_only);
    let c = from_curve_with_convention(&equity, &[], sharpe_only);

    assert_eq!(a.sharpe, b.sharpe);
    assert_ne!(a.sortino, b.sortino);
    assert_ne!(a.sharpe, c.sharpe);
    assert_eq!(a.sortino, c.sortino);
}
