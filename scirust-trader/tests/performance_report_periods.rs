use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn report_preserves_subannual_periods_from_validated_convention() {
    let convention = PerformanceConvention::for_crypto_interval("730d", 0.0, 0.0).unwrap();
    let report = from_curve_with_convention(&[100.0, 99.0, 101.0], &[], convention);
    assert_eq!(report.periods_per_year, 0.5);
}
