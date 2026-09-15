use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn independent_references_match_hand_calculated_ratios() {
    // Equity returns are +2%, -1%, +3% exactly up to normal f32 rounding.
    let equity = [100.0, 102.0, 100.98, 104.0094];
    let convention = PerformanceConvention::new(1.0, 0.005, 0.0).unwrap();
    let report = from_curve_with_convention(&equity, &[], convention);

    // mean = 0.0133333; sample sd = 0.02081666
    // Sharpe = (mean - 0.005) / sd = 0.400320...
    assert!(approx(report.sharpe, 0.40032038));

    // downside deviation around target 0 is sqrt((0^2 + 0.01^2 + 0^2)/3)
    // = 0.00577350; Sortino = mean / downside = 2.309401...
    assert!(approx(report.sortino, 2.309401));
}
