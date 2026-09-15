use scirust_trader::performance_convention::PerformanceConvention;
use scirust_trader::performance_report::from_curve_with_convention;

#[test]
fn convention_report_preserves_trade_statistics() {
    let pnls = [100.0, -50.0, 200.0, -50.0];
    let report = from_curve_with_convention(
        &[1000.0, 1010.0, 1005.0],
        &pnls,
        PerformanceConvention::new(365.0, 0.001, 0.002).unwrap(),
    );
    assert_eq!(report.trades.num_trades, 4);
    assert_eq!(report.trades.num_wins, 2);
    assert_eq!(report.trades.num_losses, 2);
    assert!((report.trades.profit_factor - 3.0).abs() < 1e-6);
}
