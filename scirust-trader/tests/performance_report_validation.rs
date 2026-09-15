use scirust_trader::performance_convention::{PerformanceConvention, PerformanceConventionError};

#[test]
fn convention_rejects_non_finite_sharpe_reference() {
    assert!(matches!(
        PerformanceConvention::new(365.0, f32::NAN, 0.0),
        Err(PerformanceConventionError::NonFiniteSharpeReference(_))
    ));
}

#[test]
fn convention_rejects_non_finite_sortino_target() {
    assert!(matches!(
        PerformanceConvention::new(365.0, 0.0, f32::INFINITY),
        Err(PerformanceConventionError::NonFiniteSortinoTarget(_))
    ));
}
