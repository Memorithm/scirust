use scirust_trader::metrics::{returns_from_equity, sharpe, sortino};
use scirust_trader::performance_convention::PerformanceConvention;

#[test]
fn convention_keeps_sharpe_reference_and_sortino_target_semantically_distinct() {
    let equity = [100.0_f32, 102.0, 100.98, 104.0094];
    let returns = returns_from_equity(&equity);
    let convention = PerformanceConvention::new(365.0, 0.0, 0.01).unwrap();

    let expected_sharpe = sharpe(
        &returns,
        convention.sharpe_reference_return_per_period(),
        convention.periods_per_year(),
    );
    let expected_sortino = sortino(
        &returns,
        convention.sortino_target_per_period(),
        convention.periods_per_year(),
    );
    let incorrectly_shared_target = sortino(
        &returns,
        convention.sharpe_reference_return_per_period(),
        convention.periods_per_year(),
    );

    assert_ne!(expected_sortino, incorrectly_shared_target);
    assert!(expected_sharpe.is_finite());
    assert!(expected_sortino.is_finite());
}
