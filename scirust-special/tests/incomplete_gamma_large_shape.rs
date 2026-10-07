//! Regression tests: regularized incomplete gamma `P(a, x)` / `Q(a, x)` at
//! large shape `a`.
//!
//! The common factor `x^a·e^(−x)/Γ(a)` used to be formed as
//! `exp(a·ln x − x − ln Γ(a))`, whose terms are of size `a·ln a` while their
//! sum near the mode is only `≈ −½·ln(2πa)`; the relative error grew like
//! `ε·a·ln a` (`P(1e10, 1e10)` was `0.4999897` instead of `0.5000013`,
//! `P(1e12, 1e12)` was `0.4990` instead of `0.5000001`). The series for `P`
//! also stopped on the last term alone, which near `x ≈ a` leaves a truncated
//! tail of relative size `≈ ε·√a`.
//!
//! References are mpmath 1.3 at 40 significant digits on the exact `f64`
//! inputs (`gammainc` for `a < 1000`, adaptive quadrature of the density
//! split at multiples of `√a` otherwise), cross-checked against
//! `P(a, a) = ½ + 1/(3·√(2πa)) + O(a^(−3/2))`. Tolerances are relative;
//! nothing is compared bit for bit.

use scirust_special::{regularized_gamma_p, regularized_gamma_q};

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let rel = ((got - want) / want).abs();
    assert!(
        got.is_finite() && rel <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {rel:e})"
    );
}

/// `(a, x, P(a, x), Q(a, x))`.
type Case = (f64, f64, f64, f64);

fn check(cases: &[Case], tol: f64) {
    for &(a, x, p, q) in cases
    {
        assert_rel(
            regularized_gamma_p(a, x),
            p,
            tol,
            &format!("P({a:e}, {x:e})"),
        );
        assert_rel(
            regularized_gamma_q(a, x),
            q,
            tol,
            &format!("Q({a:e}, {x:e})"),
        );
    }
}

/// Moderate shapes: cheap enough for Miri (a few thousand series terms).
#[test]
fn moderate_shape_keeps_full_relative_accuracy() {
    let cases: [Case; 4] = [
        (
            1e4,
            9500.0,
            1.862_454_651_795_155e-7,
            0.999_999_813_754_534_8,
        ),
        (1e4, 1e4, 0.501_329_808_339_955_2, 0.498_670_191_660_044_8),
        (
            1e4,
            10500.0,
            0.999_999_572_412_754_5,
            4.275_872_455_059_647_5e-7,
        ),
        (1e5, 1e5, 0.500_420_522_110_365_3, 0.499_579_477_889_634_8),
    ];
    check(&cases, 1e-13);
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁴–10⁶ series terms: too slow under Miri.
fn large_shape_near_the_mode() {
    let cases: [Case; 9] = [
        (
            1e6,
            995_000.0,
            2.749_580_359_270_071e-7,
            0.999_999_725_041_964_1,
        ),
        (1e6, 1e6, 0.500_132_980_760_872_6, 0.499_867_019_239_127_4),
        (
            1e6,
            1_005_000.0,
            0.999_999_701_250_986,
            2.987_490_140_114_635e-7,
        ),
        (
            1e8,
            99_950_000.0,
            2.854_642_139_958_626e-7,
            0.999_999_714_535_786,
        ),
        (1e8, 1e8, 0.500_013_298_076_014_1, 0.499_986_701_923_985_9),
        (
            1e8,
            100_050_000.0,
            0.999_999_712_157_031_3,
            2.878_429_686_852_781e-7,
        ),
        (
            1e10,
            9_999_500_000.0,
            2.865_326_545_108_890_4e-7,
            0.999_999_713_467_345_5,
        ),
        (1e10, 1e10, 0.500_001_329_807_601_3, 0.499_998_670_192_398_7),
        (
            1e10,
            10_000_500_000.0,
            0.999_999_713_229_470_5,
            2.867_705_296_367_124e-7,
        ),
    ];
    check(&cases, 4e-12);
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁷ series terms: too slow under Miri.
fn very_large_shape_near_the_mode() {
    let cases: [Case; 3] = [
        (
            1e12,
            999_995_000_000.0,
            2.866_396_783_250_204e-7,
            0.999_999_713_360_321_7,
        ),
        (1e12, 1e12, 0.500_000_132_980_760_1, 0.499_999_867_019_239_9),
        (
            1e12,
            1_000_005_000_000.0,
            0.999_999_713_336_534_1,
            2.866_634_658_372_596e-7,
        ),
    ];
    check(&cases, 3e-11);
}
