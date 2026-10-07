//! Regression tests for `beta` and the analytic-continuation edge cases of
//! `ln_beta`.
//!
//! Reference values are 50-digit `mpmath` evaluations (`mp.beta`) at the exact
//! `f64` arguments. Tolerances are relative and leave room for a few rounding
//! units of libm (and Miri's float perturbation).

use scirust_special::{beta, ln_beta};

fn rel_err(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

/// `B(a, b)` is negative whenever an odd number of `Γ(a)`, `Γ(b)`, `Γ(a+b)`
/// are negative. `beta` used to return `exp(ln |B|)`, so every negative value
/// came back with the wrong sign (`B(−½, 1)` gave `+2` instead of `−2`).
#[test]
fn beta_has_the_right_sign_for_negative_arguments() {
    let cases = [
        (-0.5, 1.0, -2.0),
        (-0.5, 2.0, -4.0),
        (-1.5, 0.25, 2.185_047_961_910_1),
        (-2.5, -0.3, -3.592_308_538_241_997_3),
        (-0.5, -0.25, -3.594_420_704_206_777),
        (0.3, -1.7, 2.828_058_204_108_712_6),
        (2.0, -1.5, 1.333_333_333_333_333_3),
        (-10.3, 4.6, -0.000_749_660_817_197_678_9),
        (-150.5, 3.25, -1.547_107_841_680_365e-7),
    ];
    for (a, b, want) in cases
    {
        for (x, y) in [(a, b), (b, a)]
        {
            let got = beta(x, y);
            assert!(
                rel_err(got, want) < 1e-13,
                "beta({x}, {y}) = {got:e}, want {want:e}"
            );
        }
    }
}

/// For moderate positive arguments `beta` is the ratio of the (accurate)
/// gamma values instead of `exp(ln B)`, whose relative error grows like
/// `ε·|ln B|` (about 1e-14 at `ln B ≈ −110`).
#[test]
fn beta_is_relatively_accurate_for_moderate_positive_arguments() {
    let cases = [
        (80.0, 80.0, 2.716_059_082_866_689e-49),
        (30.0, 40.5, 7.956_393_636_500_52e-22),
        (160.5, 9.25, 2.174_971_349_866_767_6e-16),
        (0.5, 150.0, 0.144_840_901_161_278_04),
        (1e-5, 3.5, 99_998.319_648_385_35),
        (1e-160, 1e-160, 2e160),
        (7.25, 3.0, 0.003_614_900_166_624_304_5),
        (100.0, 60.5, 2.686_217_362_569_991e-47),
        (120.0, 49.75, 1.080_981_730_390_511_9e-45),
        (2.0, 3.0, 1.0 / 12.0),
    ];
    // Natively the new errors are at most ~4e-16 and the old ones reached
    // 2.3e-14 (`B(120, 49.75)`), 8e-15 (`B(1e-160, 1e-160)`) and 6.6e-15
    // (`B(80, 80)`). Miri perturbs every libm result by a few ulp, which
    // compounds through the three gamma values to ~5e-15, so it gets a looser
    // bound.
    let tol = if cfg!(miri) { 3e-14 } else { 2e-15 };
    for (a, b, want) in cases
    {
        let got = beta(a, b);
        assert!(
            rel_err(got, want) < tol,
            "beta({a}, {b}) = {got:e}, want {want:e}"
        );
    }
}

/// At a non-positive integer `a = −m` with a positive integer `b ≤ m`, the
/// poles of `Γ(a)` and `Γ(a+b)` cancel and `B(−m, n) = (−1)ⁿ·B(m−n+1, n)` is
/// finite. `beta` returned `NaN` (`∞ − ∞` in the log sum) and `ln_beta` `NaN`.
#[test]
fn beta_takes_the_finite_limit_at_cancelling_poles() {
    let cases = [
        (-3.0, 2.0, 1.0 / 6.0, -1.791_759_469_228_055),
        (-4.0, 3.0, -1.0 / 12.0, -2.484_906_649_788_000_4),
        (-5.0, 1.0, -0.2, -1.609_437_912_434_100_3),
        (
            -20.0,
            7.0,
            -1.842_842_400_117_942e-6,
            -13.204_201_395_619_961,
        ),
        (-0.0, 0.0, f64::INFINITY, f64::INFINITY), // a genuine pole
    ];
    for (a, b, want, want_ln) in cases
    {
        for (x, y) in [(a, b), (b, a)]
        {
            let got = beta(x, y);
            let lgot = ln_beta(x, y);
            if want.is_infinite()
            {
                assert_eq!(got, f64::INFINITY, "beta({x}, {y})");
                assert_eq!(lgot, f64::INFINITY, "ln_beta({x}, {y})");
                continue;
            }
            assert!(
                rel_err(got, want) < 1e-14,
                "beta({x}, {y}) = {got:e}, want {want:e}"
            );
            assert!(
                rel_err(lgot, want_ln) < 1e-14,
                "ln_beta({x}, {y}) = {lgot}, want {want_ln}"
            );
        }
    }
}

/// Genuine poles and zeros of the continuation.
#[test]
fn beta_poles_and_zeros() {
    // Γ(a) or Γ(b) has a pole that Γ(a+b) does not cancel.
    for (a, b) in [
        (-1.0, 0.5),
        (0.0, 2.5),
        (-2.0, 3.0),
        (-1.0, 3.0),
        (-2.0, -3.0),
    ]
    {
        assert_eq!(beta(a, b), f64::INFINITY, "beta({a}, {b})");
        assert_eq!(ln_beta(a, b), f64::INFINITY, "ln_beta({a}, {b})");
    }
    // Γ(a+b) has a pole and Γ(a), Γ(b) do not: B = 0.
    for (a, b) in [(-0.5, -0.5), (0.25, -1.25), (-2.5, 0.5)]
    {
        assert_eq!(beta(a, b), 0.0, "beta({a}, {b})");
        assert_eq!(ln_beta(a, b), f64::NEG_INFINITY, "ln_beta({a}, {b})");
    }
}

/// `B(+∞, b) = lim Γ(b)·q^(−b)`: `0` for `b > 0`. Both functions returned
/// `NaN` (`ln Γ(∞) − ln Γ(∞)`).
#[test]
fn beta_with_an_infinite_argument() {
    for b in [1e-300, 0.5, 3.0, 1e300]
    {
        assert_eq!(beta(f64::INFINITY, b), 0.0, "beta(inf, {b})");
        assert_eq!(beta(b, f64::INFINITY), 0.0, "beta({b}, inf)");
        assert_eq!(ln_beta(f64::INFINITY, b), f64::NEG_INFINITY);
        assert_eq!(ln_beta(b, f64::INFINITY), f64::NEG_INFINITY);
    }
    assert!(beta(f64::NAN, 1.0).is_nan());
    assert!(beta(f64::NEG_INFINITY, 1.0).is_nan());
    assert!(ln_beta(f64::NEG_INFINITY, 1.0).is_nan());
}
