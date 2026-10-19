//! Regression tests: regularized incomplete beta `I_x(a, b)` for a tiny
//! second shape `b` on the upper side `x ≥ (a+1)/(a+b+2)`.
//!
//! That side used to be evaluated as `1 − I_{1−x}(b, a)`. For small `b` the
//! value is `I_x(a, b) ≈ b·(…)` while the subtracted term is `≈ 1`, so the
//! relative error was `≈ ε / I_x(a, b)`: `I_{0.9}(1, 1e-17)` (exactly
//! `1 − 0.1^1e-17 ≈ 2.3026e-17`) came out as `3.2e-15`, `I_{0.9}(1, 1e-15)`
//! as `1.22e-15` instead of `2.30e-15`, `I_{0.9}(0.5, 1e-12)` was off by
//! `6e-4` relative, and `b = 1e-300` gave values of either sign of size
//! `≈ 1e-14` instead of `≈ 1e-301`.
//!
//! References are `mpmath.betainc(a, b, 0, x, regularized=True)` (mpmath 1.4,
//! 50 significant digits) on the exact `f64` inputs; for `a = 1` they are
//! also the closed form `1 − (1 − x)^b`. Tolerances are relative; nothing is
//! compared bit for bit.

use scirust_special::regularized_incomplete_beta;

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let rel = ((got - want) / want).abs();
    assert!(
        got.is_finite() && rel <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {rel:e})"
    );
}

/// `(a, b, x, I_x(a, b))`, all with `x ≥ (a+1)/(a+b+2)`.
const CASES: [(f64, f64, f64, f64); 14] = [
    (1.0, 1e-17, 0.9, 2.302585092994046e-17),
    (1.0, 1e-15, 0.9, 2.3025850929940434e-15),
    (0.5, 1e-16, 0.7, 2.419870242671891e-16),
    (3.0, 1e-14, 0.95, 1.5944822735539827e-14),
    (0.5, 1e-12, 0.9, 3.6368929184559277e-12),
    (2.0, 1e-8, 0.99, 3.6151701255413136e-08),
    (10.0, 1e-10, 0.999, 4.0877690522230083e-10),
    (100.0, 1e-9, 0.999, 1.8270036164238302e-09),
    (1e-3, 1e-10, 0.6, 1.000404301011978e-07),
    (5.0, 1e-6, 0.95, 1.1050639678637691e-06),
    (1.0, 1e-3, 0.9, 0.002299936177446683),
    (2.5, 5e-4, 0.8, 0.000310727018912539),
    (
        50.0,
        1e-300,
        0.980_769_230_769_230_7,
        2.3421909473602797e-301,
    ),
    (1e-6, 1e-4, 0.75, 0.9901000976054497),
];

#[test]
fn small_b_upper_side_matches_mpmath() {
    for &(a, b, x, want) in &CASES
    {
        assert!(
            x >= (a + 1.0) / (a + b + 2.0),
            "case must be on the upper side"
        );
        let got = regularized_incomplete_beta(a, b, x);
        assert_rel(got, want, 1e-13, &format!("I_{x}({a:e}, {b:e})"));
    }
}

#[test]
fn small_b_upper_side_is_a_probability() {
    // b = 1e-300: the old complement form returned values of either sign of
    // size ~1e-14 here.
    for &a in &[0.5, 1.0, 3.0, 50.0, 1e3, 1e6]
    {
        let thr = (a + 1.0) / (a + 1e-300 + 2.0);
        for &f in &[0.0, 0.3, 0.7, 0.99]
        {
            let x = thr + (1.0 - thr) * f;
            let got = regularized_incomplete_beta(a, 1e-300, x);
            assert!(
                (0.0..1e-290).contains(&got),
                "I_{x}({a:e}, 1e-300) = {got:e}"
            );
        }
    }
}

#[test]
fn closed_form_for_a_equal_one() {
    // I_x(1, b) = 1 − (1 − x)^b = −expm1(b·ln(1 − x)).
    for &b in &[1e-300_f64, 1e-200, 1e-30, 1e-17, 1e-12, 1e-8, 1e-5, 1e-3]
    {
        for &x in &[0.7_f64, 0.9, 0.99, 0.999_999]
        {
            let want = -(b * (1.0 - x).ln()).exp_m1();
            let got = regularized_incomplete_beta(1.0, b, x);
            assert_rel(got, want, 1e-13, &format!("I_{x}(1, {b:e})"));
        }
    }
}
