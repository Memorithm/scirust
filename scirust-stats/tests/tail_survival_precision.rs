//! Regression tests: far-tail survival functions must not be computed as
//! `1 − cdf`.
//!
//! `StudentT` and `Beta` used to inherit the default `Distribution::sf`,
//! `1 − cdf(x)`, which loses all relative precision once the upper tail falls
//! below `f64::EPSILON` (≈2.2e-16) and returns exactly `0` beyond that. Because
//! the t-tests and the Pearson correlation test take their p-values from
//! `StudentT::sf`, a strongly significant result reported `p = 0`.
//!
//! References are mpmath 1.3 `betainc` at 50 significant digits, evaluated on
//! the exact `f64` inputs. Tolerances are relative and loose enough to absorb
//! the documented `ln Γ` prefactor error of `regularized_incomplete_beta`; they
//! never require bit-for-bit equality.

use scirust_stats::{Beta, Distribution, StudentT, Tail, t_test_one_sample};

fn rel_err(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        got.is_finite() && rel_err(got, want) <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {:e})",
        rel_err(got, want)
    );
}

#[test]
fn student_t_sf_keeps_relative_precision_in_the_far_tail() {
    // (ν, t, P(T > t))
    let cases = [
        (10.0, 30.0, 1.980_896_171_015_662e-11),
        (10.0, 100.0, 1.224_844_777_709_915e-16),
        (20.0, 50.0, 8.766_690_224_621_145e-23),
        (50.0, 20.0, 8.274_260_973_719_818e-26),
        (1000.0, 20.0, 2.031_144_249_762_385_7e-75),
    ];
    for (nu, t, want) in cases
    {
        let d = StudentT::new(nu);
        assert_rel(d.sf(t), want, 1e-11, &format!("StudentT({nu}).sf({t})"));
        // Lower tail by symmetry goes through the same direct evaluation.
        assert_rel(d.cdf(-t), want, 1e-11, &format!("StudentT({nu}).cdf(-{t})"));
    }
}

#[test]
fn student_t_cdf_resolves_tiny_offsets_from_the_centre() {
    // cdf(t) − ½ for t = 1e-9; the old form rounded ν/(ν+t²) to 1 and
    // returned exactly ½.
    for (nu, want) in [
        (3.0, 3.675_525_969_478_614e-10),
        (30.0, 3.956_321_848_940_978e-10),
    ]
    {
        let d = StudentT::new(nu);
        assert_rel(
            d.cdf(1e-9) - 0.5,
            want,
            1e-6,
            &format!("StudentT({nu}) centre"),
        );
        assert_rel(
            0.5 - d.sf(1e-9),
            want,
            1e-6,
            &format!("StudentT({nu}) centre sf"),
        );
    }
}

#[test]
fn student_t_sf_edge_values() {
    let d = StudentT::new(4.0);
    assert_eq!(d.sf(0.0), 0.5);
    assert_eq!(d.sf(f64::INFINITY), 0.0);
    assert_eq!(d.sf(f64::NEG_INFINITY), 1.0);
    assert!(d.sf(f64::NAN).is_nan());
    assert!(d.cdf(f64::NAN).is_nan());
}

#[test]
fn t_test_p_value_is_not_rounded_to_zero() {
    // t ≈ 3.44e4 on 5 dof: the two-sided p-value is ≈3.93e-22, which used to
    // be reported as exactly 0.
    let data = [100.0, 100.01, 99.99, 100.005, 99.995, 100.002];
    let two = t_test_one_sample(&data, 0.0, Tail::TwoSided).unwrap();
    assert_rel(two.p_value, 3.933_003_443_859_855e-22, 1e-9, "two-sided p");
    let greater = t_test_one_sample(&data, 0.0, Tail::Greater).unwrap();
    assert_rel(
        greater.p_value,
        1.966_501_721_929_927_4e-22,
        1e-9,
        "greater p",
    );
}

#[test]
fn beta_sf_keeps_relative_precision_near_one() {
    // (a, b, x, P(X > x))
    let cases = [
        (2.0, 3.0, 0.9999, 3.999_699_999_998_679e-12),
        (2.0, 5.0, 0.999, 5.995_000_000_000_027e-15),
        (10.0, 10.0, 0.999, 9.162_494_537_926_708e-26),
        (0.5, 0.5, 0.999_999, 6.366_198_784_800_777e-4),
    ];
    for (a, b, x, want) in cases
    {
        assert_rel(
            Beta::new(a, b).sf(x),
            want,
            1e-12,
            &format!("Beta({a}, {b}).sf({x})"),
        );
    }
}

#[test]
fn beta_sf_is_not_broken_for_tiny_x_with_mass_near_zero() {
    // Beta(0.001, 1) has cdf x^0.001, so at x = 1e-200 the survival function
    // is 1 − 10^(−0.2) ≈ 0.369 even though 1 − x rounds to 1.
    assert_rel(
        Beta::new(0.001, 1.0).sf(1e-200),
        0.369_042_655_519_806_76,
        1e-12,
        "Beta(0.001, 1)",
    );
}

#[test]
fn beta_sf_edge_values() {
    let d = Beta::new(2.0, 3.0);
    assert_eq!(d.sf(-1.0), 1.0);
    assert_eq!(d.sf(0.0), 1.0);
    assert_eq!(d.sf(1.0), 0.0);
    assert_eq!(d.sf(2.0), 0.0);
    assert!(d.sf(f64::NAN).is_nan());
    // Away from the tails, cdf + sf still adds to one.
    for x in [0.05, 0.3, 0.5, 0.7, 0.95]
    {
        assert!((d.cdf(x) + d.sf(x) - 1.0).abs() <= 1e-14, "x = {x}");
    }
}
