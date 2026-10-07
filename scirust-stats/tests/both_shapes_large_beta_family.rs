//! Regression tests: Beta density and cdf, Binomial cdf/sf and F cdf when
//! *both* shape parameters (or `n` and `k`) are large.
//!
//! The Beta density and the incomplete-beta prefactor were formed as
//! `exp(a·ln x + b·ln(1−x) − ln B(a, b))`, whose terms near the mode are of
//! size `≈ (a+b)·ln 2` while their sum is only `≈ ½·ln(a+b)`; the relative
//! error grew like `ε·(a+b)`. `Beta::new(1e12, 1e12).pdf(0.5)` was off by
//! 1.3e-5 relative, `Binomial::new(2e12, 0.5).cdf(1e12)` came out as `0.499975`
//! instead of `0.50000028`, and `FisherF::new(2e12, 2e12).cdf(1)` as
//! `0.499975` instead of `½`. Both now use the saddle-point (Loader /
//! TOMS 708) forms.
//!
//! References are mpmath 1.4 at 60 significant digits on the exact `f64`
//! inputs (closed-form `ln Γ` for densities; quadrature of the density on
//! ±80 standard deviations around the mode for the cdfs). Tolerances are
//! relative; nothing is compared bit for bit.

use scirust_stats::discrete::{Binomial, DiscreteDistribution};
use scirust_stats::{Beta, Distribution, FisherF};

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let rel = ((got - want) / want).abs();
    assert!(
        got.is_finite() && rel <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {rel:e})"
    );
}

#[test]
fn beta_density_with_both_shapes_large() {
    for &(a, b, x, want) in &[
        (1e12, 1e12, 0.5, 1_128_379.167_095_371_5),
        (1e10, 2e10, 1.0 / 3.0, 146_580.753_569_450_9),
        (1e8, 3e8, 0.25002, 12_026.074_038_032_89),
        (1e6, 1e6, 0.5005, 415.107_653_086_096_04),
    ]
    {
        // 1e-12: away from the mode the density's own sensitivity to one
        // ulp of `x` is already ≈ 1e-12 at these shapes.
        assert_rel(
            Beta::new(a, b).pdf(x),
            want,
            1e-12,
            &format!("Beta({a:e}, {b:e}).pdf({x})"),
        );
    }
}

#[test]
fn beta_density_small_and_mixed_shapes_unchanged() {
    // One shape at or below 2 keeps the direct form; small shapes are
    // already accurate either way.
    assert_rel(
        Beta::new(5.0, 7.0).pdf(0.3),
        2.201_330_439,
        1e-13,
        "Beta(5, 7).pdf(0.3)",
    );
    assert_rel(
        Beta::new(2.5, 1e9).pdf(2.4e-9),
        253_731_219.535_710_36,
        1e-12,
        "Beta(2.5, 1e9).pdf(2.4e-9)",
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁴ continued-fraction steps: too slow under Miri.
fn beta_cdf_with_both_shapes_large() {
    let d = Beta::new(1e10, 2e10);
    assert_rel(
        d.cdf(1.0 / 3.0),
        0.500_000_542_888_967_6,
        1e-10,
        "Beta(1e10, 2e10).cdf(1/3)",
    );
    assert_rel(
        d.sf(1.0 / 3.0),
        0.499_999_457_111_032_4,
        1e-10,
        "Beta(1e10, 2e10).sf(1/3)",
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~10³–4·10⁴ continued-fraction steps: too slow under Miri.
fn binomial_cdf_and_sf_with_huge_n() {
    for &(n, p, k, cdf, sf, tol) in &[
        (
            2_000_000_000_000_u64,
            0.5,
            1_000_000_000_000_u64,
            0.500_000_282_094_791_8,
            0.499_999_717_905_208_2,
            1e-9,
        ),
        (
            30_000_000_000,
            1.0 / 3.0,
            10_000_000_000,
            0.500_002_714_469_248_6,
            0.499_997_285_530_751_4,
            1e-10,
        ),
        (
            10_000_000_000,
            0.25,
            2_500_010_000,
            0.591_324_527_355_811_8,
            0.408_675_472_644_188_2,
            1e-10,
        ),
    ]
    {
        let d = Binomial::new(n, p);
        assert_rel(d.cdf(k), cdf, tol, &format!("Binomial({n}, {p}).cdf({k})"));
        assert_rel(d.sf(k), sf, tol, &format!("Binomial({n}, {p}).sf({k})"));
    }
}

#[test]
#[cfg_attr(miri, ignore)] // ~4·10⁴ continued-fraction steps: too slow under Miri.
fn fisher_f_cdf_with_both_dfs_huge() {
    // d1 = d2 and x = 1: the cdf is I_½(d/2, d/2) = ½ exactly.
    assert_rel(
        FisherF::new(2e12, 2e12).cdf(1.0),
        0.5,
        1e-9,
        "F(2e12, 2e12).cdf(1)",
    );
    assert_rel(
        FisherF::new(2e8, 2e8).cdf(1.0),
        0.5,
        1e-10,
        "F(2e8, 2e8).cdf(1)",
    );
}
