//! Regression tests: Student-t, F and Beta densities and tails with one very
//! large shape parameter.
//!
//! `scirust_special::ln_beta`, the Student-t normalizing constant and the
//! incomplete-beta prefactor all subtracted `ln Γ` values of size `ν·ln ν`.
//! For large `ν` (or a large Beta shape) the result lost every significant
//! digit: `StudentT::new(1e15).pdf(0)` was ≈0.159 instead of `1/√(2π)`,
//! `StudentT::new(1e15).sf(1)` was ≈0.259 instead of the normal tail ≈0.1587,
//! `FisherF::new(2, 1e15).pdf(1)` was ≈3.17 instead of `e⁻¹`, and
//! `Beta::new(0.5, 1e15).cdf(1e-15)` was ≈22.9.
//!
//! References are mpmath 1.3 at 50 significant digits on the exact `f64`
//! inputs. Tolerances are relative; nothing is compared bit for bit.

use scirust_stats::{Beta, Distribution, FisherF, StudentT};

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let rel = ((got - want) / want).abs();
    assert!(
        got.is_finite() && rel <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {rel:e})"
    );
}

#[test]
fn student_t_density_normalization_for_huge_nu() {
    assert_rel(
        StudentT::new(1e15).pdf(0.0),
        0.398_942_280_401_432_6,
        1e-13,
        "t(1e15).pdf(0)",
    );
    assert_rel(
        StudentT::new(1e12).pdf(2.0),
        0.053_990_966_513_282_536,
        1e-12,
        "t(1e12).pdf(2)",
    );
}

#[test]
fn student_t_tail_for_huge_nu() {
    // Far enough from the centre that the continued fraction is well
    // conditioned; the error came from the ln Γ prefactor.
    assert_rel(
        StudentT::new(1e15).sf(1.0),
        0.158_655_253_931_457_17,
        1e-12,
        "t(1e15).sf(1)",
    );
}

#[test]
fn fisher_f_density_with_huge_denominator_df() {
    // F(2, d₂) → χ²₂/2 as d₂ → ∞, whose density at 1 is e⁻¹.
    assert_rel(
        FisherF::new(2.0, 1e15).pdf(1.0),
        0.367_879_441_171_441_95,
        1e-12,
        "F(2,1e15).pdf(1)",
    );
}

#[test]
fn beta_density_and_cdf_with_one_huge_shape() {
    let d = Beta::new(0.5, 1e15);
    assert_rel(
        d.pdf(1e-15),
        2.075_537_487_102_974_3e14,
        1e-12,
        "Beta(0.5,1e15).pdf(1e-15)",
    );
    assert_rel(
        d.cdf(1e-15),
        0.842_700_792_949_714_9,
        1e-12,
        "Beta(0.5,1e15).cdf(1e-15)",
    );
}

/// Regression test: for large `ν` the Student-t cdf/sf lost the digits of
/// `1 − x`, `x = ν/(ν + t²)`, which `regularized_incomplete_beta` only sees
/// through `x` itself (relative error ≈ `ε·ν/t²`). `StudentT::new(1e16)
/// .cdf(1.96)` was 0.98356 (true 0.97500), `ν = 1e17` gave a cdf of 1.763
/// and a negative sf, `ν ≥ 1e50` gave NaN, and the 0.975 quantile at
/// `ν = 1e16` was √3 ≈ 1.732 instead of ≈ 1.95996. `ν = +∞` (the standard
/// normal limit) gave NaN for the cdf, sf, pdf and variance.
///
/// References: mpmath 1.4.1 `betainc(ν/2, ½, 0, ν/(ν+t²))/2` at 80
/// significant digits; for `ν ≥ 1e50` and `ν = ∞` the normal tail
/// `erfc(t/√2)/2` (the `O(t⁴/ν)` difference is far below `f64` precision).
#[test]
fn student_t_cdf_and_sf_for_huge_nu_near_the_centre() {
    // (ν, t, P(T > t)); the old relative errors were 2.1e-11, 1.6e-7,
    // 3.5e-6, 3.8e-3, 0.69, 31 (negative sf) and NaN.
    let cases = [
        (1e6, 5.0, 2.866_998_935_445_371e-7),
        (1e10, 1.96, 0.024_997_895_162_084_872),
        (1e12, 10.0, 7.619_853_043_589_387e-24),
        (1e15, 1.96, 0.024_997_895_148_220_575),
        (1e16, 3.0, 0.001_349_898_031_630_097_9),
        (1e17, 1.96, 0.024_997_895_148_220_44),
        (1e300, 5.0, 2.866_515_718_791_939e-7),
        (f64::INFINITY, 1.96, 0.024_997_895_148_220_436),
    ];
    for (nu, t, want) in cases
    {
        let d = StudentT::new(nu);
        assert_rel(d.sf(t), want, 1e-13, &format!("t({nu:e}).sf({t})"));
        assert_rel(d.cdf(-t), want, 1e-13, &format!("t({nu:e}).cdf(-{t})"));
        assert_rel(d.cdf(t), 1.0 - want, 1e-15, &format!("t({nu:e}).cdf({t})"));
    }
    // Every cdf is a probability, and cdf + sf = 1.
    for nu in [1e15, 1e16, 1e17, 1e20, 1e100, f64::MAX, f64::INFINITY]
    {
        let d = StudentT::new(nu);
        for t in [-30.0, -3.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 3.0, 30.0]
        {
            let (c, s) = (d.cdf(t), d.sf(t));
            assert!(
                (0.0..=1.0).contains(&c) && (0.0..=1.0).contains(&s),
                "t({nu:e}): cdf({t}) = {c}, sf({t}) = {s}"
            );
            assert!((c + s - 1.0).abs() <= 1e-15, "t({nu:e}) at {t}");
        }
    }
    // The centre keeps its relative precision: cdf(1e-9) − ½ ≈ φ(0)·1e-9.
    assert_rel(
        StudentT::new(1e10).cdf(1e-9) - 0.5,
        3.989_422_804_014_327e-10,
        1e-6,
        "t(1e10) centre",
    );
    // ν = +∞ is the standard normal: pdf(0) = 1/√(2π), variance 1.
    let normal = StudentT::new(f64::INFINITY);
    assert_rel(
        normal.pdf(0.0),
        0.398_942_280_401_432_7,
        1e-15,
        "t(inf).pdf(0)",
    );
    assert_rel(normal.variance(), 1.0, 1e-15, "t(inf).variance()");
    // Quantile (bisection on cdf/sf). mpmath: 1.95996398454005447 at
    // ν = 1e16 (the normal 0.975 quantile is 1.95996398454005424).
    assert_rel(
        StudentT::new(1e16).quantile(0.975),
        1.959_963_984_540_054_5,
        1e-12,
        "t(1e16).quantile(0.975)",
    );
}
