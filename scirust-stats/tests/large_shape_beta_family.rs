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
