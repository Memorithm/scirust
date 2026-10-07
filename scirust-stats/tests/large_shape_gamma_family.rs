//! Regression tests: Gamma / χ² densities and tails and Poisson tails at very
//! large shape or rate.
//!
//! `Gamma::pdf` formed `exp((k−1)·ln x − x/θ − k·ln θ − ln Γ(k))`, and the
//! regularized incomplete gamma behind `Gamma`/`ChiSquared` cdf/sf and
//! `Poisson` cdf/sf formed its prefactor the same way; the terms are of size
//! `k·ln k`, so the relative error grew like `ε·k·ln k`
//! (`Gamma::new(1e10, 1.0).pdf(1e10)` was `3.989340e-6` instead of
//! `3.989423e-6`, `Poisson::new(1e10).cdf(1e10)` was `0.4999991` instead of
//! `0.5000027`).
//!
//! References are mpmath 1.3 at 40 significant digits on the exact `f64`
//! inputs. Tolerances are relative; nothing is compared bit for bit.

use scirust_stats::{ChiSquared, DiscreteDistribution, Distribution, Gamma, Poisson};

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let rel = ((got - want) / want).abs();
    assert!(
        got.is_finite() && rel <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {rel:e})"
    );
}

#[test]
fn gamma_density_for_large_shape() {
    assert_rel(
        Gamma::new(1e4, 1.0).pdf(10050.0),
        3.504_562_865_607_79e-3,
        1e-13,
        "Gamma(1e4,1).pdf(10050)",
    );
    assert_rel(
        Gamma::new(1e6, 1.0).pdf(1e6),
        3.989_422_471_562_440_4e-4,
        1e-13,
        "Gamma(1e6,1).pdf(1e6)",
    );
    assert_rel(
        Gamma::new(1e6, 1.0).pdf(1_001_000.0),
        2.418_095_047_314_818_2e-4,
        1e-13,
        "Gamma(1e6,1).pdf(1001000)",
    );
    assert_rel(
        Gamma::new(1e10, 1.0).pdf(1e10),
        3.989_422_803_981_082e-6,
        1e-13,
        "Gamma(1e10,1).pdf(1e10)",
    );
    assert_rel(
        Gamma::new(1e12, 2.0).pdf(2e12),
        1.994_711_402_006_997_2e-7,
        1e-13,
        "Gamma(1e12,2).pdf(2e12)",
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁶ series terms: too slow under Miri.
fn gamma_and_chi_squared_cdf_for_large_shape() {
    // Gamma(k, θ).cdf(x) = P(k, x/θ); χ²(2k).cdf(2x) = P(k, x).
    assert_rel(
        Gamma::new(1e10, 3.0).cdf(3e10),
        0.500_001_329_807_601_3,
        4e-12,
        "Gamma(1e10,3).cdf(3e10)",
    );
    assert_rel(
        ChiSquared::new(2e10).cdf(2e10),
        0.500_001_329_807_601_3,
        4e-12,
        "χ²(2e10).cdf(2e10)",
    );
    assert_rel(
        ChiSquared::new(2e8).sf(200_100_000.0),
        2.878_429_686_852_781e-7,
        4e-12,
        "χ²(2e8).sf(200100000)",
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁶ series terms: too slow under Miri.
fn gamma_median_for_large_shape() {
    // Median of Gamma(a, 1) is a − 1/3 + 8/(405·a) + O(a⁻²).
    let a = 1e10;
    let want = a - 1.0 / 3.0 + 8.0 / (405.0 * a);
    assert_rel(
        Gamma::new(a, 1.0).quantile(0.5),
        want,
        1e-14,
        "Gamma(1e10,1).quantile(0.5)",
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁶ series terms: too slow under Miri.
fn poisson_tails_for_large_rate() {
    let d = Poisson::new(1e10);
    // P(X ≤ k) = Q(k + 1, λ); P(X > k) = P(k + 1, λ).
    assert_rel(
        d.cdf(10_000_000_000),
        0.500_002_659_615_202_6,
        4e-12,
        "Poisson(1e10).cdf(1e10)",
    );
    assert_rel(
        d.sf(10_000_200_000),
        2.275_013_194_727_937e-2,
        4e-12,
        "Poisson(1e10).sf(1e10 + 2e5)",
    );
}
