//! Expanded uncertainty — the reporting step of the GUM.
//!
//! [`crate::gum`] gives the combined standard uncertainty `u_c`. To state a
//! result as `y ± U` with a stated **coverage probability** `p` (usually 95 %),
//! the GUM multiplies `u_c` by a coverage factor `k`:
//!
//! ```text
//! U = k·u_c ,     k = t_{(1+p)/2}(ν_eff) ,
//! ```
//!
//! the Student-`t` quantile at the **effective degrees of freedom** given by the
//! Welch–Satterthwaite formula from each component's contribution `uᵢ` and its
//! own degrees of freedom `νᵢ`:
//!
//! ```text
//! ν_eff = u_c⁴ / Σ (uᵢ⁴ / νᵢ) ,     u_c² = Σ uᵢ² .
//! ```
//!
//! A Type B component believed exact takes `νᵢ = ∞` (no contribution to the
//! denominator), and as `ν_eff → ∞` the factor tends to the normal quantile
//! (`k → 1.96` at 95 %). The `t`-quantile inverts the Student-`t` CDF for
//! `ν_eff ≤ 1e4` (any real value), so small budgets such as a Type A term from
//! three readings (`ν = 2`) get the GUM Table G.2 coverage factor, and switches
//! to the Cornish–Fisher expansion only above that, where it is accurate to
//! better than `1e-13` relative.

use scirust_stats::{Distribution, Normal, StudentT};
use serde::{Deserialize, Serialize};

/// Above this many degrees of freedom the Cornish–Fisher series is used
/// instead of inverting the CDF: there its truncation error is below `1e-13`
/// relative, while the incomplete-beta inversion slowly loses digits as
/// `ν → ∞`.
const CORNISH_FISHER_MIN_DOF: f64 = 1e4;

/// Student-`t` quantile `t_p(ν)`: the value with `P(T ≤ t_p) = p` for `T`
/// Student-`t` with `ν` degrees of freedom.
///
/// For any real `ν ≤ 1e4` (integer or not) the Student-`t` CDF built on the
/// regularized incomplete beta ([`scirust_stats::StudentT`]) is inverted, so
/// the small `ν_eff` of a budget dominated by a Type A term from a few readings
/// gets the exact factor (`t_0.975(2) = 4.3027`). Larger `ν` uses the
/// Cornish–Fisher expansion about the normal quantile, which is accurate to
/// better than `1e-13` relative there; `ν = +∞` gives the normal quantile
/// itself (`k → 1.96` at 95 %). Both agree with `scipy.stats.t.ppf` to better
/// than `1e-9` relative. `p ≤ 0` / `p ≥ 1` give `∓∞`; a NaN `p`, a NaN `ν` or
/// `ν ≤ 0` (where the distribution is undefined) gives NaN.
pub fn t_quantile(p: f64, nu: f64) -> f64 {
    if p.is_nan() || nu.is_nan() || nu <= 0.0
    {
        return f64::NAN;
    }
    if nu <= CORNISH_FISHER_MIN_DOF
    {
        return StudentT::new(nu).quantile(p);
    }
    let z = Normal::standard().quantile(p);
    if nu == f64::INFINITY || !z.is_finite()
    {
        return z;
    }
    let z2 = z * z;
    let z3 = z2 * z;
    let z5 = z3 * z2;
    let z7 = z5 * z2;
    let z9 = z7 * z2;
    let g1 = (z3 + z) / 4.0;
    let g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0;
    let g3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0;
    let g4 = (79.0 * z9 + 776.0 * z7 + 1482.0 * z5 - 1920.0 * z3 - 945.0 * z) / 92160.0;
    z + (g1 + (g2 + (g3 + g4 / nu) / nu) / nu) / nu
}

/// Welch–Satterthwaite effective degrees of freedom from the uncertainty
/// components `(uᵢ, νᵢ)` (each contribution and its degrees of freedom;
/// `νᵢ = f64::INFINITY` for an assumed-exact Type B term). Returns `+∞` when no
/// component carries finite degrees of freedom.
pub fn effective_dof(components: &[(f64, f64)]) -> f64 {
    let uc2: f64 = components.iter().map(|&(u, _)| u * u).sum();
    let denom: f64 = components
        .iter()
        .filter(|&&(_, nu)| nu.is_finite() && nu > 0.0)
        .map(|&(u, nu)| u.powi(4) / nu)
        .sum();
    if denom <= 0.0
    {
        return f64::INFINITY;
    }
    (uc2 * uc2) / denom
}

/// Coverage factor `k = t_{(1+p)/2}(ν_eff)` for a two-sided coverage probability
/// `coverage_prob` (e.g. 0.95).
pub fn coverage_factor(nu_eff: f64, coverage_prob: f64) -> f64 {
    t_quantile(0.5 + 0.5 * coverage_prob, nu_eff)
}

/// A GUM expanded-uncertainty statement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpandedUncertainty {
    /// Combined standard uncertainty `u_c = √(Σ uᵢ²)`.
    pub combined: f64,
    /// Effective degrees of freedom `ν_eff`.
    pub effective_dof: f64,
    /// Coverage factor `k`.
    pub coverage_factor: f64,
    /// Expanded uncertainty `U = k·u_c`.
    pub expanded: f64,
    /// Lower coverage limit `value − U`.
    pub lower: f64,
    /// Upper coverage limit `value + U`.
    pub upper: f64,
}

/// Full expanded-uncertainty statement for a measured `value` from its
/// uncertainty budget `components` (`(uᵢ, νᵢ)`) at coverage probability
/// `coverage_prob`.
pub fn expanded_uncertainty(
    value: f64,
    components: &[(f64, f64)],
    coverage_prob: f64,
) -> ExpandedUncertainty {
    let combined = components.iter().map(|&(u, _)| u * u).sum::<f64>().sqrt();
    let nu_eff = effective_dof(components);
    let k = coverage_factor(nu_eff, coverage_prob);
    let u = k * combined;
    ExpandedUncertainty {
        combined,
        effective_dof: nu_eff,
        coverage_factor: k,
        expanded: u,
        lower: value - u,
        upper: value + u,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_quantile_matches_tables() {
        // Student-t 0.975 quantiles (two-sided 95 %).
        assert!((t_quantile(0.975, 10.0) - 2.2281).abs() < 5e-3);
        assert!((t_quantile(0.975, 20.0) - 2.0860).abs() < 5e-3);
        assert!((t_quantile(0.975, 30.0) - 2.0423).abs() < 5e-3);
        assert!((t_quantile(0.975, 60.0) - 2.0003).abs() < 5e-3);
        // ν → ∞ tends to the normal quantile.
        assert!((t_quantile(0.975, f64::INFINITY) - 1.95996).abs() < 1e-4);
        assert!((t_quantile(0.975, 1e9) - 1.95996).abs() < 1e-3);
    }

    #[test]
    fn welch_satterthwaite_effective_dof() {
        // u=[0.3 (ν=5), 0.4 (exact)] ⇒ u_c=0.5, ν_eff = 0.5⁴/(0.3⁴/5) = 38.58.
        let nu = effective_dof(&[(0.3, 5.0), (0.4, f64::INFINITY)]);
        assert!((nu - 38.58).abs() < 0.1, "nu_eff {nu}");
        // All-exact budget ⇒ infinite dof.
        assert!(effective_dof(&[(0.1, f64::INFINITY), (0.2, f64::INFINITY)]).is_infinite());
    }

    #[test]
    fn expanded_uncertainty_is_k_times_uc() {
        let e = expanded_uncertainty(100.0, &[(0.3, 5.0), (0.4, f64::INFINITY)], 0.95);
        assert!((e.combined - 0.5).abs() < 1e-12);
        // k ≈ t_0.975(38.6) ≈ 2.02.
        assert!(
            (e.coverage_factor - 2.02).abs() < 0.03,
            "k {}",
            e.coverage_factor
        );
        assert!((e.expanded - e.coverage_factor * e.combined).abs() < 1e-12);
        assert!((e.upper - e.lower - 2.0 * e.expanded).abs() < 1e-12);
        assert!(e.lower < 100.0 && e.upper > 100.0);
    }

    /// Relative-error check (never bit-for-bit).
    fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
        let rel = ((got - want) / want).abs();
        assert!(
            rel < tol,
            "{what}: got {got}, want {want} (rel err {rel:e})"
        );
    }

    #[test]
    fn t_quantile_is_exact_for_small_dof() {
        // Reference: scipy.stats.t.ppf(p, nu), SciPy 1.18.1. Small and
        // non-integer ν are where the Cornish–Fisher series used to fall short
        // (ν = 1 at 0.975 gave 11.30 instead of 12.71).
        let cases = [
            (0.975, 1.0, 12.706204736174694),
            (0.975, 2.0, 4.302652729749462),
            (0.975, 3.0, 3.1824463052837078),
            (0.975, 1.5, 6.016663104427929),
            (0.975, 2.5, 3.5746548420036817),
            (0.995, 1.0, 63.656741162871526),
            (0.995, 2.0, 9.924843200918287),
            (0.995, 3.0, 5.840909309733355),
            (0.995, 1.5, 17.820310514462804),
            (0.995, 2.5, 7.163728138948783),
            (0.975, 10.0, 2.228138851986274),
            (0.975, 60.0, 2.00029782201426),
        ];
        for (p, nu, want) in cases
        {
            assert_rel(t_quantile(p, nu), want, 1e-9, &format!("t_{p}({nu})"));
            // Symmetry: t_{1-p}(ν) = -t_p(ν).
            assert_rel(
                t_quantile(1.0 - p, nu),
                -want,
                1e-9,
                &format!("t_{}({nu})", 1.0 - p),
            );
        }
    }

    #[test]
    fn t_quantile_limits_and_invalid_inputs() {
        // ν = ∞ is the normal quantile; huge finite ν approaches it smoothly.
        let z = 1.959963984540054; // scipy.stats.norm.ppf(0.975)
        assert_rel(t_quantile(0.975, f64::INFINITY), z, 1e-12, "nu = inf");
        assert_rel(t_quantile(0.975, 1e9), z, 1e-8, "nu = 1e9");
        // Both sides of the inversion / Cornish–Fisher switch at ν = 1e4
        // (scipy.stats.t.ppf(0.975, ν)).
        assert_rel(t_quantile(0.975, 1e4), 1.960201239890626, 1e-11, "nu = 1e4");
        assert_rel(
            t_quantile(0.975, 1e5),
            1.9599877075346095,
            1e-11,
            "nu = 1e5",
        );
        assert_rel(
            t_quantile(0.9995, 1e5),
            3.2906240314119137,
            1e-11,
            "nu = 1e5, 0.9995",
        );
        assert_rel(t_quantile(0.975, 1e15), z, 1e-8, "nu = 1e15");
        assert!(t_quantile(0.5, 3.0).abs() < 1e-12);
        assert_eq!(t_quantile(1.0, 3.0), f64::INFINITY);
        assert_eq!(t_quantile(0.0, 3.0), f64::NEG_INFINITY);
        // Undefined distribution or probability: NaN, never a panic.
        assert!(t_quantile(0.975, 0.0).is_nan());
        assert!(t_quantile(0.975, -2.0).is_nan());
        assert!(t_quantile(0.975, f64::NAN).is_nan());
        assert!(t_quantile(f64::NAN, 5.0).is_nan());
    }

    #[test]
    fn coverage_factor_matches_gum_table_g2_at_small_dof() {
        // JCGM 100:2008 (GUM) Table G.2 lists k = 13.97 for ν = 1 at
        // p = 95.45 % and k = 19.21 for ν = 2 at p = 99.73 %; the precise
        // values are scipy.stats.t.ppf((1 + p)/2, ν).
        assert_rel(
            coverage_factor(1.0, 0.9545),
            13.96781148750255,
            1e-9,
            "k(1, 95.45 %)",
        );
        assert_rel(
            coverage_factor(2.0, 0.9973),
            19.206015887811365,
            1e-9,
            "k(2, 99.73 %)",
        );
    }

    #[test]
    fn expanded_uncertainty_is_not_understated_for_three_readings() {
        // A Type A term from three repeated readings has ν = 2. At 99 %
        // coverage, k = t_0.995(2) = 9.9248 (scipy.stats.t.ppf); the
        // Cornish–Fisher series used to give 9.514, i.e. U about 4 % too small.
        let e = expanded_uncertainty(10.0, &[(0.1, 2.0)], 0.99);
        assert_rel(e.effective_dof, 2.0, 1e-12, "nu_eff");
        assert_rel(e.coverage_factor, 9.924843200918287, 1e-9, "k");
        assert_rel(e.expanded, 0.9924843200918287, 1e-9, "U");
    }

    #[test]
    fn large_dof_gives_k_near_two() {
        // Many-degrees-of-freedom budget ⇒ k ≈ 1.96 at 95 %.
        let e = expanded_uncertainty(0.0, &[(1.0, 1000.0)], 0.95);
        assert!(
            (e.coverage_factor - 1.96).abs() < 0.01,
            "k {}",
            e.coverage_factor
        );
    }
}
