//! Non-normal statistical tolerancing.
//!
//! The **inertia** `I = √(δ² + σ²)` is *distribution-free* — it is the
//! root-mean-square deviation from target, `√(E[(X−T)²])`, and holds for any
//! distribution. What *does* depend on the distribution shape is the
//! **conformity** (out-of-spec fraction) and the classical capability indices,
//! which the [`crate::capability`] module derives under a normal assumption.
//!
//! This module lifts that assumption using the first four moments — mean `μ`,
//! standard deviation `σ`, skewness `S`, excess kurtosis `K`:
//!
//! - [`cornish_fisher_quantile`] — the Cornish–Fisher expansion, giving a
//!   distribution's `p`-quantile from its moments.
//! - [`nonnormal_ppm`] — non-conformity in ppm, by inverting the expansion at
//!   each spec limit (reduces to the normal tail when `S = K = 0`).
//! - [`clements_capability`] — the Clements (1989) percentile method for
//!   `Cp`/`Cpk` on skewed data (reduces to the classical indices for a normal
//!   process).

use crate::special::{inv_normal_cdf, normal_cdf, normal_sf};
use serde::{Deserialize, Serialize};

/// Cornish–Fisher standardised deviate `w(z)` for skewness `s` and excess
/// kurtosis `k`, so that the `p`-quantile is `μ + σ·w(Φ⁻¹(p))`:
///
/// ```text
/// w = z + (z²−1)·S/6 + (z³−3z)·K/24 − (2z³−5z)·S²/36 .
/// ```
fn cf_w(z: f64, s: f64, k: f64) -> f64 {
    z + (z * z - 1.0) * s / 6.0 + (z * z * z - 3.0 * z) * k / 24.0
        - (2.0 * z * z * z - 5.0 * z) * s * s / 36.0
}

/// Derivative `dw/dz`, which bounds the monotone branch used by
/// [`cornish_fisher_quantile`] and [`nonnormal_ppm`].
fn cf_w_prime(z: f64, s: f64, k: f64) -> f64 {
    1.0 + (2.0 * z) * s / 6.0 + (3.0 * z * z - 3.0) * k / 24.0 - (6.0 * z * z - 5.0) * s * s / 36.0
}

/// Edges `[zl, zh]` of the **monotone branch** of the Cornish–Fisher map `w`
/// around `z = 0`, i.e. the interval containing `0` on which `w'(z) > 0`,
/// capped to `[−40, 40]`. Returns `None` when `w'(0) ≤ 0` (the expansion is
/// degenerate, not increasing even at the centre).
///
/// `w'(z) = a·z² + b·z + c` with `a = K/8 − S²/6`, `b = S/3`,
/// `c = 1 − K/8 + 5S²/36`, so the edges are the roots of that quadratic nearest
/// to `0` on either side (computed with the cancellation-free form).
fn cf_branch(s: f64, k: f64) -> Option<(f64, f64)> {
    const CAP: f64 = 40.0;
    let c = cf_w_prime(0.0, s, k);
    if c <= 0.0
    {
        return None;
    }
    let a = k / 8.0 - s * s / 6.0;
    let b = s / 3.0;
    let (mut zl, mut zh) = (-CAP, CAP);
    let mut take = |r: f64| {
        if r.is_finite()
        {
            if r < 0.0 && r > zl
            {
                zl = r;
            }
            if r > 0.0 && r < zh
            {
                zh = r;
            }
        }
    };
    if a == 0.0
    {
        if b != 0.0
        {
            take(-c / b);
        }
    }
    else
    {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0
        {
            let q = -0.5 * (b + b.signum() * disc.sqrt());
            // q ≠ 0: q = 0 would need b = 0 and disc = −4ac = 0, impossible
            // with a ≠ 0 < c (`signum(0.0)` is 1).
            take(q / a);
            take(c / q);
        }
    }
    Some((zl, zh))
}

/// The Cornish–Fisher approximation of a distribution's `p`-quantile from its
/// mean `μ`, standard deviation `σ`, skewness `s` and excess kurtosis `k`:
/// `x_p = μ + σ·w(Φ⁻¹(p))`. Exact for a normal process (`s = k = 0`, where it
/// is `μ + σ·Φ⁻¹(p)`).
///
/// The map `w` is a cubic and is increasing only on its monotone branch
/// `[zl, zh]` around `z = 0` (see [`nonnormal_ppm`]); beyond it the raw
/// expansion turns back, so for strongly skewed moments a far-tail quantile
/// could land on the wrong side of a central one. `Φ⁻¹(p)` is therefore
/// clamped to that branch, which keeps the result non-decreasing in `p` and
/// consistent with the inversion used by [`nonnormal_ppm`]: quantiles beyond
/// the branch saturate at its endpoint. If the expansion is degenerate
/// (`w'(0) ≤ 0`) the plain normal quantile `μ + σ·Φ⁻¹(p)` is returned, matching
/// the fallback of [`nonnormal_ppm`].
pub fn cornish_fisher_quantile(mean: f64, sd: f64, skew: f64, ex_kurtosis: f64, p: f64) -> f64 {
    let z = inv_normal_cdf(p);
    match cf_branch(skew, ex_kurtosis)
    {
        Some((zl, zh)) => mean + sd * cf_w(z.clamp(zl, zh), skew, ex_kurtosis),
        None => mean + sd * z,
    }
}

/// Invert the Cornish–Fisher expansion: find the standard-normal deviate `z`
/// whose mapped value equals the standardised target `t = (x − μ)/σ`, i.e.
/// solve `w(z) = t`.
///
/// The map `w` is a cubic, so it is only invertible on the **monotone branch**
/// around `z = 0` (between the turning points where `w'(z) = 0`, see
/// `cf_branch`). This routine clamps a target outside its range to the
/// nearest branch endpoint — an extreme spec limit thereby yields a negligible
/// tail rather than a spurious root — and bisects for the unique root inside
/// it. If `w'(0) ≤ 0` the expansion is degenerate (very strong non-normality)
/// and it falls back to the plain normal deviate `t`.
fn cf_inverse_z(t: f64, s: f64, k: f64) -> f64 {
    let Some((zl, zh)) = cf_branch(s, k)
    else
    {
        return t; // degenerate: not monotone even at the centre
    };
    let wl = cf_w(zl, s, k);
    let wh = cf_w(zh, s, k);
    if t <= wl
    {
        return zl;
    }
    if t >= wh
    {
        return zh;
    }
    // Bisection for the unique root in [zl, zh] (w is increasing there).
    let (mut a, mut b) = (zl, zh);
    for _ in 0..200
    {
        let m = 0.5 * (a + b);
        if cf_w(m, s, k) < t
        {
            a = m;
        }
        else
        {
            b = m;
        }
        if b - a < 1e-14
        {
            break;
        }
    }
    0.5 * (a + b)
}

/// Non-conformity of a non-normal characteristic against `[lsl, usl]`, in parts
/// per million, from its first four moments. Each tail is obtained by inverting
/// the Cornish–Fisher expansion at the limit and reading the normal CDF:
///
/// ```text
/// PPM = 10⁶ · [ Φ(z_L) + (1 − Φ(z_U)) ] ,
/// w(z_L) = (LSL − μ)/σ ,  w(z_U) = (USL − μ)/σ .
/// ```
///
/// Reduces to [`crate::capability::nonconformity_ppm`] for a normal process
/// (`skew = ex_kurtosis = 0`). A degenerate `σ = 0` yields `0` inside the
/// interval, `10⁶` outside.
///
/// Cornish–Fisher is a moment-based approximation valid for **moderate**
/// skewness/kurtosis and spec limits within the distribution's bulk (a few σ,
/// the usual capability regime). For a limit far in the tail the expansion
/// leaves its valid monotone branch and the internal `cf_inverse_z` falls back
/// to the normal deviate there (a negligible-tail limit contributes ≈ 0), so the
/// result stays sane rather than diverging.
pub fn nonnormal_ppm(mean: f64, sd: f64, skew: f64, ex_kurtosis: f64, lsl: f64, usl: f64) -> f64 {
    if sd <= 0.0
    {
        return if mean >= lsl && mean <= usl { 0.0 } else { 1e6 };
    }
    let z_lo = cf_inverse_z((lsl - mean) / sd, skew, ex_kurtosis);
    let z_hi = cf_inverse_z((usl - mean) / sd, skew, ex_kurtosis);
    let below = normal_cdf(z_lo);
    let above = normal_sf(z_hi);
    ((below + above).clamp(0.0, 1.0)) * 1e6
}

/// Non-normal capability indices by the Clements (1989) percentile method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClementsCapability {
    /// `Cp = (USL − LSL) / (U_{0.99865} − L_{0.00135})` — potential capability
    /// using the estimated 6σ-equivalent percentile spread.
    pub cp: f64,
    /// `Cpk = min(Cpu, Cpl)` about the estimated median.
    pub cpk: f64,
    /// Upper index `Cpu = (USL − M) / (U_{0.99865} − M)`.
    pub cpu: f64,
    /// Lower index `Cpl = (M − LSL) / (M − L_{0.00135})`.
    pub cpl: f64,
    /// Estimated median `M` (the `p = 0.5` percentile).
    pub median: f64,
}

/// Clements' percentile capability from the first four moments and the
/// bilateral spec. The 0.135 %, 50 %, 99.865 % percentiles are estimated by the
/// Cornish–Fisher expansion (a moment-based stand-in for Clements' Pearson-curve
/// tables). Reduces to the classical `Cp`/`Cpk` for a normal process, where the
/// percentile spread is exactly `6σ`.
pub fn clements_capability(
    mean: f64,
    sd: f64,
    skew: f64,
    ex_kurtosis: f64,
    lsl: f64,
    usl: f64,
) -> ClementsCapability {
    let up = cornish_fisher_quantile(mean, sd, skew, ex_kurtosis, 0.998_65);
    let lp = cornish_fisher_quantile(mean, sd, skew, ex_kurtosis, 0.001_35);
    let m = cornish_fisher_quantile(mean, sd, skew, ex_kurtosis, 0.5);
    let spread = up - lp;
    let cp = if spread > 0.0
    {
        (usl - lsl) / spread
    }
    else
    {
        f64::INFINITY
    };
    let cpu = if up > m
    {
        (usl - m) / (up - m)
    }
    else
    {
        f64::INFINITY
    };
    let cpl = if m > lp
    {
        (m - lsl) / (m - lp)
    }
    else
    {
        f64::INFINITY
    };
    ClementsCapability {
        cp,
        cpk: cpu.min(cpl),
        cpu,
        cpl,
        median: m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{cp, cpk, nonconformity_ppm};
    use approx::assert_relative_eq;

    #[test]
    fn reduces_to_normal_quantile_when_symmetric() {
        // s = k = 0 ⇒ x_p = μ + σ·Φ⁻¹(p).
        let q = cornish_fisher_quantile(10.0, 2.0, 0.0, 0.0, 0.975);
        assert_relative_eq!(q, 10.0 + 2.0 * 1.959_963_984_540_054, epsilon = 1e-6);
    }

    #[test]
    fn ppm_reduces_to_normal_when_symmetric() {
        // Matches the normal ppm to tight tolerance for s = k = 0.
        let (mean, sd, lsl, usl) = (0.0, 1.0, -3.0, 3.0);
        let nn = nonnormal_ppm(mean, sd, 0.0, 0.0, lsl, usl);
        let normal = nonconformity_ppm(mean, sd, lsl, usl);
        assert_relative_eq!(nn, normal, epsilon = 1e-3);
    }

    #[test]
    fn clements_reduces_to_classical_indices_when_normal() {
        // Normal ⇒ percentile spread = 6σ ⇒ Cp = (USL−LSL)/(6σ), Cpk classic.
        let (mean, sd, lsl, usl) = (10.5, 1.0, 7.0, 13.0);
        let c = clements_capability(mean, sd, 0.0, 0.0, lsl, usl);
        assert_relative_eq!(c.cp, cp(sd, lsl, usl), epsilon = 1e-3);
        assert_relative_eq!(c.cpk, cpk(mean, sd, lsl, usl), epsilon = 1e-3);
        assert_relative_eq!(c.median, mean, epsilon = 1e-6);
    }

    #[test]
    fn skew_shifts_the_tails_the_right_way() {
        // Right-skew (s>0) fattens the upper tail: more nonconformity above USL
        // than a symmetric process with the same μ, σ.
        let (mean, sd, lsl, usl) = (0.0, 1.0, -3.0, 3.0);
        let sym = nonnormal_ppm(mean, sd, 0.0, 0.0, lsl, usl);
        let skewed = nonnormal_ppm(mean, sd, 1.0, 2.0, lsl, usl);
        assert!(
            skewed > sym,
            "skewed {skewed} should exceed symmetric {sym}"
        );
        // Clements Cpk drops for the skewed process (median pulled, tail fattened).
        let c = clements_capability(mean, sd, 1.0, 2.0, lsl, usl);
        assert!(c.cpk < clements_capability(mean, sd, 0.0, 0.0, lsl, usl).cpk);
    }

    #[test]
    fn cornish_fisher_inversion_round_trips() {
        // w(cf_inverse_z(t)) ≈ t.
        for &t in &[-2.5, -1.0, 0.3, 1.7]
        {
            let z = cf_inverse_z(t, 0.6, 1.5);
            assert_relative_eq!(cf_w(z, 0.6, 1.5), t, epsilon = 1e-10);
        }
    }

    #[test]
    fn upper_tail_keeps_relative_precision_far_from_the_mean() {
        // Regression: the upper tail was `1 − Φ(z)`, which loses relative
        // precision past z ≈ 5 and is exactly 0 past z ≈ 8.3. Reference values
        // (mpmath, 40 digits): 10⁶·Q(7) = 1.279812543885835e-6,
        // 10⁶·Q(9) = 1.128588405953841e-13; the lower limit at −40σ adds ~0.
        let p7 = nonnormal_ppm(0.0, 1.0, 0.0, 0.0, -40.0, 7.0);
        assert_relative_eq!(p7, 1.279_812_543_885_835e-6, max_relative = 1e-9);
        let p9 = nonnormal_ppm(0.0, 1.0, 0.0, 0.0, -40.0, 9.0);
        assert_relative_eq!(p9, 1.128_588_405_953_841e-13, max_relative = 1e-9);
        // Both tails are now treated symmetrically.
        let mirrored = nonnormal_ppm(0.0, 1.0, 0.0, 0.0, -9.0, 40.0);
        assert_relative_eq!(p9, mirrored, max_relative = 1e-12);
    }

    #[test]
    fn quantile_is_non_decreasing_in_p_for_skewed_moments() {
        // Regression: the raw cubic `w(Φ⁻¹(p))` turns back outside its monotone
        // branch. With exponential-like moments (S = 2, K = 6) the 0.135 %
        // quantile came out at −0.500σ, above the 10 % quantile (−0.877σ); with
        // S = K = 2 it was +2.50σ, above the median (−0.333σ).
        let ps = [
            1e-9,
            0.001_35,
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
            0.99,
            0.998_65,
            1.0 - 1e-9,
        ];
        for &(s, k) in &[
            (2.0, 6.0),
            (2.0, 2.0),
            (1.5, 0.5),
            (1.0, 0.0),
            (-1.5, 0.5),
            (0.5, 4.0),
        ]
        {
            let q: Vec<f64> = ps
                .iter()
                .map(|&p| cornish_fisher_quantile(0.0, 1.0, s, k, p))
                .collect();
            for w in q.windows(2)
            {
                assert!(w[0] <= w[1], "s={s} k={k}: quantiles not monotone: {q:?}");
            }
        }
        let lo = cornish_fisher_quantile(0.0, 1.0, 2.0, 6.0, 0.001_35);
        let p10 = cornish_fisher_quantile(0.0, 1.0, 2.0, 6.0, 0.1);
        assert!(lo < p10, "{lo} should lie below {p10}");
        // Clements on S = K = 2: the lower percentile now lies below the median,
        // so Cpl is finite (it was +∞, with Cp ≈ 18 from a 0.33σ "spread").
        let c = clements_capability(0.0, 1.0, 2.0, 2.0, -3.0, 3.0);
        assert!(c.cpl.is_finite() && c.cp < 2.0, "{c:?}");
    }

    #[test]
    fn quantile_round_trips_through_ppm_on_the_branch() {
        // Inside the monotone branch the quantile and `nonnormal_ppm` invert
        // each other: the lower-tail ppm at x_p is 10⁶·p.
        for &(s, k) in &[(2.0, 6.0), (1.0, 1.5), (0.6, 1.5)]
        {
            let (zl, _) = cf_branch(s, k).unwrap();
            for &p in &[0.02, 0.1, 0.3]
            {
                if inv_normal_cdf(p) <= zl
                {
                    continue; // saturated: outside the branch
                }
                let x = cornish_fisher_quantile(0.0, 1.0, s, k, p);
                let ppm = nonnormal_ppm(0.0, 1.0, s, k, x, 1e3);
                assert_relative_eq!(ppm, 1e6 * p, max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn branch_edges_are_turning_points() {
        for &(s, k) in &[(2.0, 6.0), (2.0, 2.0), (1.5, 0.5), (-1.0, 0.0), (0.0, -1.0)]
        {
            let (zl, zh) = cf_branch(s, k).unwrap();
            assert!(zl < 0.0 && zh > 0.0);
            for z in [zl, zh]
            {
                if z.abs() < 40.0
                {
                    assert!(cf_w_prime(z, s, k).abs() < 1e-12, "w'({z}) for s={s} k={k}");
                }
            }
            for i in 1..100
            {
                let z = zl + (zh - zl) * i as f64 / 100.0;
                assert!(cf_w_prime(z, s, k) > 0.0);
            }
        }
        // Normal: w' ≡ 1, no turning points.
        assert_eq!(cf_branch(0.0, 0.0), Some((-40.0, 40.0)));
    }
}
