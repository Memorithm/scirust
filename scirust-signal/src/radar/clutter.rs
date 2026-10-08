//! Radar clutter amplitude statistics.
//!
//! CFAR ([`super::cfar`]) sets a detection threshold to hold a chosen false-alarm
//! rate — *given a model of the clutter amplitude distribution*. Over calm
//! conditions that clutter is Rayleigh (the envelope of complex-Gaussian
//! returns), and a cell-averaging CFAR is optimal; over a rough sea or cluttered
//! land the amplitude becomes **spiky**, with a heavier tail that a Rayleigh
//! threshold badly under-estimates, inflating the false-alarm rate. This module
//! provides the amplitude distributions used to model that clutter and to set
//! thresholds against it:
//!
//! - **Rayleigh** — homogeneous, noise-like clutter.
//! - **Weibull** — the workhorse spiky-clutter model; its shape `c` tunes the
//!   tail (`c = 2` recovers Rayleigh, `c = 1` is exponential, `c < 2` is
//!   spikier).
//! - **Log-normal** — very spiky clutter with a long high-amplitude tail.
//!
//! Each family gives its PDF, CDF, and (where elementary) quantile in closed
//! form; the log-normal uses a self-contained error function. Dependency-free.

use std::f64::consts::PI;

/// The **error function** `erf(x)` via the Abramowitz & Stegun 7.1.26 rational
/// approximation (absolute error < 1.5·10⁻⁷).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ── Rayleigh ────────────────────────────────────────────────────────────────

/// The **Rayleigh** amplitude PDF `f(x) = (x/σ²)·e^{−x²/2σ²}` for `x ≥ 0`, scale
/// `sigma`. `0.0` for `x < 0`.
///
/// Evaluated through the normalised amplitude `z = x/σ` as `(z/σ)·e^{−z²/2}`, so
/// the result does not depend on the units of `x` and `sigma`: forming `x²` and
/// `σ²` separately underflows to `0/0` (NaN) for scales near `1e-170` and
/// overflows to `∞/∞` (NaN) near `1e170`.
pub fn rayleigh_pdf(x: f64, sigma: f64) -> f64 {
    if x < 0.0 || sigma <= 0.0
    {
        return 0.0;
    }
    let z = x / sigma;
    (z / sigma) * (-0.5 * z * z).exp()
}

/// The Rayleigh CDF `F(x) = 1 − e^{−x²/2σ²}`.
///
/// Computed as `−expm1(−z²/2)` with `z = x/σ`. This keeps full relative
/// accuracy in the lower tail, where `1 − e^{−u}` cancels (it returns exactly
/// `0.0` once `u` drops below about `1.1e-16`), and avoids the `0/0` and `∞/∞`
/// NaNs that squaring `x` and `σ` separately produces at extreme scales.
pub fn rayleigh_cdf(x: f64, sigma: f64) -> f64 {
    if x <= 0.0 || sigma <= 0.0
    {
        return 0.0;
    }
    let z = x / sigma;
    -(-0.5 * z * z).exp_m1()
}

/// The Rayleigh quantile (inverse CDF) `x = σ·√(−2·ln(1−p))` for `p ∈ [0, 1)`.
///
/// `ln(1−p)` is evaluated as `ln_1p(−p)`, so small probabilities keep full
/// relative accuracy instead of rounding `1 − p` to `1` (which returned `0.0`
/// for every `p` below about `1.1e-16`). `p` is clamped to `[0, 1 − ε]`, so
/// `p ≥ 1` returns a large finite amplitude rather than `∞`.
pub fn rayleigh_quantile(p: f64, sigma: f64) -> f64 {
    let p = p.clamp(0.0, 1.0 - f64::EPSILON);
    sigma * (-2.0 * (-p).ln_1p()).sqrt()
}

// ── Weibull ─────────────────────────────────────────────────────────────────

/// The **Weibull** amplitude PDF `f(x) = (c/b)·(x/b)^{c−1}·e^{−(x/b)^c}` for
/// `x ≥ 0`, scale `b`, shape `c`. `c = 2` is Rayleigh (with `b = σ√2`), `c = 1`
/// exponential; smaller `c` is spikier.
pub fn weibull_pdf(x: f64, scale: f64, shape: f64) -> f64 {
    if x < 0.0 || scale <= 0.0 || shape <= 0.0
    {
        return 0.0;
    }
    let z = x / scale;
    (shape / scale) * z.powf(shape - 1.0) * (-z.powf(shape)).exp()
}

/// The Weibull CDF `F(x) = 1 − e^{−(x/b)^c}`.
///
/// Computed as `−expm1(−(x/b)^c)`, which keeps full relative accuracy in the
/// lower tail where `1 − e^{−u}` cancels (and returns exactly `0.0` once `u`
/// drops below about `1.1e-16`).
pub fn weibull_cdf(x: f64, scale: f64, shape: f64) -> f64 {
    if x <= 0.0 || scale <= 0.0 || shape <= 0.0
    {
        return 0.0;
    }
    -(-(x / scale).powf(shape)).exp_m1()
}

/// The Weibull quantile `x = b·(−ln(1−p))^{1/c}` for `p ∈ [0, 1)`.
///
/// `ln(1−p)` is evaluated as `ln_1p(−p)`, so small probabilities keep full
/// relative accuracy instead of rounding `1 − p` to `1` (which returned `0.0`
/// for every `p` below about `1.1e-16`). `p` is clamped to `[0, 1 − ε]`, so
/// `p ≥ 1` returns a large finite amplitude rather than `∞`.
pub fn weibull_quantile(p: f64, scale: f64, shape: f64) -> f64 {
    let p = p.clamp(0.0, 1.0 - f64::EPSILON);
    scale * (-(-p).ln_1p()).powf(1.0 / shape)
}

// ── Log-normal ──────────────────────────────────────────────────────────────

/// The **log-normal** amplitude PDF for `x > 0` with log-mean `mu` and log-std
/// `sigma`: `f(x) = 1/(x·σ√2π)·e^{−(ln x − μ)²/2σ²}`. `0.0` for `x ≤ 0`.
pub fn lognormal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 || sigma <= 0.0
    {
        return 0.0;
    }
    let d = (x.ln() - mu) / sigma;
    1.0 / (x * sigma * (2.0 * PI).sqrt()) * (-0.5 * d * d).exp()
}

/// The log-normal CDF `F(x) = ½·[1 + erf((ln x − μ)/(σ√2))]`.
pub fn lognormal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 || sigma <= 0.0
    {
        return 0.0;
    }
    0.5 * (1.0 + erf((x.ln() - mu) / (sigma * std::f64::consts::SQRT_2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trapezoidal integral of `f` over `[lo, hi]` with `n` intervals.
    fn integrate(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
        let step = (hi - lo) / n as f64;
        let mut s = 0.5 * (f(lo) + f(hi));
        for i in 1..n
        {
            s += f(lo + i as f64 * step);
        }
        s * step
    }

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.0)).abs() < 1e-9);
        assert!((erf(10.0) - 1.0).abs() < 1e-9);
        assert!((erf(-10.0) + 1.0).abs() < 1e-9);
        assert!((erf(0.5) - 0.520_499_9).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-6);
        assert!((erf(-0.7) + erf(0.7)).abs() < 1e-12); // odd symmetry
    }

    #[test]
    fn rayleigh_cdf_pdf_and_quantile_are_consistent() {
        let sigma = 2.0;
        // CDF runs 0 → 1 monotonically.
        assert_eq!(rayleigh_cdf(0.0, sigma), 0.0);
        assert!(rayleigh_cdf(1.0, sigma) < rayleigh_cdf(3.0, sigma));
        assert!(rayleigh_cdf(100.0, sigma) > 0.999);
        // Quantile inverts the CDF.
        let q = rayleigh_quantile(0.7, sigma);
        assert!((rayleigh_cdf(q, sigma) - 0.7).abs() < 1e-9);
        // PDF integrates to 1.
        let mass = integrate(0.0, 40.0, 20_000, |x| rayleigh_pdf(x, sigma));
        assert!((mass - 1.0).abs() < 1e-4, "mass {mass}");
    }

    #[test]
    fn weibull_shape_two_recovers_rayleigh() {
        // Weibull(shape = 2, scale = σ√2) is exactly Rayleigh(σ).
        let sigma = 1.5;
        let scale = sigma * std::f64::consts::SQRT_2;
        for &x in &[0.3, 1.0, 2.5, 4.0]
        {
            assert!((weibull_pdf(x, scale, 2.0) - rayleigh_pdf(x, sigma)).abs() < 1e-12);
            assert!((weibull_cdf(x, scale, 2.0) - rayleigh_cdf(x, sigma)).abs() < 1e-12);
        }
    }

    #[test]
    fn weibull_quantile_inverts_and_lower_shape_is_spikier() {
        let scale = 1.0;
        let q = weibull_quantile(0.9, scale, 1.5);
        assert!((weibull_cdf(q, scale, 1.5) - 0.9).abs() < 1e-9);
        // A smaller shape ⇒ heavier tail ⇒ larger high quantile (spikier clutter).
        let spiky = weibull_quantile(0.999, scale, 0.8);
        let mild = weibull_quantile(0.999, scale, 2.0);
        assert!(spiky > mild, "spiky {spiky} vs mild {mild}");
        // PDF integrates to 1.
        let mass = integrate(0.0, 60.0, 40_000, |x| weibull_pdf(x, scale, 1.2));
        assert!((mass - 1.0).abs() < 1e-3, "mass {mass}");
    }

    #[test]
    fn lognormal_cdf_is_a_valid_distribution() {
        let (mu, sigma) = (0.0, 0.5);
        assert_eq!(lognormal_cdf(0.0, mu, sigma), 0.0);
        // Median is e^μ, where the CDF is 0.5.
        assert!((lognormal_cdf(mu.exp(), mu, sigma) - 0.5).abs() < 1e-6);
        assert!(lognormal_cdf(1.0, mu, sigma) < lognormal_cdf(5.0, mu, sigma));
        assert!(lognormal_cdf(1e6, mu, sigma) > 0.999);
        // PDF integrates to 1.
        let mass = integrate(1e-4, 60.0, 60_000, |x| lognormal_pdf(x, mu, sigma));
        assert!((mass - 1.0).abs() < 1e-3, "mass {mass}");
    }

    /// Relative error `|a − b| / |b|`.
    /// Tolerances are loose (1e-11) so Miri's float noise cannot trip them;
    /// the old formulas were off by 8e-8 to 1e-4, or returned 0.0 / NaN.
    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn lower_tail_cdfs_keep_relative_accuracy() {
        // F(x) ≈ x²/2σ² for x ≪ σ. The old `1 − e^{−u}` form returned 0.0
        // here and lost about 8e-8 relative accuracy at u = 1e-9.
        let x = 1e-9;
        let exact = 0.5 * x * x; // σ = 1; the next series term is ~1e-37 relative
        assert!(rayleigh_cdf(x, 1.0) > 0.0);
        assert!(rel(rayleigh_cdf(x, 1.0), exact) < 1e-11);
        // Weibull with u = (x/b)^c = 1e-9: F = u − u²/2 + …
        let u: f64 = 1e-9;
        let w = weibull_cdf(1e-6, 1.0, 1.5);
        assert!(rel(w, u - 0.5 * u * u) < 1e-11, "weibull cdf {w}");
        // Exponential case (c = 1) far into the lower tail.
        assert!(rel(weibull_cdf(1e-20, 1.0, 1.0), 1e-20) < 1e-11);
    }

    #[test]
    fn lower_tail_quantiles_keep_relative_accuracy() {
        // −ln(1−p) = p + p²/2 + … ; the old `(1 − p).ln()` returned 0.0 for
        // p = 1e-20 and was ~1e-4 off in relative terms at p = 1e-12.
        for &p in &[1e-20_f64, 1e-12, 1e-8]
        {
            let neg_log = p + 0.5 * p * p;
            let r = rayleigh_quantile(p, 3.0);
            assert!(rel(r, 3.0 * (2.0 * neg_log).sqrt()) < 1e-11, "p {p}: {r}");
            let w = weibull_quantile(p, 2.0, 0.7);
            assert!(rel(w, 2.0 * neg_log.powf(1.0 / 0.7)) < 1e-11, "p {p}: {w}");
        }
        // Quantile inverts the CDF in the lower tail as well.
        for &x in &[1e-10_f64, 1e-6, 1e-3]
        {
            let p = rayleigh_cdf(x, 1.0);
            assert!(rel(rayleigh_quantile(p, 1.0), x) < 1e-11, "x {x}");
            let p = weibull_cdf(x, 1.0, 1.5);
            assert!(rel(weibull_quantile(p, 1.0, 1.5), x) < 1e-11, "x {x}");
        }
    }

    #[test]
    fn rayleigh_is_finite_and_scale_invariant_at_extreme_scales() {
        // Squaring x and σ separately gave 0/0 or ∞/∞ (NaN) at these scales.
        let cdf_one = 1.0 - (-0.5_f64).exp();
        let pdf_one = (-0.5_f64).exp();
        for &s in &[1e-170_f64, 1e-100, 1.0, 1e100, 1e170]
        {
            let c = rayleigh_cdf(s, s);
            assert!(rel(c, cdf_one) < 1e-11, "scale {s}: cdf {c}");
            // f(x; σ) = f(x/σ; 1)/σ.
            let d = rayleigh_pdf(s, s);
            assert!(
                d.is_finite() && rel(d * s, pdf_one) < 1e-11,
                "scale {s}: pdf {d}"
            );
            assert!(rel(rayleigh_quantile(cdf_one, s), s) < 1e-11, "scale {s}");
        }
    }

    #[test]
    fn negative_support_guards() {
        assert_eq!(rayleigh_pdf(-1.0, 1.0), 0.0);
        assert_eq!(weibull_pdf(-1.0, 1.0, 1.5), 0.0);
        assert_eq!(lognormal_pdf(-1.0, 0.0, 1.0), 0.0);
        assert_eq!(lognormal_pdf(0.0, 0.0, 1.0), 0.0);
    }
}
