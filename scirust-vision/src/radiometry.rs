//! Thermal-infrared radiometry and sensor performance metrics.
//!
//! The [`optics`](crate::optics) module characterises an imager's *spatial*
//! response (PSF, MTF). This module adds its *radiometric* and *sensitivity*
//! counterparts — the physics that decides how small a temperature difference an
//! EO/IR sensor can see:
//!
//! - **Radiometry** — Planck's law for spectral radiance, the Stefan–Boltzmann
//!   total exitance `σT⁴` and its derivative, Wien's peak-wavelength law, and
//!   band-integrated radiance / thermal contrast obtained by quadrature.
//! - **Sensitivity** — **NETD** (noise-equivalent temperature difference: the
//!   ΔT that produces a signal equal to the detector noise) from the sensor's
//!   f-number, detector specific-detectivity `D*`, and in-band thermal contrast,
//!   and **MRTD** (minimum resolvable temperature difference), the NETD-over-MTF
//!   trade-off that combines this thermal sensitivity with the [`optics`] spatial
//!   response into the headline thermal-imager spec.
//!
//! [`optics`]: crate::optics
//! Dependency-free.

use std::f64::consts::PI;

/// Planck constant `h` (J·s).
const PLANCK_H: f64 = 6.626_070_15e-34;
/// Speed of light `c` (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;
/// Boltzmann constant `k_B` (J/K).
const K_B: f64 = 1.380_649e-23;
/// Stefan–Boltzmann constant `σ` (W·m⁻²·K⁻⁴).
const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
/// Wien displacement constant `b` (m·K).
const WIEN_B: f64 = 2.897_771_955e-3;

/// Spectral radiance of a blackbody, Planck's law
/// `L(λ, T) = 2hc²/λ⁵ · 1/(exp(hc/λk_BT) − 1)` in W·m⁻²·sr⁻¹·m⁻¹, for wavelength
/// `wavelength` (m) and temperature `temperature` (K).
///
/// The Bose–Einstein factor `1/(eˣ − 1)` is evaluated as `e⁻ˣ/(1 − e⁻ˣ)` with
/// `exp_m1`, so the Wien tail (`x = hc/λk_BT` above ≈ 709, where `eˣ`
/// overflows `f64`) still yields the small but representable radiance instead
/// of a flushed zero, and the Rayleigh–Jeans limit (`x → 0`) keeps full
/// relative precision instead of cancelling in `eˣ − 1`. Non-positive
/// wavelength or temperature yields `0`.
pub fn planck_radiance(wavelength: f64, temperature: f64) -> f64 {
    if wavelength <= 0.0 || temperature <= 0.0
    {
        return 0.0;
    }
    let c1 = 2.0 * PLANCK_H * C_LIGHT * C_LIGHT;
    let x = PLANCK_H * C_LIGHT / (wavelength * K_B * temperature);
    let prefactor = c1 / wavelength.powi(5);
    if x < WIEN_TAIL_X
    {
        let occupancy = bose_einstein_occupancy(x);
        if !occupancy.is_finite()
        {
            // `x` underflowed to zero: use the Rayleigh–Jeans form `2ck_BT/λ⁴`.
            return 2.0 * C_LIGHT * K_B * temperature / wavelength.powi(4);
        }
        return prefactor * occupancy;
    }
    // Wien tail: `e⁻ˣ` alone would be subnormal (or zero) even though the
    // product with the prefactor is representable, so apply it in two halves
    // to keep every intermediate normal. Here `1 − e⁻ˣ` is 1 to full precision.
    let half = (-0.5 * x).exp();
    if half == 0.0
    {
        return 0.0;
    }
    if prefactor.is_finite()
    {
        return prefactor * half * half / -(-x).exp_m1();
    }
    // `λ⁵` underflowed (absurdly short wavelength): work in log space.
    (c1.ln() - 5.0 * wavelength.ln() - x).exp() / -(-x).exp_m1()
}

/// Above this `x = hc/λk_BT`, `e⁻ˣ` approaches the subnormal range, so
/// [`planck_radiance`] splits the exponential to keep relative precision.
const WIEN_TAIL_X: f64 = 600.0;

/// The Bose–Einstein occupancy `1/(eˣ − 1)` for `x > 0`, written as
/// `e⁻ˣ/(1 − e⁻ˣ)` so it neither overflows for large `x` nor cancels for
/// small `x`.
fn bose_einstein_occupancy(x: f64) -> f64 {
    (-x).exp() / -(-x).exp_m1()
}

/// `x·eˣ/(eˣ − 1) = x/(1 − e⁻ˣ)` for `x ≥ 0`, the dimensionless factor in
/// `∂L/∂T`; tends to `1` as `x → 0` and to `x` for large `x`.
fn planck_dt_factor(x: f64) -> f64 {
    if x == 0.0
    {
        return 1.0;
    }
    x / -(-x).exp_m1()
}

/// The temperature derivative `∂L/∂T` of the Planck spectral radiance, in
/// W·m⁻²·sr⁻¹·m⁻¹·K⁻¹. Analytic: `∂L/∂T = L · x·eˣ / (T·(eˣ − 1))` with
/// `x = hc/λk_BT`, evaluated as `L · x/(T·(1 − e⁻ˣ))` so it stays finite
/// (and tends to `0`) in the Wien tail where `eˣ` overflows, rather than
/// forming `∞/∞`. Non-positive wavelength or temperature yields `0`.
pub fn planck_radiance_dt(wavelength: f64, temperature: f64) -> f64 {
    if wavelength <= 0.0 || temperature <= 0.0
    {
        return 0.0;
    }
    let radiance = planck_radiance(wavelength, temperature);
    if radiance == 0.0
    {
        // Radiance underflowed in the deep Wien tail (where `x` may even be
        // infinite); the derivative is equally negligible.
        return 0.0;
    }
    let x = PLANCK_H * C_LIGHT / (wavelength * K_B * temperature);
    radiance * planck_dt_factor(x) / temperature
}

/// The hemispherical radiant exitance of a blackbody, Stefan–Boltzmann
/// `M = σT⁴` (W·m⁻²).
pub fn radiant_exitance(temperature: f64) -> f64 {
    STEFAN_BOLTZMANN * temperature.powi(4)
}

/// The temperature derivative of the radiant exitance, `dM/dT = 4σT³`
/// (W·m⁻²·K⁻¹).
pub fn exitance_derivative(temperature: f64) -> f64 {
    4.0 * STEFAN_BOLTZMANN * temperature.powi(3)
}

/// The wavelength of peak spectral exitance, Wien's displacement law
/// `λ_peak = b/T` (m).
pub fn peak_wavelength(temperature: f64) -> f64 {
    if temperature <= 0.0
    {
        return f64::INFINITY;
    }
    WIEN_B / temperature
}

/// Trapezoidal integral of `f` over `[lo, hi]` with `n` intervals.
fn integrate(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
    if n == 0 || hi <= lo
    {
        return 0.0;
    }
    let step = (hi - lo) / n as f64;
    let mut sum = 0.5 * (f(lo) + f(hi));
    for i in 1..n
    {
        sum += f(lo + i as f64 * step);
    }
    sum * step
}

/// The in-band radiance `∫_{λ₁}^{λ₂} L(λ, T) dλ` (W·m⁻²·sr⁻¹) by `n`-interval
/// quadrature over the wavelength band `[lambda_lo, lambda_hi]` (m).
pub fn band_radiance(lambda_lo: f64, lambda_hi: f64, temperature: f64, n: usize) -> f64 {
    integrate(lambda_lo, lambda_hi, n, |l| planck_radiance(l, temperature))
}

/// The in-band **thermal contrast** `∫_{λ₁}^{λ₂} ∂L/∂T dλ`
/// (W·m⁻²·sr⁻¹·K⁻¹) — the rate at which in-band radiance rises with target
/// temperature, the quantity that drives an infrared sensor's sensitivity.
pub fn thermal_contrast(lambda_lo: f64, lambda_hi: f64, temperature: f64, n: usize) -> f64 {
    integrate(lambda_lo, lambda_hi, n, |l| {
        planck_radiance_dt(l, temperature)
    })
}

/// **NETD** — the noise-equivalent temperature difference (K), the target/
/// background ΔT that produces a signal equal to the detector noise:
///
/// `NETD = 4·F² · √Δf / (π · √A_d · τ_o · D* · (∂L/∂T)_band)`
///
/// from the optics f-number `f_number`, detector area `detector_area` (m²),
/// noise-equivalent bandwidth `noise_bandwidth` (Hz), specific detectivity
/// `d_star` (m·√Hz·W⁻¹), optical transmission `tau_optics`, and the in-band
/// thermal contrast `contrast` (from [`thermal_contrast`]). Smaller is better.
/// Non-positive/degenerate inputs yield `f64::INFINITY`.
pub fn netd(
    f_number: f64,
    detector_area: f64,
    noise_bandwidth: f64,
    d_star: f64,
    tau_optics: f64,
    contrast: f64,
) -> f64 {
    let denom = PI * detector_area.sqrt() * tau_optics * d_star * contrast;
    if denom <= 0.0 || detector_area <= 0.0
    {
        return f64::INFINITY;
    }
    4.0 * f_number * f_number * noise_bandwidth.sqrt() / denom
}

/// **MRTD** — the minimum resolvable temperature difference (K) at a spatial
/// frequency whose system MTF is `mtf`: the thermal-sensitivity/resolution
/// trade-off `MRTD = k · NETD / MTF`, where the perception factor `k` folds the
/// observer SNR threshold and the eye/temporal integration geometry. As the MTF
/// rolls off toward zero the resolvable ΔT diverges. `mtf ≤ 0` yields
/// `f64::INFINITY`.
pub fn mrtd(netd: f64, mtf: f64, perception: f64) -> f64 {
    if mtf <= 0.0
    {
        return f64::INFINITY;
    }
    perception * netd / mtf
}

#[cfg(test)]
mod tests {
    use super::*;

    // Common infrared bands (m): MWIR 3–5 µm, LWIR 8–12 µm.
    const LWIR: (f64, f64) = (8e-6, 12e-6);

    #[test]
    fn planck_integral_recovers_stefan_boltzmann() {
        // Integrating spectral radiance over (almost) all wavelengths and
        // multiplying by π (Lambertian hemisphere) recovers M = σT⁴.
        let t = 300.0;
        let l_total = band_radiance(1e-7, 2e-4, t, 20_000);
        let m = PI * l_total;
        let expect = radiant_exitance(t);
        assert!((m - expect).abs() / expect < 1e-3, "{m} vs {expect}");
    }

    #[test]
    fn exitance_and_its_derivative_match_closed_forms() {
        let t = 320.0;
        assert!((radiant_exitance(t) - STEFAN_BOLTZMANN * t.powi(4)).abs() < 1e-9);
        // dM/dT = 4σT³ = 4·M/T; also matches a central finite difference.
        let analytic = exitance_derivative(t);
        assert!((analytic - 4.0 * radiant_exitance(t) / t).abs() < 1e-9);
        let fd = (radiant_exitance(t + 0.01) - radiant_exitance(t - 0.01)) / 0.02;
        assert!(
            (analytic - fd).abs() / analytic < 1e-4,
            "{analytic} vs {fd}"
        );
    }

    #[test]
    fn wien_peak_shifts_inversely_with_temperature() {
        // Hotter ⇒ shorter peak wavelength; and the Planck curve peaks there.
        let (cool, hot) = (300.0, 600.0);
        let (pc, ph) = (peak_wavelength(cool), peak_wavelength(hot));
        assert!(
            (pc / ph - 2.0).abs() < 1e-9,
            "peak should halve when T doubles"
        );
        let peak = peak_wavelength(cool);
        let at_peak = planck_radiance(peak, cool);
        assert!(at_peak > planck_radiance(0.5 * peak, cool));
        assert!(at_peak > planck_radiance(2.0 * peak, cool));
    }

    #[test]
    fn planck_dt_matches_a_finite_difference() {
        let (l, t) = (10e-6, 300.0);
        let analytic = planck_radiance_dt(l, t);
        let fd = (planck_radiance(l, t + 0.01) - planck_radiance(l, t - 0.01)) / 0.02;
        assert!(analytic > 0.0);
        assert!(
            (analytic - fd).abs() / analytic < 1e-4,
            "{analytic} vs {fd}"
        );
    }

    #[test]
    fn thermal_contrast_is_positive_and_grows_with_temperature() {
        let cool = thermal_contrast(LWIR.0, LWIR.1, 300.0, 400);
        let warm = thermal_contrast(LWIR.0, LWIR.1, 330.0, 400);
        assert!(cool > 0.0 && warm > cool, "contrast {cool} -> {warm}");
    }

    #[test]
    fn netd_obeys_its_scaling_laws() {
        let base = netd(2.0, 4e-10, 1e5, 3e10, 0.8, 5.0);
        assert!(base.is_finite() && base > 0.0);
        // ∝ F²: doubling the f-number quadruples NETD.
        assert!((netd(4.0, 4e-10, 1e5, 3e10, 0.8, 5.0) / base - 4.0).abs() < 1e-9);
        // ∝ 1/D*: doubling detectivity halves NETD.
        assert!((netd(2.0, 4e-10, 1e5, 6e10, 0.8, 5.0) / base - 0.5).abs() < 1e-9);
        // ∝ 1/contrast: doubling thermal contrast halves NETD.
        assert!((netd(2.0, 4e-10, 1e5, 3e10, 0.8, 10.0) / base - 0.5).abs() < 1e-9);
        // ∝ 1/√A_d: quadrupling detector area halves NETD.
        assert!((netd(2.0, 16e-10, 1e5, 3e10, 0.8, 5.0) / base - 0.5).abs() < 1e-9);
        // Degenerate contrast ⇒ infinite NETD.
        assert!(netd(2.0, 4e-10, 1e5, 3e10, 0.8, 0.0).is_infinite());
    }

    /// Relative error helper for comparisons against mpmath references.
    fn rel_err(got: f64, want: f64) -> f64 {
        ((got - want) / want).abs()
    }

    #[test]
    fn planck_wien_tail_stays_finite_where_exp_overflows() {
        // λ = 1 µm, T = 20 K ⇒ x = hc/λk_BT ≈ 719.4 > ln(f64::MAX) ≈ 709.8, so
        // `exp(x)` overflows. Before the fix the radiance flushed to 0 and
        // ∂L/∂T became `0 · ∞/∞ = NaN`. References: mpmath, 50 digits.
        let (l, t) = (1e-6, 20.0);
        let radiance = planck_radiance(l, t);
        let derivative = planck_radiance_dt(l, t);
        assert!(
            rel_err(radiance, 4.461_677_095_938_368_5e-299) < 1e-12,
            "L = {radiance}"
        );
        assert!(
            rel_err(derivative, 1.604_839_460_131_256_3e-297) < 1e-12,
            "dL/dT = {derivative}"
        );
    }

    #[test]
    fn thermal_contrast_over_a_wide_band_is_finite_for_cold_scenes() {
        // A wide band starting in the UV at T = 200 K puts the first
        // quadrature node at x ≈ 719 (overflowing `exp`). Before the fix one
        // NaN sample poisoned the whole integral, and NETD turned NaN too.
        // Over (almost) all wavelengths ∫ ∂L/∂T dλ = (dM/dT)/π = 4σT³/π.
        let t = 200.0;
        let contrast = thermal_contrast(1e-7, 2e-4, t, 20_000);
        assert!(contrast.is_finite(), "contrast = {contrast}");
        let expect = exitance_derivative(t) / PI;
        assert!(rel_err(contrast, expect) < 1e-3, "{contrast} vs {expect}");
        let sensitivity = netd(2.0, 4e-10, 1e5, 3e10, 0.8, contrast);
        assert!(sensitivity.is_finite() && sensitivity > 0.0);
    }

    #[test]
    fn planck_rayleigh_jeans_limit_keeps_relative_precision() {
        // λ = 1 km, T = 10⁴ K ⇒ x ≈ 1.44e-9: `exp(x) − 1` cancels (the old code
        // was off by about 2e-9 relative here); `exp_m1` keeps full precision.
        let (l, t) = (1e3, 1e4);
        let radiance = planck_radiance(l, t);
        let derivative = planck_radiance_dt(l, t);
        assert!(
            rel_err(radiance, 8.278_163_140_949_625e-23) < 1e-12,
            "L = {radiance}"
        );
        assert!(
            rel_err(derivative, 8.278_163_146_904_84e-27) < 1e-12,
            "dL/dT = {derivative}"
        );
    }

    #[test]
    fn planck_matches_mpmath_in_the_thermal_infrared() {
        // Ordinary regime (x ≈ 4.8) is unchanged by the reformulation.
        let (l, t) = (10e-6, 300.0);
        assert!(rel_err(planck_radiance(l, t), 9_924_033.330_070_695) < 1e-12);
        assert!(rel_err(planck_radiance_dt(l, t), 159_971.567_251_321_94) < 1e-12);
    }

    #[test]
    fn planck_extreme_inputs_never_produce_nan() {
        for &(l, t) in &[(1e-70, 1.0), (1e-9, 1e-3), (1e70, 1e10), (1e-6, 1e-300)]
        {
            let radiance = planck_radiance(l, t);
            let derivative = planck_radiance_dt(l, t);
            assert!(
                !radiance.is_nan() && radiance >= 0.0,
                "L({l}, {t}) = {radiance}"
            );
            assert!(
                !derivative.is_nan() && derivative >= 0.0,
                "dL/dT({l}, {t}) = {derivative}"
            );
        }
    }

    #[test]
    fn mrtd_rises_as_the_mtf_falls() {
        let netd_val = 0.05;
        // At full MTF, MRTD = k·NETD.
        assert!((mrtd(netd_val, 1.0, 2.0) - 0.1).abs() < 1e-12);
        // Halving the MTF doubles the resolvable ΔT.
        assert!((mrtd(netd_val, 0.5, 2.0) - 0.2).abs() < 1e-12);
        // At the MTF cutoff the resolvable ΔT diverges.
        assert!(mrtd(netd_val, 0.0, 2.0).is_infinite());
    }
}
