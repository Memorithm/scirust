//! GUM uncertainty propagation.
//!
//! For a measurement model `y = f(x₁,…,xₙ)` with input standard uncertainties
//! `u(xᵢ)`, the GUM law of propagation gives the combined standard uncertainty
//! `u_c²(y) = Σ (∂f/∂xᵢ)² u²(xᵢ)` (independent inputs), with sensitivity
//! coefficients here taken by central differences. A Monte-Carlo evaluation
//! (GUM Supplement 1) cross-checks it without the linearity assumption.

/// Combined standard uncertainty by the GUM sensitivity-coefficient method.
///
/// Each sensitivity coefficient `∂f/∂xᵢ` is a central difference with step
/// `h` (in the units of `xᵢ`), so `h` should be small relative to every
/// `xᵢ` but well above its rounding (about `∛ε · |xᵢ|` balances both). A step `h` that is not finite and positive falls back to
/// `∛ε · max(|xᵢ|, uᵢ)`, a step relative to each input. The step is used as
/// given: it used to be floored to an absolute `1e-9`, which for an input in
/// small SI units (a 100 pF capacitance, `1e-10` F) moved the evaluation
/// points across `xᵢ = 0` and returned an uncertainty about 100× too small.
pub fn combined_uncertainty(f: impl Fn(&[f64]) -> f64, x: &[f64], u: &[f64], h: f64) -> f64 {
    let mut var = 0.0;
    for i in 0..x.len()
    {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let step = if h.is_finite() && h > 0.0
        {
            h
        }
        else
        {
            relative_step(x[i], u[i])
        };
        xp[i] += step;
        xm[i] -= step;
        // Divide by the spacing actually represented in `f64`, not `2·step`.
        let partial = (f(&xp) - f(&xm)) / (xp[i] - xm[i]);
        var += (partial * u[i]).powi(2);
    }
    var.sqrt()
}

/// Fallback finite-difference step `∛ε · max(|x|, u)` (`∛ε` alone when both
/// are zero), the usual balance of truncation and rounding error for a
/// central difference.
fn relative_step(x: f64, u: f64) -> f64 {
    let scale = x.abs().max(u.abs());
    let cbrt_eps = f64::EPSILON.cbrt();
    if scale.is_finite() && scale > 0.0
    {
        cbrt_eps * scale
    }
    else
    {
        cbrt_eps
    }
}

/// GUM Supplement 1 Monte-Carlo propagation: sample each input `xᵢ ~ N(xᵢ,uᵢ)`
/// and return the `(mean, standard deviation)` of `f`. Deterministic (seeded).
pub fn monte_carlo(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    u: &[f64],
    n: usize,
    seed: u64,
) -> (f64, f64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut normal = || {
        let u1 = next_u01(&mut state).max(1e-12);
        let u2 = next_u01(&mut state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    };
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi: Vec<f64> = x
            .iter()
            .zip(u)
            .map(|(&xv, &uv)| xv + uv * normal())
            .collect();
        samples.push(f(&xi));
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, var.sqrt())
}

fn next_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_uncertainty_matches_analytic() {
        // y = x1·x2, x = [10, 5], u = [0.1, 0.05].
        // u_y = sqrt((x2·u1)² + (x1·u2)²) = sqrt(0.25 + 0.25) = 0.7071.
        let f = |x: &[f64]| x[0] * x[1];
        let uc = combined_uncertainty(f, &[10.0, 5.0], &[0.1, 0.05], 1e-4);
        assert!(
            (uc - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-4,
            "uc {uc}"
        );
    }

    #[test]
    fn monte_carlo_agrees_with_the_sensitivity_method() {
        let f = |x: &[f64]| x[0] * x[1];
        let x = [10.0, 5.0];
        let u = [0.1, 0.05];
        let uc = combined_uncertainty(f, &x, &u, 1e-4);
        let (mean, sd) = monte_carlo(f, &x, &u, 100_000, 0xABCDE);
        assert!((mean - 50.0).abs() < 0.05, "mean {mean}");
        assert!((sd - uc).abs() < 0.02, "MC sd {sd} vs GUM {uc}");
    }

    #[test]
    fn sum_uncertainties_add_in_quadrature() {
        // y = x1 + x2: sensitivities are 1, so u_c = sqrt(u1² + u2²).
        // u1=3, u2=4 → u_c = 5 (the classic 3-4-5).
        let f = |x: &[f64]| x[0] + x[1];
        let uc = combined_uncertainty(f, &[10.0, 5.0], &[3.0, 4.0], 1e-4);
        assert!((uc - 5.0).abs() < 1e-6, "uc {uc}");
    }

    #[test]
    fn honours_a_small_step_for_inputs_in_small_si_units() {
        // Capacitive reactance X = 1/(2π f C) of C = 100 pF ± 1 pF at f = 1 MHz:
        // X ≈ 1591.55 Ω, u_X = X · u_C / C ≈ 15.9155 Ω (1 % relative).
        // The caller asks for a step of 1e-15 F; it used to be floored to an
        // absolute 1e-9 F, ten times C itself, so the difference straddled
        // the pole at C = 0 and gave u_X ≈ 0.161 Ω (a positive slope, 99× too
        // small).
        let freq = 1.0e6;
        let f = |x: &[f64]| 1.0 / (2.0 * core::f64::consts::PI * freq * x[0]);
        let (c, u_c) = (100.0e-12, 1.0e-12);
        let x_c = f(&[c]);
        let expected = x_c * u_c / c;
        let uc = combined_uncertainty(f, &[c], &[u_c], 1.0e-15);
        assert!(
            (uc - expected).abs() <= 1e-6 * expected,
            "u_X {uc}, expected {expected}"
        );
    }

    #[test]
    fn invalid_step_falls_back_to_a_step_relative_to_the_input() {
        // h ≤ 0 or non-finite cannot be honoured; the fallback step scales
        // with the input, so the same 100 pF budget still gives ≈ 15.9155 Ω
        // and an ordinary-scale input still gives the analytic value.
        let freq = 1.0e6;
        let f = |x: &[f64]| 1.0 / (2.0 * core::f64::consts::PI * freq * x[0]);
        let (c, u_c) = (100.0e-12, 1.0e-12);
        let expected = f(&[c]) * u_c / c;
        for h in [0.0, -1.0, f64::NAN, f64::INFINITY]
        {
            let uc = combined_uncertainty(f, &[c], &[u_c], h);
            assert!(
                (uc - expected).abs() <= 1e-6 * expected,
                "h = {h}: u_X {uc}, expected {expected}"
            );
        }
        let g = |x: &[f64]| x[0].sqrt();
        let uc = combined_uncertainty(g, &[4.0], &[0.1], 0.0);
        assert!((uc - 0.025).abs() < 1e-9, "uc {uc}");
    }

    #[test]
    fn nonlinear_model_propagates() {
        // y = sqrt(x), at x=4, u=0.1 -> u_y ≈ u/(2 sqrt x) = 0.025.
        let f = |x: &[f64]| x[0].sqrt();
        let uc = combined_uncertainty(f, &[4.0], &[0.1], 1e-5);
        assert!((uc - 0.025).abs() < 1e-4, "uc {uc}");
    }
}
