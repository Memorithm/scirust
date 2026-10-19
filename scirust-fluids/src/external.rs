//! External flow: drag on immersed bodies and terminal settling velocity.

use crate::error::{FluidsError, in_range, non_negative, positive};

/// Upper Reynolds-number bound of the Clift–Gauvin sphere drag correlation
/// used by [`sphere_drag_coefficient`] (onset of the drag crisis).
const SPHERE_DRAG_RE_MAX: f64 = 3.0e5;

/// Stokes drag force on a sphere in creeping flow, `F = 3π μ d V` \[N\]
/// (valid for Re ≲ 0.1).
pub fn stokes_drag(dyn_viscosity: f64, diameter: f64, speed: f64) -> Result<f64, FluidsError> {
    positive("dyn_viscosity", dyn_viscosity)?;
    positive("diameter", diameter)?;
    non_negative("speed", speed)?;
    Ok(3.0 * std::f64::consts::PI * dyn_viscosity * diameter * speed)
}

/// Drag force from a drag coefficient, `F = C_d ρ V² A / 2` \[N\]
/// (A = reference/frontal area).
pub fn drag_force(cd: f64, density: f64, speed: f64, area: f64) -> Result<f64, FluidsError> {
    positive("cd", cd)?;
    positive("density", density)?;
    non_negative("speed", speed)?;
    positive("area", area)?;
    Ok(0.5 * cd * density * speed * speed * area)
}

/// Drag coefficient of a smooth sphere from the **Clift–Gauvin**
/// correlation (standard drag curve, sub-critical regime):
///
/// `C_d = (24/Re)(1 + 0.15 Re^0.687) + 0.42/(1 + 42500/Re^1.16)`
///
/// Valid for `0 < Re ≤ 3×10⁵` (below the drag crisis); reduces to
/// Stokes `24/Re` as Re → 0. Reproduces the standard curve within ~6 %.
pub fn sphere_drag_coefficient(re: f64) -> Result<f64, FluidsError> {
    in_range("re", re, f64::MIN_POSITIVE, SPHERE_DRAG_RE_MAX)?;
    Ok(24.0 / re * (1.0 + 0.15 * re.powf(0.687)) + 0.42 / (1.0 + 42_500.0 / re.powf(1.16)))
}

/// Terminal settling velocity of a sphere in a still fluid \[m/s\],
/// from the force balance `(ρ_p − ρ_f) g π d³/6 = C_d(Re) ρ_f V² π d²/8`
/// with [`sphere_drag_coefficient`] for `C_d`.
///
/// Solved by deterministic bracketing + 200 bisection steps (the residual
/// is strictly increasing in V), so identical inputs give bit-identical
/// results on every platform. Requires `particle_density > fluid_density`
/// (a settling, not rising, particle) and a terminal Reynolds number
/// within the correlation's validity (`Re ≤ 3×10⁵`).
///
/// The bracket is searched only over speeds whose Reynolds number stays
/// inside that range, so a Stokes-law estimate far above `Re = 3×10⁵`
/// (e.g. a centimetre-sized steel ball in water) does not by itself cause
/// an error. When the terminal Reynolds number itself exceeds `3×10⁵`, the
/// function returns [`FluidsError::OutOfRange`] for `re`, whose `value` is
/// the Stokes-law Reynolds number `ρ_f V_Stokes d / μ` (an upper bound on
/// the terminal Reynolds number).
pub fn sphere_terminal_velocity(
    diameter: f64,
    particle_density: f64,
    fluid_density: f64,
    dyn_viscosity: f64,
    gravity: f64,
) -> Result<f64, FluidsError> {
    positive("diameter", diameter)?;
    positive("particle_density", particle_density)?;
    positive("fluid_density", fluid_density)?;
    positive("dyn_viscosity", dyn_viscosity)?;
    positive("gravity", gravity)?;
    if particle_density <= fluid_density
    {
        return Err(FluidsError::OutOfRange {
            name: "particle_density",
            value: particle_density,
            min: fluid_density,
            max: f64::MAX,
        });
    }

    // Residual h(V) = C_d(Re) V² − 4 g d (ρ_p − ρ_f)/(3 ρ_f); h is strictly
    // increasing in V and h(0⁺) < 0.
    let rhs = 4.0 * gravity * diameter * (particle_density - fluid_density) / (3.0 * fluid_density);
    let residual = |v: f64| -> Result<f64, FluidsError> {
        let re = fluid_density * v * diameter / dyn_viscosity;
        Ok(sphere_drag_coefficient(re)? * v * v - rhs)
    };

    // Largest speed whose Reynolds number is inside the correlation's
    // range (stepped down by ulps if rounding pushes Re just above it).
    let mut v_re_max = SPHERE_DRAG_RE_MAX * dyn_viscosity / (fluid_density * diameter);
    for _ in 0..8
    {
        if fluid_density * v_re_max * diameter / dyn_viscosity <= SPHERE_DRAG_RE_MAX
        {
            break;
        }
        v_re_max *= 1.0 - f64::EPSILON;
    }

    // Bracket: grow `hi` from the Stokes estimate (capped at `v_re_max`)
    // until the residual is positive (bounded number of doublings keeps
    // this deterministic).
    let v_stokes =
        gravity * diameter * diameter * (particle_density - fluid_density) / (18.0 * dyn_viscosity);
    // h(0⁺) < 0, and bisection never evaluates `lo` itself.
    let mut lo = 0.0;
    let mut hi = v_stokes.max(1e-12).min(v_re_max);
    let mut bracketed = false;
    for _ in 0..200
    {
        if residual(hi)? > 0.0
        {
            bracketed = true;
            break;
        }
        if hi >= v_re_max
        {
            // Even at Re = 3e5 the drag is below the weight: the terminal
            // Re is out of domain for this correlation.
            return Err(FluidsError::OutOfRange {
                name: "re",
                value: fluid_density * v_stokes * diameter / dyn_viscosity,
                min: f64::MIN_POSITIVE,
                max: SPHERE_DRAG_RE_MAX,
            });
        }
        lo = hi;
        hi = (2.0 * hi).min(v_re_max);
    }
    if !bracketed
    {
        return Err(FluidsError::NoConvergence {
            what: "terminal-velocity bracket",
        });
    }
    for _ in 0..200
    {
        let mid = 0.5 * (lo + hi);
        if residual(mid)? > 0.0
        {
            hi = mid;
        }
        else
        {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stokes_limit_of_standard_curve() {
        // As Re → 0 the correlation reduces to 24/Re.
        let re = 1e-3;
        let cd = sphere_drag_coefficient(re).unwrap();
        assert!((cd - 24.0 / re).abs() / (24.0 / re) < 2e-3, "cd = {cd}");
    }

    #[test]
    fn standard_curve_known_points() {
        // Standard drag curve (Clift, Grace & Weber tables):
        // Re = 100 → C_d ≈ 1.09 ; Re = 1000 → C_d ≈ 0.46 ; plateau ≈ 0.44.
        let cd100 = sphere_drag_coefficient(100.0).unwrap();
        assert!((cd100 - 1.09).abs() < 0.05, "cd(100) = {cd100}");
        let cd1000 = sphere_drag_coefficient(1000.0).unwrap();
        assert!((cd1000 - 0.46).abs() < 0.05, "cd(1000) = {cd1000}");
        let cd1e5 = sphere_drag_coefficient(1.0e5).unwrap();
        assert!((0.4..0.55).contains(&cd1e5), "cd(1e5) = {cd1e5}");
    }

    #[test]
    fn drag_force_car_example() {
        // C_d = 0.3, A = 2 m², air 1.2 kg/m³, 30 m/s:
        // F = 0.5·0.3·1.2·900·2 = 324 N.
        let f = drag_force(0.3, 1.2, 30.0, 2.0).unwrap();
        assert!((f - 324.0).abs() < 1e-9);
    }

    #[test]
    fn stokes_drag_matches_cd_route() {
        // In creeping flow the two formulations must agree:
        // F = 3πμdV = C_d(24/Re) ρV² (πd²/4)/2.
        let (mu, d, v, rho) = (1.0e-3, 1.0e-5, 1.0e-4, 1000.0);
        let f_stokes = stokes_drag(mu, d, v).unwrap();
        let re = rho * v * d / mu;
        let area = std::f64::consts::PI * d * d / 4.0;
        let f_cd = drag_force(24.0 / re, rho, v, area).unwrap();
        assert!((f_stokes - f_cd).abs() / f_stokes < 1e-12);
    }

    #[test]
    fn terminal_velocity_stokes_regime() {
        // 10 µm water droplet in air: analytic Stokes velocity
        // V = g d² Δρ / (18 μ) ≈ 3.02 mm/s, terminal Re ≈ 2e-3.
        let v = sphere_terminal_velocity(1.0e-5, 1000.0, 1.2, 1.8e-5, 9.81).unwrap();
        let v_stokes = 9.81 * 1.0e-10 * (1000.0 - 1.2) / (18.0 * 1.8e-5);
        assert!(
            (v - v_stokes).abs() / v_stokes < 0.01,
            "v = {v}, stokes = {v_stokes}"
        );
    }

    #[test]
    fn terminal_velocity_balances_forces() {
        // 1 mm sand grain in water — outside Stokes; check the balance.
        let (d, rp, rf, mu, g) = (1.0e-3, 2650.0, 1000.0, 1.0e-3, 9.81);
        let v = sphere_terminal_velocity(d, rp, rf, mu, g).unwrap();
        let re = rf * v * d / mu;
        let cd = sphere_drag_coefficient(re).unwrap();
        let weight = (rp - rf) * g * std::f64::consts::PI * d * d * d / 6.0;
        let drag = 0.5 * cd * rf * v * v * std::f64::consts::PI * d * d / 4.0;
        assert!(
            (weight - drag).abs() / weight < 1e-9,
            "unbalanced: {weight} vs {drag}"
        );
    }

    #[test]
    fn terminal_velocity_when_stokes_estimate_exceeds_correlation_range() {
        // 10 mm steel ball settling in water: the Stokes estimate
        // g d² Δρ / (18 μ) ≈ 370.6 m/s corresponds to Re ≈ 3.7e6, outside
        // the Clift–Gauvin range, but the actual terminal Reynolds number is
        // ≈ 1.43e4, inside it. Reference: SciPy `brentq` on the same force
        // balance and correlation gives V = 1.4261228151 m/s.
        let (d, rp, rf, mu, g) = (1.0e-2, 7800.0, 1000.0, 1.0e-3, 9.81);
        let v = sphere_terminal_velocity(d, rp, rf, mu, g).unwrap();
        assert!(
            (v - 1.426_122_815_126).abs() / 1.426_122_815_126 < 1e-9,
            "v = {v}"
        );
        let re = rf * v * d / mu;
        let cd = sphere_drag_coefficient(re).unwrap();
        let weight = (rp - rf) * g * std::f64::consts::PI * d * d * d / 6.0;
        let drag = 0.5 * cd * rf * v * v * std::f64::consts::PI * d * d / 4.0;
        assert!(
            (weight - drag).abs() / weight < 1e-9,
            "unbalanced: {weight} vs {drag}"
        );
        // 7 mm gravel grain in water: Stokes Re ≈ 3.08e5, terminal Re ≈ 4.38e3;
        // SciPy `brentq` reference V = 0.6262792472 m/s.
        let v = sphere_terminal_velocity(7.0e-3, 2650.0, 1000.0, 1.0e-3, 9.81).unwrap();
        assert!(
            (v - 0.626_279_247_155).abs() / 0.626_279_247_155 < 1e-9,
            "v = {v}"
        );
    }

    #[test]
    fn terminal_velocity_rejects_terminal_re_above_correlation_range() {
        // 0.5 m steel ball falling in air: terminal Re far above 3e5.
        let err = sphere_terminal_velocity(0.5, 7800.0, 1.2, 1.8e-5, 9.81).unwrap_err();
        assert!(
            matches!(err, FluidsError::OutOfRange { name: "re", .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rejects_rising_particle_and_bad_re() {
        assert!(sphere_terminal_velocity(1e-3, 800.0, 1000.0, 1e-3, 9.81).is_err());
        assert!(sphere_drag_coefficient(0.0).is_err());
        assert!(sphere_drag_coefficient(1.0e6).is_err());
    }
}
