//! Quadrature de Simpson adaptative récursive.
//!
//! Idée : on calcule l'intégrale par Simpson sur `[a, b]`, puis sur les deux
//! moitiés `[a, m]` et `[m, b]`. Si la différence entre la somme des deux
//! et la valeur globale est dans la tolérance, on accepte. Sinon, on
//! subdivise récursivement chaque moitié.
//!
//! - Tolérance scale-aware : si l'échelle initiale
//!   `(b−a)·max(|f(a)|, |f(b)|, |f(m)|)` est `≥ tol`, on garde le critère
//!   absolu classique `|err| < tol` ; si l'intégrande est déjà mesurée en
//!   unités minuscules (`f ↦ s·f` avec `|s|·(b−a) ≲ tol`), le test devient
//!   relatif `|err| ≤ 10⁻⁸ · scale₀` (budget aussi subdivisé par moitié),
//!   pour ne pas accepter une estimation grossière dont l'erreur absolue
//!   est petite seulement parce que `f` l'est.

use crate::{SolverError, SolverResult};

/// Relative factor used when the integrand scale sits below `tol`.
/// Matches [`crate::Tolerance::default`].rel.
const ERR_REL: f64 = 1e-8;

/// Scale-aware acceptance for a Richardson error estimate.
///
/// Under a uniform residual rescaling `f ↦ s·f`, a fixed absolute
/// `|err| < tol` floor falsely accepts the coarsest panel once
/// `|s|·(b−a) ≲ tol` (e.g. a Lorentzian peak scaled by `1e-20` with the
/// common `tol = 1e-10` returns a value ~4× too large). When the initial
/// integrand scale is already below `tol`, the test switches to a relative
/// criterion against that scale. `tol` is the per-interval absolute budget
/// (halved on subdivision); `tol0` is the caller-facing absolute floor used
/// only to choose absolute vs relative mode.
fn error_acceptable(err: f64, scale0: f64, tol: f64, tol0: f64) -> bool {
    let e = err.abs();
    if scale0 >= tol0 || scale0 == 0.0
    {
        e < tol
    }
    else
    {
        // Same halving schedule as the absolute budget: each half-interval
        // receives half of the relative allowance.
        let rel_tol = ERR_REL * scale0 * (tol / tol0);
        e <= rel_tol
    }
}

/// Intègre `f` sur `[a, b]` par Simpson adaptatif.
///
/// `tol` : tolérance absolue cible sur l'estimation d'erreur lorsque
/// l'intégrande est d'échelle `≳ tol`. Si `(b−a)·max(|f|)` aux nœuds
/// initiaux est déjà sous `tol`, l'acceptation bascule sur un critère
/// relatif (`≈ 10⁻⁸` de cette échelle) — un plancher absolu seul
/// accepterait n'importe quelle estimation grossière.
/// `max_depth` : profondeur maximale de récursion (en pratique 30 suffit).
pub fn simpson_adaptive<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> SolverResult<f64> {
    validate_inputs(a, b, tol)?;
    if a == b
    {
        return Ok(0.0);
    }
    let (a, b, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let fa = eval_finite(&f, a)?;
    let fb = eval_finite(&f, b)?;
    let m = 0.5 * (a + b);
    let fm = eval_finite(&f, m)?;
    let whole = simpson_step(a, b, fa, fb, fm);
    let scale0 = (b - a) * fa.abs().max(fb.abs()).max(fm.abs());
    let val = recurse(&f, a, b, fa, fb, fm, whole, tol, tol, scale0, max_depth)?;
    Ok(sign * val)
}

fn validate_inputs(a: f64, b: f64, tol: f64) -> SolverResult<()> {
    if !a.is_finite() || !b.is_finite()
    {
        return Err(SolverError::InvalidInput(
            "integration bounds must be finite".into(),
        ));
    }
    if !tol.is_finite() || tol <= 0.0
    {
        return Err(SolverError::InvalidInput(
            "tol must be finite and > 0".into(),
        ));
    }
    Ok(())
}

fn eval_finite<F: Fn(f64) -> f64>(f: &F, x: f64) -> SolverResult<f64> {
    let value = f(x);
    if value.is_finite()
    {
        Ok(value)
    }
    else
    {
        Err(SolverError::NanDetected { iter: 0, value })
    }
}

#[inline]
fn simpson_step(a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
    (b - a) / 6.0 * (fa + 4.0 * fm + fb)
}

#[allow(clippy::too_many_arguments)]
fn recurse<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    tol0: f64,
    scale0: f64,
    depth: usize,
) -> SolverResult<f64> {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = eval_finite(f, lm)?;
    let frm = eval_finite(f, rm)?;
    let left = simpson_step(a, m, fa, fm, flm);
    let right = simpson_step(m, b, fm, fb, frm);
    let sum = left + right;
    let err = (sum - whole) / 15.0; // estimateur de Richardson
    if depth == 0 || error_acceptable(err, scale0, tol, tol0)
    {
        Ok(sum + err)
    }
    else
    {
        let half_tol = tol * 0.5;
        Ok(recurse(
            f,
            a,
            m,
            fa,
            fm,
            flm,
            left,
            half_tol,
            tol0,
            scale0,
            depth - 1,
        )? + recurse(
            f,
            m,
            b,
            fm,
            fb,
            frm,
            right,
            half_tol,
            tol0,
            scale0,
            depth - 1,
        )?)
    }
}

/// Variante qui renvoie aussi une erreur explicite si la subdivision atteint
/// la profondeur max sans satisfaire la tolérance.
pub fn simpson_adaptive_strict<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> SolverResult<f64> {
    validate_inputs(a, b, tol)?;
    if max_depth == 0
    {
        return Err(SolverError::InvalidInput("max_depth must be > 0".into()));
    }
    if a == b
    {
        return Ok(0.0);
    }
    let (a, b, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let fa = eval_finite(&f, a)?;
    let fb = eval_finite(&f, b)?;
    let m = 0.5 * (a + b);
    let fm = eval_finite(&f, m)?;
    let whole = simpson_step(a, b, fa, fb, fm);
    let scale0 = (b - a) * fa.abs().max(fb.abs()).max(fm.abs());
    recurse_strict(&f, a, b, fa, fb, fm, whole, tol, tol, scale0, max_depth).map(|v| sign * v)
}

#[allow(clippy::too_many_arguments)]
fn recurse_strict<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    tol0: f64,
    scale0: f64,
    depth: usize,
) -> SolverResult<f64> {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = eval_finite(f, lm)?;
    let frm = eval_finite(f, rm)?;
    let left = simpson_step(a, m, fa, fm, flm);
    let right = simpson_step(m, b, fm, fb, frm);
    let sum = left + right;
    let err = (sum - whole) / 15.0;
    if error_acceptable(err, scale0, tol, tol0)
    {
        return Ok(sum + err);
    }
    if depth == 0
    {
        return Err(SolverError::IntegrationFailed(format!(
            "adaptive Simpson depth exhausted on [{a}, {b}]: estimated error {} exceeds tolerance {tol}",
            err.abs()
        )));
    }
    let half_tol = tol * 0.5;
    Ok(recurse_strict(
        f,
        a,
        m,
        fa,
        fm,
        flm,
        left,
        half_tol,
        tol0,
        scale0,
        depth - 1,
    )? + recurse_strict(
        f,
        m,
        b,
        fm,
        fb,
        frm,
        right,
        half_tol,
        tol0,
        scale0,
        depth - 1,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn integrate_sin_0_pi() {
        // ∫₀^π sin(x) dx = 2
        let v = simpson_adaptive(|x| x.sin(), 0.0, PI, 1e-12, 30).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn integrate_x_cubed() {
        // ∫₀¹ x³ dx = 0.25 (exact pour Simpson, c'est un polynôme degré 3)
        let v = simpson_adaptive(|x| x.powi(3), 0.0, 1.0, 1e-12, 10).unwrap();
        assert_relative_eq!(v, 0.25, epsilon = 1e-14);
    }

    #[test]
    fn integrate_runge_function() {
        // ∫₋₅⁵ 1/(1+x²) dx = 2·atan(5) ≈ 2.7468
        // (fonction de Runge, classique pour tester l'adaptativité)
        let v = simpson_adaptive(|x: f64| 1.0 / (1.0 + x * x), -5.0, 5.0, 1e-10, 30).unwrap();
        let exact = 2.0 * 5.0_f64.atan();
        assert_relative_eq!(v, exact, epsilon = 1e-9);
    }

    #[test]
    fn integrate_oscillating() {
        // ∫₀^{2π} sin(10x) dx = 0
        let v = simpson_adaptive(|x| (10.0 * x).sin(), 0.0, 2.0 * PI, 1e-10, 30).unwrap();
        assert!(v.abs() < 1e-8, "expected ~0, got {}", v);
    }

    #[test]
    fn reverse_interval() {
        // ∫_a^b = -∫_b^a
        let forward = simpson_adaptive(|x| x.sin(), 0.0, PI, 1e-10, 20).unwrap();
        let backward = simpson_adaptive(|x| x.sin(), PI, 0.0, 1e-10, 20).unwrap();
        assert_relative_eq!(forward, -backward, epsilon = 1e-10);
    }

    #[test]
    fn simpson_adaptive_strict_rejects_max_depth_zero() {
        let result = simpson_adaptive_strict(|x| x, 0.0, 1.0, 1e-10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn simpson_adaptive_strict_works_like_adaptive() {
        let v = simpson_adaptive_strict(|x| x.sin(), 0.0, PI, 1e-12, 30).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn strict_reports_depth_exhaustion_instead_of_returning_wrong_value() {
        let result = simpson_adaptive_strict(|x| (1000.0 * x).sin(), 0.0, 1.0, 1e-30, 1);
        assert!(matches!(result, Err(SolverError::IntegrationFailed(_))));
    }

    #[test]
    fn rejects_invalid_numeric_inputs() {
        assert!(simpson_adaptive(|x| x, 0.0, 1.0, 0.0, 10).is_err());
        assert!(simpson_adaptive(|x| x, f64::NAN, 1.0, 1e-6, 10).is_err());
        assert!(simpson_adaptive(|_| f64::NAN, 0.0, 1.0, 1e-6, 10).is_err());
    }

    /// Regression: an absolute `|err| < tol` check falsely accepted the
    /// coarsest Simpson panel when the integrand was measured in tiny units.
    /// For `f(x) = s / (ε² + (x − ½)²)` on `[0, 1]` any scale `s ≲ tol`
    /// made the Richardson error look "small" in absolute terms while the
    /// peak was still unresolved, returning a value several times too large
    /// instead of refining toward `s · (2/ε)·arctan(1/(2ε))`. When the
    /// initial scale `(b−a)·max(|f|)` is below `tol`, acceptance is now
    /// relative to that scale.
    #[test]
    fn tiny_scale_peak_does_not_false_converge_via_absolute_err() {
        let eps2 = 1e-4_f64;
        let s = 1e-20_f64;
        let tol = 1e-10_f64;
        let f = |x: f64| s / (eps2 + (x - 0.5).powi(2));
        let exact = s * (2.0 / eps2.sqrt()) * (0.5 / eps2.sqrt()).atan();
        let v = simpson_adaptive(f, 0.0, 1.0, tol, 30).expect("must refine via relative err");
        // Absolute floor alone returned ~4× the truth; require 1e-6 relative.
        assert!(
            (v - exact).abs() <= 1e-6 * exact.abs(),
            "got {v:e}, exact {exact:e}, rel {}",
            (v - exact).abs() / exact.abs()
        );
    }

    #[test]
    fn tiny_scale_peak_strict_matches_adaptive() {
        let eps2 = 1e-4_f64;
        let s = 1e-20_f64;
        let f = |x: f64| s / (eps2 + (x - 0.5).powi(2));
        let exact = s * (2.0 / eps2.sqrt()) * (0.5 / eps2.sqrt()).atan();
        let v = simpson_adaptive_strict(f, 0.0, 1.0, 1e-10, 30).unwrap();
        assert!((v - exact).abs() <= 1e-6 * exact.abs());
    }
}
