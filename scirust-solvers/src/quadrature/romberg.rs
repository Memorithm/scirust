//! Quadrature de Romberg : extrapolation de Richardson sur la règle du
//! trapèze. Converge à l'ordre 2k pour k niveaux d'extrapolation. Très
//! efficace sur fonctions très lisses (analytiques).
//!
//! - Tolérance scale-aware : si l'échelle initiale
//!   `(b−a)·max(|f|)` aux nœuds du premier trapèze (extrémités + milieu)
//!   est `≥ tol`, on garde le critère absolu classique `|Δ| < tol` ; si
//!   l'intégrande est déjà mesurée en unités minuscules (`f ↦ s·f` avec
//!   `|s|·(b−a) ≲ tol`), le test devient relatif `|Δ| ≤ 10⁻⁸ · scale₀`,
//!   pour ne pas accepter une extrapolation grossière dont l'erreur
//!   absolue est petite seulement parce que `f` l'est.

use crate::{SolverError, SolverResult};

/// Relative factor used when the integrand scale sits below `tol`.
/// Matches [`crate::Tolerance::default`].rel.
const ERR_REL: f64 = 1e-8;

fn validate_inputs(a: f64, b: f64, tol: f64, max_levels: usize) -> SolverResult<()> {
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
    if max_levels < 2
    {
        return Err(SolverError::InvalidInput(
            "max_levels must be >= 2 (a Richardson error estimate needs at least two levels)"
                .into(),
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

/// Scale-aware acceptance for a Romberg diagonal residual.
///
/// Under a uniform residual rescaling `f ↦ s·f`, a fixed absolute
/// `|Δ| < tol` floor falsely stops at a coarse level once
/// `|s|·(b−a) ≲ tol` (e.g. a Lorentzian peak scaled by `1e-20` with the
/// common `tol = 1e-10` returns a value several times too large). When the
/// initial integrand scale is already below `tol`, the test switches to a
/// relative criterion against that scale.
fn error_acceptable(residual: f64, scale0: f64, tol: f64) -> bool {
    let e = residual.abs();
    if scale0 >= tol || scale0 == 0.0
    {
        e < tol
    }
    else
    {
        e <= ERR_REL * scale0
    }
}

/// Intègre f sur `[a,b]` par Romberg. `max_levels` ~ 10-15 suffit en pratique.
///
/// `tol` est la tolérance absolue cible sur l'erreur estimée (différence
/// entre deux niveaux successifs) lorsque l'intégrande est d'échelle
/// `≳ tol`. Si `(b−a)·max(|f|)` aux nœuds initiaux est déjà sous `tol`,
/// l'acceptation bascule sur un critère relatif (`≈ 10⁻⁸` de cette
/// échelle) — un plancher absolu seul arrêterait trop tôt. Best-effort :
/// si la tolérance n'est pas atteinte après `max_levels` niveaux, renvoie
/// tout de même la dernière estimation (voir [`romberg_strict`] pour une
/// variante qui signale l'échec).
///
/// # Errors
/// [`SolverError::InvalidInput`] si les bornes/`tol`/`max_levels` sont
/// invalides ; [`SolverError::NanDetected`] si `f` renvoie une valeur non
/// finie.
pub fn romberg<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> SolverResult<f64> {
    validate_inputs(a, b, tol, max_levels)?;
    romberg_table(&f, a, b, tol, max_levels).map(|(value, _, _)| value)
}

/// Variante stricte : renvoie [`SolverError::NoConvergence`] si la tolérance
/// n'est pas atteinte après `max_levels` niveaux, plutôt que de renvoyer
/// silencieusement la meilleure estimation disponible.
///
/// # Errors
/// Comme [`romberg`], plus [`SolverError::NoConvergence`] sur non-convergence.
pub fn romberg_strict<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> SolverResult<f64> {
    validate_inputs(a, b, tol, max_levels)?;
    let (value, converged, residual) = romberg_table(&f, a, b, tol, max_levels)?;
    if converged
    {
        Ok(value)
    }
    else
    {
        Err(SolverError::NoConvergence {
            iterations: max_levels,
            residual,
        })
    }
}

/// Runs the Romberg table and returns `(value, converged, last_residual)`.
fn romberg_table<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> SolverResult<(f64, bool, f64)> {
    let mut r = vec![vec![0.0_f64; max_levels]; max_levels];

    // Niveau 0 : trapèze simple (+ milieu pour l'échelle initiale, comme
    // Simpson adaptatif : trois nœuds sur le premier panneau).
    let fa = eval_finite(f, a)?;
    let fb = eval_finite(f, b)?;
    let m = 0.5 * (a + b);
    let fm = eval_finite(f, m)?;
    r[0][0] = 0.5 * (b - a) * (fa + fb);
    let scale0 = (b - a) * fa.abs().max(fb.abs()).max(fm.abs());

    let mut last_residual = f64::INFINITY;
    for i in 1..max_levels
    {
        // Trapèze composé avec 2^i intervalles
        let n = 1usize << (i - 1); // nombre de NOUVEAUX points
        let h = (b - a) / (1usize << i) as f64;
        let mut sum = 0.0;
        for k in 0..n
        {
            // i == 1 réévalue le milieu déjà lu pour scale0 — un point
            // redondant, négligeable face à la table Richardson.
            sum += eval_finite(f, a + (2 * k + 1) as f64 * h)?;
        }
        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        // Extrapolation de Richardson
        for j in 1..=i
        {
            let denom = (1usize << (2 * j)) as f64 - 1.0;
            r[i][j] = r[i][j - 1] + (r[i][j - 1] - r[i - 1][j - 1]) / denom;
        }

        // Convergence : différence entre deux niveaux d'extrapolation
        if i >= 2
        {
            last_residual = (r[i][i] - r[i - 1][i - 1]).abs();
            if error_acceptable(last_residual, scale0, tol)
            {
                return Ok((r[i][i], true, last_residual));
            }
        }
    }
    Ok((r[max_levels - 1][max_levels - 1], false, last_residual))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn romberg_sin() {
        let v = romberg(|x| x.sin(), 0.0, PI, 1e-14, 15).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn romberg_exp() {
        // ∫₀¹ exp(x) dx = e - 1
        let v = romberg(|x: f64| x.exp(), 0.0, 1.0, 1e-14, 15).unwrap();
        assert_relative_eq!(v, std::f64::consts::E - 1.0, epsilon = 1e-13);
    }

    #[test]
    fn rejects_invalid_numeric_inputs() {
        assert!(romberg(|x| x, 0.0, 1.0, 0.0, 15).is_err());
        assert!(romberg(|x| x, f64::NAN, 1.0, 1e-10, 15).is_err());
        assert!(romberg(|_| f64::NAN, 0.0, 1.0, 1e-10, 15).is_err());
    }

    #[test]
    fn rejects_max_levels_below_two() {
        assert!(matches!(
            romberg(|x| x, 0.0, 1.0, 1e-10, 1),
            Err(SolverError::InvalidInput(_))
        ));
        assert!(matches!(
            romberg(|x| x, 0.0, 1.0, 1e-10, 0),
            Err(SolverError::InvalidInput(_))
        ));
    }

    /// `romberg` is best-effort: an unreachable tolerance still returns the
    /// best available estimate rather than erroring.
    #[test]
    fn romberg_returns_best_effort_on_unreachable_tolerance() {
        let v = romberg(|x| x.sin(), 0.0, PI, 1e-300, 5).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-6);
    }

    /// `romberg_strict` reports the same unreachable tolerance as an error
    /// instead of silently returning the best-effort value.
    #[test]
    fn romberg_strict_reports_non_convergence() {
        let result = romberg_strict(|x| x.sin(), 0.0, PI, 1e-300, 5);
        assert!(matches!(result, Err(SolverError::NoConvergence { .. })));
    }

    #[test]
    fn romberg_strict_matches_romberg_on_convergence() {
        let v = romberg_strict(|x| x.sin(), 0.0, PI, 1e-14, 15).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
    }

    /// Regression: an absolute `|Δ| < tol` check falsely stopped Romberg at a
    /// coarse Richardson level when the integrand was measured in tiny units.
    /// For `f(x) = s / (ε² + (x − ½)²)` on `[0, 1]` any scale `s ≲ tol`
    /// made the diagonal residual look "small" in absolute terms while the
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
        let v = romberg(f, 0.0, 1.0, tol, 15).expect("must refine via relative err");
        // Absolute floor alone returned ~4× the truth; require 1e-6 relative.
        assert!(
            (v - exact).abs() <= 1e-6 * exact.abs(),
            "got {v:e}, exact {exact:e}, rel {}",
            (v - exact).abs() / exact.abs()
        );
    }

    #[test]
    fn tiny_scale_peak_strict_matches_romberg() {
        let eps2 = 1e-4_f64;
        let s = 1e-20_f64;
        let f = |x: f64| s / (eps2 + (x - 0.5).powi(2));
        let exact = s * (2.0 / eps2.sqrt()) * (0.5 / eps2.sqrt()).atan();
        let v = romberg_strict(f, 0.0, 1.0, 1e-10, 15).unwrap();
        assert!((v - exact).abs() <= 1e-6 * exact.abs());
    }
}
