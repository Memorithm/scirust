//! Méthode de la sécante : équivalent de Newton mais sans dérivée — utilise
//! deux points pour approximer la pente. Ordre de convergence ≈ 1.618.
//!
//! ## Sécurité numérique
//! - Stationnarité du résidu scale-aware : si `|f₁⁰| ≥ tol.abs` on garde le
//!   critère absolu classique `|f| < tol.abs` ; si `|f₁⁰|` est déjà sous
//!   `tol.abs` (fonction mesurée en unités minuscules, `f ↦ s·f` avec
//!   `|s| ≲ tol.abs`), le test devient relatif `|f| ≤ tol.rel · |f₁⁰|` pour
//!   qu'un départ loin de la racine ne paraisse pas convergé.
//! - Racine exacte (`f(x1) == 0`) → solution acceptée.
//! - Sécante horizontale (`f(x1) == f(x0)`) → `StepUnderflow`. Pas de seuil
//!   absolu sur `|f(x1) − f(x0)|` : il dépend des unités de `f`.
//! - Convergence en pas scale-aware : si `|δ₀| ≥ tol.abs + tol.rel · |x|`
//!   on garde le plancher classique `|pas| < tol.abs + tol.rel · |x|` ; si le
//!   premier pas est déjà sous ce plancher (paramètre micrométrique,
//!   `|x★| ≲ tol.abs`, départ près de l'origine), le test devient relatif
//!   `|pas| ≤ tol.rel · |δ₀|` — un plancher absolu seul acceptait le premier
//!   pas de sécante incomplet sur `f(x) = (x − x★)³` (retourne ≈ `x★/3`)
//! - Stagnation : `x2 == x1` sans tolérance atteinte → StepUnderflow
//!   (critère relatif à `x`, pas un seuil absolu de `1e-16`).
//! - check_finite sur f(x0), f(x1), x2

use crate::{Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

fn check_finite(v: f64, _label: &str) -> Result<(), SolverError> {
    if !v.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: v });
    }
    Ok(())
}

/// Scale-aware stationarity test for the scalar residual `|f(x)|`.
///
/// Under a uniform residual rescaling `f ↦ s·f`, a fixed absolute
/// `|f| < tol.abs` floor falsely accepts any start once `|s| ≲ tol.abs`
/// (e.g. `1e-20·(x² − 2)` with the default `tol.abs = 1e-10`). When the
/// initial residual is already below `tol.abs`, the test switches to a
/// relative reduction against `|f₁⁰|`; otherwise the classical absolute
/// floor is kept (so well-scaled problems keep the same terminal accuracy).
fn residual_stationary(res: f64, res0: f64, tol: Tolerance) -> bool {
    if res == 0.0 || res0 == 0.0
    {
        return true;
    }
    if res0 >= tol.abs
    {
        res < tol.abs
    }
    else
    {
        res <= tol.rel * res0
    }
}

/// Scale-aware secant step stationarity (`|δ| < tol.abs + tol.rel · |x|`).
///
/// On a tiny-scale root (`|x★| ≲ tol.abs`, starts near the origin) the first
/// secant step itself sits below `tol.abs`, so a fixed absolute floor
/// accepts after iteration 0 even when the update is incomplete — e.g.
/// `f(x) = (x − x★)³` has Newton/secant factor ≈ `2/3`, so the first step
/// lands near `x★/3` and the absolute check returns that point with ~67 %
/// relative error. When `|δ₀|` already sits below the classical floor, stop
/// only after a relative reduction against `|δ₀|`; otherwise the classical
/// floor is kept.
fn step_stationary(step: f64, x_abs: f64, step0: f64, tol: Tolerance) -> bool {
    if step == 0.0
    {
        return true;
    }
    let classical = tol.abs + tol.rel * x_abs;
    if step0 >= classical || step0 == 0.0
    {
        step < classical
    }
    else
    {
        step <= tol.rel * step0
    }
}

/// Cherche une racine en partant de deux estimations `x0, x1`.
pub fn secant<F: Fn(f64) -> f64>(
    f: F,
    mut x0: f64,
    mut x1: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>> {
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    check_finite(f0, "f(x0)")?;
    check_finite(f1, "f(x1)")?;

    let res0 = f1.abs();
    let mut step0: Option<f64> = None;

    for k in 0..tol.max_iter
    {
        let res = f1.abs();
        if residual_stationary(res, res0, tol)
        {
            return Ok(Solution::new(x1, k, res));
        }

        let denom = f1 - f0;
        if denom == 0.0
        {
            warn!(
                target: "solver",
                "Secant: denominator {:.3e} near-zero at iteration {} — aborting",
                denom, k
            );
            return Err(SolverError::StepUnderflow { step: denom });
        }

        let x2 = x1 - f1 * (x1 - x0) / denom;
        check_finite(x2, "x2")?;

        let step = (x2 - x1).abs();
        let step0_val = *step0.get_or_insert(step);
        // Scale-aware step floor (see `step_stationary`); not a bare tol.abs.
        if step_stationary(step, x2.abs(), step0_val, tol)
        {
            let f2 = f(x2);
            return Ok(Solution::new(x2, k + 1, f2.abs()));
        }

        if x2 == x1
        {
            warn!(
                target: "solver",
                "Secant: step no longer moves x={:.6e} at iteration {}",
                x1, k
            );
            return Err(SolverError::StepUnderflow { step });
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
        check_finite(f1, "f(x1)")?;
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: f1.abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn secant_cubic() {
        let s = secant(
            |x| x.powi(3) - 2.0 * x - 5.0,
            2.0,
            3.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-10);
    }

    #[test]
    fn secant_transcendental() {
        let s = secant(|x| x.exp() - 3.0 * x, 0.0, 1.0, Tolerance::default()).unwrap();
        assert!((s.value - 0.6190612867).abs() < 1e-6);
    }

    /// Roots of magnitude below ~0.5 with a relative tolerance near machine
    /// precision: the old absolute `|x2 − x1| < 1e-16` stagnation test fired
    /// before the convergence test (`0.02`, `2e-6`, `2e-20` all failed).
    #[test]
    fn secant_small_roots_with_relative_tolerance() {
        let tol = Tolerance::new(0.0, 1e-15, 200);
        for c in [0.02_f64, 2e-6, 2.0, 2e-20]
        {
            let s =
                secant(|x| x * x - c, 1.0, 0.9, tol).unwrap_or_else(|e| panic!("c = {c}: {e:?}"));
            assert_relative_eq!(s.value, c.sqrt(), max_relative = 4e-15);
        }
    }

    /// `1e-35 · (x² − 2)` with a residual tolerance for that scale: the
    /// secant denominator is ~1e-35, which the old absolute `1e-30` cutoff
    /// rejected as a division by zero.
    #[test]
    fn secant_small_scale_function() {
        let tol = Tolerance::new(1e-60, 1e-12, 200);
        let s = secant(|x| (x * x - 2.0) * 1e-35, 1.0, 2.0, tol).unwrap();
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-10);
    }

    /// Regression: an absolute `|f| < tol.abs` check falsely accepted a
    /// far-from-root start when the residual was measured in tiny units.
    /// For `f(x) = s·(x² − 2)` any scale `s ≲ tol.abs` made even an O(1)
    /// distance to `√2` look "converged" at iteration 0 (e.g. `x1 = 2`).
    /// When `|f₁⁰| < tol.abs` the stationarity test is now relative to `|f₁⁰|`.
    #[test]
    fn tiny_scale_residual_does_not_false_converge_via_absolute_f() {
        let scale = 1e-20_f64;
        let tol = Tolerance::new(1e-10, 1e-12, 200);
        let s = secant(|x| (x * x - 2.0) * scale, 1.0, 2.0, tol).unwrap();
        assert!(
            s.info.iterations > 0,
            "must actually iterate; absolute |f| floor used to return at iter 0 with x1=2"
        );
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-10);
    }

    /// Regression: the classical step floor `|δ| < tol.abs + tol.rel · |x|`
    /// is dominated by `tol.abs` whenever the coordinate scale is ≲
    /// `tol.abs`. For `f(x) = (x − x★)³` with `x★ = 1e-10` and starts near
    /// 0, the first secant step lands near `x★/3` with `|δ| ≈ 3.3e-11 <
    /// tol.abs`, so the buggy check returned that point (~67 % relative
    /// error) without iterating further. When `|δ₀|` already sits below the
    /// classical floor, step stationarity is relative to `|δ₀|`.
    #[test]
    fn tiny_scale_parameter_does_not_false_converge_via_absolute_step() {
        let x_star = 1e-10_f64;
        let s = secant(
            move |x| {
                let d = x - x_star;
                d * d * d
            },
            0.0,
            1e-16,
            Tolerance::default(),
        )
        .expect("tiny-scale cubic residual must be solvable");
        assert!(
            (s.value - x_star).abs() < 1e-2 * x_star.abs(),
            "expected x≈{x_star}, got {} (iters={})",
            s.value,
            s.info.iterations
        );
        assert!(
            s.info.iterations > 1,
            "must continue past the incomplete first secant step; absolute tol.abs floor used to return at iter 1 with x≈x★/3"
        );
    }

    #[test]
    fn secant_still_reports_flat_secant() {
        let r = secant(|x| x * x + 1.0, -1.0, 1.0, Tolerance::default());
        assert!(matches!(r, Err(SolverError::StepUnderflow { .. })));
    }
}
