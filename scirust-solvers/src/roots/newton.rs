//! Méthode de Newton 1D avec dérivée (autodiff ou explicite).
//!
//! ## Sécurité numérique
//! - Stationnarité du résidu scale-aware : si `|f₀| ≥ tol.abs` on garde le
//!   critère absolu classique `|f| < tol.abs` ; si `|f₀|` est déjà sous
//!   `tol.abs` (fonction mesurée en unités minuscules, `f ↦ s·f` avec
//!   `|s| ≲ tol.abs`), le test devient relatif `|f| ≤ tol.rel · |f₀|` pour
//!   qu'un départ loin de la racine ne paraisse pas convergé.
//! - Racine exacte (`f(x) == 0`) → solution acceptée.
//! - Dérivée nulle (`f'(x) == 0`) ou pas `f(x)/f'(x)` non fini →
//!   `SolverError::ZeroDerivative`. Il n'y a pas de seuil absolu sur `|f'|` :
//!   une pente de `1e-18` est légitime (par ex. `ln(x) − 40` près de `x ≈ 2,35e17`).
//! - Convergence : `|pas| < tol.abs + tol.rel · |x|`, testée avant toute
//!   détection de stagnation.
//! - Stagnation : si le pas ne change plus `x` en `f64` sans que la tolérance
//!   soit atteinte (tolérance plus fine que la précision machine en `x`) →
//!   `SolverError::StepUnderflow`. Le critère est relatif à `x`, pas un seuil
//!   absolu, pour que les racines de petite amplitude (`1e-10`) convergent.
//! - check_finite sur fx et dfx.

use crate::{Solution, SolverError, SolverResult, Tolerance};
use scirust_autodiff::Dual;
use tracing::warn;

fn check_finite(v: f64, _label: &str) -> Result<(), SolverError> {
    if !v.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: v });
    }
    Ok(())
}

/// Newton step `f(x) / f'(x)`, or `None` when the derivative is zero or so
/// small relative to `f(x)` that the step is not finite.
///
/// There is deliberately no absolute floor on `|f'(x)|`: the derivative's
/// scale depends on the units of `x` and `f`, and only an exactly flat slope
/// or an overflowing step makes the Newton update undefined.
fn newton_step(fx: f64, dfx: f64) -> Option<f64> {
    if dfx == 0.0
    {
        return None;
    }
    let step = fx / dfx;
    step.is_finite().then_some(step)
}

/// Scale-aware stationarity test for the scalar residual `|f(x)|`.
///
/// Under a uniform residual rescaling `f ↦ s·f`, a fixed absolute
/// `|f| < tol.abs` floor falsely accepts any start once `|s| ≲ tol.abs`
/// (e.g. `1e-20·(x² − 2)` with the default `tol.abs = 1e-10`). When the
/// initial residual is already below `tol.abs`, the test switches to a
/// relative reduction against `|f₀|`; otherwise the classical absolute
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

/// Newton avec dérivée calculée automatiquement par dual numbers.
pub fn newton<F>(f: F, x0: f64, tol: Tolerance) -> SolverResult<Solution<f64>>
where
    F: Fn(Dual) -> Dual,
{
    let mut x = x0;
    let mut f0 = f64::NAN;
    for k in 0..tol.max_iter
    {
        let d = f(Dual::new(x, 1.0));
        let fx = d.value;
        let dfx = d.deriv;
        check_finite(fx, "fx")?;
        check_finite(dfx, "dfx")?;

        let res = fx.abs();
        if k == 0
        {
            f0 = res;
        }
        if residual_stationary(res, f0, tol)
        {
            return Ok(Solution::new(x, k, res));
        }
        let Some(step) = newton_step(fx, dfx)
        else
        {
            warn!(target: "solver", "Newton 1D: zero derivative at x={x:.6e}");
            return Err(SolverError::ZeroDerivative { x });
        };

        let x_new = x - step;
        if step.abs() < tol.abs + tol.rel * x_new.abs()
        {
            let fx2 = f(Dual::new(x_new, 0.0)).value;
            return Ok(Solution::new(x_new, k + 1, fx2.abs()));
        }
        if x_new == x
        {
            warn!(target: "solver", "Newton 1D: step {step:.3e} no longer moves x={x:.6e} at iteration {k}");
            return Err(SolverError::StepUnderflow { step });
        }
        x = x_new;
    }
    let fx = f(Dual::new(x, 0.0)).value;
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: fx.abs(),
    })
}

/// Variante quand l'utilisateur fournit `f` et `f'` séparément.
pub fn newton_with_derivative<F, G>(
    f: F,
    df: G,
    x0: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut x = x0;
    let mut f0 = f64::NAN;
    for k in 0..tol.max_iter
    {
        let fx = f(x);
        let dfx = df(x);
        check_finite(fx, "fx")?;
        check_finite(dfx, "dfx")?;

        let res = fx.abs();
        if k == 0
        {
            f0 = res;
        }
        if residual_stationary(res, f0, tol)
        {
            return Ok(Solution::new(x, k, res));
        }
        let Some(step) = newton_step(fx, dfx)
        else
        {
            warn!(target: "solver", "Newton 1D (explicit): zero derivative at x={x:.6e}");
            return Err(SolverError::ZeroDerivative { x });
        };

        let x_new = x - step;
        if step.abs() < tol.abs + tol.rel * x_new.abs()
        {
            let fx2 = f(x_new);
            return Ok(Solution::new(x_new, k + 1, fx2.abs()));
        }
        if x_new == x
        {
            return Err(SolverError::StepUnderflow { step });
        }
        x = x_new;
    }
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: f(x).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirust_autodiff::Dual;

    #[test]
    fn newton_autodiff_cubic() {
        let s = newton(
            |x: Dual| x.powi(3) - x * 2.0 - 5.0,
            2.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-12);
        assert!(s.info.iterations < 10);
    }

    #[test]
    fn newton_sqrt2() {
        let s = newton(|x: Dual| x * x - 2.0, 1.0, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn newton_explicit_derivative() {
        let s = newton_with_derivative(
            |x: f64| x.tan() - x,
            |x: f64| 1.0 / x.cos().powi(2) - 1.0,
            4.5,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 4.493_409_457_909_064, epsilon = 1e-9);
    }

    /// `ln(x) = 40` has its root at `x = e^40 ≈ 2.354e17`, where the slope
    /// is `1/x ≈ 4.2e-18`. The old absolute `|f'| < 1e-15` cutoff reported
    /// `ZeroDerivative` at the starting point although Newton converges.
    #[test]
    fn newton_accepts_small_but_nonzero_slope() {
        let root = 40.0_f64.exp();
        let s = newton(|x: Dual| x.ln() - 40.0, 2e17, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, root, max_relative = 1e-12);
        let s = newton_with_derivative(|x| x.ln() - 40.0, |x| 1.0 / x, 2e17, Tolerance::default())
            .unwrap();
        assert_relative_eq!(s.value, root, max_relative = 1e-12);
    }

    /// A function measured in small units, `1e-20 · (x² − 2)`, with a residual
    /// tolerance chosen for that scale. The slope `≈ 2.8e-20` is not zero.
    #[test]
    fn newton_small_scale_function() {
        let tol = Tolerance::new(1e-40, 1e-12, 200);
        let s = newton(|x: Dual| (x * x - 2.0) * 1e-20, 1.0, tol).unwrap();
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-10);
        let s = newton_with_derivative(|x| (x * x - 2.0) * 1e-20, |x| 2e-20 * x, 1.0, tol).unwrap();
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-10);
    }

    /// With a purely relative tolerance near machine precision, roots of
    /// magnitude below ~0.5 used to fail with `StepUnderflow` because the
    /// final Newton step is below the absolute `1e-16` cutoff even though it
    /// satisfies `|step| < rel · |x|`. An exact zero residual (`f(x) == 0`)
    /// with `abs = 0` was also rejected.
    #[test]
    fn newton_small_roots_with_relative_tolerance() {
        let tol = Tolerance::new(0.0, 1e-15, 200);
        for c in [0.02_f64, 2e-6, 2.0, 2e-20]
        {
            let root = c.sqrt();
            let s =
                newton(|x: Dual| x * x - c, 1.0, tol).unwrap_or_else(|e| panic!("c = {c}: {e:?}"));
            assert_relative_eq!(s.value, root, max_relative = 4e-15);
            let s = newton_with_derivative(|x| x * x - c, |x| 2.0 * x, 1.0, tol)
                .unwrap_or_else(|e| panic!("c = {c}: {e:?}"));
            assert_relative_eq!(s.value, root, max_relative = 4e-15);
        }
    }

    /// Regression: an absolute `|f| < tol.abs` check falsely accepted a
    /// far-from-root start when the residual was measured in tiny units.
    /// For `f(x) = s·(x² − 2)` any scale `s ≲ tol.abs` made even an O(1)
    /// distance to `√2` look "converged" at iteration 0. When `|f₀| < tol.abs`
    /// the stationarity test is now relative to `|f₀|`.
    #[test]
    fn tiny_scale_residual_does_not_false_converge_via_absolute_f() {
        let scale = 1e-20_f64;
        let tol = Tolerance::new(1e-10, 1e-12, 200);
        let s = newton(|x: Dual| (x * x - 2.0) * scale, 1.0, tol).unwrap();
        assert!(
            s.info.iterations > 0,
            "must actually iterate; absolute |f| floor used to return at iter 0 with x=1"
        );
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-10);
        let s = newton_with_derivative(|x| (x * x - 2.0) * scale, |x| 2.0 * scale * x, 1.0, tol)
            .unwrap();
        assert!(
            s.info.iterations > 0,
            "must actually iterate; absolute |f| floor used to return at iter 0 with x=1"
        );
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-10);
    }

    #[test]
    fn newton_still_reports_flat_derivative() {
        let r = newton(|x: Dual| x * x + 1.0, 0.0, Tolerance::default());
        assert!(matches!(r, Err(SolverError::ZeroDerivative { .. })));
        let r = newton_with_derivative(|x| x * x + 1.0, |x| 2.0 * x, 0.0, Tolerance::default());
        assert!(matches!(r, Err(SolverError::ZeroDerivative { .. })));
    }

    /// A zero tolerance cannot be met at an irrational root: the solver must
    /// stop with an error (stagnation or iteration limit), not report success.
    #[test]
    fn newton_unreachable_tolerance_is_an_error() {
        let tol = Tolerance::new(0.0, 0.0, 200);
        let r = newton(|x: Dual| x * x - 2.0, 1.0, tol);
        match r
        {
            Ok(s) => assert_eq!(s.info.residual, 0.0),
            Err(e) => assert!(
                matches!(
                    e,
                    SolverError::StepUnderflow { .. } | SolverError::NoConvergence { .. }
                ),
                "{e:?}"
            ),
        }
    }
}
