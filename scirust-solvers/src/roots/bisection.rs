//! Méthode de bissection : robuste, garantit la convergence dès qu'il y a un
//! changement de signe sur l'intervalle. Convergence linéaire (1 bit / itér).
//!
//! ## Sécurité numérique
//! - Stationnarité du résidu scale-aware : si `|f₀| ≥ tol.abs` on garde le
//!   critère absolu classique `|f| < tol.abs` ; si `|f₀|` est déjà sous
//!   `tol.abs` (fonction mesurée en unités minuscules, `f ↦ s·f` avec
//!   `|s| ≲ tol.abs`), le test devient relatif `|f| ≤ tol.rel · |f₀|` pour
//!   qu'un milieu d'intervalle encore large ne paraisse pas convergé.
//! - Racine exacte (`f == 0`) → solution acceptée.
//! - Convergence en largeur : `|b − a| < tol.abs + tol.rel · |mid|`.

use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};

/// Scale-aware stationarity test for the scalar residual `|f(x)|`.
///
/// Under a uniform residual rescaling `f ↦ s·f`, a fixed absolute
/// `|f| < tol.abs` floor falsely accepts the first midpoint once
/// `|s| ≲ tol.abs` (e.g. `1e-20·(x² − 2)` on `[1, 2]` with the default
/// `tol.abs = 1e-10` returns `1.5` instead of `√2`). When the initial
/// residual is already below `tol.abs`, the test switches to a relative
/// reduction against `|f₀|`; otherwise the classical absolute floor is
/// kept (so well-scaled problems keep the same terminal accuracy).
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

/// Trouve une racine de `f` dans `[a, b]` sachant que `f(a)·f(b) < 0`.
pub fn bisection<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>> {
    if a > b
    {
        std::mem::swap(&mut a, &mut b);
    }
    let mut fa = f(a);
    let fb = f(b);
    if fa.signum() == fb.signum() && fa != 0.0 && fb != 0.0
    {
        return Err(SolverError::NoSignChange { a, b, fa, fb });
    }

    if fa == 0.0
    {
        return Ok(Solution::new(a, 0, 0.0));
    }
    if fb == 0.0
    {
        return Ok(Solution::new(b, 0, 0.0));
    }

    // Initial residual scale: max of the two endpoint magnitudes.
    let res0 = fa.abs().max(fb.abs());

    for k in 0..tol.max_iter
    {
        let mid = 0.5 * (a + b);
        let fm = f(mid);
        let res = fm.abs();
        if residual_stationary(res, res0, tol) || (b - a).abs() < tol.abs + tol.rel * mid.abs()
        {
            return Ok(Solution {
                value: mid,
                info: ConvergenceInfo {
                    iterations: k + 1,
                    residual: res,
                    converged: true,
                },
            });
        }
        if fa.signum() == fm.signum()
        {
            a = mid;
            fa = fm;
        }
        else
        {
            b = mid;
            // pas besoin de tracker fb : on connaît son signe (opposé de fa)
        }
    }
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: (b - a).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn root_of_cos() {
        // racine de cos dans [0, pi] : x = pi/2
        let s = bisection(|x| x.cos(), 0.0, PI, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, PI / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn cubic_root() {
        // x^3 - 2x - 5 = 0 dans [2, 3] ; racine ~ 2.0945514815...
        let s = bisection(
            |x| x.powi(3) - 2.0 * x - 5.0,
            2.0,
            3.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-9);
    }

    #[test]
    fn no_sign_change_detected() {
        // x^2 + 1 n'a pas de racine réelle
        assert!(bisection(|x| x * x + 1.0, -1.0, 1.0, Tolerance::default()).is_err());
    }

    /// Regression: an absolute `|f| < tol.abs` check falsely accepted the
    /// first midpoint when the residual was measured in tiny units.
    /// For `f(x) = s·(x² − 2)` on `[1, 2]` any scale `s ≲ tol.abs` made
    /// mid = 1.5 look "converged" (e.g. `|f(1.5)| = 2.5e-21 ≪ 1e-10`)
    /// instead of iterating toward `√2`. When `|f₀| < tol.abs` the
    /// stationarity test is now relative to `|f₀|` (and the interval-width
    /// criterion still drives termination for tiny-scale problems).
    #[test]
    fn tiny_scale_residual_does_not_false_converge_via_absolute_f() {
        let scale = 1e-20_f64;
        let tol = Tolerance::new(1e-10, 1e-12, 200);
        let s = bisection(|x| (x * x - 2.0) * scale, 1.0, 2.0, tol).unwrap();
        assert!(
            s.info.iterations > 1,
            "must actually bisect; absolute |f| floor used to return mid=1.5 at iter 1"
        );
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), max_relative = 1e-9);
    }
}
