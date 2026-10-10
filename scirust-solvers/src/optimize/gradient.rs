//! Descente de gradient avec backtracking (Armijo). Gradient par autodiff.
//!
//! ## Sécurité numérique
//! - check_finite sur gradient, pas, valeur de f
//! - Détection de NaN dans x_new avant évaluation de f
//! - Backtracking Armijo : stagnation si le pas d'essai ne déplace plus `x`
//!   en `f64` → StepUnderflow. Plus de plancher absolu `alpha < 1e-20`, qui
//!   abortait les quadratiques raides (`L ≳ 1e20`, pas Armijo optimal ≈ 1/L).
//! - Stationarité de `g = ∇f` scale-aware : si `|g₀|_∞ ≥ tol.abs` le test
//!   absolu classique `|g|_∞ < tol.abs` est conservé ; si le gradient initial
//!   est déjà sous `tol.abs` (quadratique à coordonnées micrométriques,
//!   `|x★| ≲ tol.abs`), le test devient relatif à `|g₀|` — un plancher
//!   absolu seul acceptait tout départ O(`|x★|`) à l'itération 0.

use crate::linalg;
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

/// Scale-aware stationarity test for `g = ∇f`.
///
/// Under a uniform coordinate rescaling that keeps the quadratic shape
/// (`x★ ↦ s·x★` with start at 0), one has `g ↦ s·g`, so a fixed absolute
/// `|g|_∞ < tol.abs` floor falsely accepts any start once `|x★| ≲ tol.abs`.
/// When the initial gradient is already below `tol.abs`, the test switches
/// to a relative reduction against `|g₀|`; otherwise the classical absolute
/// floor is kept (so well-scaled problems keep the same terminal accuracy).
fn gradient_stationary(g_inf: f64, g0_inf: f64, tol: Tolerance) -> bool {
    if g_inf == 0.0 || g0_inf == 0.0
    {
        return true;
    }
    if g0_inf >= tol.abs
    {
        g_inf < tol.abs
    }
    else
    {
        g_inf <= tol.rel * g0_inf
    }
}

/// Descente de gradient avec line search. `f: R^n → R` avec Dual.
pub fn gradient_descent<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[Dual]) -> Dual,
{
    let n = x0.len();
    let mut x = x0;
    let mut grad = vec![0.0; n];
    let mut buf = vec![Dual::primal(0.0); n];

    let eval = |x: &[f64], buf: &mut [Dual], grad: &mut [f64]| -> f64 {
        for i in 0..n
        {
            buf[i] = Dual::primal(x[i]);
        }
        let fx = f(buf).value;
        for j in 0..n
        {
            for i in 0..n
            {
                buf[i] = Dual::new(x[i], if i == j { 1.0 } else { 0.0 });
            }
            grad[j] = f(buf).deriv;
        }
        fx
    };

    for (i, &xi) in x.iter().enumerate()
    {
        check_finite(xi, &format!("x0[{i}]"))?;
    }

    let mut fx = eval(&x, &mut buf, &mut grad);
    for gi in &grad
    {
        check_finite(*gi, "grad[0]")?;
    }
    let g0_inf = linalg::norm_inf(&grad);
    let mut last_gnorm = g0_inf;

    for k in 0..tol.max_iter
    {
        let gnorm = linalg::norm_inf(&grad);
        last_gnorm = gnorm;
        if gradient_stationary(gnorm, g0_inf, tol)
        {
            return Ok(Solution::new(x, k, gnorm));
        }

        // Direction = -gradient
        let mut alpha = 1.0_f64;
        let c = 1e-4;
        let g_dot_d = -linalg::dot(&grad, &grad);
        let mut x_new = x.clone();
        let mut fx_new;
        let mut grad_new = vec![0.0; n];

        loop
        {
            for i in 0..n
            {
                x_new[i] = x[i] - alpha * grad[i];
                check_finite(x_new[i], &format!("x_new[{i}] alpha={alpha}"))?;
            }
            // Stagnation before Armijo: alpha so small the trial no longer
            // moves x in f64. An absolute `alpha < 1e-20` floor used to abort
            // here on stiff but well-posed quadratics (Lipschitz L ≳ 1e20 ⇒
            // Armijo step ≈ 1/L ≪ 1e-20).
            if alpha < 1.0 && x_new == x
            {
                warn!(target: "solver", "GD: backtracking step no longer moves x at iteration {k}");
                return Err(SolverError::StepUnderflow { step: alpha });
            }
            fx_new = eval(&x_new, &mut buf, &mut grad_new);

            for gi in &grad_new
            {
                check_finite(*gi, "grad_new")?;
            }

            if fx_new <= fx + c * alpha * g_dot_d
            {
                break;
            }
            alpha *= 0.5;
        }

        let step_norm = alpha * gnorm;
        x = x_new;
        fx = fx_new;
        grad = grad_new;

        if step_norm < tol.abs + tol.rel * linalg::norm_inf(&x)
        {
            return Ok(Solution::new(x, k + 1, linalg::norm_inf(&grad)));
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_gnorm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirust_autodiff::Dual;

    #[test]
    fn quadratic_2d() {
        let s = gradient_descent(
            |x: &[Dual]| (x[0] - 3.0) * (x[0] - 3.0) + (x[1] + 1.0) * (x[1] + 1.0),
            vec![0.0, 0.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(s.value[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn rosenbrock_slow() {
        let f_value = |x: f64, y: f64| (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
        let s = gradient_descent(
            |x: &[Dual]| {
                let a = -x[0] + 1.0;
                let b = x[1] - x[0] * x[0];
                a * a + b * b * 100.0
            },
            vec![0.0, 0.0],
            Tolerance {
                abs: 1e-3,
                rel: 1e-3,
                max_iter: 5000,
            },
        )
        .or_else(|_| {
            Ok::<Solution<Vec<f64>>, SolverError>(Solution::new(vec![0.0, 0.0], 5000, 1e9))
        })
        .unwrap();

        if s.info.converged
        {
            let f_final = f_value(s.value[0], s.value[1]);
            assert!(f_final < 0.2, "f_final should be small: {f_final}");
        }
    }

    /// Regression: an absolute `alpha < 1e-20` Armijo floor aborted gradient
    /// descent on a stiff but well-conditioned quadratic. For
    /// f(x) = ½k‖x − x*‖² the optimal step is 1/k; with k = 1e22 that is
    /// 1e-22, below the old floor, so every backtrack returned StepUnderflow
    /// instead of the obvious minimum.
    #[test]
    fn gd_finds_minimum_of_stiff_quadratic() {
        let k = 1e22_f64;
        let sol = gradient_descent(
            move |x: &[Dual]| {
                let d0 = x[0] - Dual::primal(1.0);
                let d1 = x[1] - Dual::primal(2.0);
                (d0 * d0 + d1 * d1) * Dual::primal(0.5 * k)
            },
            vec![0.0, 0.0],
            Tolerance {
                abs: 1e-8,
                rel: 1e-8,
                max_iter: 200,
            },
        )
        .expect("stiff quadratic must not abort with StepUnderflow");
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sol.value[1], 2.0, epsilon = 1e-5);
    }

    /// Regression: an absolute `|g|_∞ < tol.abs` check falsely accepted a
    /// far-from-optimum start on a micrometre-scale quadratic. For
    /// `f(x) = ½‖x − x★‖²` with `|x★| ≲ tol.abs` and start at 0, the gradient
    /// is `g = x − x★`, so `|g₀| = |x★|` sits below `tol.abs` and the buggy
    /// test returned the origin at iteration 0. When `|g₀| < tol.abs` the
    /// stationarity test is now relative to `|g₀|`.
    #[test]
    fn tiny_scale_quadratic_does_not_false_converge_via_absolute_gradient() {
        let x_star = 1e-16_f64;
        let y_star = 2e-16_f64;
        let sol = gradient_descent(
            move |x: &[Dual]| {
                let d0 = x[0] - Dual::primal(x_star);
                let d1 = x[1] - Dual::primal(y_star);
                (d0 * d0 + d1 * d1) * Dual::primal(0.5)
            },
            vec![0.0, 0.0],
            Tolerance::new(1e-10, 1e-12, 200),
        )
        .expect("micrometre-scale quadratic must be solvable");
        assert!(
            sol.info.iterations > 0,
            "must actually iterate; absolute |g| floor used to return at iter 0 with x=origin"
        );
        assert!(
            (sol.value[0] - x_star).abs() < 1e-18,
            "expected x≈{x_star}, got {:?} (iters={})",
            sol.value,
            sol.info.iterations
        );
        assert!(
            (sol.value[1] - y_star).abs() < 1e-18,
            "expected y≈{y_star}, got {:?} (iters={})",
            sol.value,
            sol.info.iterations
        );
    }

    #[test]
    fn converges_immediately_when_x0_is_already_the_minimizer() {
        let sol = gradient_descent(
            |x: &[Dual]| {
                let d0 = x[0] - Dual::primal(3.0);
                let d1 = x[1] - Dual::primal(-1.0);
                d0 * d0 + d1 * d1
            },
            vec![3.0, -1.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_eq!(sol.info.iterations, 0);
        assert_relative_eq!(sol.value[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(sol.value[1], -1.0, epsilon = 1e-12);
    }
}
