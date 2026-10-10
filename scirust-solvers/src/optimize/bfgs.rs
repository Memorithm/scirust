//! BFGS — Broyden–Fletcher–Goldfarb–Shanno.
//!
//! ## Sécurité numérique
//! - `check_finite` sur gradient, direction, pas
//! - Guard de positivité : si `H` devient non-définie positive (dir_deriv ≥ 0),
//!   on réinitialise H = I
//! - Courbure relative `sy > 1e-12 · ‖s‖ · ‖y‖` avant mise à jour rang-2
//!   (évite divisions par zéro sans dépendre de l'échelle absolue de `s`/`y`)
//! - Backtracking Armijo : stagnation si le pas d'essai ne déplace plus `x`
//!   en `f64` → StepUnderflow (plus de plancher absolu `alpha < 1e-20`, qui
//!   abortait les objectifs raides dont le pas Armijo optimal est ≪ 1e-20)
//! - Plus de `.unwrap()` sur matvec (remplacé par produit manuel)

use crate::linalg::{self, Matrix};
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

/// BFGS avec line search backtracking (Armijo). Gradient via autodiff Dual.
pub fn bfgs<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[Dual]) -> Dual,
{
    let n = x0.len();
    let mut x = x0;
    let mut grad = vec![0.0; n];
    let mut buf = vec![Dual::primal(0.0); n];

    let eval_fg = |x: &[f64], buf: &mut [Dual], grad: &mut [f64]| -> f64 {
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

    // Vérifier que x0 est fini
    for (i, &xi) in x.iter().enumerate()
    {
        check_finite(xi, &format!("x0[{i}]"))?;
    }

    let mut fx = eval_fg(&x, &mut buf, &mut grad);
    for gi in &grad
    {
        check_finite(*gi, "grad[0]")?;
    }

    let mut h = Matrix::identity(n);
    let mut last_gnorm = linalg::norm_inf(&grad);

    for k in 0..tol.max_iter
    {
        let gnorm = linalg::norm_inf(&grad);
        last_gnorm = gnorm;
        if gnorm < tol.abs
        {
            return Ok(Solution::new(x, k, gnorm));
        }

        // Direction p = -H · g (produit manuel, pas de matvec avec unwrap)
        let mut p = vec![0.0; n];
        for i in 0..n
        {
            let mut s = 0.0;
            for j in 0..n
            {
                s += h[(i, j)] * grad[j];
            }
            p[i] = -s;
            check_finite(p[i], &format!("p[{i}]"))?;
        }

        // Vérifier que p est une direction de descente
        let dir_deriv: f64 = linalg::dot(&grad, &p);
        if dir_deriv >= 0.0
        {
            warn!(target: "solver", "BFGS: H lost positive-definiteness at iteration {k} — resetting to I");
            h = Matrix::identity(n);
            for i in 0..n
            {
                p[i] = -grad[i];
            }
        }

        // Line search backtracking Armijo
        let c1 = 1e-4;
        let mut alpha = 1.0_f64;
        let dir_deriv_g: f64 = linalg::dot(&grad, &p);
        let mut x_new = vec![0.0; n];
        let mut grad_new = vec![0.0; n];
        let mut fx_new;
        loop
        {
            for i in 0..n
            {
                x_new[i] = x[i] + alpha * p[i];
                check_finite(x_new[i], &format!("x_new[{i}] alpha={alpha}"))?;
            }
            // Stagnation before Armijo: alpha so small the trial no longer
            // moves x in f64. An absolute `alpha < 1e-20` floor used to abort
            // here on stiff but well-posed objectives (Lipschitz L ≳ 1e20 ⇒
            // Armijo step ≈ 1/L ≪ 1e-20).
            if alpha < 1.0 && x_new == x
            {
                warn!(target: "solver", "BFGS: backtracking step no longer moves x at iteration {k}");
                return Err(SolverError::StepUnderflow { step: alpha });
            }
            fx_new = eval_fg(&x_new, &mut buf, &mut grad_new);

            for gi in &grad_new
            {
                check_finite(*gi, "grad_new")?;
            }

            if fx_new <= fx + c1 * alpha * dir_deriv_g
            {
                break;
            }
            alpha *= 0.5;
        }

        // Mises à jour BFGS
        let mut s_vec = vec![0.0; n];
        let mut y_vec = vec![0.0; n];
        for i in 0..n
        {
            s_vec[i] = x_new[i] - x[i];
            y_vec[i] = grad_new[i] - grad[i];
            check_finite(s_vec[i], &format!("s[{i}]"))?;
            check_finite(y_vec[i], &format!("y[{i}]"))?;
        }

        let sy: f64 = linalg::dot(&s_vec, &y_vec);
        // Curvature must be meaningfully positive relative to ‖s‖·‖y‖. An
        // absolute `sy > 1e-12` floor skipped every update along the weak
        // eigendirection of ill-conditioned micrometre-scale quadratics
        // (‖s‖·‖y‖ ≪ 1e-12 even when the angle between s and y is fine).
        let s_norm = linalg::norm2(&s_vec);
        let y_norm = linalg::norm2(&y_vec);
        let sy_floor = 1e-12 * s_norm * y_norm.max(f64::MIN_POSITIVE);

        if sy > sy_floor
        {
            let rho = 1.0 / sy;
            check_finite(rho, "rho")?;

            // Hy = H · y (produit manuel)
            let mut hy = vec![0.0; n];
            for i in 0..n
            {
                let mut sum = 0.0;
                for j in 0..n
                {
                    sum += h[(i, j)] * y_vec[j];
                }
                hy[i] = sum;
                check_finite(hy[i], &format!("hy[{i}]"))?;
            }

            let yhy: f64 = linalg::dot(&y_vec, &hy);
            check_finite(yhy, "yHy")?;

            // Mise à jour rang-2 de l'inverse Hessienne
            let factor = rho * (1.0 + rho * yhy);
            check_finite(factor, "factor")?;

            for i in 0..n
            {
                for j in 0..n
                {
                    let delta =
                        -rho * (s_vec[i] * hy[j] + hy[i] * s_vec[j]) + factor * s_vec[i] * s_vec[j];
                    h[(i, j)] += delta;
                    check_finite(h[(i, j)], &format!("h[{i},{j}]"))?;
                }
            }
        }
        else
        {
            warn!(target: "solver", "BFGS: sy={sy:.3e} below relative floor {sy_floor:.3e} at iteration {k} — skipping Hessian update");
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
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
    fn bfgs_quadratic() {
        let s = bfgs(
            |x: &[Dual]| (x[0] - 3.0) * (x[0] - 3.0) + (x[1] + 1.0) * (x[1] + 1.0),
            vec![0.0, 0.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(s.value[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn bfgs_rosenbrock() {
        let s = bfgs(
            |x: &[Dual]| {
                let a = -x[0] + 1.0;
                let b = x[1] - x[0] * x[0];
                a * a + b * b * 100.0
            },
            vec![-1.2, 1.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(s.value[1], 1.0, epsilon = 1e-6);
        assert!(s.info.iterations < 100);
    }

    #[test]
    fn bfgs_himmelblau() {
        let s = bfgs(
            |x: &[Dual]| {
                let a = x[0] * x[0] + x[1] - 11.0;
                let b = x[0] + x[1] * x[1] - 7.0;
                a * a + b * b
            },
            vec![1.0, 1.0],
            Tolerance::default(),
        )
        .unwrap();
        let v = (s.value[0] - 3.0).powi(2) + (s.value[1] - 2.0).powi(2);
        assert!(v < 1e-8, "didn't reach (3,2): {:?}", s.value);
    }

    /// Regression: an absolute `sy > 1e-12` floor skipped every BFGS update
    /// along the weak eigendirection of an ill-conditioned quadratic once the
    /// coordinate scale dropped below ~1e-6 (micrometre-scale least squares).
    /// Relative curvature `sy > ε · ‖s‖ · ‖y‖` keeps the update at any scale.
    #[test]
    fn bfgs_illconditioned_micrometre_scale() {
        let a = 1e-7_f64;
        let s = bfgs(
            move |x: &[Dual]| {
                let dx = x[0] - Dual::primal(a);
                let dy = x[1] - Dual::primal(a);
                dx * dx * 1e4 + dy * dy
            },
            vec![0.0, 0.0],
            Tolerance {
                // Gradient scale near the optimum is O(1e4 · δx); ask for
                // about 1e-6 relative accuracy on the position.
                abs: 1e4 * a * 1e-6,
                rel: 1e-8,
                max_iter: 200,
            },
        )
        .expect("BFGS should converge on a tiny-scale ill-conditioned quadratic");
        let e0 = (s.value[0] - a).abs() / a;
        let e1 = (s.value[1] - a).abs() / a;
        assert!(
            e0 < 1e-4 && e1 < 1e-4,
            "expected relative error < 1e-4, got ({e0}, {e1}) at {:?}",
            s.value
        );
    }

    /// Regression: an absolute `alpha < 1e-20` Armijo floor aborted BFGS on a
    /// stiff quadratic before any accepted step. With H₀ = I the first search
    /// direction is −∇f, so the optimal Armijo step is again 1/k; k = 1e22
    /// needs α ≈ 1e-22 and must not be treated as underflow.
    #[test]
    fn bfgs_finds_minimum_of_stiff_quadratic() {
        let k = 1e22_f64;
        let sol = bfgs(
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
}
