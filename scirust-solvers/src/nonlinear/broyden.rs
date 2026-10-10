//! Méthode de Broyden ("bad" / Broyden 1, formule par-bon-pas).
//!
//! ## Sécurité numérique
//! - `check_finite` après chaque évaluation de F
//! - Vérification NaN sur le pas de correction `delta`
//! - Stationarity of `F` is scale-aware: when `|F₀|_∞ ≥ tol.abs` the
//!   classical absolute `|F|_∞ < tol.abs` test is kept; when the initial
//!   residual is already below `tol.abs` (tiny-scale `F ↦ s·F` with
//!   `|s| ≲ tol.abs`), the test becomes relative `|F|_∞ ≤ tol.rel · |F₀|_∞`
//!   so a far start cannot look converged
//! - Convergence en pas : `‖δ‖ < tol.abs + tol.rel · ‖x‖`, testée avant
//!   toute détection de stagnation
//! - Stagnation : si le pas ne change plus `x` en `f64` → `StepUnderflow`
//!   (plus de seuil absolu `‖δ‖ < 1e-16`, qui abortait les systèmes
//!   micrométriques bien conditionnés)
//! - `.unwrap()` sur `matvec` et `solve` remplacé par propagation `?`
//! - Réinitialisation de la jacobienne si singulière (DF)

use crate::linalg::{self, Matrix};
use crate::{Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

const JACOBIAN_H: f64 = 1e-7;

fn check_finite(value: f64, _label: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Scale-aware residual stationarity for Broyden.
///
/// Under a uniform residual rescaling `F ↦ s·F` a fixed absolute
/// `|F|_∞ < tol.abs` floor falsely accepts any start once `|s| ≲ tol.abs`.
/// When the initial residual is already below `tol.abs`, the test switches
/// to a relative reduction against `|F₀|`; otherwise the classical absolute
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

/// Calcule la jacobienne par différences finies au point `x`.
fn finite_diff_jacobian<F>(f: &F, x: &[f64], fx: &[f64]) -> Matrix
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = x.len();
    let mut b = Matrix::zeros(n, n);
    let mut x_pert = x.to_vec();
    let mut fx_pert = vec![0.0; n];
    for j in 0..n
    {
        let xj_orig = x[j];
        let hj = JACOBIAN_H * xj_orig.abs().max(1.0);
        x_pert[j] = xj_orig + hj;
        f(&x_pert, &mut fx_pert);
        x_pert[j] = xj_orig;
        for i in 0..n
        {
            b[(i, j)] = (fx_pert[i] - fx[i]) / hj;
        }
    }
    b
}

pub fn broyden<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = x0.len();
    let mut x = x0;
    let mut fx = vec![0.0; n];
    f(&x, &mut fx);

    let res0 = linalg::norm_inf(&fx);
    if residual_stationary(res0, res0, tol)
    {
        return Ok(Solution::new(x, 0, res0));
    }

    for fi in &fx
    {
        check_finite(*fi, "fx[0]")?;
    }

    // Jacobienne initiale par différences finies
    let mut b = finite_diff_jacobian(&f, &x, &fx);

    let mut last_res = res0;
    for k in 0..tol.max_iter
    {
        // Résous B · δ = -F
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = match linalg::solve(b.clone(), &rhs)
        {
            Ok(d) => d,
            Err(_) =>
            {
                warn!(target: "solver", "Broyden: jacobian singular at iteration {k} — re-initializing via FD");
                b = finite_diff_jacobian(&f, &x, &fx);
                continue;
            },
        };

        // Vérifier que delta est fini
        for (i, &d) in delta.iter().enumerate()
        {
            check_finite(d, &format!("delta[{i}] Broyden k={k}"))?;
        }

        let step_norm = linalg::norm_inf(&delta);

        // x_{k+1} = x_k + delta
        let mut x_new = x.clone();
        for i in 0..n
        {
            x_new[i] += delta[i];
            check_finite(x_new[i], &format!("x_new[{i}] Broyden k={k}"))?;
        }

        // Convergence before stagnation — relative to ‖x‖, not a fixed 1e-16.
        if step_norm < tol.abs + tol.rel * linalg::norm_inf(&x_new)
        {
            let mut fx_new = vec![0.0; n];
            f(&x_new, &mut fx_new);
            return Ok(Solution::new(x_new, k + 1, linalg::norm_inf(&fx_new)));
        }
        if x_new == x
        {
            warn!(target: "solver", "Broyden: step {step_norm:.3e} no longer moves x at iteration {k}");
            return Err(SolverError::StepUnderflow { step: step_norm });
        }

        let mut fx_new = vec![0.0; n];
        f(&x_new, &mut fx_new);
        for fi in &fx_new
        {
            check_finite(*fi, &format!("fx_new Broyden k={k}"))?;
        }

        let res = linalg::norm_inf(&fx_new);
        last_res = res;

        if residual_stationary(res, res0, tol)
        {
            return Ok(Solution::new(x_new, k + 1, res));
        }

        // Mise à jour de B par rang-1 (Broyden "good")
        let mut df = vec![0.0; n];
        for i in 0..n
        {
            df[i] = fx_new[i] - fx[i];
        }

        // B·δ avec propagation d'erreur (plus de .unwrap())
        let bdelta = match b.matvec(&delta)
        {
            Ok(v) => v,
            Err(e) =>
            {
                warn!(target: "solver", "Broyden: matvec failed at iteration {k}: {e} — re-initializing");
                b = finite_diff_jacobian(&f, &x, &fx);
                x = x_new;
                fx = fx_new;
                continue;
            },
        };

        let denom = linalg::dot(&delta, &delta);
        // Any nonzero step can update B; a fixed 1e-30 floor skipped the
        // rank-1 correction on micrometre-scale steps (‖δ‖² ≪ 1e-30).
        if denom > 0.0
        {
            for i in 0..n
            {
                let coef = (df[i] - bdelta[i]) / denom;
                check_finite(coef, &format!("Broyden rank-1 coef[{i}] k={k}"))?;
                for j in 0..n
                {
                    b[(i, j)] += coef * delta[j];
                }
            }
        }

        x = x_new;
        fx = fx_new;
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_res,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn broyden_circle_diagonal() {
        let s = broyden(
            |x, out| {
                out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
                out[1] = x[0] - x[1];
            },
            vec![1.0, 0.5],
            Tolerance::default(),
        )
        .unwrap();
        let expected = (0.5_f64).sqrt();
        assert_relative_eq!(s.value[0], expected, epsilon = 1e-8);
        assert_relative_eq!(s.value[1], expected, epsilon = 1e-8);
    }

    #[test]
    fn broyden_brown_func() {
        let n = 3;
        let s = broyden(
            move |x, out| {
                let sum: f64 = x.iter().sum();
                for i in 0..(n - 1)
                {
                    out[i] = x[i] + sum - (n as f64 + 1.0);
                }
                out[n - 1] = x.iter().product::<f64>() - 1.0;
            },
            vec![0.5, 0.5, 0.5],
            Tolerance::default(),
        )
        .unwrap();
        for v in &s.value
        {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-6);
        }
    }

    /// Regression: an absolute `step_norm < 1e-16` floor aborted Broyden on a
    /// well-conditioned micrometre-scale linear system before any update.
    #[test]
    fn broyden_solves_a_tiny_scale_linear_system() {
        let target = 1e-18;
        let s = broyden(
            move |x, out| {
                out[0] = x[0] - target;
                out[1] = x[1] - 2.0 * target;
            },
            vec![2.0 * target, 4.0 * target],
            Tolerance {
                abs: 1e-30,
                rel: 1e-12,
                max_iter: 20,
            },
        )
        .expect("a regular tiny-scale system must not abort with StepUnderflow");
        assert!((s.value[0] - target).abs() <= 1e-30, "x0={}", s.value[0]);
        assert!(
            (s.value[1] - 2.0 * target).abs() <= 1e-30,
            "x1={}",
            s.value[1]
        );
    }

    /// Regression: an absolute `|F|_∞ < tol.abs` check falsely accepted a
    /// far-from-root start when the residual was measured in tiny units.
    /// For `F(x) = s·(x − x★)` any scale `s ≲ tol.abs` made an O(1)
    /// displacement look "converged" and Broyden returned the start at
    /// iteration 0. When `|F₀| < tol.abs` stationarity is now relative to
    /// `|F₀|`.
    #[test]
    fn tiny_scale_residual_does_not_false_converge_via_absolute_norm() {
        let scale = 1e-12_f64;
        let x_star = 3.0_f64;
        let s = broyden(
            move |x, out| {
                out[0] = scale * (x[0] - x_star);
            },
            vec![0.0],
            Tolerance::new(1e-10, 1e-12, 50),
        )
        .expect("tiny-scale linear residual must be solvable");
        assert!(
            (s.value[0] - x_star).abs() < 1e-9 * (1.0 + x_star.abs()),
            "expected x≈{x_star}, got {:?} (iters={})",
            s.value,
            s.info.iterations
        );
        assert!(
            s.info.iterations > 0,
            "must actually iterate; absolute |F| floor used to return at iter 0"
        );
    }

    /// Same absolute-residual regression on a 2-D scaled linear system.
    #[test]
    fn tiny_scale_residual_2d_does_not_false_converge_via_absolute_norm() {
        let scale = 1e-12_f64;
        let x_star = 3.0_f64;
        let y_star = 4.0_f64;
        let s = broyden(
            move |x, out| {
                out[0] = scale * (x[0] - x_star);
                out[1] = scale * (x[1] - y_star);
            },
            vec![0.0, 0.0],
            Tolerance::new(1e-10, 1e-12, 50),
        )
        .expect("tiny-scale 2-D residual must be solvable");
        assert!(
            (s.value[0] - x_star).abs() < 1e-9 * (1.0 + x_star.abs()),
            "expected x≈{x_star}, got {:?} (iters={})",
            s.value,
            s.info.iterations
        );
        assert!(
            (s.value[1] - y_star).abs() < 1e-9 * (1.0 + y_star.abs()),
            "expected y≈{y_star}, got {:?} (iters={})",
            s.value,
            s.info.iterations
        );
        assert!(
            s.info.iterations > 0,
            "must actually iterate; absolute |F| floor used to return at iter 0"
        );
    }

    /// Already-at-root must still exit immediately (iters == 0).
    #[test]
    fn converges_immediately_when_x0_is_already_the_root() {
        let s = broyden(
            |x, out| {
                out[0] = x[0] - 3.0;
                out[1] = x[1] - 4.0;
            },
            vec![3.0, 4.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_eq!(s.info.iterations, 0);
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(s.value[1], 4.0, epsilon = 1e-12);
    }
}
