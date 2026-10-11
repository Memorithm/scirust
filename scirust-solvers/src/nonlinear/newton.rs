//! Newton-Raphson pour systèmes `F: R^n → R^n`.
//!
//! ## Sécurité numérique
//! - `check_finite` sur F et J à chaque itération
//! - Stationarity of `F` is scale-aware: when `|F₀|_∞ ≥ tol.abs` the
//!   classical absolute `|F|_∞ < tol.abs` test is kept; when the initial
//!   residual is already below `tol.abs` (tiny-scale `F ↦ s·F` with
//!   `|s| ≲ tol.abs`), the test becomes relative `|F|_∞ ≤ tol.rel · |F₀|_∞`
//!   so a far start cannot look converged
//! - Convergence en pas scale-aware : si `‖δ₀‖ ≥ tol.abs + tol.rel · ‖x‖`
//!   on garde le plancher classique `‖δ‖ < tol.abs + tol.rel · ‖x‖` ; si le
//!   premier pas est déjà sous ce plancher (paramètre micrométrique,
//!   `|x★| ≲ tol.abs`, départ à l'origine), le test devient relatif
//!   `‖δ‖ ≤ tol.rel · ‖δ₀‖` — un plancher absolu seul acceptait le premier
//!   pas de Newton incomplet sur `F(x) = (x − x★)³` (retourne `x★/3`)
//! - Convergence en pas testée avant toute détection de stagnation
//! - Stagnation : si le pas ne change plus `x` en `f64` sans que la
//!   tolérance soit atteinte → `SolverError::StepUnderflow`. Le critère
//!   n'est pas un seuil absolu (l'ancien `‖δ‖ < 1e-16` abortait les
//!   systèmes micrométriques bien conditionnés)
//! - Pas de `.unwrap()` — propagation d'erreur via `?`
//! - Newton avec backtracking si le résidu augmente (linesearch simple)
//!
//! Deux entrées :
//!   - `newton_system` : autodiff via Dual numbers
//!   - `newton_system_jac` : F et J fournies séparément

use crate::linalg::{self, Matrix};
use crate::{Solution, SolverError, SolverResult, Tolerance};
use scirust_autodiff::Dual;
use tracing::warn;

fn check_finite(value: f64, _location: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Scale-aware residual stationarity for nonlinear Newton.
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

/// Scale-aware Newton step stationarity (`‖δ‖ < tol.abs + tol.rel · ‖x‖`).
///
/// On a tiny-scale root (`|x★| ≲ tol.abs`, start at the origin) the first
/// Newton step itself sits below `tol.abs`, so a fixed absolute floor
/// accepts after iteration 0 even when the update is incomplete — e.g.
/// `F(x) = (x − x★)³` has Newton factor `2/3`, so the first step lands at
/// `x★/3` and the absolute check returns that point with ~67 % relative
/// error. When `‖δ₀‖` already sits below the classical floor, stop only
/// after a relative reduction against `‖δ₀‖`; otherwise the classical
/// floor is kept.
fn step_stationary(step: f64, x_norm: f64, step0: f64, tol: Tolerance) -> bool {
    if step == 0.0
    {
        return true;
    }
    let classical = tol.abs + tol.rel * x_norm;
    if step0 >= classical || step0 == 0.0
    {
        step < classical
    }
    else
    {
        step <= tol.rel * step0
    }
}

/// Newton multivarié avec jacobienne automatique via dual numbers.
pub fn newton_system<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[Dual], &mut [Dual]),
{
    let n = x0.len();
    let mut x = x0;
    let mut buf_in = vec![Dual::primal(0.0); n];
    let mut buf_out = vec![Dual::primal(0.0); n];
    let mut fx = vec![0.0; n];
    let mut jac = Matrix::zeros(n, n);
    let mut last_res = f64::INFINITY;
    let mut res0 = f64::NAN;
    let mut step0: Option<f64> = None;

    for k in 0..tol.max_iter
    {
        // Évalue F et J colonne par colonne via Dual
        for j in 0..n
        {
            for i in 0..n
            {
                buf_in[i] = Dual::new(x[i], if i == j { 1.0 } else { 0.0 });
            }
            f(&buf_in, &mut buf_out);
            for i in 0..n
            {
                let deriv = buf_out[i].deriv;
                check_finite(deriv, &format!("J[{i},{j}] Newton k={k}"))?;
                jac[(i, j)] = deriv;
                if j == 0
                {
                    let val = buf_out[i].value;
                    check_finite(val, &format!("fx[{i}] Newton k={k}"))?;
                    fx[i] = val;
                }
            }
        }

        let res = linalg::norm_inf(&fx);
        last_res = res;
        if k == 0
        {
            res0 = res;
        }
        if residual_stationary(res, res0, tol)
        {
            return Ok(Solution::new(x, k, res));
        }

        // Résous J · δ = -F
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = linalg::solve(jac.clone(), &rhs).map_err(|e| {
            warn!(target: "solver", "Newton: jacobian singular at iteration {k}: {e:?}");
            e
        })?;

        for (i, &d) in delta.iter().enumerate()
        {
            check_finite(d, &format!("delta[{i}] Newton k={k}"))?;
        }

        let step_norm = linalg::norm_inf(&delta);
        // Convergence on the Newton step is checked after the linesearch
        // update below (relative to ‖x‖). An absolute ‖δ‖ < 1e-16 floor used
        // to abort here on well-conditioned micrometre-scale systems.

        // Mise à jour avec backtracking linesearch simple
        // On essaie d'abord le pas complet, on réduit de moitié si le résidu augmente
        let mut lambda = 1.0;
        let mut x_candidate = vec![0.0; n];
        let mut fx_candidate = vec![0.0; n];
        let mut best_fx_norm = f64::INFINITY;
        let mut best_x = x.clone();
        let mut backtrack_success = false;

        for _ in 0..8
        {
            for i in 0..n
            {
                x_candidate[i] = x[i] + lambda * delta[i];
                check_finite(x_candidate[i], &format!("x_cand[{i}] lambda={lambda}"))?;
            }
            // Évaluer F au candidat via Dual
            for i in 0..n
            {
                buf_in[i] = Dual::primal(x_candidate[i]);
            }
            f(&buf_in, &mut buf_out);
            for i in 0..n
            {
                fx_candidate[i] = buf_out[i].value;
            }
            let cand_res = linalg::norm_inf(&fx_candidate);

            if cand_res < best_fx_norm
            {
                best_fx_norm = cand_res;
                best_x.copy_from_slice(&x_candidate);
                backtrack_success = true;
            }

            if cand_res < res
            {
                // Pas acceptable
                x.copy_from_slice(&x_candidate);
                fx.copy_from_slice(&fx_candidate);
                backtrack_success = true;
                break;
            }
            lambda *= 0.5;
        }

        if !backtrack_success
        {
            warn!(target: "solver", "Newton: backtracking failed at iteration {k}, using best candidate");
            x.copy_from_slice(&best_x);
            // Évaluer fx au best_x
            for i in 0..n
            {
                buf_in[i] = Dual::primal(best_x[i]);
            }
            f(&buf_in, &mut buf_out);
            for i in 0..n
            {
                fx[i] = buf_out[i].value;
            }
        }

        let final_norm = linalg::norm_inf(&fx);
        let x_norm = linalg::norm_inf(&x);
        let final_step = linalg::norm_inf(&delta);
        let step0_val = *step0.get_or_insert(final_step);

        if residual_stationary(final_norm, res0, tol)
            || step_stationary(final_step, x_norm, step0_val, tol)
        {
            // Dernière évaluation propre
            for i in 0..n
            {
                buf_in[i] = Dual::primal(x[i]);
            }
            f(&buf_in, &mut buf_out);
            let res_final = buf_out.iter().fold(0.0f64, |a, d| a.max(d.value.abs()));
            return Ok(Solution::new(x, k + 1, res_final));
        }

        // Stagnation: the full Newton step no longer moves any coordinate in
        // f64 (tolerance finer than machine precision at this scale).
        if delta.iter().all(|&d| d == 0.0) || step_norm == 0.0
        {
            warn!(target: "solver", "Newton: step {step_norm:.3e} no longer moves x at iteration {k}");
            return Err(SolverError::StepUnderflow { step: step_norm });
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_res,
    })
}

/// Variante où l'utilisateur fournit F et J séparément (pas d'autodiff).
pub fn newton_system_jac<F, J>(
    f: F,
    jac: J,
    x0: Vec<f64>,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
    J: Fn(&[f64], &mut Matrix),
{
    let n = x0.len();
    let mut x = x0;
    let mut fx = vec![0.0; n];
    let mut j_mat = Matrix::zeros(n, n);
    let mut res0 = f64::NAN;
    let mut step0: Option<f64> = None;

    for k in 0..tol.max_iter
    {
        f(&x, &mut fx);
        for fi in &fx
        {
            check_finite(*fi, &format!("fx J Newton k={k}"))?;
        }

        let res = linalg::norm_inf(&fx);
        if k == 0
        {
            res0 = res;
        }
        if residual_stationary(res, res0, tol)
        {
            return Ok(Solution::new(x, k, res));
        }

        jac(&x, &mut j_mat);
        // Vérifier que J est finie
        for i in 0..n
        {
            for j in 0..n
            {
                check_finite(j_mat[(i, j)], &format!("J[{i},{j}] Newton k={k}"))?;
            }
        }

        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = linalg::solve(j_mat.clone(), &rhs).map_err(|e| {
            warn!(target: "solver", "Newton(J): solve failed at iteration {k}: {e:?}");
            e
        })?;

        for (i, &d) in delta.iter().enumerate()
        {
            check_finite(d, &format!("delta[{i}] Newton(J) k={k}"))?;
        }

        let step_norm = linalg::norm_inf(&delta);
        let mut x_new = x.clone();
        for i in 0..n
        {
            x_new[i] += delta[i];
            check_finite(x_new[i], &format!("x[{i}] Newton(J) k={k}"))?;
        }

        // Convergence before stagnation — scale-aware step floor (see
        // `step_stationary`); not a fixed 1e-16 and not a bare tol.abs.
        let x_norm = linalg::norm_inf(&x_new);
        let step0_val = *step0.get_or_insert(step_norm);
        if step_stationary(step_norm, x_norm, step0_val, tol)
        {
            f(&x_new, &mut fx);
            return Ok(Solution::new(x_new, k + 1, linalg::norm_inf(&fx)));
        }
        if x_new == x
        {
            warn!(target: "solver", "Newton(J): step {step_norm:.3e} no longer moves x at iteration {k}");
            return Err(SolverError::StepUnderflow { step: step_norm });
        }
        x = x_new;
    }

    f(&x, &mut fx);
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: linalg::norm_inf(&fx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn newton_2d_circle_diagonal() {
        let s = newton_system(
            |x, out| {
                out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
                out[1] = x[0] - x[1];
            },
            vec![1.0, 0.5],
            Tolerance::default(),
        )
        .unwrap();
        let expected = (0.5_f64).sqrt();
        assert_relative_eq!(s.value[0], expected, epsilon = 1e-10);
        assert_relative_eq!(s.value[1], expected, epsilon = 1e-10);
    }

    #[test]
    fn rosenbrock_root() {
        let s = newton_system(
            |x, out| {
                out[0] = (x[1] - x[0] * x[0]) * 10.0;
                out[1] = -x[0] + 1.0;
            },
            vec![-1.2, 1.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(s.value[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn powell_badly_scaled() {
        let s = newton_system(
            |x, out| {
                out[0] = x[0] * x[1] * 1e4 - 1.0;
                out[1] = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
            },
            vec![0.0, 1.0],
            Tolerance {
                abs: 1e-8,
                rel: 1e-6,
                max_iter: 200,
            },
        )
        .unwrap();
        let f0 = s.value[0] * s.value[1] * 1e4 - 1.0;
        let f1 = (-s.value[0]).exp() + (-s.value[1]).exp() - 1.0001;
        assert!(f0.abs() < 1e-6);
        assert!(f1.abs() < 1e-6);
    }

    /// Regression: an absolute `step_norm < 1e-16` floor aborted Newton on a
    /// well-conditioned micrometre-scale linear system before any update.
    /// The Newton step equals the (tiny) residual; it must not be treated as
    /// underflow when it still moves `x` in `f64` and the tolerance has not
    /// been met.
    #[test]
    fn newton_jac_solves_a_tiny_scale_linear_system() {
        let target = 1e-18;
        let s = newton_system_jac(
            move |x, out| {
                out[0] = x[0] - target;
                out[1] = x[1] - 2.0 * target;
            },
            |_x, j| {
                for i in 0..2
                {
                    for jj in 0..2
                    {
                        j[(i, jj)] = 0.0;
                    }
                }
                j[(0, 0)] = 1.0;
                j[(1, 1)] = 1.0;
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

    /// Same absolute-step regression through the autodiff Newton entry point.
    #[test]
    fn newton_system_solves_a_tiny_scale_linear_system() {
        let target = 1e-18;
        let s = newton_system(
            move |x, out| {
                out[0] = x[0] - Dual::primal(target);
                out[1] = x[1] - Dual::primal(2.0 * target);
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
    /// displacement look "converged" and Newton returned the start at
    /// iteration 0. When `|F₀| < tol.abs` stationarity is now relative to
    /// `|F₀|`.
    #[test]
    fn tiny_scale_residual_does_not_false_converge_via_absolute_norm() {
        let scale = 1e-12_f64;
        let x_star = 3.0_f64;
        let s = newton_system(
            move |x, out| {
                out[0] = (x[0] - Dual::primal(x_star)) * Dual::primal(scale);
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

    /// Same absolute-residual regression through the explicit-Jacobian entry.
    #[test]
    fn tiny_scale_residual_newton_jac_does_not_false_converge() {
        let scale = 1e-12_f64;
        let x_star = 3.0_f64;
        let s = newton_system_jac(
            move |x, out| {
                out[0] = scale * (x[0] - x_star);
            },
            move |_x, j| {
                j[(0, 0)] = scale;
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

    /// Already-at-root must still exit immediately (iters == 0).
    #[test]
    fn converges_immediately_when_x0_is_already_the_root() {
        let s = newton_system(
            |x, out| {
                out[0] = x[0] - Dual::primal(3.0);
                out[1] = x[1] - Dual::primal(4.0);
            },
            vec![3.0, 4.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_eq!(s.info.iterations, 0);
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(s.value[1], 4.0, epsilon = 1e-12);
    }

    /// Regression: the classical step floor `‖δ‖ < tol.abs + tol.rel · ‖x‖`
    /// is dominated by `tol.abs` whenever the coordinate scale is ≲
    /// `tol.abs`. For `F(x) = (x − x★)³` with `x★ = 1e-10` and start at 0,
    /// Newton's first step lands at `x★/3` with `‖δ‖ ≈ 3.3e-11 < tol.abs`,
    /// so the buggy check returned that point (~67 % relative error)
    /// without iterating further. When `‖δ₀‖` already sits below the
    /// classical floor, step stationarity is relative to `‖δ₀‖`.
    #[test]
    fn tiny_scale_parameter_does_not_false_converge_via_absolute_step() {
        let x_star = 1e-10_f64;
        let s = newton_system(
            move |x, out| {
                let d = x[0] - Dual::primal(x_star);
                out[0] = d * d * d;
            },
            vec![0.0],
            Tolerance::default(),
        )
        .expect("tiny-scale cubic residual must be solvable");
        assert!(
            (s.value[0] - x_star).abs() < 1e-2 * x_star.abs(),
            "expected x≈{x_star}, got {:?} (iters={})",
            s.value,
            s.info.iterations
        );
        assert!(
            s.info.iterations > 1,
            "must continue past the incomplete first Newton step; absolute tol.abs floor used to return at iter 1 with x=x★/3"
        );
    }

    /// Same absolute-step regression through the explicit-Jacobian entry.
    #[test]
    fn tiny_scale_parameter_newton_jac_does_not_false_converge_via_absolute_step() {
        let x_star = 1e-10_f64;
        let s = newton_system_jac(
            move |x, out| {
                let d = x[0] - x_star;
                out[0] = d * d * d;
            },
            move |x, j| {
                let d = x[0] - x_star;
                j[(0, 0)] = 3.0 * d * d;
            },
            vec![0.0],
            Tolerance::default(),
        )
        .expect("tiny-scale cubic residual must be solvable");
        assert!(
            (s.value[0] - x_star).abs() < 1e-2 * x_star.abs(),
            "expected x≈{x_star}, got {:?} (iters={})",
            s.value,
            s.info.iterations
        );
        assert!(
            s.info.iterations > 1,
            "must continue past the incomplete first Newton step; absolute tol.abs floor used to return at iter 1 with x=x★/3"
        );
    }
}
