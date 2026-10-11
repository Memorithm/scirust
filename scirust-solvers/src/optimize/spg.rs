//! Gradient projeté spectral (Spectral Projected Gradient, SPG) pour
//! l'optimisation sous contraintes de boîte `l ≤ x ≤ u`.
//!
//! Combine le pas de Barzilai-Borwein (approximation de second ordre sans
//! Hessienne) avec une recherche linéaire d'Armijo **non monotone** (fenêtre
//! de mémoire fixe sur les dernières valeurs de `f`) et une projection sur
//! la boîte. Plus léger qu'un L-BFGS-B complet mais couvre le même besoin
//! (calibration de modèle bornée, QP de boîte pour MPC) avec une preuve de
//! convergence globale sur ensembles convexes.
//!
//! Référence : E.G. Birgin, J.M. Martínez, M. Raydan, « Nonmonotone
//! Spectral Projected Gradient Methods on Convex Sets », SIAM J. Optim.
//! 10(4), 2000. Voir aussi Nocedal & Wright, *Numerical Optimization*,
//! 2e éd., chap. 17 pour le cadre gradient projeté.
//!
//! ## Déterminisme
//! Fenêtre de mémoire non monotone (`MEMORY`), facteur de rétrécissement du
//! backtracking (`0.5`) et nombre max de retours en arrière (`MAX_BACKTRACK`)
//! tous fixes — aucun critère temporel.
//!
//! ## Sécurité numérique
//! Le pas spectral de Barzilai–Borwein `α = ‖s‖² / (s·y)` a la dimension
//! d'une inverse de courbure. Les garde-fous `ALPHA_MIN` / `ALPHA_MAX` sont
//! donc volontairement larges (`1e-30` … `1e30`, cf. Birgin–Martínez–Raydan) :
//! un clamp absolu plus serré (`1e-10` … `1e10`) forçait un pas trop grand
//! sur les quadratiques raides (`k ≳ 1e10`) et trop petit sur les
//! quadratiques plates (`k ≲ 1e-10`), empêchant la convergence. La recherche
//! linéaire d'Armijo et la projection sur la boîte restent les garde-fous
//! actifs contre les pas extrêmes.
//!
//! ## Stopping test
//! Stationarity uses the unit-step projected gradient
//! `d = P(x − ∇f) − x` (not the spectral Barzilai–Borwein trial, whose
//! `α = 1/‖∇f‖` would inflate a micrometre-scale gradient into a
//! box-truncated step of size `O(bound)`). The test is scale-aware: when
//! `|d₀| ≥ tol.abs` the classical
//! `‖d‖₂ ≤ tol.abs + tol.rel · ‖x‖₂` floor is kept (no `max(‖x‖, 1)`);
//! when `|d₀|` is already below `tol.abs` (tiny-scale minimum with
//! `|x★| ≲ tol.abs`), it becomes relative `‖d‖₂ ≤ tol.rel · |d₀|` so a far
//! start cannot look converged. A minimum at the origin is still reached
//! through the absolute floor on well-scaled starts. The spectral step
//! remains the search direction only.

use crate::linalg::{dot, norm2};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};

/// Scale-aware stationarity test for the unit-step projected gradient
/// `d = P(x − ∇f) − x`.
///
/// Under a tiny-scale box-constrained minimum (`|x★| ≲ tol.abs`) a fixed
/// absolute `‖d‖₂ ≤ tol.abs + tol.rel · ‖x‖₂` floor falsely accepts the start
/// once `‖P(x₀ − ∇f) − x₀‖` itself sits below `tol.abs` (e.g. `x0 = 0`,
/// `x★ = 1e-20` with default `tol.abs = 1e-10`). Stopping on the *spectral*
/// trial `P(x − α∇f) − x` with `α = 1/‖∇f‖` is worse still: the oversized
/// step is truncated to the box bound (e.g. `1e-12`), which also sits below
/// `tol.abs` and returns the origin at iteration 0. When the initial
/// unit-step projected gradient is already below `tol.abs`, the test
/// switches to a relative reduction against `|d₀|`; otherwise the classical
/// abs+rel-in-`x` floor is kept. The relative term never uses a
/// `max(‖x‖, 1)` floor.
fn projected_stationary(dnorm: f64, d0: f64, x_norm: f64, tol: Tolerance) -> bool {
    if dnorm == 0.0 || d0 == 0.0
    {
        return true;
    }
    if d0 >= tol.abs
    {
        dnorm <= tol.abs + tol.rel * x_norm
    }
    else
    {
        dnorm <= tol.rel * d0
    }
}

const MEMORY: usize = 10;
const GAMMA: f64 = 1e-4;
const MAX_BACKTRACK: usize = 30;
/// Lower safeguard on the Barzilai–Borwein spectral step (1/curvature).
/// Must stay far below the curvature of stiff but well-posed objectives —
/// see `spg_finds_minimum_of_stiff_quadratic`.
const ALPHA_MIN: f64 = 1e-30;
/// Upper safeguard on the spectral step. Symmetric of `ALPHA_MIN` for flat
/// objectives — see `spg_finds_minimum_of_flat_quadratic`.
const ALPHA_MAX: f64 = 1e30;

fn project_box(x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lower)
        .zip(upper)
        .map(|((&xi, &lo), &hi)| xi.clamp(lo, hi))
        .collect()
}

/// Minimise `f` sous `lower ≤ x ≤ upper` par gradient projeté spectral.
///
/// `grad` doit renvoyer le gradient exact ou une approximation cohérente de
/// `f`. `x0` est projeté sur la boîte avant la première itération.
pub fn spg<F, G>(
    f: F,
    grad: G,
    x0: Vec<f64>,
    lower: &[f64],
    upper: &[f64],
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if lower.len() != n || upper.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: lower.len().min(upper.len()),
        });
    }
    for i in 0..n
    {
        if lower[i] > upper[i]
        {
            return Err(SolverError::InvalidInput(format!(
                "spg: lower[{i}] ({}) > upper[{i}] ({})",
                lower[i], upper[i]
            )));
        }
    }

    let mut x = project_box(&x0, lower, upper);
    let mut fx = f(&x);
    if !fx.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: fx });
    }
    let mut g = grad(&x);
    let mut alpha = {
        let gn = norm2(&g);
        if gn > 1e-300
        {
            (1.0 / gn).clamp(ALPHA_MIN, ALPHA_MAX)
        }
        else
        {
            1.0
        }
    };
    let mut history = vec![fx];
    let mut d0: Option<f64> = None;

    for k in 0..tol.max_iter
    {
        // Unit-step projected gradient for the stopping test (physical
        // scale of ∇f). The spectral trial below is the search direction
        // only — using it for stopping false-converges on tiny-scale
        // problems whose box truncates α·∇f below tol.abs.
        let mut trial_pg = vec![0.0; n];
        for i in 0..n
        {
            trial_pg[i] = x[i] - g[i];
        }
        let x_pg = project_box(&trial_pg, lower, upper);
        let mut d_pg = vec![0.0; n];
        for i in 0..n
        {
            d_pg[i] = x_pg[i] - x[i];
        }
        let dnorm = norm2(&d_pg);
        let d0_val = *d0.get_or_insert(dnorm);

        if projected_stationary(dnorm, d0_val, norm2(&x), tol)
        {
            return Ok(Solution {
                value: x,
                info: ConvergenceInfo {
                    iterations: k,
                    residual: dnorm,
                    converged: true,
                },
            });
        }

        let mut trial = vec![0.0; n];
        for i in 0..n
        {
            trial[i] = x[i] - alpha * g[i];
        }
        let x_trial = project_box(&trial, lower, upper);
        let mut d = vec![0.0; n];
        for i in 0..n
        {
            d[i] = x_trial[i] - x[i];
        }

        let gd = dot(&g, &d);
        let f_max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut lambda = 1.0f64;
        let mut x_new = vec![0.0; n];
        let mut f_new = fx;
        let mut accepted = false;
        for _ in 0..MAX_BACKTRACK
        {
            for i in 0..n
            {
                x_new[i] = x[i] + lambda * d[i];
            }
            f_new = f(&x_new);
            if !f_new.is_finite()
            {
                return Err(SolverError::NanDetected {
                    iter: k,
                    value: f_new,
                });
            }
            if f_new <= f_max + GAMMA * lambda * gd
            {
                accepted = true;
                break;
            }
            lambda *= 0.5;
        }
        if !accepted
        {
            return Err(SolverError::StepUnderflow { step: lambda });
        }

        let g_new = grad(&x_new);
        let mut s = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n
        {
            s[i] = x_new[i] - x[i];
            y[i] = g_new[i] - g[i];
        }
        let sy = dot(&s, &y);
        alpha = if sy > 1e-300
        {
            (dot(&s, &s) / sy).clamp(ALPHA_MIN, ALPHA_MAX)
        }
        else
        {
            ALPHA_MAX
        };

        x = x_new;
        g = g_new;
        fx = f_new;
        history.push(fx);
        if history.len() > MEMORY
        {
            history.remove(0);
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: fx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spg_finds_unconstrained_minimum_inside_box() {
        // min (x-1)^2 + (y-2)^2 sur [-5,5]x[-5,5] : optimum non contraint (1,2).
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
        let sol = spg(
            f,
            grad,
            vec![0.0, 0.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sol.value[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn spg_clamps_to_active_bound() {
        // min (x-3)^2 + (y+2)^2 sur [0,1]x[-1,1] : optimum au coin (1,-1).
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)];
        let sol = spg(
            f,
            grad,
            vec![0.5, 0.0],
            &[0.0, -1.0],
            &[1.0, 1.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sol.value[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn spg_projects_initial_point_into_box() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64]| vec![2.0 * x[0]];
        // x0=10 est hors boîte [-1,1] ; doit converger vers 0 (minimum global, dans la boîte).
        let sol = spg(f, grad, vec![10.0], &[-1.0], &[1.0], Tolerance::default()).unwrap();
        assert_relative_eq!(sol.value[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn spg_rejects_inverted_bounds() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64]| vec![2.0 * x[0]];
        let res = spg(f, grad, vec![0.0], &[1.0], &[-1.0], Tolerance::default());
        assert!(res.is_err());
    }

    /// Regression: the Barzilai–Borwein step α = ‖s‖² / (s·y) equals 1/k for
    /// the quadratic ½k‖x − x*‖². Absolute clamps α ∈ [1e-10, 1e10] forced
    /// α = 1e-10 on a well-conditioned stiff problem (k = 1e12 ⇒ α* = 1e-12),
    /// so the projected step overshot, Armijo rejected every trial, and SPG
    /// returned NoConvergence instead of the obvious minimum inside the box.
    #[test]
    fn spg_finds_minimum_of_stiff_quadratic() {
        let k = 1e12_f64;
        let f = move |x: &[f64]| 0.5 * k * ((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2));
        let grad = move |x: &[f64]| vec![k * (x[0] - 1.0), k * (x[1] - 2.0)];
        let sol = spg(
            f,
            grad,
            vec![0.0, 0.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            Tolerance::new(1e-8, 1e-8, 200),
        )
        .expect("stiff box-constrained quadratic must converge");
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sol.value[1], 2.0, epsilon = 1e-5);
    }

    /// Same clamp bug on the flat side: k = 1e-12 ⇒ α* = 1e12, which the old
    /// ALPHA_MAX = 1e10 truncated so the spectral step crawled instead of
    /// taking the Newton-like BB step.
    #[test]
    fn spg_finds_minimum_of_flat_quadratic() {
        let k = 1e-12_f64;
        let f = move |x: &[f64]| 0.5 * k * (x[0] - 1.0).powi(2);
        let grad = move |x: &[f64]| vec![k * (x[0] - 1.0)];
        let sol = spg(
            f,
            grad,
            vec![0.0],
            &[-5.0],
            &[5.0],
            Tolerance::new(1e-8, 1e-8, 50),
        )
        .expect("flat box-constrained quadratic must converge");
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
    }

    /// Regression: the stopping test used `tol.rel · max(‖x‖, 1)`, which
    /// turns `tol.rel` into an absolute floor whenever ‖x‖ < 1. With
    /// `Tolerance::default()` the threshold is ≈ `1.01e-8`. On a micrometre-
    /// scale quadratic whose box truncates the first projected step to that
    /// magnitude, the buggy check accepted `x₀ = 0` at iteration 0 (100 %
    /// error) instead of walking to `x★`. The relative term is now
    /// `tol.rel · ‖x‖` with no floor.
    #[test]
    fn small_scale_box_minimum_is_not_accepted_at_the_start() {
        let x_star = 2e-9_f64;
        let f = move |x: &[f64]| 0.5 * (x[0] - x_star).powi(2);
        let grad = move |x: &[f64]| vec![x[0] - x_star];
        // Tight box: initial α = 1/|g₀| ≈ 5e8 would step to ~1, but projection
        // truncates to the upper bound 1e-8 — exactly the floored threshold.
        let sol = spg(f, grad, vec![0.0], &[-1e-8], &[1e-8], Tolerance::default())
            .expect("small-scale box quadratic must converge");
        assert!(
            sol.info.iterations > 0,
            "buggy max(|x|,1) floor accepts the start at iteration 0"
        );
        assert_relative_eq!(sol.value[0], x_star, max_relative = 1e-6);
    }

    /// Same floor bug in 2-D: projected first step lands at the box corner
    /// with ‖d‖₂ ≈ 9.9e-9 while ‖x₀‖ = 0, so the floored test stops immediately.
    #[test]
    fn small_scale_2d_box_converges_to_true_minimum() {
        let x_star = [2.25e-9_f64, 3.5e-9];
        let f = move |x: &[f64]| 0.5 * ((x[0] - x_star[0]).powi(2) + (x[1] - x_star[1]).powi(2));
        let grad = move |x: &[f64]| vec![x[0] - x_star[0], x[1] - x_star[1]];
        // Bound 7e-9 so ‖d₀‖₂ = √2·7e-9 ≈ 9.9e-9 ≤ floored threshold 1.01e-8
        // (with 1e-8 the corner step already exceeds the floor and slips past
        // iteration 0).
        let sol = spg(
            f,
            grad,
            vec![0.0, 0.0],
            &[-7e-9, -7e-9],
            &[7e-9, 7e-9],
            Tolerance::default(),
        )
        .expect("small-scale 2-D box quadratic must converge");
        assert!(sol.info.iterations > 0);
        assert_relative_eq!(sol.value[0], x_star[0], max_relative = 1e-6);
        assert_relative_eq!(sol.value[1], x_star[1], max_relative = 1e-6);
    }

    /// Origin minimum must still be accepted via `tol.abs` once the projected
    /// step is below the absolute floor (no relative contribution at x = 0).
    #[test]
    fn minimum_at_origin_still_converges_via_absolute_floor() {
        let f = |x: &[f64]| 0.5 * x[0] * x[0];
        let grad = |x: &[f64]| vec![x[0]];
        let sol = spg(f, grad, vec![1e-4], &[-1.0], &[1.0], Tolerance::default())
            .expect("origin minimum must converge");
        assert!(sol.value[0].abs() < 1e-8);
        assert!(sol.info.converged);
    }

    /// Regression: an absolute `‖d‖₂ ≤ tol.abs + tol.rel · ‖x‖₂` check on the
    /// *spectral* projected trial falsely accepted a far-from-minimum start
    /// once the box truncated `α·∇f` (with `α = 1/‖∇f‖`) below `tol.abs`.
    /// With `x★ = 1e-20`, bounds `±1e-12` and default `tol.abs = 1e-10`, that
    /// trial has ‖d‖ = 1e-12 while the relative error is 100 %. Stopping now
    /// uses the unit-step projected gradient (scale-aware): when `|d₀| <
    /// tol.abs` the criterion is relative to `|d₀|`.
    #[test]
    fn tiny_scale_box_does_not_false_converge_via_absolute_projected_step() {
        let x_star = 1e-20_f64;
        let f = move |x: &[f64]| 0.5 * (x[0] - x_star).powi(2);
        let grad = move |x: &[f64]| vec![x[0] - x_star];
        // Spectral α ≈ 1e20 would step to ~1, projection to 1e-12 ≪ tol.abs —
        // the buggy spectral absolute floor stopped at iteration 0.
        let sol = spg(
            f,
            grad,
            vec![0.0],
            &[-1e-12],
            &[1e-12],
            Tolerance::default(),
        )
        .expect("tiny-scale box quadratic must not false-converge at the origin");
        assert!(
            sol.info.iterations > 0,
            "buggy absolute spectral ‖d‖ floor accepts the start at iteration 0"
        );
        assert!(
            (sol.value[0] - x_star).abs() <= 1e-6 * x_star.abs(),
            "got {} want {}",
            sol.value[0],
            x_star
        );
    }

    /// Same absolute-floor bug in 2-D: spectral box-corner step
    /// ‖d‖₂ = √2·1e-12 sits below `tol.abs` while `x★` is orders of magnitude
    /// smaller; the unit-step projected gradient exposes the true scale.
    #[test]
    fn tiny_scale_2d_box_does_not_false_converge_via_absolute_projected_step() {
        let x_star = [1.5e-20_f64, 2.5e-20];
        let f = move |x: &[f64]| 0.5 * ((x[0] - x_star[0]).powi(2) + (x[1] - x_star[1]).powi(2));
        let grad = move |x: &[f64]| vec![x[0] - x_star[0], x[1] - x_star[1]];
        let sol = spg(
            f,
            grad,
            vec![0.0, 0.0],
            &[-1e-12, -1e-12],
            &[1e-12, 1e-12],
            Tolerance::default(),
        )
        .expect("tiny-scale 2-D box quadratic must not false-converge at the origin");
        assert!(sol.info.iterations > 0);
        assert!(
            (sol.value[0] - x_star[0]).abs() <= 1e-6 * x_star[0].abs(),
            "got {} want {}",
            sol.value[0],
            x_star[0]
        );
        assert!(
            (sol.value[1] - x_star[1]).abs() <= 1e-6 * x_star[1].abs(),
            "got {} want {}",
            sol.value[1],
            x_star[1]
        );
    }
}
