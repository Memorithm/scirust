//! Accélération d'Anderson (Walker & Ni, « Anderson Acceleration for
//! Fixed-Point Iterations », SIAM J. Numer. Anal. 49(4), 2011) : accélère
//! une itération à point fixe `x_{k+1} = g(x_k)` en recombinant les
//! dernières itérations plutôt qu'en n'utiliser que la plus récente —
//! utile pour les boucles de couplage de type Picard (multi-physique) ou
//! pour accélérer la convergence lente de Newton/Broyden
//! (`crate::nonlinear`) sur des problèmes mal conditionnés.
//!
//! ## Algorithme (formulation « Type-I » à contrainte de somme unitaire)
//! Avec `f_i = g(x_i) - x_i` le résidu à l'itération `i`, la version
//! Anderson(m) choisit, parmi une fenêtre des `m+1` dernières itérations,
//! les coefficients `α_i` (Σα_i = 1) minimisant `‖Σ α_i f_i‖`, puis pose
//! `x_{k+1} = Σ α_i g(x_i)`. En éliminant `α_0 = 1 - Σ_{i≥1} α_i`, ce
//! problème sous contrainte devient les moindres carrés sans contrainte
//! `min_γ ‖ΔF·γ + f_base‖` (résolus ici par la QR déjà présente dans ce
//! crate — `crate::linalg::solve_qr_least_squares`), avec
//! `x_{k+1} = g_base + ΔG·γ`. C'est mathématiquement équivalent à la
//! formulation par différences la plus souvent citée, mais se lit
//! directement comme « combinaison affine des sorties passées de `g` qui
//! annule au mieux le résidu combiné ».
//!
//! ## Stopping test
//! Stationarity of `‖g(x) − x‖₂` is scale-aware: when the initial residual
//! `|f₀| ≥ tol.abs` the classical test
//! `‖f‖₂ ≤ tol.abs + tol.rel · ‖x‖₂` is kept (no `max(‖x‖, 1)` floor, so
//! `tol.rel` stays relative for small-scale fixed points); when `|f₀|` is
//! already below `tol.abs` (tiny-scale fixed point with `|x★| ≲ tol.abs`),
//! the test becomes relative `‖f‖₂ ≤ tol.rel · |f₀|` so a far start cannot
//! look converged. A fixed point at the origin is still reached through the
//! absolute floor on well-scaled starts.
//!
//! ## Déterminisme
//! Fenêtre de mémoire `m` fixe, moindres carrés résolus par une QR
//! déterministe (pas d'aléa), nombre max d'itérations fixe.

use crate::linalg::{Matrix, norm2, qr_decompose, solve_qr_least_squares};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};

fn check_finite_slice(v: &[f64], iter: usize) -> SolverResult<()> {
    for &x in v
    {
        if !x.is_finite()
        {
            return Err(SolverError::NanDetected { iter, value: x });
        }
    }
    Ok(())
}

/// Scale-aware residual stationarity for Anderson acceleration.
///
/// Under a tiny-scale fixed point (`|x★| ≲ tol.abs`) a fixed absolute
/// `‖g(x) − x‖₂ ≤ tol.abs + tol.rel · ‖x‖₂` floor falsely accepts the start
/// once the residual itself sits below `tol.abs` (e.g. `x0 = 0` while
/// `x★ = 1e-20`). When the initial residual is already below `tol.abs`, the
/// test switches to a relative reduction against `|f₀|`; otherwise the
/// classical abs+rel-in-`x` floor is kept (so well-scaled problems keep the
/// same terminal accuracy, and a fixed point at the origin is still reached
/// through `tol.abs`).
fn residual_stationary(res: f64, res0: f64, x_norm: f64, tol: Tolerance) -> bool {
    if res == 0.0 || res0 == 0.0
    {
        return true;
    }
    if res0 >= tol.abs
    {
        res <= tol.abs + tol.rel * x_norm
    }
    else
    {
        res <= tol.rel * res0
    }
}

/// Accélère l'itération à point fixe `x_{k+1} = g(x_k)` par Anderson(m).
///
/// `m` est la taille de la fenêtre de mémoire (nombre d'itérations passées
/// recombinées ; typiquement 3 à 10). Automatiquement plafonnée à la
/// dimension de `x0` (au-delà, le système de moindres carrés serait
/// sous-déterminé).
pub fn anderson_accelerate<G>(
    g: G,
    x0: Vec<f64>,
    m: usize,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0
    {
        return Err(SolverError::InvalidInput(
            "anderson_accelerate: x0 must be non-empty".to_string(),
        ));
    }
    if m == 0
    {
        return Err(SolverError::InvalidInput(
            "anderson_accelerate: m must be >= 1".to_string(),
        ));
    }
    check_finite_slice(&x0, 0)?;
    let window = m.min(n);

    let mut x = x0;
    let mut gx = g(&x);
    check_finite_slice(&gx, 0)?;
    let mut fx: Vec<f64> = gx.iter().zip(&x).map(|(gi, xi)| gi - xi).collect();

    // Historique (x, g(x), f(x)) de la fenêtre courante, du plus ancien au
    // plus récent.
    let mut history: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> =
        vec![(x.clone(), gx.clone(), fx.clone())];
    let res0 = norm2(&fx);

    for k in 0..tol.max_iter
    {
        let residual = norm2(&fx);
        if residual_stationary(residual, res0, norm2(&x), tol)
        {
            return Ok(Solution {
                value: x,
                info: ConvergenceInfo {
                    iterations: k,
                    residual,
                    converged: true,
                },
            });
        }

        if history.len() == 1
        {
            // Pas encore d'historique : itération à point fixe ordinaire.
            x = gx.clone();
        }
        else
        {
            let (base_x, base_g, base_f) = &history[0];
            let _ = base_x;
            let cols = history.len() - 1;
            let mut delta_f = Matrix::zeros(n, cols);
            let mut delta_g = Matrix::zeros(n, cols);
            for (c, (_, g_i, f_i)) in history[1..].iter().enumerate()
            {
                for i in 0..n
                {
                    delta_f[(i, c)] = f_i[i] - base_f[i];
                    delta_g[(i, c)] = g_i[i] - base_g[i];
                }
            }
            let neg_base_f: Vec<f64> = base_f.iter().map(|v| -v).collect();
            let qr = qr_decompose(delta_f)?;
            let gamma = solve_qr_least_squares(&qr, &neg_base_f)?;

            let mut x_next = base_g.clone();
            for c in 0..cols
            {
                for i in 0..n
                {
                    x_next[i] += delta_g[(i, c)] * gamma[c];
                }
            }
            x = x_next;
        }

        gx = g(&x);
        check_finite_slice(&gx, k + 1)?;
        fx = gx.iter().zip(&x).map(|(gi, xi)| gi - xi).collect();
        check_finite_slice(&fx, k + 1)?;

        history.push((x.clone(), gx.clone(), fx.clone()));
        if history.len() > window + 1
        {
            history.remove(0);
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: norm2(&fx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn accelerates_scalar_cosine_fixed_point() {
        // x = cos(x) -> le nombre de Dottie, ≈0.7390851332151607.
        let sol = anderson_accelerate(
            |x: &[f64]| vec![x[0].cos()],
            vec![0.5],
            5,
            Tolerance::new(1e-12, 1e-12, 200),
        )
        .unwrap();
        assert_relative_eq!(sol.value[0], 0.739_085_133_215_160_7, epsilon = 1e-9);
    }

    #[test]
    fn converges_to_the_true_fixed_point_of_a_linear_map() {
        // g(x) = A x + b, rayon spectral < 1 -> point fixe (I-A)^-1 b.
        // A = [[0.4, 0.1], [0.2, 0.3]], b = [1, 2].
        let g = |x: &[f64]| vec![0.4 * x[0] + 0.1 * x[1] + 1.0, 0.2 * x[0] + 0.3 * x[1] + 2.0];
        let sol = anderson_accelerate(g, vec![0.0, 0.0], 4, Tolerance::default()).unwrap();
        // Résolu à la main : (I-A) x = b.
        let x0 = sol.value[0];
        let x1 = sol.value[1];
        assert_relative_eq!(0.6 * x0 - 0.1 * x1, 1.0, epsilon = 1e-6);
        assert_relative_eq!(-0.2 * x0 + 0.7 * x1, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn accelerates_convergence_versus_plain_fixed_point_iteration() {
        // Itération de Picard nue pour comparaison : x_{k+1} = g(x_k).
        fn plain_iterations(
            g: impl Fn(&[f64]) -> Vec<f64>,
            mut x: Vec<f64>,
            tol: f64,
            max_iter: usize,
        ) -> usize {
            for k in 0..max_iter
            {
                let gx = g(&x);
                let res: f64 = gx
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if res <= tol
                {
                    return k;
                }
                x = gx;
            }
            max_iter
        }
        let g = |x: &[f64]| vec![0.9 * x[0].cos() + 0.1];
        let plain = plain_iterations(g, vec![0.0], 1e-10, 500);
        let sol = anderson_accelerate(g, vec![0.0], 5, Tolerance::new(1e-10, 1e-10, 500)).unwrap();
        assert!(
            sol.info.iterations <= plain,
            "Anderson ({}) should not need more iterations than plain Picard ({plain})",
            sol.info.iterations
        );
    }

    /// Regression: the relative term of the stopping test used
    /// `tol.rel · max(‖x‖, 1)`, so `tol.rel` silently acted as an absolute
    /// tolerance whenever `‖x‖ < 1`. With the default tolerance
    /// (`abs = 1e-10`, `rel = 1e-8`) the threshold was `≈1.01e-8`, and a fixed
    /// point at `x★ = 2e-9` was "found" at the start `x0 = 0` after zero
    /// iterations (100 % relative error).
    #[test]
    fn small_scale_fixed_point_is_not_accepted_at_the_start() {
        let x_star = 2e-9_f64;
        // g(x) = x★ + 0.5·(x − x★): contraction with fixed point x★.
        let g = move |x: &[f64]| vec![x_star + 0.5 * (x[0] - x_star)];
        let sol = anderson_accelerate(g, vec![0.0], 3, Tolerance::default()).unwrap();
        assert!(
            sol.info.iterations > 0,
            "must iterate; the start x0 = 0 is a 100 % error (got {:?})",
            sol.value
        );
        assert_relative_eq!(sol.value[0], x_star, max_relative = 1e-6);
    }

    /// Same regression in 2-D with a coupled linear map at micro-scale.
    #[test]
    fn small_scale_linear_map_converges_to_true_fixed_point() {
        // g(x) = A x + s·b with A = [[0.4, 0.1], [0.2, 0.3]], b = [1, 2].
        let s = 1e-9_f64;
        let g = move |x: &[f64]| {
            vec![
                0.4 * x[0] + 0.1 * x[1] + s,
                0.2 * x[0] + 0.3 * x[1] + 2.0 * s,
            ]
        };
        let sol = anderson_accelerate(g, vec![0.0, 0.0], 4, Tolerance::default()).unwrap();
        // (I − A) x = s·b  =>  x = s·[0.9/0.4, 1.4/0.4] = s·[2.25, 3.5].
        assert_relative_eq!(sol.value[0], 2.25 * s, max_relative = 1e-6);
        assert_relative_eq!(sol.value[1], 3.5 * s, max_relative = 1e-6);
    }

    /// Regression: an absolute `‖f‖₂ ≤ tol.abs + tol.rel · ‖x‖₂` check falsely
    /// accepted the start when the fixed point (and thus the residual) was
    /// measured in tiny units. For `g(x) = x★ + ½(x − x★)` with
    /// `x★ = 1e-20` and `x0 = 0`, the residual `½|x★|` sits below the default
    /// `tol.abs = 1e-10` while the relative error is 100 %. When `|f₀| < tol.abs`
    /// stationarity is now relative to `|f₀|`.
    #[test]
    fn tiny_scale_residual_does_not_false_converge_via_absolute_f() {
        let x_star = 1e-20_f64;
        let g = move |x: &[f64]| vec![x_star + 0.5 * (x[0] - x_star)];
        let sol = anderson_accelerate(g, vec![0.0], 3, Tolerance::default())
            .expect("tiny-scale fixed point must be solvable");
        assert!(
            sol.info.iterations > 0,
            "must iterate; the start x0 = 0 is a 100 % error (got {:?})",
            sol.value
        );
        assert_relative_eq!(sol.value[0], x_star, max_relative = 1e-6);
    }

    /// Same absolute-residual regression in 2-D with a coupled linear map
    /// whose fixed point sits below `tol.abs`.
    #[test]
    fn tiny_scale_linear_map_does_not_false_converge_via_absolute_f() {
        // g(x) = A x + s·b with A = [[0.4, 0.1], [0.2, 0.3]], b = [1, 2].
        let s = 1e-20_f64;
        let g = move |x: &[f64]| {
            vec![
                0.4 * x[0] + 0.1 * x[1] + s,
                0.2 * x[0] + 0.3 * x[1] + 2.0 * s,
            ]
        };
        let sol = anderson_accelerate(g, vec![0.0, 0.0], 4, Tolerance::default())
            .expect("tiny-scale 2-D fixed point must be solvable");
        // (I − A) x = s·b  =>  x = s·[2.25, 3.5].
        assert!(
            sol.info.iterations > 0,
            "must iterate; start is 100 % away (got {:?})",
            sol.value
        );
        assert_relative_eq!(sol.value[0], 2.25 * s, max_relative = 1e-6);
        assert_relative_eq!(sol.value[1], 3.5 * s, max_relative = 1e-6);
    }

    /// A fixed point at the origin still converges through the absolute floor.
    #[test]
    fn fixed_point_at_origin_still_converges_via_absolute_floor() {
        let sol = anderson_accelerate(
            |x: &[f64]| vec![0.5 * x[0].sin()],
            vec![1.0],
            3,
            Tolerance::default(),
        )
        .unwrap();
        assert!(sol.value[0].abs() < 1e-9, "got {:?}", sol.value);
    }

    #[test]
    fn rejects_zero_window() {
        assert!(
            anderson_accelerate(|x: &[f64]| x.to_vec(), vec![0.0], 0, Tolerance::default())
                .is_err()
        );
    }

    #[test]
    fn rejects_empty_initial_point() {
        assert!(
            anderson_accelerate(|x: &[f64]| x.to_vec(), vec![], 3, Tolerance::default()).is_err()
        );
    }
}
