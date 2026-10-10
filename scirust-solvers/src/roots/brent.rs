//! Méthode de Brent — combine bissection, sécante et interpolation
//! quadratique inverse. Garantit la convergence (comme la bissection) et est
//! super-linéaire en pratique.
//!
//! ## Stopping test
//! Accepts when the bracket half-width `|m| = ½|c − b|` satisfies
//! `|m| ≤ 2ε|b| + ½(tol.abs + tol.rel · |b|)` (or `f(b) = 0` exactly).
//! This is the same abs+rel scale as bisection/secant for the full bracket,
//! plus Brent's classical `2ε|b|` safeguard. The previous formula used only
//! `½·tol.abs + 2ε|b|` and ignored `tol.rel`, so with a zero absolute floor
//! Brent kept iterating down to working precision and could exhaust
//! `max_iter` even when `|m|` was already far below `tol.rel · |b|`.
//!
//! ## Sécurité numérique
//! - Détection d'oscillation stérile : si l'interpolation est refusée pendant
//!   `MAX_OSCILLATION` itérations consécutives, on force un pas de bissection
//!   pour casser le Zeno
//! - Division par zéro dans `p/q` protégée par `|q| < tol_act · 1e-10`
//!   (force alors un pas de bissection)
//! - check_finite sur chaque évaluation de f
//!
//! Référence : Brent, *Algorithms for Minimization Without Derivatives*, 1973.

use crate::{Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

/// Compteur d'oscillation : si l'interpolation stagne, on force bissection.
const MAX_OSCILLATION: u32 = 5;

fn check_finite(v: f64, _label: &str) -> Result<(), SolverError> {
    if !v.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: v });
    }
    Ok(())
}

/// Méthode de Brent dans `[a, b]`. Requiert `f(a)·f(b) < 0`.
pub fn brent<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>> {
    let mut fa = f(a);
    let mut fb = f(b);
    check_finite(fa, "fa")?;
    check_finite(fb, "fb")?;

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

    if fa.abs() < fb.abs()
    {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    let mut oscillation_count = 0u32;

    for k in 0..tol.max_iter
    {
        if fc.signum() == fb.signum()
        {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs()
        {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        // Half-width acceptance uses the same abs+rel scale as bisection /
        // secant for the full bracket, plus the classical 2ε|b| safeguard.
        // Ignoring `tol.rel` (previous `0.5 · tol.abs + 2ε|b|` only) forced a
        // machine-eps bracket whenever `tol.abs = 0` and could exhaust
        // `max_iter` despite `|m|` already satisfying `tol.rel · |b|`.
        let tol_act = 2.0 * f64::EPSILON * b.abs() + 0.5 * (tol.abs + tol.rel * b.abs());
        let m = 0.5 * (c - b);
        if m.abs() <= tol_act || fb == 0.0
        {
            return Ok(Solution::new(b, k, fb.abs()));
        }

        let mut use_bisection = false;

        // Tente interpolation si la condition de Brent le permet
        if e.abs() >= tol_act && fa.abs() > fb.abs()
        {
            let s = fb / fa;
            let (mut p, mut q): (f64, f64);
            check_finite(s, "s")?;

            if a == c
            {
                // Sécante
                p = 2.0 * m * s;
                q = 1.0 - s;
            }
            else
            {
                // Interpolation quadratique inverse
                let qq = fa / fc;
                let r = fb / fc;
                check_finite(qq, "qq")?;
                check_finite(r, "r")?;
                p = s * (2.0 * m * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }

            check_finite(p, "p")?;
            check_finite(q, "q")?;

            if p > 0.0
            {
                q = -q;
            }
            else
            {
                p = -p;
            }

            // Vérifier que la division par q est sûre
            if q.abs() < tol_act * 1e-10
            {
                use_bisection = true;
            }
            else if 2.0 * p < (3.0 * m * q - (tol_act * q).abs()).min((e * q).abs())
            {
                e = d;
                d = p / q;
                check_finite(d, "d=p/q")?;
                oscillation_count = 0;
            }
            else
            {
                use_bisection = true;
            }
        }
        else
        {
            use_bisection = true;
        }

        // Détection d'oscillation stérile — forcer bissection
        if use_bisection
        {
            oscillation_count += 1;
            if oscillation_count >= MAX_OSCILLATION
            {
                warn!(
                    target: "solver",
                    "Brent: oscillation detected ({oscillation_count} consecutive bisections), forcing pure bisection step"
                );
                // Forcer un pas de bissection (plus fiable)
                d = m;
                e = d;
                oscillation_count = 0;
            }
            else
            {
                d = m;
                e = d;
            }
        }

        a = b;
        fa = fb;

        let step = if d.abs() > tol_act
        {
            d
        }
        else if m > 0.0
        {
            tol_act
        }
        else
        {
            -tol_act
        };
        check_finite(step, "step")?;
        b += step;
        check_finite(b, "b")?;

        fb = f(b);
        check_finite(fb, "fb")?;
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: fb.abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn brent_cos() {
        let s = brent(|x| x.cos(), 0.0, PI, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, PI / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn brent_cubic_fast() {
        let s = brent(
            |x| x.powi(3) - 2.0 * x - 5.0,
            2.0,
            3.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-12);
        assert!(s.info.iterations < 20);
    }

    #[test]
    fn brent_difficult_function() {
        let f = |x: f64| (1..=10).map(|i| x - i as f64).product::<f64>();
        let s = brent(f, 4.5, 5.5, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, 5.0, epsilon = 1e-9);
    }

    /// Regression: with `tol.abs = 0` the previous `tol_act = 2ε|b| + ½·tol.abs`
    /// ignored `tol.rel` and demanded a machine-eps bracket. On this cubic a
    /// modest `max_iter` then returned `NoConvergence` with `|f(b)|` already
    /// ~1e-14 — far tighter than `tol.rel` — because `|m|` was still ~1e-8
    /// (above `2ε|b|`, below `tol.rel · |b|`). Half-width acceptance now uses
    /// `½ · (tol.abs + tol.rel · |b|)`.
    #[test]
    fn brent_honors_relative_tolerance_when_abs_floor_is_zero() {
        let tol = Tolerance::new(0.0, 1e-8, 6);
        let s = brent(|x| x.powi(3) - 2.0 * x - 5.0, 2.0, 3.0, tol)
            .expect("must stop via tol.rel once |m| ≲ tol.rel·|b|; previously NoConvergence");
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, max_relative = 1e-7);
        assert!(
            s.info.iterations <= 6,
            "expected early accept via tol.rel, got {} iterations",
            s.info.iterations
        );
    }
}
