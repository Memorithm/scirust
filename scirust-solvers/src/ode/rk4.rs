//! Runge-Kutta classique d'ordre 4 à pas fixe, avec guards NaN.
//!
//! ```text
//! k1 = f(t, y)
//! k2 = f(t + h/2, y + h/2 · k1)
//! k3 = f(t + h/2, y + h/2 · k2)
//! k4 = f(t + h,   y + h   · k3)
//! y_{n+1} = y_n + h/6 · (k1 + 2 k2 + 2 k3 + k4)
//! ```

use crate::{SolverError, SolverResult};

/// Intègre `dy/dt = f(t, y)` de `t0` à `t_end` par pas `h`.
/// Renvoie `Vec<(t, y)>` avec tous les points de discrétisation.
///
/// Les instants de sortie sont `t0 + k·h`, calculés à partir de l'indice de
/// pas (l'arrondi ne s'accumule pas), et le dernier vaut exactement `t_end`.
/// Lorsque `(t_end - t0) / h` est à `1e-9` (relatif) d'un entier, ce nombre
/// de pas égaux est utilisé, sans pas résiduel minuscule ; sinon le dernier
/// pas est raccourci. `t_end == t0` renvoie le seul point initial.
///
/// # Errors
///
/// [`SolverError::InvalidInput`] si `t0`/`t_end` ne sont pas finis,
/// `t_end < t0`, `h` n'est pas fini et strictement positif, si `h` est
/// inférieur à la résolution flottante de `t` sur `[t0, t_end]` (`t + h`
/// s'arrondirait en `t`), ou si la trajectoire demandée est trop grande ;
/// [`SolverError::NanDetected`] si l'état ou une dérivée devient non fini.
pub fn rk4_fixed<F>(
    f: F,
    t0: f64,
    t_end: f64,
    y0: Vec<f64>,
    h: f64,
) -> SolverResult<Vec<(f64, Vec<f64>)>>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    if !t0.is_finite() || !t_end.is_finite()
    {
        return Err(SolverError::InvalidInput(
            "t0 and t_end must be finite".into(),
        ));
    }
    if t_end < t0
    {
        return Err(SolverError::InvalidInput("t_end must be >= t0".into()));
    }
    if !h.is_finite() || h <= 0.0
    {
        return Err(SolverError::InvalidInput("h must be finite and > 0".into()));
    }
    if let Some(&value) = y0.iter().find(|value| !value.is_finite())
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }

    let n = y0.len();
    let span = t_end - t0;
    if !span.is_finite()
    {
        return Err(SolverError::InvalidInput(
            "integration interval is too large".into(),
        ));
    }
    // `t + h` must actually advance the clock at the magnitude of `t0` /
    // `t_end`; otherwise every step would be a zero step at a frozen `t`.
    let t_mag = t0.abs().max(t_end.abs());
    if span > 0.0 && t_mag + h == t_mag
    {
        return Err(below_resolution(h, t_mag));
    }
    // `span / h` rounded up, except that a quotient within a relative 1e-9 of
    // an integer is rounded to it: `h = 0.1` on `[0, 1]` is ten equal steps.
    let q = span / h;
    let nsteps_f = if (q - q.round()).abs() <= 1e-9 * q.max(1.0)
    {
        q.round()
    }
    else
    {
        q.ceil()
    };
    if !nsteps_f.is_finite() || nsteps_f > (usize::MAX - 1) as f64
    {
        return Err(SolverError::InvalidInput(
            "requested fixed-step trajectory is too large".into(),
        ));
    }
    // At least one step whenever the span is non-empty (a span far shorter
    // than `h` would otherwise round to zero steps).
    let nsteps = if span > 0.0
    {
        (nsteps_f as usize).max(1)
    }
    else
    {
        0
    };
    let capacity = nsteps.checked_add(1).ok_or_else(|| {
        SolverError::InvalidInput("requested fixed-step trajectory is too large".into())
    })?;
    let mut out = Vec::new();
    out.try_reserve(capacity).map_err(|_| {
        SolverError::IntegrationFailed("unable to allocate fixed-step trajectory".into())
    })?;
    let mut t = t0;
    let mut y = y0;
    out.push((t, y.clone()));

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut ytmp = vec![0.0; n];

    for step in 0..nsteps
    {
        // Grid time from the step index; the last one is exactly `t_end`.
        let t_next = if step + 1 >= nsteps
        {
            t_end
        }
        else
        {
            (t0 + (step + 1) as f64 * h).min(t_end)
        };
        let h_act = t_next - t;
        if h_act <= 0.0
        {
            return Err(below_resolution(h, t));
        }

        f(t, &y, &mut k1);
        if let Some(&value) = k1.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + 0.5 * h_act * k1[i];
        }
        if let Some(&value) = ytmp.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }
        f(t + 0.5 * h_act, &ytmp, &mut k2);
        if let Some(&value) = k2.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + 0.5 * h_act * k2[i];
        }
        if let Some(&value) = ytmp.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }
        f(t + 0.5 * h_act, &ytmp, &mut k3);
        if let Some(&value) = k3.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + h_act * k3[i];
        }
        if let Some(&value) = ytmp.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }
        f(t + h_act, &ytmp, &mut k4);
        if let Some(&value) = k4.iter().find(|value| !value.is_finite())
        {
            return Err(SolverError::NanDetected { iter: step, value });
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + h_act / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            if !ytmp[i].is_finite()
            {
                // NaN détecté après mise à jour — on stoppe proprement
                return Err(SolverError::NanDetected {
                    iter: step,
                    value: ytmp[i],
                });
            }
        }
        y.copy_from_slice(&ytmp);
        t = t_next;
        out.push((t, y.clone()));
    }
    Ok(out)
}

fn below_resolution(h: f64, t: f64) -> SolverError {
    SolverError::InvalidInput(format!(
        "step size h = {h:e} is below the floating-point resolution of t near {t:e}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exponential_decay() {
        let out = rk4_fixed(|_, y, dy| dy[0] = -y[0], 0.0, 1.0, vec![1.0], 0.01).unwrap();
        let (t_final, y_final) = out.last().unwrap();
        assert_relative_eq!(*t_final, 1.0, epsilon = 1e-12);
        assert_relative_eq!(y_final[0], (-1.0_f64).exp(), epsilon = 1e-8);
    }

    #[test]
    fn harmonic_oscillator() {
        let out = rk4_fixed(
            |_, y, dy| {
                dy[0] = y[1];
                dy[1] = -y[0];
            },
            0.0,
            std::f64::consts::PI,
            vec![1.0, 0.0],
            0.001,
        )
        .unwrap();
        let (_, y_final) = out.last().unwrap();
        assert_relative_eq!(y_final[0], -1.0, epsilon = 1e-8);
        assert!(y_final[1].abs() < 1e-7);
    }

    /// Régression : lorsqu'un NaN/inf apparaît lors de la mise à jour finale,
    /// le point non-fini ne doit PAS être enregistré (guard NaN propre).
    ///
    /// La dérivée renvoie une constante `f64::MAX / 2.0` : chaque `k_i` et
    /// chaque étape intermédiaire restent finis, mais la somme pondérée
    /// `k1 + 2·k2 + 2·k3 + k4 = 6·C` déborde vers `+inf`, ce qui rend `y`
    /// non-fini uniquement au moment de la mise à jour finale.
    /// Avant le correctif, le `break` ne quittait que la boucle interne et le
    /// point infini était tout de même poussé dans `out`.
    #[test]
    fn nan_on_final_update_is_not_recorded() {
        let out = rk4_fixed(|_, _, dy| dy[0] = f64::MAX / 2.0, 0.0, 1.0, vec![0.0], 1e-6);
        // Seul le point initial doit subsister : le pas ayant produit l'infini
        // est rejeté proprement.
        assert!(matches!(out, Err(SolverError::NanDetected { .. })));
        // Aucun point enregistré ne doit contenir de valeur non-finie.
    }

    #[test]
    fn invalid_step_and_bounds_are_reported_before_step_count() {
        let f = |_: f64, _: &[f64], _: &mut [f64]| {};
        assert!(matches!(
            rk4_fixed(f, 0.0, 1.0, vec![1.0], 0.0),
            Err(SolverError::InvalidInput(_))
        ));
        assert!(rk4_fixed(f, 0.0, 1.0, vec![1.0], f64::NAN).is_err());
        assert!(rk4_fixed(f, 0.0, 1.0, vec![1.0], f64::INFINITY).is_err());
        assert!(rk4_fixed(f, 1.0, 0.0, vec![1.0], 0.1).is_err());
        assert!(rk4_fixed(f, f64::NAN, 1.0, vec![1.0], 0.1).is_err());
    }

    /// Regression: the clock was advanced with `t += h`, so rounding piled up
    /// and the trajectory stopped short of `t_end` (`0.9999999999999999` for
    /// `h = 0.1` on `[0, 1]`, `2.99999999999978` for `h = 1e-3` on `[0, 3]`).
    #[test]
    fn trajectory_ends_exactly_at_t_end_on_an_even_grid() {
        let f = |_: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
        for &(t_end, h, steps) in &[(1.0, 0.1, 10), (3.0, 1e-3, 3000), (5.0, 0.05, 100)]
        {
            let out = rk4_fixed(f, 0.0, t_end, vec![1.0], h).unwrap();
            assert_eq!(out.len(), steps + 1, "t_end={t_end}, h={h}");
            assert_eq!(out.last().unwrap().0, t_end, "t_end={t_end}, h={h}");
            for (k, (t, _)) in out.iter().enumerate().take(steps)
            {
                assert_eq!(*t, k as f64 * h);
            }
        }
    }

    /// A span that is not a multiple of `h` still ends on `t_end`, with a
    /// single shortened last step.
    #[test]
    fn uneven_span_shortens_only_the_last_step() {
        let out = rk4_fixed(|_, _, dy| dy[0] = 1.0, 0.0, 1.0, vec![0.0], 0.3).unwrap();
        let ts: Vec<f64> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(ts.len(), 5);
        assert_eq!(ts[4], 1.0);
        assert!((ts[4] - ts[3] - 0.1).abs() < 1e-12);
        assert!((out[4].1[0] - 1.0).abs() < 1e-12);
    }

    /// Regression: with `h` below the resolution of `t` (`ulp(1e10) ≈ 1.9e-6`),
    /// `t += h` left `t` frozen at `t0`, so the solver returned ten million
    /// copies of the initial point ending at `t0`, far from `t_end`.
    #[test]
    fn step_below_time_resolution_is_rejected() {
        let f = |_: f64, _: &[f64], dy: &mut [f64]| dy[0] = 1.0;
        assert!(matches!(
            rk4_fixed(f, 1e10, 1e10 + 1.0, vec![0.0], 1e-7),
            Err(SolverError::InvalidInput(_))
        ));
        // An empty span is still fine and returns the initial point.
        let out = rk4_fixed(f, 1e10, 1e10, vec![0.0], 1e-7).unwrap();
        assert_eq!(out.len(), 1);
    }

    /// A span much shorter than `h` is integrated in one step to `t_end`.
    #[test]
    fn span_shorter_than_h_takes_one_step() {
        let out = rk4_fixed(|_, _, dy| dy[0] = 2.0, 0.0, 1e-12, vec![0.0], 1.0).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].0, 1e-12);
        assert!((out[1].1[0] - 2e-12).abs() < 1e-24);
    }
}
