//! Interacting Multiple Models (IMM) filter.
//!
//! Many systems switch regimes — a target flying straight then turning, a process
//! in steady state then upset. The IMM runs a bank of Kalman filters (one per
//! model), mixes their estimates each step, and updates a probability for each
//! mode from how well it explains the measurement. It tracks across regime
//! changes that a single model lags, and exposes *which* regime is active.
//!
//! Scalar measurement (`m = 1`) — the usual tracking case.

use crate::linalg::Mat;

/// One mode's linear-Gaussian model.
#[derive(Debug, Clone)]
pub struct ImmModel {
    pub a: Mat,
    pub q: Mat,
    pub h: Mat,
    pub r: Mat,
}

/// Interacting Multiple Models filter.
pub struct Imm {
    models: Vec<ImmModel>,
    x: Vec<Vec<f64>>,
    p: Vec<Mat>,
    mu: Vec<f64>,
    pi: Vec<Vec<f64>>,
}

impl Imm {
    /// Build from per-mode `models`, a shared initial `x0`/`p0`, initial mode
    /// probabilities `mu0`, and the Markov mode-transition matrix `pi`
    /// (`pi[i][j] = P(mode i → mode j)`).
    pub fn new(
        models: Vec<ImmModel>,
        x0: Vec<f64>,
        p0: Mat,
        mu0: Vec<f64>,
        pi: Vec<Vec<f64>>,
    ) -> Self {
        let k = models.len();
        Self {
            x: vec![x0; k],
            p: vec![p0; k],
            mu: mu0,
            pi,
            models,
        }
    }

    /// Overall (mode-probability-weighted) state estimate.
    pub fn estimate(&self) -> Vec<f64> {
        let n = self.x[0].len();
        let mut out = vec![0.0; n];
        for (mu, xi) in self.mu.iter().zip(&self.x)
        {
            for (o, &v) in out.iter_mut().zip(xi)
            {
                *o += mu * v;
            }
        }
        out
    }

    /// Current mode probabilities.
    pub fn mode_probabilities(&self) -> &[f64] {
        &self.mu
    }

    /// One IMM cycle for scalar measurement `z`.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, z: f64) {
        let k = self.models.len();
        let n = self.x[0].len();

        // 1. Predicted mode probabilities and mixing weights.
        let cbar: Vec<f64> = (0..k)
            .map(|j| (0..k).map(|i| self.pi[i][j] * self.mu[i]).sum())
            .collect();

        // 2. Mixed initial conditions per mode j.
        let mut x_mix = vec![vec![0.0; n]; k];
        let mut p_mix = vec![Mat::zeros(n, n); k];
        for j in 0..k
        {
            if cbar[j] <= 0.0
            {
                x_mix[j] = self.x[j].clone();
                p_mix[j] = self.p[j].clone();
                continue;
            }
            for i in 0..k
            {
                let w = self.pi[i][j] * self.mu[i] / cbar[j];
                for d in 0..n
                {
                    x_mix[j][d] += w * self.x[i][d];
                }
            }
            for i in 0..k
            {
                let w = self.pi[i][j] * self.mu[i] / cbar[j];
                let dx: Vec<f64> = self.x[i]
                    .iter()
                    .zip(&x_mix[j])
                    .map(|(a, b)| a - b)
                    .collect();
                // P_mix += w (P_i + dx dxᵀ)
                for r0 in 0..n
                {
                    for c0 in 0..n
                    {
                        p_mix[j].data[r0 * n + c0] += w * (self.p[i].get(r0, c0) + dx[r0] * dx[c0]);
                    }
                }
            }
        }

        // 3. Mode-matched Kalman predict + update; collect log-likelihoods.
        // `None` marks a degenerate mode (innovation variance `s <= 0`).
        let mut log_likelihood: Vec<Option<f64>> = vec![None; k];
        for j in 0..k
        {
            let m = &self.models[j];
            // Predict.
            let xp = m.a.matvec(&x_mix[j]);
            let pp = m.a.matmul(&p_mix[j]).matmul(&m.a.t()).add(&m.q);
            // Update (scalar measurement).
            let hx = m.h.matvec(&xp)[0];
            let y = z - hx;
            let ht = m.h.t();
            let s = m.h.matmul(&pp).matmul(&ht).add(&m.r).get(0, 0);
            if s <= 0.0
            {
                self.x[j] = xp;
                self.p[j] = pp;
                continue;
            }
            // Kalman gain K = P Hᵀ / s.
            let pht = pp.matmul(&ht); // n×1
            let kgain: Vec<f64> = (0..n).map(|i| pht.get(i, 0) / s).collect();
            self.x[j] = xp.iter().zip(&kgain).map(|(a, b)| a + b * y).collect();
            // P = (I - K H) P.
            let mut kh = Mat::zeros(n, n);
            for r0 in 0..n
            {
                for c0 in 0..n
                {
                    kh.data[r0 * n + c0] = kgain[r0] * m.h.get(0, c0);
                }
            }
            self.p[j] = Mat::identity(n).sub(&kh).matmul(&pp);
            // Gaussian log-likelihood of the innovation. Kept in log space: the
            // linear-space density underflows to 0 once |y| exceeds ~38 sqrt(s),
            // which would otherwise zero every mode at once on a large jump.
            log_likelihood[j] =
                Some(-0.5 * y * y / s - 0.5 * (2.0 * core::f64::consts::PI * s).ln());
        }

        // 4. Mode-probability update, mu_j ∝ cbar_j * L_j, normalized in log space
        // (subtract the largest log-weight before exponentiating). A degenerate
        // mode carries no likelihood and gets zero weight, unless every mode is
        // degenerate, in which case the likelihoods are uninformative (equal) and
        // the predicted probabilities `cbar` are kept.
        let all_degenerate = log_likelihood.iter().all(Option::is_none);
        let log_w: Vec<f64> = (0..k)
            .map(|j| {
                let ll = match log_likelihood[j]
                {
                    Some(ll) => ll,
                    None if all_degenerate => 0.0,
                    None => f64::NEG_INFINITY,
                };
                ll + cbar[j].ln()
            })
            .collect();
        let max_lw = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut norm = 0.0;
        let mut new_mu = vec![0.0; k];
        if max_lw.is_finite()
        {
            for j in 0..k
            {
                new_mu[j] = (log_w[j] - max_lw).exp();
                norm += new_mu[j];
            }
        }
        if norm > 0.0
        {
            for m in new_mu.iter_mut()
            {
                *m /= norm;
            }
            self.mu = new_mu;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv_model(dt: f64, q: f64, r: f64) -> ImmModel {
        ImmModel {
            a: Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]),
            q: Mat::diag(&[q, q]),
            h: Mat::new(1, 2, vec![1.0, 0.0]),
            r: Mat::new(1, 1, vec![r]),
        }
    }

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn normal(&mut self, sd: f64) -> f64 {
            let u = |st: &mut u64| {
                *st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = *st;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
            };
            let (u1, u2) = (u(&mut self.s).max(1e-9), u(&mut self.s));
            sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn imm_single_model_step_matches_scalar_kalman() {
        // Single mode reduces IMM to a scalar KF.
        // predict: xp=0, pp = a*p*a + q = 1.
        // update z=4: y = 4-0 = 4, s = h*pp*h + r = 1+1 = 2, K = pp*h/s = 1/2,
        //   x = 0 + 0.5*4 = 2.0. estimate = mu0*x = 1*2.0 = 2.0.
        // With one mode the normalized mode prob is always 1.0.
        let m = ImmModel {
            a: Mat::new(1, 1, vec![1.0]),
            q: Mat::new(1, 1, vec![0.0]),
            h: Mat::new(1, 1, vec![1.0]),
            r: Mat::new(1, 1, vec![1.0]),
        };
        let mut imm = Imm::new(
            vec![m],
            vec![0.0],
            Mat::new(1, 1, vec![1.0]),
            vec![1.0],
            vec![vec![1.0]],
        );
        imm.step(4.0);
        assert!((imm.estimate()[0] - 2.0).abs() < 1e-12);
        assert!((imm.mode_probabilities()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mode_probability_shifts_to_the_maneuver_model() {
        let dt = 1.0;
        // Mode 0: quiet CV (low Q). Mode 1: agile (high Q) for maneuvers.
        let models = vec![cv_model(dt, 1e-4, 0.25), cv_model(dt, 1.0, 0.25)];
        let pi = vec![vec![0.95, 0.05], vec![0.1, 0.9]];
        let mut imm = Imm::new(
            models,
            vec![0.0, 1.0],
            Mat::diag(&[1.0, 1.0]),
            vec![0.5, 0.5],
            pi,
        );

        let mut rng = Rng::new(0x1117);
        let (mut pos, mut vel) = (0.0, 1.0);
        let mut quiet_mode1 = 0.0f64;
        let mut maneuver_mode1 = 0.0f64;
        for k in 0..120
        {
            // Sudden velocity reversal at k = 60 (a maneuver).
            if k == 60
            {
                vel = -2.0;
            }
            pos += vel * dt;
            imm.step(pos + rng.normal(0.5));
            if (40..60).contains(&k)
            {
                quiet_mode1 = imm.mode_probabilities()[1];
            }
            if (62..80).contains(&k)
            {
                maneuver_mode1 = maneuver_mode1.max(imm.mode_probabilities()[1]);
            }
        }
        // The agile model's probability rises after the maneuver.
        assert!(
            maneuver_mode1 > quiet_mode1 + 0.2,
            "maneuver p1 {maneuver_mode1} vs quiet p1 {quiet_mode1}"
        );
    }

    /// Two scalar random-walk modes (quiet `q = 0.01`, agile `q = 100`, `r = 1`)
    /// and a jump `z = 500` that is > 38 innovation standard deviations away for
    /// *both* modes, so both Gaussian likelihoods underflow to 0 in linear space.
    fn jump_imm() -> Imm {
        let rw = |q: f64| ImmModel {
            a: Mat::new(1, 1, vec![1.0]),
            q: Mat::new(1, 1, vec![q]),
            h: Mat::new(1, 1, vec![1.0]),
            r: Mat::new(1, 1, vec![1.0]),
        };
        Imm::new(
            vec![rw(0.01), rw(100.0)],
            vec![0.0],
            Mat::new(1, 1, vec![1.0]),
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
        )
    }

    #[test]
    fn large_innovation_does_not_freeze_mode_probabilities() {
        let mut imm = jump_imm();
        imm.step(500.0);
        // s_quiet = 1 + 0.01 + 1 = 2.01, s_agile = 1 + 100 + 1 = 102.
        // log L_quiet - log L_agile = -0.5*500^2*(1/2.01 - 1/102) - 0.5*ln(2.01/102)
        //   ≈ -6.0962e4, so the exact posterior is mu = [exp(-6.0962e4), 1] ≈ [0, 1].
        let mu = imm.mode_probabilities();
        assert!(mu[0] < 1e-12, "quiet mode probability {}", mu[0]);
        assert!(
            (mu[1] - 1.0).abs() < 1e-12,
            "agile mode probability {}",
            mu[1]
        );
        assert!(((mu[0] + mu[1]) - 1.0).abs() < 1e-12);
        // Agile-mode update: x = 0 + (101/102)*500 = 495.098..., which is the
        // overall estimate once mu = [0, 1] (the quiet mode would give 251.24).
        let expected = 101.0 / 102.0 * 500.0;
        let est = imm.estimate()[0];
        assert!(
            (est - expected).abs() < 1e-9,
            "estimate {est} vs {expected}"
        );
    }

    #[test]
    fn moderate_innovation_mode_probabilities_match_closed_form() {
        // z = 5: both likelihoods are representable; check against the closed form
        // mu_j ∝ cbar_j * N(5; 0, s_j) with cbar = [0.5, 0.5].
        let mut imm = jump_imm();
        imm.step(5.0);
        let ll = |s: f64| -0.5 * 25.0 / s - 0.5 * (2.0 * core::f64::consts::PI * s).ln();
        let (l0, l1) = (ll(2.01), ll(102.0));
        let p1 = 1.0 / (1.0 + (l0 - l1).exp());
        let mu = imm.mode_probabilities();
        assert!((mu[1] - p1).abs() < 1e-12, "agile {} vs {}", mu[1], p1);
        assert!(
            (mu[0] - (1.0 - p1)).abs() < 1e-12,
            "quiet {} vs {}",
            mu[0],
            1.0 - p1
        );
    }

    #[test]
    fn degenerate_modes_get_zero_weight_unless_all_are_degenerate() {
        // r = -10 makes the innovation variance s = 1 + q - 10 < 0 (degenerate).
        let rw = |q: f64, r: f64| ImmModel {
            a: Mat::new(1, 1, vec![1.0]),
            q: Mat::new(1, 1, vec![q]),
            h: Mat::new(1, 1, vec![1.0]),
            r: Mat::new(1, 1, vec![r]),
        };
        let pi = vec![vec![0.9, 0.1], vec![0.3, 0.7]];
        // One degenerate mode next to a valid one: all weight on the valid mode.
        let mut imm = Imm::new(
            vec![rw(0.0, -10.0), rw(1.0, 1.0)],
            vec![0.0],
            Mat::new(1, 1, vec![1.0]),
            vec![0.5, 0.5],
            pi.clone(),
        );
        imm.step(500.0);
        assert_eq!(imm.mode_probabilities()[0], 0.0);
        assert!((imm.mode_probabilities()[1] - 1.0).abs() < 1e-12);
        // Every mode degenerate: the predicted probabilities cbar are kept,
        // cbar = [0.9*0.5 + 0.3*0.5, 0.1*0.5 + 0.7*0.5] = [0.6, 0.4].
        let mut imm = Imm::new(
            vec![rw(0.0, -10.0), rw(1.0, -10.0)],
            vec![0.0],
            Mat::new(1, 1, vec![1.0]),
            vec![0.5, 0.5],
            pi,
        );
        imm.step(1.0);
        assert!((imm.mode_probabilities()[0] - 0.6).abs() < 1e-12);
        assert!((imm.mode_probabilities()[1] - 0.4).abs() < 1e-12);
    }
}
