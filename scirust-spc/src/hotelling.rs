//! Hotelling T² — multivariate process monitoring.
//!
//! A single statistic `T² = (x − μ)ᵀ Σ⁻¹ (x − μ)` watches several correlated
//! quality variables at once, catching shifts that per-variable charts miss
//! (because they account for the correlation). Compare against an upper control
//! limit — a χ²ₚ critical value for phase-II individuals.

use scirust_estimation::Mat;
use serde::{Deserialize, Serialize};

/// Hotelling T² monitor (mean vector + inverse covariance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotellingT2 {
    mean: Vec<f64>,
    cov_inv: Mat,
}

impl HotellingT2 {
    /// Fit on in-control multivariate samples (each row a length-p observation).
    /// Returns `None` if the covariance is singular: a variable with zero (or
    /// non-finite) variance, or variables that are (numerically) collinear.
    /// The singularity check is independent of the units of each variable.
    pub fn fit(data: &[Vec<f64>]) -> Option<Self> {
        let n = data.len();
        if n < 2
        {
            return None;
        }
        let p = data[0].len();
        let mut mean = vec![0.0; p];
        for row in data
        {
            for (m, &v) in mean.iter_mut().zip(row)
            {
                *m += v;
            }
        }
        for m in mean.iter_mut()
        {
            *m /= n as f64;
        }
        let mut cov = Mat::zeros(p, p);
        for row in data
        {
            for i in 0..p
            {
                for j in 0..p
                {
                    cov.data[i * p + j] += (row[i] - mean[i]) * (row[j] - mean[j]);
                }
            }
        }
        for v in cov.data.iter_mut()
        {
            *v /= (n - 1) as f64;
        }
        // Invert the correlation matrix `R = D⁻¹ Σ D⁻¹` (`D = diag(σᵢ)`) and
        // map back with `Σ⁻¹ = D⁻¹ R⁻¹ D⁻¹`. The singularity test of
        // `Mat::inverse` is relative to the largest entry of the matrix, so on
        // the raw covariance a variable measured in large units (Pa, σ ≈ 1e3)
        // made a perfectly regular one in small units (m, σ ≈ 1e-5) look like
        // a zero pivot and `fit` returned `None`. T² does not depend on the
        // units of each variable, and `R` has a unit diagonal, so the test is
        // now applied where it only reflects genuine (near-)collinearity.
        let sd: Vec<f64> = (0..p).map(|i| cov.data[i * p + i].sqrt()).collect();
        if sd.iter().any(|&s| !(s.is_finite() && s > 0.0))
        {
            return None;
        }
        let mut corr = Mat::zeros(p, p);
        for i in 0..p
        {
            for j in 0..p
            {
                corr.data[i * p + j] = cov.data[i * p + j] / sd[i] / sd[j];
            }
        }
        let mut cov_inv = corr.inverse()?;
        for i in 0..p
        {
            for j in 0..p
            {
                cov_inv.data[i * p + j] /= sd[i] * sd[j];
            }
        }
        Some(Self { mean, cov_inv })
    }

    /// `T²` statistic for an observation.
    pub fn t2(&self, x: &[f64]) -> f64 {
        let d: Vec<f64> = x.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
        let sd = self.cov_inv.matvec(&d);
        d.iter().zip(&sd).map(|(a, b)| a * b).sum()
    }

    /// Whether the observation exceeds the upper control limit `ucl`
    /// (a χ²ₚ critical value).
    pub fn is_out_of_control(&self, x: &[f64], ucl: f64) -> bool {
        self.t2(x) > ucl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn u01(&mut self) -> f64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.u01().max(1e-9), self.u01());
            (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn flags_correlated_out_of_control_point() {
        // Two correlated variables: x2 ≈ x1 + small noise.
        let mut rng = Rng::new(0x7C2);
        let data: Vec<Vec<f64>> = (0..2000)
            .map(|_| {
                let a = rng.normal();
                vec![a, a + 0.3 * rng.normal()]
            })
            .collect();
        let h = HotellingT2::fit(&data).expect("nonsingular");
        let ucl = 9.21; // χ²(2) at α = 0.01

        // T² at the mean is 0.
        assert!(h.t2(&h.mean) < 1e-6);
        // An in-control sample is (almost surely) below the UCL.
        assert!(!h.is_out_of_control(&[0.2, 0.25], ucl));
        // A point that violates the correlation (x1 high, x2 low) is far in
        // Mahalanobis distance even though each coordinate is individually modest.
        assert!(
            h.is_out_of_control(&[2.5, -2.5], ucl),
            "T² {}",
            h.t2(&[2.5, -2.5])
        );
    }

    #[test]
    fn fit_accepts_variables_measured_in_very_different_units() {
        // Pressure in Pa (σ ≈ 1e3) next to a length in metres (σ ≈ 1e-5):
        // the covariance diagonal spans 1e6 .. 1e-10, a ratio of 1e16, although
        // the variables are only mildly correlated. T² is invariant to the
        // units of each variable, so this must fit and give the same T² as the
        // same data expressed in kPa and µm (both σ ≈ 1–10).
        let mut rng = Rng::new(0x51C);
        let unit: Vec<Vec<f64>> = (0..400)
            .map(|_| {
                let a = rng.normal();
                vec![a, 0.4 * a + rng.normal()]
            })
            .collect();
        let si: Vec<Vec<f64>> = unit
            .iter()
            .map(|r| vec![101_325.0 + 1.0e3 * r[0], 0.05 + 1.0e-5 * r[1]])
            .collect();
        let scaled: Vec<Vec<f64>> = unit
            .iter()
            .map(|r| vec![101.325 + r[0], 50_000.0 + 10.0 * r[1]])
            .collect();
        let h_ref = HotellingT2::fit(&scaled).expect("well-conditioned in kPa/µm");
        let h_si = HotellingT2::fit(&si).expect("same data in Pa/m is not singular");
        for r in [[0.0, 0.0], [1.5, -0.7], [-2.0, 3.0], [3.0, 3.0]]
        {
            let t_ref = h_ref.t2(&[101.325 + r[0], 50_000.0 + 10.0 * r[1]]);
            let t_si = h_si.t2(&[101_325.0 + 1.0e3 * r[0], 0.05 + 1.0e-5 * r[1]]);
            assert!(
                (t_si - t_ref).abs() <= 1e-8 * t_ref.max(1.0),
                "point {r:?}: T² {t_si} in Pa/m vs {t_ref} in kPa/µm"
            );
        }
    }

    #[test]
    fn fit_still_rejects_a_collinear_covariance() {
        // x2 = 3·x1 exactly: the covariance is singular whatever the units.
        let data: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let a = (i as f64 * 0.37).sin();
                vec![a, 3.0 * a]
            })
            .collect();
        assert!(HotellingT2::fit(&data).is_none());
        // A constant variable has zero variance.
        let flat: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64, 2.0]).collect();
        assert!(HotellingT2::fit(&flat).is_none());
    }
}
