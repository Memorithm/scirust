//! Deterministic log-log scaling fits for positive paired observations.
//!
//! This module provides an ordinary-least-squares primitive for power-law
//! style relationships of the form y = a * x^b. The fit is performed in
//! natural-log space. Inputs are canonically sorted before reduction so the
//! result is independent of original row order for the same finite values.
//! This is a descriptive regression, not a power-law distribution test or a
//! causal growth model. Bitwise repeatability is scoped to the same target
//! and floating-point implementation.

use core::fmt;

/// Errors returned by log_log_scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingError {
    /// Predictor and response slices have different lengths.
    LengthMismatch,
    /// Fewer than two paired observations were supplied.
    TooFewSamples,
    /// At least one observation is NaN or infinite.
    NonFinite,
    /// At least one observation is zero or negative.
    NonPositive,
    /// All predictor values are identical in log space.
    DegeneratePredictor,
}

impl fmt::Display for ScalingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::LengthMismatch => write!(f, "predictor and response lengths differ"),
            Self::TooFewSamples => write!(f, "at least two paired observations are required"),
            Self::NonFinite => write!(f, "all observations must be finite"),
            Self::NonPositive => write!(f, "all observations must be strictly positive"),
            Self::DegeneratePredictor => write!(f, "predictor has zero variance in log space"),
        }
    }
}

impl std::error::Error for ScalingError {}

/// Ordinary least-squares fit in natural-log space.
///
/// If ln(y) = intercept + slope * ln(x), the equivalent multiplicative model
/// is y = exp(intercept) * x.powf(slope).
///
/// r_squared is None when every response value is identical in log space,
/// because the usual coefficient of determination has zero total variance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogLogFit {
    /// Number of paired observations used by the fit.
    pub n: usize,
    /// Scaling exponent.
    pub slope: f64,
    /// Natural-log intercept.
    pub intercept: f64,
    /// Coefficient of determination when defined.
    pub r_squared: Option<f64>,
    /// Root-mean-square residual in natural-log response units.
    pub residual_rms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct KahanSum {
    sum: f64,
    correction: f64,
}

impl KahanSum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }

    fn finish(self) -> f64 {
        self.sum
    }
}

/// Fit a deterministic log-log scaling relation to positive paired samples.
///
/// Paired observations are sorted lexicographically by ln(x) then ln(y) before
/// reduction, making the fit independent of caller row ordering on the same
/// target. Complexity is O(n log n) time and O(n) auxiliary storage.
/// No observations are silently dropped or imputed.
///
/// # Errors
///
/// Returns ScalingError for length mismatch, too few observations, non-finite
/// or non-positive values, or a zero-variance predictor.
///
/// # Examples
///
/// ~~~
/// use scirust_stats::scaling::log_log_scaling;
/// let x = [1.0, 2.0, 4.0, 8.0];
/// let y = [3.0, 3.0 * 2.0_f64.powf(1.5), 24.0, 3.0 * 8.0_f64.powf(1.5)];
/// let fit = log_log_scaling(&x, &y).unwrap();
/// assert!((fit.slope - 1.5).abs() < 1e-12);
/// ~~~
pub fn log_log_scaling(x: &[f64], y: &[f64]) -> Result<LogLogFit, ScalingError> {
    if x.len() != y.len()
    {
        return Err(ScalingError::LengthMismatch);
    }
    if x.len() < 2
    {
        return Err(ScalingError::TooFewSamples);
    }

    let mut pairs = Vec::with_capacity(x.len());
    for (&predictor, &response) in x.iter().zip(y)
    {
        if !predictor.is_finite() || !response.is_finite()
        {
            return Err(ScalingError::NonFinite);
        }
        if predictor <= 0.0 || response <= 0.0
        {
            return Err(ScalingError::NonPositive);
        }
        pairs.push((predictor.ln(), response.ln()));
    }

    pairs.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    // A rounded mean can differ from an exactly repeated value. Testing the
    // actual range first prevents artificial nonzero variance and false fits.
    if pairs[0].0 == pairs[pairs.len() - 1].0
    {
        return Err(ScalingError::DegeneratePredictor);
    }
    if pairs.iter().all(|&(_, ly)| ly == pairs[0].1)
    {
        return Ok(LogLogFit {
            n: pairs.len(),
            slope: 0.0,
            intercept: pairs[0].1,
            r_squared: None,
            residual_rms: 0.0,
        });
    }

    let mut sum_x = KahanSum::default();
    let mut sum_y = KahanSum::default();
    for &(lx, ly) in &pairs
    {
        sum_x.add(lx);
        sum_y.add(ly);
    }
    let n = pairs.len();
    let mean_x = sum_x.finish() / n as f64;
    let mean_y = sum_y.finish() / n as f64;

    let mut sxx = KahanSum::default();
    let mut sxy = KahanSum::default();
    let mut syy = KahanSum::default();
    for &(lx, ly) in &pairs
    {
        let dx = lx - mean_x;
        let dy = ly - mean_y;
        sxx.add(dx * dx);
        sxy.add(dx * dy);
        syy.add(dy * dy);
    }

    let sxx = sxx.finish();
    if sxx <= 0.0
    {
        return Err(ScalingError::DegeneratePredictor);
    }
    let syy = syy.finish();
    let slope = sxy.finish() / sxx;
    let intercept = mean_y - slope * mean_x;

    let mut residual_sum_squares = KahanSum::default();
    for &(lx, ly) in &pairs
    {
        let residual = ly - (intercept + slope * lx);
        residual_sum_squares.add(residual * residual);
    }
    let ss_res = residual_sum_squares.finish();
    let residual_rms = (ss_res / n as f64).sqrt();
    let r_squared = if syy > 0.0
    {
        Some((1.0 - ss_res / syy).min(1.0))
    }
    else
    {
        None
    };

    Ok(LogLogFit {
        n,
        slope,
        intercept,
        r_squared,
        residual_rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_power_law_recovers_exponent_and_intercept() {
        let x = [1.0_f64, 2.0, 4.0, 8.0, 16.0];
        let y = x.map(|value| 3.0 * value.powf(1.5));
        let fit = log_log_scaling(&x, &y).unwrap();

        assert_eq!(fit.n, 5);
        assert!((fit.slope - 1.5).abs() < 1e-12);
        assert!((fit.intercept - 3.0_f64.ln()).abs() < 1e-12);
        assert!(fit.residual_rms < 1e-12);
        assert!((fit.r_squared.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn input_row_order_does_not_change_bits() {
        let x_a = [1.0, 2.0, 3.0, 5.0, 8.0];
        let y_a = [2.0, 4.2, 5.7, 11.1, 17.0];
        let x_b = [8.0, 1.0, 5.0, 3.0, 2.0];
        let y_b = [17.0, 2.0, 11.1, 5.7, 4.2];

        let a = log_log_scaling(&x_a, &y_a).unwrap();
        let b = log_log_scaling(&x_b, &y_b).unwrap();

        assert_eq!(a.slope.to_bits(), b.slope.to_bits());
        assert_eq!(a.intercept.to_bits(), b.intercept.to_bits());
        assert_eq!(a.residual_rms.to_bits(), b.residual_rms.to_bits());
        assert_eq!(
            a.r_squared.unwrap().to_bits(),
            b.r_squared.unwrap().to_bits()
        );
    }

    #[test]
    fn constant_response_has_undefined_r_squared() {
        let fit = log_log_scaling(&[1.0, 2.0, 4.0], &[7.0, 7.0, 7.0]).unwrap();
        assert_eq!(fit.r_squared, None);
        assert!(fit.slope.abs() < 1e-12);
    }

    #[test]
    fn constant_ranges_are_detected_before_mean_rounding() {
        for value in [1.1, 2.0, 7.0, 1.0e200]
        {
            for n in 2..64
            {
                let constant = vec![value; n];
                let varying: Vec<f64> = (1..=n).map(|x| x as f64).collect();
                assert_eq!(
                    log_log_scaling(&constant, &varying).unwrap_err(),
                    ScalingError::DegeneratePredictor
                );
                let fit = log_log_scaling(&varying, &constant).unwrap();
                assert_eq!(fit.r_squared, None);
                assert_eq!(fit.slope, 0.0);
                assert_eq!(fit.residual_rms, 0.0);
            }
        }
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert_eq!(
            log_log_scaling(&[1.0, 2.0], &[1.0]).unwrap_err(),
            ScalingError::LengthMismatch
        );
        assert_eq!(
            log_log_scaling(&[1.0], &[1.0]).unwrap_err(),
            ScalingError::TooFewSamples
        );
        assert_eq!(
            log_log_scaling(&[1.0, f64::NAN], &[1.0, 2.0]).unwrap_err(),
            ScalingError::NonFinite
        );
        assert_eq!(
            log_log_scaling(&[0.0, 1.0], &[1.0, 2.0]).unwrap_err(),
            ScalingError::NonPositive
        );
        assert_eq!(
            log_log_scaling(&[2.0, 2.0], &[1.0, 2.0]).unwrap_err(),
            ScalingError::DegeneratePredictor
        );
    }
}
