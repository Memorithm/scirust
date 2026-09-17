//! Bounded resampling of caller-declared independent units.
//!
//! The caller supplies one paired contrast per independent unit (or cluster).
//! This layer cannot infer independence from seeds, row counts or task names.
//! Percentile intervals use the existing type-7 quantile. They are exploratory
//! unless a protocol prescribed this method before observing results. No
//! coverage, equivalence, superiority or stopping decision is inferred.

use crate::{SplitMix64, describe};
use std::fmt;

/// Validation or arithmetic failure; no invalid numeric result is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResearchStatsError {
    /// Shape, count, confidence, probability or design constraints failed.
    InvalidInput,
    /// Inputs or intermediate arithmetic are not finite.
    NonFinite,
    /// The requested number of scalar operations exceeds the documented bound.
    BudgetExceeded,
    /// A variance-based estimate has no varying reference sample.
    ZeroVariance,
}

impl fmt::Display for ResearchStatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ResearchStatsError {}

/// Mean paired effect and a two-sided percentile bootstrap interval.
#[derive(Clone, Debug, PartialEq)]
pub struct PercentileInterval {
    /// Arithmetic mean over equally weighted independent units.
    pub estimate: f64,
    /// Type-7 lower percentile, in the input contrast's units.
    pub lower: f64,
    /// Type-7 upper percentile, in the input contrast's units.
    pub upper: f64,
    /// Requested two-sided confidence level, strictly between zero and one.
    pub confidence: f64,
    /// Number of independent units supplied by the caller.
    pub units: usize,
    /// Number of bootstrap resamples actually computed.
    pub resamples: usize,
}

pub(crate) fn checked_mean(values: &[f64]) -> Result<f64, ResearchStatsError> {
    if values.is_empty()
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    if values.iter().any(|x| !x.is_finite())
    {
        return Err(ResearchStatsError::NonFinite);
    }
    let n = values.len() as f64;
    // Sum before dividing to preserve representable subnormal means. Only
    // overflow triggers scaling; scaling every tiny contribution in advance
    // can otherwise lose a representable result to underflow.
    let result = if let Some(total) = compensated_sum(values.iter().copied())
    {
        total / n
    }
    else
    {
        let scale = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
        (compensated_sum(values.iter().map(|x| x / scale)).ok_or(ResearchStatsError::NonFinite)?
            / n)
            * scale
    };
    if !result.is_finite()
    {
        return Err(ResearchStatsError::NonFinite);
    }
    Ok(result)
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (mut sum, mut correction): (f64, f64) = (0.0, 0.0);
    for value in values
    {
        let next = sum + value;
        if !next.is_finite()
        {
            return None;
        }
        correction += if sum.abs() >= value.abs()
        {
            (sum - next) + value
        }
        else
        {
            (value - next) + sum
        };
        if !correction.is_finite()
        {
            return None;
        }
        sum = next;
    }
    let total = sum + correction;
    total.is_finite().then_some(total)
}

fn uniform_index(rng: &mut SplitMix64, size: usize) -> usize {
    let size = size as u64;
    // Rejection removes modulo bias. Each call owns its stream, with no global
    // RNG or data-dependent seed selection.
    let threshold = size.wrapping_neg() % size;
    loop
    {
        let value = rng.next_u64();
        if value >= threshold
        {
            return (value % size) as usize;
        }
    }
}

/// Bootstrap the mean of paired contrasts over equally weighted units.
///
/// Requires 2..=10,000 finite contrasts, 100..=100,000 resamples and at most
/// 10,000,000 sampled scalar contributions. Rejects non-finite arithmetic.
/// A constant contrast has a degenerate interval. The caller must aggregate
/// repeated observations at the declared independent-unit level first.
/// Determinism is scoped to this algorithm, seed, input order and build/target.
///
/// ```
/// use scirust_stats::resampling::paired_mean_percentile;
/// let ci = paired_mean_percentile(&[2.0, 2.0, 2.0], 1000, 0.95, 7).unwrap();
/// assert_eq!((ci.estimate, ci.lower, ci.upper), (2.0, 2.0, 2.0));
/// assert!(paired_mean_percentile(&[1.0], 1000, 0.95, 7).is_err());
/// ```
pub fn paired_mean_percentile(
    contrasts: &[f64],
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> Result<PercentileInterval, ResearchStatsError> {
    if !((2..=10_000).contains(&contrasts.len())
        && (100..=100_000).contains(&resamples)
        && confidence.is_finite()
        && 0.0 < confidence
        && confidence < 1.0)
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    if contrasts.len() * resamples > 10_000_000
    {
        return Err(ResearchStatsError::BudgetExceeded);
    }
    let estimate = checked_mean(contrasts)?;
    let mut rng = SplitMix64::new(seed);
    let n = contrasts.len();
    let mut means = Vec::with_capacity(resamples);
    let mut sample = vec![0.0; n];
    for _ in 0..resamples
    {
        for value in &mut sample
        {
            *value = contrasts[uniform_index(&mut rng, n)];
        }
        means.push(checked_mean(&sample)?);
    }
    let tail = (1.0 - confidence) / 2.0;
    let bounds = describe::quantiles(&means, &[tail, 1.0 - tail]);
    let (lower, upper) = (bounds[0], bounds[1]);
    if !lower.is_finite() || !upper.is_finite()
    {
        return Err(ResearchStatsError::NonFinite);
    }
    Ok(PercentileInterval {
        estimate,
        lower,
        upper,
        confidence,
        units: n,
        resamples,
    })
}

/// Holm step-down adjusted p-values, restored to the caller's input order.
///
/// Accepts 1..=10,000 finite p-values in `[0,1]`. The caller must declare the
/// comparison family before examination; this function does not legitimize a
/// post-hoc choice of family and does not manufacture p-values from intervals.
///
/// ```
/// use scirust_stats::resampling::holm_adjust;
/// assert_eq!(holm_adjust(&[0.01, 0.04, 0.03]).unwrap(), vec![0.03, 0.06, 0.06]);
/// assert!(holm_adjust(&[f64::NAN]).is_err());
/// ```
pub fn holm_adjust(p_values: &[f64]) -> Result<Vec<f64>, ResearchStatsError> {
    if p_values.is_empty() || p_values.len() > 10_000
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    if p_values.iter().any(|x| !x.is_finite())
    {
        return Err(ResearchStatsError::NonFinite);
    }
    if p_values.iter().any(|x| !(0.0..=1.0).contains(x))
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    let mut order: Vec<usize> = (0..p_values.len()).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));
    let mut adjusted = vec![0.0; order.len()];
    let mut previous: f64 = 0.0;
    for (rank, &index) in order.iter().enumerate()
    {
        previous = previous.max((p_values[index] * (order.len() - rank) as f64).min(1.0));
        adjusted[index] = previous;
    }
    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn means_preserve_subnormals_and_avoid_finite_input_overflow() {
        for value in [1e-320, -1e-320, f64::MAX, -f64::MAX]
        {
            let result = paired_mean_percentile(&vec![value; 10_000], 100, 0.95, 19).unwrap();
            assert_eq!(result.estimate, value);
            assert_eq!(result.lower, value);
            assert_eq!(result.upper, value);
        }
        assert_eq!(checked_mean(&[1e308, 1e308, -1e308, -1e308]), Ok(0.0));
        assert_eq!(checked_mean(&[1e20, 3.0, -1e20]), Ok(1.0));
        assert_eq!(checked_mean(&[1e308, 1e-320, -1e308]), Ok(1e-320 / 3.0));
    }

    #[test]
    fn preserves_subnormal_constant_mean_and_interval() {
        let contrasts = vec![1.0e-320; 10_000];
        assert_eq!(checked_mean(&contrasts).unwrap(), 1.0e-320);
        let result = paired_mean_percentile(&contrasts, 100, 0.95, 23).unwrap();
        assert_eq!(
            (result.estimate, result.lower, result.upper),
            (1.0e-320, 1.0e-320, 1.0e-320)
        );
    }

    #[test]
    fn rejects_invalid_and_excessive_resampling() {
        assert_eq!(
            paired_mean_percentile(&[1.0, f64::NAN], 100, 0.95, 0),
            Err(ResearchStatsError::NonFinite)
        );
        assert!(paired_mean_percentile(&[1.0, 2.0], 100, 1.0, 0).is_err());
        assert_eq!(
            paired_mean_percentile(&vec![0.0; 10_000], 100_000, 0.95, 0),
            Err(ResearchStatsError::BudgetExceeded)
        );
    }

    #[test]
    fn analytical_two_unit_distribution_and_determinism() {
        // The complete empirical mean distribution has masses 1/4,1/2,1/4 at
        // 0,1,2. Both 2.5% endpoints are exactly known for this bounded fixture.
        let result = paired_mean_percentile(&[0.0, 2.0], 10_000, 0.95, 19).unwrap();
        assert_eq!(
            (result.estimate, result.lower, result.upper),
            (1.0, 0.0, 2.0)
        );
        assert_eq!(
            result,
            paired_mean_percentile(&[0.0, 2.0], 10_000, 0.95, 19).unwrap()
        );
    }

    #[test]
    fn holm_preserves_order_ties_and_bounds() {
        assert_eq!(
            holm_adjust(&[0.7, 0.01, 0.01, 0.9]).unwrap(),
            vec![1.0, 0.04, 0.04, 1.0]
        );
        assert!(holm_adjust(&[-0.1]).is_err());
        assert!(holm_adjust(&[]).is_err());
    }
}
