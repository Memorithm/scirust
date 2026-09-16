//! Bounded Morris screening and first/total-order Sobol estimators.
//!
//! Inputs are evaluations of a caller-owned deterministic function on declared
//! independent uniform factors. Designs, domains, budgets and model evidence
//! belong to the consumer. This module neither evaluates models nor chooses a
//! scientific decision. SALib 1.5.2 is the differential reference; constant
//! Sobol outputs are explicitly rejected instead of returning undefined ratios.

use crate::resampling::{ResearchStatsError, checked_mean};

/// Per-factor elementary-effect summaries in output units per unit input range.
#[derive(Clone, Debug, PartialEq)]
pub struct MorrisSummary {
    /// Signed mean elementary effect.
    pub mu: Vec<f64>,
    /// Mean absolute elementary effect; screening magnitude, not a causal claim.
    pub mu_star: Vec<f64>,
    /// Sample standard deviation (ddof=1) across independent trajectories.
    pub sigma: Vec<f64>,
    /// Number of complete trajectories actually used.
    pub trajectories: usize,
}

/// Summarize complete one-factor-at-a-time trajectories in `[0,1]^d`.
///
/// Every consecutive block of `d+1` rows must change each factor exactly once.
/// Requires 2..=10,000 trajectories, 1..=32 factors and at most 1,000,000 input
/// scalars. Invalid, duplicate, truncated or non-finite designs are rejected.
/// The signed step may be negative; its actual magnitude is used. For standard
/// Morris inference the caller must use a declared Morris design (typically an
/// even-level grid), not choose trajectories in response to observed effects.
///
/// ```
/// use scirust_stats::sensitivity::morris_effects;
/// let x = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
/// let result = morris_effects(&x, &[0.0, 3.0, 3.0, 0.0]).unwrap();
/// assert_eq!(result.mu_star, vec![3.0]);
/// assert_eq!(result.sigma, vec![0.0]);
/// assert!(morris_effects(&x, &[0.0]).is_err());
/// ```
pub fn morris_effects(
    inputs: &[Vec<f64>],
    outputs: &[f64],
) -> Result<MorrisSummary, ResearchStatsError> {
    let dimensions = inputs.first().map_or(0, Vec::len);
    if !(1..=32).contains(&dimensions)
        || inputs.len() != outputs.len()
        || inputs.len() % (dimensions + 1) != 0
        || !(2..=10_000).contains(&(inputs.len() / (dimensions + 1)))
        || inputs.iter().any(|row| row.len() != dimensions)
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    if inputs.len() * dimensions > 1_000_000
    {
        return Err(ResearchStatsError::BudgetExceeded);
    }
    if outputs
        .iter()
        .chain(inputs.iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err(ResearchStatsError::NonFinite);
    }
    if inputs.iter().flatten().any(|x| !(0.0..=1.0).contains(x))
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    let trajectories = inputs.len() / (dimensions + 1);
    let mut effects = vec![Vec::with_capacity(trajectories); dimensions];
    for start in (0..inputs.len()).step_by(dimensions + 1)
    {
        let mut seen = vec![false; dimensions];
        for i in start..start + dimensions
        {
            let changed: Vec<usize> = (0..dimensions)
                .filter(|&j| inputs[i][j] != inputs[i + 1][j])
                .collect();
            if changed.len() != 1 || seen[changed[0]]
            {
                return Err(ResearchStatsError::InvalidInput);
            }
            let j = changed[0];
            seen[j] = true;
            let effect = (outputs[i + 1] - outputs[i]) / (inputs[i + 1][j] - inputs[i][j]);
            if !effect.is_finite()
            {
                return Err(ResearchStatsError::NonFinite);
            }
            effects[j].push(effect);
        }
    }
    let mut result = MorrisSummary {
        mu: Vec::new(),
        mu_star: Vec::new(),
        sigma: Vec::new(),
        trajectories,
    };
    for values in effects
    {
        let mu = checked_mean(&values)?;
        let mu_star = checked_mean(&values.iter().map(|x| x.abs()).collect::<Vec<_>>())?;
        let sigma = (values
            .iter()
            .map(|x| ((x - mu) / ((trajectories - 1) as f64).sqrt()).powi(2))
            .sum::<f64>())
        .sqrt();
        if !sigma.is_finite()
        {
            return Err(ResearchStatsError::NonFinite);
        }
        result.mu.push(mu);
        result.mu_star.push(mu_star);
        result.sigma.push(sigma);
    }
    Ok(result)
}

/// Saltelli 2010 first-order and total-order sample estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct SobolSummary {
    /// First-order estimates; sampling error can place them outside `[0,1]`.
    pub first: Vec<f64>,
    /// Total-order estimates; estimates are not clipped or renormalized.
    pub total: Vec<f64>,
    /// Number of rows in each base matrix A and B.
    pub base_samples: usize,
}

/// Estimate Sobol indices from a Saltelli cross-sampled output layout.
///
/// `a[i]` and `b[i]` are base-matrix outputs. `ab[j][i]` evaluates A's row i
/// with factor j replaced by B's. Factors must be independent; correlated or
/// conditionally constrained inputs require another estimator. The complete
/// output set is centered; variance uses the combined A/B sample with ddof=0.
/// This is the first/total-order Saltelli 2010 convention used by SALib 1.5.2
/// with `calc_second_order=False`. No uncertainty interval is inferred here.
///
/// Requires 2..=100,000 base rows, 1..=32 factors, at most 1,000,000 outputs
/// and a strictly positive finite scaled A/B variance. Negative estimates are
/// retained. Degenerate data returns `ZeroVariance`, never a success-shaped NaN.
///
/// ```
/// use scirust_stats::sensitivity::sobol_first_total;
/// let a = [0.0, 1.0, 0.0, 1.0];
/// let b = [0.0, 0.0, 1.0, 1.0];
/// let s = sobol_first_total(&a, &b, &[b.to_vec()]).unwrap();
/// assert_eq!(s.first, vec![1.0]);
/// assert_eq!(s.total, vec![1.0]);
/// assert!(sobol_first_total(&[1.0, 1.0], &[1.0, 1.0], &[vec![1.0, 1.0]]).is_err());
/// ```
pub fn sobol_first_total(
    a: &[f64],
    b: &[f64],
    ab: &[Vec<f64>],
) -> Result<SobolSummary, ResearchStatsError> {
    if !(2..=100_000).contains(&a.len())
        || b.len() != a.len()
        || !(1..=32).contains(&ab.len())
        || ab.iter().any(|row| row.len() != a.len())
    {
        return Err(ResearchStatsError::InvalidInput);
    }
    let count = a.len() * (ab.len() + 2);
    if count > 1_000_000
    {
        return Err(ResearchStatsError::BudgetExceeded);
    }
    if a.iter()
        .chain(b)
        .chain(ab.iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err(ResearchStatsError::NonFinite);
    }
    let scale = a
        .iter()
        .chain(b)
        .chain(ab.iter().flatten())
        .map(|x| x.abs())
        .fold(0.0, f64::max);
    if scale == 0.0
    {
        return Err(ResearchStatsError::ZeroVariance);
    }
    let center = a
        .iter()
        .chain(b)
        .chain(ab.iter().flatten())
        .map(|x| (x / scale) / count as f64)
        .sum::<f64>();
    let normalize = |x: &f64| x / scale - center;
    let av: Vec<f64> = a.iter().map(normalize).collect();
    let bv: Vec<f64> = b.iter().map(normalize).collect();
    let mean_ab = av
        .iter()
        .chain(&bv)
        .map(|x| x / (2 * a.len()) as f64)
        .sum::<f64>();
    let variance = av
        .iter()
        .chain(&bv)
        .map(|x| (x - mean_ab).powi(2) / (2 * a.len()) as f64)
        .sum::<f64>();
    if !variance.is_finite() || variance <= 0.0
    {
        return Err(ResearchStatsError::ZeroVariance);
    }
    let mut result = SobolSummary {
        first: Vec::new(),
        total: Vec::new(),
        base_samples: a.len(),
    };
    for cross in ab
    {
        let first = (0..a.len())
            .map(|i| bv[i] * (normalize(&cross[i]) - av[i]) / a.len() as f64)
            .sum::<f64>()
            / variance;
        let total = (0..a.len())
            .map(|i| 0.5 * (av[i] - normalize(&cross[i])).powi(2) / a.len() as f64)
            .sum::<f64>()
            / variance;
        if !first.is_finite() || !total.is_finite()
        {
            return Err(ResearchStatsError::NonFinite);
        }
        result.first.push(first);
        result.total.push(total);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morris_rejects_missing_duplicate_and_multifactor_steps() {
        for x in [
            vec![vec![0.0, 0.0]; 6],
            vec![
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
        ]
        {
            assert!(morris_effects(&x, &[0.0; 6]).is_err());
        }
    }

    #[test]
    fn sobol_rejects_invalid_and_retains_signed_estimates() {
        assert_eq!(
            sobol_first_total(&[2.0; 4], &[2.0; 4], &[vec![2.0; 4]]),
            Err(ResearchStatsError::ZeroVariance)
        );
        assert!(sobol_first_total(&[0.0, f64::NAN], &[1.0, 2.0], &[vec![1.0, 2.0]]).is_err());
        let result = sobol_first_total(&[0.0, 1.0], &[1.0, 0.0], &[vec![-1.0, 2.0]]).unwrap();
        assert!(result.first[0] < 0.0);
    }
}
