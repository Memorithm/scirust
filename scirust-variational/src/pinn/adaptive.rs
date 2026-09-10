use scirust_core::nn::rng::PcgEngine;

use crate::error::{Result, VariationalError};
use crate::pinn::collocation::CollocationPoints;
use crate::pinn::domain::Domain1D;

#[derive(Debug, Clone)]
pub struct AdaptiveSamplingConfig {
    pub initial_points: usize,
    pub candidate_pool_size: usize,
    pub add_per_round: usize,
    pub max_points: usize,
    pub refinement_rounds: usize,
    pub seed: u64,
}

impl Default for AdaptiveSamplingConfig {
    fn default() -> Self {
        Self {
            initial_points: 100,
            candidate_pool_size: 1000,
            add_per_round: 50,
            max_points: 500,
            refinement_rounds: 5,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveRefinementHistory {
    pub rounds: Vec<RoundInfo>,
}

#[derive(Debug, Clone)]
pub struct RoundInfo {
    pub round: usize,
    pub n_points_before: usize,
    pub n_points_after: usize,
    pub max_residual: f32,
    pub mean_residual: f32,
}

pub struct AdaptiveSampler;

impl AdaptiveSampler {
    /// Refines collocation points while validating configuration, domain and residual values.
    pub fn try_refine<R>(
        domain: &Domain1D,
        residual_fn: R,
        config: &AdaptiveSamplingConfig,
    ) -> Result<(CollocationPoints, AdaptiveRefinementHistory)>
    where
        R: Fn(f32) -> f32,
    {
        validate_domain(domain)?;
        validate_config(config)?;

        let mut rng = PcgEngine::new(config.seed);
        let mut points =
            CollocationPoints::from_random_uniform(domain, config.initial_points, config.seed);
        let mut history = AdaptiveRefinementHistory { rounds: Vec::new() };

        for round in 0..config.refinement_rounds
        {
            if points.len() >= config.max_points
            {
                break;
            }

            let mut max_res = 0.0f32;
            let mut residual_sum = 0.0f64;
            for point in &points.points
            {
                let residual = checked_residual(&residual_fn, point[0])?;
                max_res = max_res.max(residual);
                residual_sum += residual as f64;
            }
            let mean_res = if points.is_empty()
            {
                0.0
            }
            else
            {
                let mean = residual_sum / points.len() as f64;
                if !mean.is_finite() || mean > f32::MAX as f64
                {
                    return Err(VariationalError::NonFiniteValue {
                        component: "adaptive sampling mean residual",
                        value: mean as f32,
                    });
                }
                mean as f32
            };

            let before = points.len();
            let span = domain.end - domain.start;
            let mut candidates = Vec::with_capacity(config.candidate_pool_size);
            for _ in 0..config.candidate_pool_size
            {
                let x = domain.start + rng.float() * span;
                if !x.is_finite()
                {
                    return Err(VariationalError::NonFiniteValue {
                        component: "adaptive sampling candidate",
                        value: x,
                    });
                }
                let residual = checked_residual(&residual_fn, x)?;
                candidates.push((x, residual));
            }

            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

            let mut to_add = 0;
            for &(x, _) in &candidates
            {
                if to_add >= config.add_per_round || points.len() >= config.max_points
                {
                    break;
                }
                let is_duplicate = points.points.iter().any(|pt| (pt[0] - x).abs() < 1e-4);
                if !is_duplicate
                {
                    points.points.push(vec![x]);
                    to_add += 1;
                }
            }

            history.rounds.push(RoundInfo {
                round,
                n_points_before: before,
                n_points_after: points.len(),
                max_residual: max_res,
                mean_residual: mean_res,
            });

            if points.len() == before
            {
                break;
            }
        }

        Ok((points, history))
    }

    pub fn refine<R>(
        domain: &Domain1D,
        residual_fn: R,
        config: &AdaptiveSamplingConfig,
    ) -> (CollocationPoints, AdaptiveRefinementHistory)
    where
        R: Fn(f32) -> f32,
    {
        Self::try_refine(domain, residual_fn, config)
            .expect("AdaptiveSampler::refine requires valid finite adaptive-sampling inputs")
    }
}

fn validate_domain(domain: &Domain1D) -> Result<()> {
    if !domain.start.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "adaptive sampling domain start",
            value: domain.start,
        });
    }
    if !domain.end.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "adaptive sampling domain end",
            value: domain.end,
        });
    }
    if domain.start >= domain.end
    {
        return Err(VariationalError::InvalidInterval {
            start: domain.start,
            end: domain.end,
        });
    }
    if !(domain.end - domain.start).is_finite()
    {
        return Err(VariationalError::UnsupportedOperation {
            details: "adaptive sampling domain span is not representable as finite f32".into(),
        });
    }
    Ok(())
}

fn validate_config(config: &AdaptiveSamplingConfig) -> Result<()> {
    if config.max_points == 0
    {
        return Err(VariationalError::UnsupportedOperation {
            details: "adaptive sampling max_points must be greater than zero".into(),
        });
    }
    if config.initial_points > config.max_points
    {
        return Err(VariationalError::UnsupportedOperation {
            details: format!(
                "adaptive sampling initial_points ({}) exceeds max_points ({})",
                config.initial_points, config.max_points
            ),
        });
    }
    Ok(())
}

fn checked_residual<R>(residual_fn: &R, x: f32) -> Result<f32>
where
    R: Fn(f32) -> f32,
{
    let raw = residual_fn(x);
    if !raw.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "adaptive sampling residual",
            value: raw,
        });
    }
    let residual = raw.abs();
    if !residual.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "adaptive sampling absolute residual",
            value: residual,
        });
    }
    Ok(residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_refinement() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let config = AdaptiveSamplingConfig {
            initial_points: 20,
            candidate_pool_size: 100,
            add_per_round: 10,
            max_points: 60,
            refinement_rounds: 3,
            seed: 42,
        };
        let (points, history) = AdaptiveSampler::refine(&domain, |x| (x - 0.5).abs(), &config);
        assert!(points.len() >= 20);
        assert!(!history.rounds.is_empty());
    }

    #[test]
    fn checked_refinement_rejects_non_finite_residual() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let config = AdaptiveSamplingConfig {
            initial_points: 4,
            candidate_pool_size: 8,
            add_per_round: 2,
            max_points: 8,
            refinement_rounds: 2,
            seed: 1,
        };
        assert!(AdaptiveSampler::try_refine(&domain, |_| f32::NAN, &config).is_err());
    }

    #[test]
    fn checked_refinement_rejects_initial_count_above_limit() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let config = AdaptiveSamplingConfig {
            initial_points: 10,
            max_points: 5,
            ..AdaptiveSamplingConfig::default()
        };
        assert!(AdaptiveSampler::try_refine(&domain, |_| 0.0, &config).is_err());
    }

    #[test]
    fn checked_refinement_revalidates_mutated_domain() {
        let mut domain = Domain1D::new(0.0, 1.0).unwrap();
        domain.end = f32::NAN;
        assert!(
            AdaptiveSampler::try_refine(&domain, |_| 0.0, &AdaptiveSamplingConfig::default())
                .is_err()
        );
    }

    #[test]
    fn refinement_never_exceeds_max_points() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let config = AdaptiveSamplingConfig {
            initial_points: 4,
            candidate_pool_size: 100,
            add_per_round: 100,
            max_points: 5,
            refinement_rounds: 3,
            seed: 2,
        };
        let (points, _) = AdaptiveSampler::try_refine(&domain, |x| x, &config).unwrap();
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn refinement_stops_after_round_without_progress() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let config = AdaptiveSamplingConfig {
            initial_points: 4,
            candidate_pool_size: 0,
            add_per_round: 2,
            max_points: 10,
            refinement_rounds: 5,
            seed: 3,
        };
        let (_, history) = AdaptiveSampler::try_refine(&domain, |x| x, &config).unwrap();
        assert_eq!(history.rounds.len(), 1);
    }
}
