use scirust_core::nn::rng::PcgEngine;

use crate::error::{Result, VariationalError};
use crate::pinn::domain::Domain1D;

#[derive(Debug, Clone)]
pub struct CollocationPoints {
    pub points: Vec<Vec<f32>>,
    pub ndim: usize,
}

impl CollocationPoints {
    /// Builds a one-dimensional uniform collocation grid after validating the domain.
    pub fn try_from_uniform_grid(domain: &Domain1D, n_points: usize) -> Result<Self> {
        let xs = domain.try_uniform_points(n_points)?;
        let points: Vec<Vec<f32>> = xs.into_iter().map(|x| vec![x]).collect();
        Ok(Self { points, ndim: 1 })
    }

    pub fn from_uniform_grid(domain: &Domain1D, n_points: usize) -> Self {
        Self::try_from_uniform_grid(domain, n_points)
            .expect("CollocationPoints::from_uniform_grid requires a valid finite domain")
    }

    /// Samples one-dimensional collocation points uniformly from validated finite bounds.
    pub fn try_from_random_uniform(domain: &Domain1D, n_points: usize, seed: u64) -> Result<Self> {
        domain.try_uniform_points(0)?;
        let mut rng = PcgEngine::new(seed);
        let span = domain.end - domain.start;
        let mut points = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            let x = domain.start + rng.float() * span;
            if !x.is_finite() {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN random collocation point",
                    value: x,
                });
            }
            points.push(vec![x]);
        }
        Ok(Self { points, ndim: 1 })
    }

    pub fn from_random_uniform(domain: &Domain1D, n_points: usize, seed: u64) -> Self {
        Self::try_from_random_uniform(domain, n_points, seed)
            .expect("CollocationPoints::from_random_uniform requires a valid finite domain")
    }

    /// Builds Latin-hypercube collocation points from finite, strictly ordered bounds.
    pub fn try_from_latin_hypercube(
        bounds: &[(f32, f32)],
        n_points: usize,
        seed: u64,
    ) -> Result<Self> {
        if bounds.is_empty() {
            return Err(VariationalError::UnsupportedOperation {
                details: "Latin-hypercube collocation requires at least one dimension".into(),
            });
        }
        for &(lo, hi) in bounds {
            if !lo.is_finite() {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN collocation lower bound",
                    value: lo,
                });
            }
            if !hi.is_finite() {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN collocation upper bound",
                    value: hi,
                });
            }
            if lo >= hi {
                return Err(VariationalError::InvalidInterval { start: lo, end: hi });
            }
            if !(hi - lo).is_finite() {
                return Err(VariationalError::UnsupportedOperation {
                    details: "Latin-hypercube bound span is not representable as finite f32".into(),
                });
            }
        }

        let ndim = bounds.len();
        if n_points == 0 {
            return Ok(Self {
                points: Vec::new(),
                ndim,
            });
        }

        let mut rng = PcgEngine::new(seed);
        let mut points: Vec<Vec<f32>> = (0..n_points).map(|_| vec![0.0; ndim]).collect();

        for d in 0..ndim {
            let (lo, hi) = bounds[d];
            let bin_width = (hi - lo) / n_points as f32;
            if !bin_width.is_finite() || bin_width <= 0.0 {
                return Err(VariationalError::UnsupportedOperation {
                    details: format!(
                        "Latin-hypercube bin width for dimension {d} is not a finite positive f32"
                    ),
                });
            }

            let mut perm: Vec<usize> = (0..n_points).collect();
            for i in (1..perm.len()).rev() {
                let j = (rng.float() * (i as f32 + 1.0)) as usize;
                perm.swap(i, j.min(i));
            }
            for i in 0..n_points {
                let bin_start = lo + perm[i] as f32 * bin_width;
                let value = bin_start + rng.float() * bin_width;
                if !value.is_finite() {
                    return Err(VariationalError::NonFiniteValue {
                        component: "PINN Latin-hypercube point",
                        value,
                    });
                }
                points[i][d] = value;
            }
        }

        Ok(Self { points, ndim })
    }

    pub fn from_latin_hypercube(bounds: &[(f32, f32)], n_points: usize, seed: u64) -> Self {
        Self::try_from_latin_hypercube(bounds, n_points, seed)
            .expect("CollocationPoints::from_latin_hypercube requires valid finite bounds")
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.points.len().saturating_mul(self.ndim));
        for pt in &self.points {
            flat.extend_from_slice(pt);
        }
        flat
    }

    /// Converts the current public point state into a tensor payload only if shape invariants hold.
    pub fn try_to_batched_tensor(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.ndim == 0 {
            return Err(VariationalError::TrainingFailure {
                details: "collocation dimension must be greater than zero".into(),
            });
        }
        for point in &self.points {
            if point.len() != self.ndim {
                return Err(VariationalError::DimensionMismatch {
                    expected: self.ndim,
                    got: point.len(),
                    context: "CollocationPoints::try_to_batched_tensor".into(),
                });
            }
            if let Some(&value) = point.iter().find(|value| !value.is_finite()) {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN collocation point",
                    value,
                });
            }
        }

        let expected_len = self
            .points
            .len()
            .checked_mul(self.ndim)
            .ok_or_else(|| VariationalError::UnsupportedOperation {
                details: "collocation tensor element count overflow".into(),
            })?;
        let flat = self.to_flat();
        if flat.len() != expected_len {
            return Err(VariationalError::DimensionMismatch {
                expected: expected_len,
                got: flat.len(),
                context: "CollocationPoints flattened tensor".into(),
            });
        }
        Ok((flat, vec![self.points.len(), self.ndim]))
    }

    pub fn to_batched_tensor(&self) -> (Vec<f32>, Vec<usize>) {
        self.try_to_batched_tensor()
            .expect("CollocationPoints::to_batched_tensor requires consistent finite points")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinn::domain::Domain1D;

    #[test]
    fn test_uniform_collocation() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let pts = CollocationPoints::from_uniform_grid(&domain, 10);
        assert_eq!(pts.len(), 10);
    }

    #[test]
    fn test_deterministic_sampling() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let a = CollocationPoints::from_random_uniform(&domain, 100, 42);
        let b = CollocationPoints::from_random_uniform(&domain, 100, 42);
        assert_eq!(a.to_flat(), b.to_flat());
    }

    #[test]
    fn test_latin_hypercube() {
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let pts = CollocationPoints::from_latin_hypercube(&bounds, 10, 42);
        assert_eq!(pts.len(), 10);
        assert_eq!(pts.ndim, 2);
    }

    #[test]
    fn checked_sampling_rejects_invalid_bounds() {
        assert!(CollocationPoints::try_from_latin_hypercube(&[], 10, 42).is_err());
        assert!(
            CollocationPoints::try_from_latin_hypercube(&[(0.0, f32::NAN)], 10, 42).is_err()
        );
        assert!(CollocationPoints::try_from_latin_hypercube(&[(1.0, 0.0)], 10, 42).is_err());
    }

    #[test]
    fn random_sampling_revalidates_mutated_domain() {
        let mut domain = Domain1D::new(0.0, 1.0).unwrap();
        domain.end = f32::NAN;
        assert!(CollocationPoints::try_from_random_uniform(&domain, 4, 42).is_err());
    }

    #[test]
    fn batched_tensor_rejects_mutated_point_dimensions() {
        let points = CollocationPoints {
            points: vec![vec![0.0], vec![1.0, 2.0]],
            ndim: 1,
        };
        assert!(points.try_to_batched_tensor().is_err());
    }

    #[test]
    fn batched_tensor_rejects_non_finite_points() {
        let points = CollocationPoints {
            points: vec![vec![f32::NAN]],
            ndim: 1,
        };
        assert!(points.try_to_batched_tensor().is_err());
    }
}
