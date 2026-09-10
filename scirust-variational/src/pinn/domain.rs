use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct Domain1D {
    pub start: f32,
    pub end: f32,
}

impl Domain1D {
    pub fn new(start: f32, end: f32) -> Result<Self> {
        validate_interval(start, end, "Domain1D")?;
        Ok(Self { start, end })
    }

    pub fn contains(&self, x: f32) -> bool {
        x >= self.start && x <= self.end
    }

    /// Returns uniformly spaced points after revalidating the mutable public domain bounds.
    pub fn try_uniform_points(&self, n: usize) -> Result<Vec<f32>> {
        validate_interval(self.start, self.end, "Domain1D::try_uniform_points")?;
        if n == 0
        {
            return Ok(Vec::new());
        }
        if n == 1
        {
            return Ok(vec![0.5 * (self.start + self.end)]);
        }
        let dx = (self.end - self.start) / (n - 1) as f32;
        if !dx.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "Domain1D uniform spacing",
                value: dx,
            });
        }
        let points: Vec<f32> = (0..n).map(|i| self.start + i as f32 * dx).collect();
        if let Some(&value) = points.iter().find(|value| !value.is_finite())
        {
            return Err(VariationalError::NonFiniteValue {
                component: "Domain1D uniform point",
                value,
            });
        }
        Ok(points)
    }

    pub fn uniform_points(&self, n: usize) -> Vec<f32> {
        self.try_uniform_points(n)
            .expect("Domain1D::uniform_points requires valid finite bounds")
    }
}

#[derive(Debug, Clone)]
pub struct DomainRect {
    pub ndim: usize,
    pub bounds: Vec<(f32, f32)>,
}

impl DomainRect {
    pub fn new(bounds: Vec<(f32, f32)>) -> Result<Self> {
        validate_rect_bounds(&bounds, "DomainRect")?;
        Ok(Self {
            ndim: bounds.len(),
            bounds,
        })
    }

    /// Builds a uniform Cartesian grid after validating dimensions and mutable public bounds.
    pub fn try_uniform_grid(&self, points_per_dim: &[usize]) -> Result<Vec<Vec<f32>>> {
        if self.ndim == 0 || self.bounds.len() != self.ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.ndim,
                got: self.bounds.len(),
                context: "DomainRect::try_uniform_grid bounds".into(),
            });
        }
        validate_rect_bounds(&self.bounds, "DomainRect::try_uniform_grid")?;
        if points_per_dim.len() != self.ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.ndim,
                got: points_per_dim.len(),
                context: "DomainRect::try_uniform_grid points_per_dim".into(),
            });
        }
        if points_per_dim.iter().any(|&count| count == 0)
        {
            return Ok(Vec::new());
        }

        let total = points_per_dim.iter().try_fold(1usize, |total, &count| {
            total
                .checked_mul(count)
                .ok_or_else(|| VariationalError::UnsupportedOperation {
                    details: "DomainRect uniform-grid point count overflow".into(),
                })
        })?;
        let mut grid = Vec::with_capacity(total);
        for idx in 0..total
        {
            let mut point = Vec::with_capacity(self.ndim);
            let mut remaining = idx;
            for d in (0..self.ndim).rev()
            {
                let dim_count = points_per_dim[d];
                let coord_idx = remaining % dim_count;
                remaining /= dim_count;
                let (lo, hi) = self.bounds[d];
                let x = if dim_count == 1
                {
                    0.5 * (lo + hi)
                }
                else
                {
                    lo + coord_idx as f32 * (hi - lo) / (dim_count - 1) as f32
                };
                if !x.is_finite()
                {
                    return Err(VariationalError::NonFiniteValue {
                        component: "DomainRect uniform-grid coordinate",
                        value: x,
                    });
                }
                point.push(x);
            }
            point.reverse();
            grid.push(point);
        }
        Ok(grid)
    }

    pub fn uniform_grid(&self, points_per_dim: &[usize]) -> Vec<Vec<f32>> {
        self.try_uniform_grid(points_per_dim)
            .expect("DomainRect::uniform_grid requires valid bounds and grid dimensions")
    }
}

fn validate_interval(start: f32, end: f32, context: &str) -> Result<()> {
    if !start.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "domain start",
            value: start,
        });
    }
    if !end.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "domain end",
            value: end,
        });
    }
    if start >= end
    {
        return Err(VariationalError::InvalidInterval { start, end });
    }
    let span = end - start;
    if !span.is_finite()
    {
        return Err(VariationalError::UnsupportedOperation {
            details: format!("{context} span is not representable as finite f32"),
        });
    }
    Ok(())
}

fn validate_rect_bounds(bounds: &[(f32, f32)], context: &str) -> Result<()> {
    if bounds.is_empty()
    {
        return Err(VariationalError::UnsupportedOperation {
            details: format!("{context} requires at least one dimension"),
        });
    }
    for &(lo, hi) in bounds
    {
        validate_interval(lo, hi, context)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain1d() {
        let d = Domain1D::new(0.0, 1.0).unwrap();
        assert!(d.contains(0.5));
        assert!(!d.contains(-0.1));
        let pts = d.uniform_points(5);
        assert_eq!(pts.len(), 5);
        assert!((pts[0] - 0.0).abs() < 1e-6);
        assert!((pts[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_domain_rect() {
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let d = DomainRect::new(bounds).unwrap();
        let grid = d.uniform_grid(&[3, 3]);
        assert_eq!(grid.len(), 9);
    }

    #[test]
    fn test_invalid_interval() {
        assert!(Domain1D::new(1.0, 0.0).is_err());
    }

    #[test]
    fn domains_reject_non_finite_bounds() {
        assert!(Domain1D::new(f32::NAN, 1.0).is_err());
        assert!(Domain1D::new(0.0, f32::INFINITY).is_err());
        assert!(DomainRect::new(vec![(0.0, f32::NAN)]).is_err());
    }

    #[test]
    fn uniform_points_revalidate_mutated_public_bounds() {
        let mut domain = Domain1D::new(0.0, 1.0).unwrap();
        domain.end = f32::NAN;
        assert!(domain.try_uniform_points(4).is_err());
    }

    #[test]
    fn uniform_grid_rejects_mutated_dimension_state() {
        let mut domain = DomainRect::new(vec![(0.0, 1.0), (0.0, 1.0)]).unwrap();
        domain.ndim = 3;
        assert!(domain.try_uniform_grid(&[2, 2, 2]).is_err());
    }

    #[test]
    fn uniform_grid_reports_dimension_mismatch_without_panicking() {
        let domain = DomainRect::new(vec![(0.0, 1.0), (0.0, 1.0)]).unwrap();
        assert!(domain.try_uniform_grid(&[2]).is_err());
    }
}
