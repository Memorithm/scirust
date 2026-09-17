use crate::{NeuralOperatorError, Result};

/// Uniform periodic one-dimensional grid on `[0, length)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicGrid1d {
    points: usize,
    length: f64,
}

impl PeriodicGrid1d {
    /// Construct a periodic grid. The current spectral backend requires a
    /// radix-2 number of points because it uses SciRust's deterministic FFT.
    pub fn new(points: usize, length: f64) -> Result<Self> {
        if points == 0
        {
            return Err(NeuralOperatorError::Empty { what: "grid" });
        }
        if !points.is_power_of_two()
        {
            return Err(NeuralOperatorError::NonRadix2 { len: points });
        }
        if !length.is_finite() || length <= 0.0
        {
            return Err(NeuralOperatorError::InvalidDomainLength { length });
        }
        Ok(Self { points, length })
    }

    /// Return the number of grid points.
    pub fn points(self) -> usize {
        self.points
    }
    /// Return the physical period length.
    pub fn length(self) -> f64 {
        self.length
    }
    /// Return the uniform grid spacing `length / points`.
    pub fn spacing(self) -> f64 {
        self.length / self.points as f64
    }

    /// Coordinates `x_j = j L/N`, excluding the duplicate endpoint `L`.
    pub fn coordinates(self) -> Vec<f64> {
        let dx = self.spacing();
        (0..self.points).map(|j| j as f64 * dx).collect()
    }

    /// Signed angular wavenumber for FFT bin `index`.
    pub fn angular_wavenumber(self, index: usize) -> f64 {
        assert!(index < self.points);
        let n = self.points as isize;
        let i = index as isize;
        let signed = if i <= n / 2 { i } else { i - n };
        std::f64::consts::TAU * signed as f64 / self.length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_grid_has_expected_spacing_and_signed_modes() {
        let g = PeriodicGrid1d::new(8, std::f64::consts::TAU).unwrap();
        assert!((g.spacing() - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        assert!((g.angular_wavenumber(1) - 1.0).abs() < 1e-15);
        assert!((g.angular_wavenumber(7) + 1.0).abs() < 1e-15);
    }
}
