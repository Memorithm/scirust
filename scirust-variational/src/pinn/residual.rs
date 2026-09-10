use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub enum DifferentialOperator {
    /// ∂u/∂x (first derivative w.r.t. coordinate index)
    FirstDerivative(usize),
    /// ∂²u/∂x² (second derivative w.r.t. coordinate index)
    SecondDerivative(usize),
    /// ∇u (gradient magnitude squared)
    Gradient,
    /// Δu (Laplacian, sum of second derivatives)
    Laplacian,
    /// ∂u/∂t (time derivative)
    TimeDerivative,
}

impl std::fmt::Display for DifferentialOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstDerivative(i) => write!(f, "d/dx{i}"),
            Self::SecondDerivative(i) => write!(f, "d²/dx{i}²"),
            Self::Gradient => write!(f, "∇"),
            Self::Laplacian => write!(f, "Δ"),
            Self::TimeDerivative => write!(f, "∂/∂t"),
        }
    }
}

/// Computes first and second 1-D central differences with checked finite inputs/evaluations.
pub fn try_central_difference_1d<F: Fn(f32) -> f32>(
    f: F,
    x: f32,
    h: f32,
) -> Result<(f32, f32)> {
    validate_step(h, "try_central_difference_1d")?;
    if !x.is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference coordinate",
            value: x,
        });
    }

    let xp = x + h;
    let xm = x - h;
    if !xp.is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference positive perturbation",
            value: xp,
        });
    }
    if !xm.is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference negative perturbation",
            value: xm,
        });
    }

    let fp = checked_eval(f(xp), "central-difference f(x+h)")?;
    let fm = checked_eval(f(xm), "central-difference f(x-h)")?;
    let f0 = checked_eval(f(x), "central-difference f(x)")?;
    checked_derivatives(fp, fm, f0, h)
}

pub fn central_difference_1d<F: Fn(f32) -> f32>(f: F, x: f32, h: f32) -> (f32, f32) {
    try_central_difference_1d(f, x, h)
        .expect("central_difference_1d requires finite inputs, evaluations and positive step")
}

/// Computes first and second central differences along one checked coordinate axis.
pub fn try_central_difference<F: Fn(&[f32]) -> f32>(
    f: F,
    x: &[f32],
    axis: usize,
    h: f32,
) -> Result<(f32, f32)> {
    validate_step(h, "try_central_difference")?;
    if axis >= x.len() {
        return Err(VariationalError::DimensionMismatch {
            expected: x.len(),
            got: axis.saturating_add(1),
            context: "try_central_difference axis".into(),
        });
    }
    if let Some(&value) = x.iter().find(|value| !value.is_finite()) {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference coordinate",
            value,
        });
    }

    let mut xp = x.to_vec();
    xp[axis] += h;
    if !xp[axis].is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference positive perturbation",
            value: xp[axis],
        });
    }
    let fp = checked_eval(f(&xp), "central-difference f(x+h)")?;

    let mut xm = x.to_vec();
    xm[axis] -= h;
    if !xm[axis].is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference negative perturbation",
            value: xm[axis],
        });
    }
    let fm = checked_eval(f(&xm), "central-difference f(x-h)")?;
    let f0 = checked_eval(f(x), "central-difference f(x)")?;

    checked_derivatives(fp, fm, f0, h)
}

pub fn central_difference<F: Fn(&[f32]) -> f32>(
    f: F,
    x: &[f32],
    axis: usize,
    h: f32,
) -> (f32, f32) {
    try_central_difference(f, x, axis, h)
        .expect("central_difference requires a valid axis and finite positive-step evaluations")
}

fn validate_step(h: f32, context: &str) -> Result<()> {
    if !h.is_finite() || h <= 0.0 {
        return Err(VariationalError::UnsupportedOperation {
            details: format!("{context} requires a finite positive step, got {h}"),
        });
    }
    let h_sq = h * h;
    if !h_sq.is_finite() || h_sq == 0.0 {
        return Err(VariationalError::UnsupportedOperation {
            details: format!("{context} step squared is not a finite non-zero f32"),
        });
    }
    Ok(())
}

fn checked_eval(value: f32, component: &'static str) -> Result<f32> {
    if !value.is_finite() {
        return Err(VariationalError::NonFiniteValue { component, value });
    }
    Ok(value)
}

fn checked_derivatives(fp: f32, fm: f32, f0: f32, h: f32) -> Result<(f32, f32)> {
    let df = (fp - fm) / (2.0 * h);
    let d2f = (fp - 2.0 * f0 + fm) / (h * h);
    if !df.is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference first derivative",
            value: df,
        });
    }
    if !d2f.is_finite() {
        return Err(VariationalError::NonFiniteValue {
            component: "central-difference second derivative",
            value: d2f,
        });
    }
    Ok((df, d2f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_difference_sin() {
        let (df, d2f) = central_difference_1d(|x| x.sin(), 0.5, 1e-2);
        assert!((df - 0.5_f32.cos()).abs() < 1e-3);
        assert!((d2f + 0.5_f32.sin()).abs() < 1e-2);
    }

    #[test]
    fn test_central_difference_quadratic() {
        let (df, d2f) = central_difference_1d(|x| x * x, 2.0, 1e-2);
        assert!((df - 4.0).abs() < 1e-3);
        assert!((d2f - 2.0).abs() < 1e-2);
    }

    #[test]
    fn checked_difference_rejects_zero_or_non_finite_step() {
        assert!(try_central_difference_1d(|x| x, 0.0, 0.0).is_err());
        assert!(try_central_difference_1d(|x| x, 0.0, f32::NAN).is_err());
    }

    #[test]
    fn checked_difference_rejects_axis_out_of_bounds() {
        assert!(try_central_difference(|x| x[0], &[1.0], 1, 1e-2).is_err());
    }

    #[test]
    fn checked_difference_rejects_non_finite_coordinates() {
        assert!(try_central_difference(|x| x[0], &[f32::NAN], 0, 1e-2).is_err());
    }

    #[test]
    fn checked_difference_rejects_non_finite_callback_value() {
        assert!(try_central_difference_1d(|_| f32::NAN, 0.5, 1e-2).is_err());
    }

    #[test]
    fn checked_difference_preserves_valid_multidimensional_result() {
        let (df, d2f) = try_central_difference(|x| x[0] * x[0] + x[1], &[2.0, 3.0], 0, 1e-2)
            .unwrap();
        assert!((df - 4.0).abs() < 1e-3);
        assert!((d2f - 2.0).abs() < 1e-2);
    }
}
