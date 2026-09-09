use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct LossValue {
    pub total: f32,
    pub components: Vec<(String, f32)>,
}

impl LossValue {
    pub fn new(total: f32) -> Self {
        Self {
            total,
            components: Vec::new(),
        }
    }

    pub fn with_component(mut self, name: &str, value: f32) -> Self {
        self.components.push((name.to_string(), value));
        self
    }
}

pub fn mse_loss(predicted: &[f32], target: &[f32]) -> f32 {
    try_mse_loss(predicted, target).unwrap_or_else(|err| panic!("mse_loss: {err}"))
}

/// Compute mean-squared error for equal-length, non-empty finite vectors.
pub fn try_mse_loss(predicted: &[f32], target: &[f32]) -> Result<f32> {
    if predicted.len() != target.len()
    {
        return Err(VariationalError::DimensionMismatch {
            expected: predicted.len(),
            got: target.len(),
            context: "try_mse_loss predicted/target".into(),
        });
    }
    if predicted.is_empty()
    {
        return Err(VariationalError::TrainingFailure {
            details: "mean-squared error requires at least one value".into(),
        });
    }

    let mut sum = 0.0f32;
    for (&prediction, &truth) in predicted.iter().zip(target.iter())
    {
        validate_finite("loss prediction", prediction)?;
        validate_finite("loss target", truth)?;
        let residual = prediction - truth;
        validate_finite("loss residual", residual)?;
        sum += residual * residual;
        validate_finite("squared-error sum", sum)?;
    }

    let loss = sum / predicted.len() as f32;
    validate_finite("mean-squared error", loss)?;
    Ok(loss)
}

pub fn acceleration_loss(predicted_ddq: &[f32], target_ddq: &[f32]) -> LossValue {
    try_acceleration_loss(predicted_ddq, target_ddq)
        .unwrap_or_else(|err| panic!("acceleration_loss: {err}"))
}

/// Compute the acceleration MSE through the checked equal-shape loss path.
pub fn try_acceleration_loss(predicted_ddq: &[f32], target_ddq: &[f32]) -> Result<LossValue> {
    let loss = try_mse_loss(predicted_ddq, target_ddq)?;
    Ok(LossValue::new(loss).with_component("acceleration", loss))
}

pub fn euler_lagrange_residual_loss(el_residual: &[f32]) -> LossValue {
    try_euler_lagrange_residual_loss(el_residual)
        .unwrap_or_else(|err| panic!("euler_lagrange_residual_loss: {err}"))
}

/// Compute the mean squared Euler-Lagrange residual for a non-empty finite vector.
pub fn try_euler_lagrange_residual_loss(el_residual: &[f32]) -> Result<LossValue> {
    if el_residual.is_empty()
    {
        return Err(VariationalError::TrainingFailure {
            details: "Euler-Lagrange residual loss requires at least one value".into(),
        });
    }

    let mut sum = 0.0f32;
    for &residual in el_residual
    {
        validate_finite("Euler-Lagrange residual", residual)?;
        sum += residual * residual;
        validate_finite("Euler-Lagrange squared residual sum", sum)?;
    }
    let loss = sum / el_residual.len() as f32;
    validate_finite("Euler-Lagrange residual loss", loss)?;
    Ok(LossValue::new(loss).with_component("el_residual", loss))
}

pub fn parameter_norm(params: &[Vec<f32>]) -> f32 {
    try_parameter_norm(params).unwrap_or_else(|err| panic!("parameter_norm: {err}"))
}

/// Compute the Euclidean parameter norm while rejecting non-finite inputs or overflow.
pub fn try_parameter_norm(params: &[Vec<f32>]) -> Result<f32> {
    let mut sum_sq = 0.0f32;
    for &value in params.iter().flat_map(|param| param.iter())
    {
        validate_finite("parameter value", value)?;
        sum_sq += value * value;
        validate_finite("parameter norm squared", sum_sq)?;
    }
    let norm = sum_sq.sqrt();
    validate_finite("parameter norm", norm)?;
    Ok(norm)
}

pub fn hessian_conditioning_penalty(condition_number: f32, threshold: f32) -> f32 {
    try_hessian_conditioning_penalty(condition_number, threshold)
        .unwrap_or_else(|err| panic!("hessian_conditioning_penalty: {err}"))
}

/// Compute the conditioning penalty for finite non-negative inputs and a positive threshold.
pub fn try_hessian_conditioning_penalty(condition_number: f32, threshold: f32) -> Result<f32> {
    validate_finite("Hessian condition number", condition_number)?;
    validate_finite("Hessian conditioning threshold", threshold)?;
    if condition_number < 0.0
    {
        return Err(VariationalError::TrainingFailure {
            details: "Hessian condition number must be non-negative".into(),
        });
    }
    if threshold <= 0.0
    {
        return Err(VariationalError::TrainingFailure {
            details: "Hessian conditioning threshold must be strictly positive".into(),
        });
    }

    let penalty = if condition_number > threshold
    {
        (condition_number / threshold - 1.0).powi(2)
    }
    else
    {
        0.0
    };
    validate_finite("Hessian conditioning penalty", penalty)?;
    Ok(penalty)
}

fn validate_finite(component: &'static str, value: f32) -> Result<()> {
    if !value.is_finite()
    {
        return Err(VariationalError::NonFiniteValue { component, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_identical() {
        let loss = mse_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((loss).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_different() {
        let loss = mse_loss(&[1.0, 2.0], &[1.0, 3.0]);
        assert!((loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parameter_norm() {
        let params = vec![vec![3.0, 4.0]];
        let norm = parameter_norm(&params);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn checked_mse_rejects_length_mismatch() {
        let err = try_mse_loss(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    #[test]
    fn checked_mse_rejects_empty_input() {
        let err = try_mse_loss(&[], &[]).unwrap_err();
        assert!(matches!(err, VariationalError::TrainingFailure { .. }));
    }

    #[test]
    fn checked_mse_rejects_non_finite_values() {
        let err = try_mse_loss(&[f32::NAN], &[0.0]).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "loss prediction",
                ..
            }
        ));
    }

    #[test]
    fn checked_residual_loss_rejects_empty_vector() {
        let err = try_euler_lagrange_residual_loss(&[]).unwrap_err();
        assert!(matches!(err, VariationalError::TrainingFailure { .. }));
    }

    #[test]
    fn checked_parameter_norm_rejects_non_finite_values() {
        let err = try_parameter_norm(&[vec![f32::INFINITY]]).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "parameter value",
                ..
            }
        ));
    }

    #[test]
    fn checked_conditioning_penalty_rejects_zero_threshold() {
        let err = try_hessian_conditioning_penalty(10.0, 0.0).unwrap_err();
        assert!(matches!(err, VariationalError::TrainingFailure { .. }));
    }

    #[test]
    fn checked_conditioning_penalty_matches_legacy_formula() {
        let penalty = try_hessian_conditioning_penalty(20.0, 10.0).unwrap();
        assert!((penalty - 1.0).abs() < 1e-6);
    }
}
