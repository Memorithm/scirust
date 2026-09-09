use crate::error::{Result, VariationalError};
use crate::mechanics::invariants::ConservationReport;

pub fn compare_conservation(trajectory: &[(f32, Vec<f32>)], ndim: usize) -> ConservationReport {
    try_compare_conservation(trajectory, ndim)
        .unwrap_or_else(|err| panic!("compare_conservation: {err}"))
}

/// Compute the trajectory conservation report with exact phase-space validation.
pub fn try_compare_conservation(
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
) -> Result<ConservationReport> {
    let expected = ndim
        .checked_mul(2)
        .ok_or_else(|| VariationalError::UnsupportedOperation {
            details: "phase-space dimension overflow".into(),
        })?;

    if trajectory.is_empty()
    {
        return Ok(ConservationReport::new(&[0.0]));
    }

    let mut jacobi_integrals = Vec::with_capacity(trajectory.len());

    for (index, (_, state)) in trajectory.iter().enumerate()
    {
        if state.len() != expected
        {
            return Err(VariationalError::DimensionMismatch {
                expected,
                got: state.len(),
                context: format!("try_compare_conservation state {index}"),
            });
        }

        let mut sum = 0.0;
        for i in 0..ndim
        {
            let qi = state[i];
            let pi = state[i + ndim];
            if !qi.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "generalized coordinate",
                    value: qi,
                });
            }
            if !pi.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "generalized momentum",
                    value: pi,
                });
            }
            sum += pi * qi;
        }

        if !sum.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "conservation integral",
                value: sum,
            });
        }
        jacobi_integrals.push(sum);
    }

    Ok(ConservationReport::new(&jacobi_integrals))
}

pub fn detect_drift(values: &[f32], drift_threshold: f32) -> (bool, f32) {
    try_detect_drift(values, drift_threshold).unwrap_or_else(|err| panic!("detect_drift: {err}"))
}

/// Detect endpoint drift after validating the series and non-negative threshold.
pub fn try_detect_drift(values: &[f32], drift_threshold: f32) -> Result<(bool, f32)> {
    if !drift_threshold.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "drift threshold",
            value: drift_threshold,
        });
    }
    if drift_threshold < 0.0
    {
        return Err(VariationalError::UnsupportedOperation {
            details: "drift threshold must be non-negative".into(),
        });
    }

    for &value in values
    {
        if !value.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "drift series",
                value,
            });
        }
    }

    if values.len() < 2
    {
        return Ok((false, 0.0));
    }

    let initial = values[0];
    let final_val = values[values.len() - 1];
    let drift = (final_val - initial).abs();
    if !drift.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "drift",
            value: drift,
        });
    }

    Ok((drift > drift_threshold, drift))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detection() {
        let values = vec![1.0, 1.01, 1.02, 1.03];
        let (has_drift, drift) = detect_drift(&values, 0.01);
        assert!(has_drift);
        assert!((drift - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_no_drift() {
        let values = vec![1.0, 1.0, 1.0];
        let (has_drift, _) = detect_drift(&values, 0.01);
        assert!(!has_drift);
    }

    #[test]
    fn checked_conservation_preserves_empty_trajectory_convention() {
        let report = try_compare_conservation(&[], 1).unwrap();
        assert_eq!(report.initial, 0.0);
        assert_eq!(report.final_value, 0.0);
        assert_eq!(report.max_abs_drift, 0.0);
    }

    #[test]
    fn checked_conservation_rejects_short_phase_state() {
        let trajectory = vec![(0.0, vec![1.0])];
        let err = try_compare_conservation(&trajectory, 1).unwrap_err();
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
    fn checked_conservation_rejects_extra_phase_components() {
        let trajectory = vec![(0.0, vec![1.0, 2.0, 3.0])];
        let err = try_compare_conservation(&trajectory, 1).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 2,
                got: 3,
                ..
            }
        ));
    }

    #[test]
    fn checked_conservation_rejects_non_finite_state() {
        let trajectory = vec![(0.0, vec![f32::NAN, 1.0])];
        let err = try_compare_conservation(&trajectory, 1).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "generalized coordinate",
                ..
            }
        ));
    }

    #[test]
    fn checked_drift_rejects_negative_threshold() {
        let err = try_detect_drift(&[1.0, 1.1], -0.1).unwrap_err();
        assert!(matches!(err, VariationalError::UnsupportedOperation { .. }));
    }

    #[test]
    fn checked_drift_rejects_non_finite_threshold() {
        let err = try_detect_drift(&[1.0, 1.1], f32::NAN).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "drift threshold",
                ..
            }
        ));
    }

    #[test]
    fn checked_drift_rejects_non_finite_series() {
        let err = try_detect_drift(&[1.0, f32::INFINITY], 0.1).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "drift series",
                ..
            }
        ));
    }
}
