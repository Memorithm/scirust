use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct ConservationReport {
    pub initial: f32,
    pub final_value: f32,
    pub max_abs_drift: f32,
    pub relative_drift: f32,
    pub rms_drift: f32,
}

impl ConservationReport {
    pub fn new(values: &[f32]) -> Self {
        Self::try_new(values).unwrap_or_else(|err| panic!("ConservationReport::new: {err}"))
    }

    /// Build a conservation report from a non-empty finite value series.
    pub fn try_new(values: &[f32]) -> Result<Self> {
        if values.is_empty()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                context: "ConservationReport::try_new".into(),
            });
        }
        for &value in values
        {
            if !value.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "conservation value",
                    value,
                });
            }
        }

        let initial = values[0];
        let final_value = values[values.len() - 1];
        let max_abs_drift = values
            .iter()
            .map(|&v| (v - initial).abs())
            .fold(0.0, f32::max);
        let relative_drift = if initial.abs() > 1e-10
        {
            max_abs_drift / initial.abs()
        }
        else
        {
            max_abs_drift
        };
        let rms_drift = (values.iter().map(|&v| (v - initial).powi(2)).sum::<f32>()
            / values.len() as f32)
            .sqrt();

        Ok(Self {
            initial,
            final_value,
            max_abs_drift,
            relative_drift,
            rms_drift,
        })
    }

    pub fn is_conserved_within(&self, tolerance: f32) -> bool {
        self.max_abs_drift < tolerance
    }
}

pub fn compute_energy<F>(lagrangian: &F, q: &[f32], dq: &[f32], t: f32) -> f32
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    try_compute_energy(lagrangian, q, dq, t)
        .unwrap_or_else(|err| panic!("compute_energy: {err}"))
}

/// Compute the Legendre energy while validating dimensions and finite values.
pub fn try_compute_energy<F>(lagrangian: &F, q: &[f32], dq: &[f32], t: f32) -> Result<f32>
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    if q.len() != dq.len()
    {
        return Err(VariationalError::DimensionMismatch {
            expected: q.len(),
            got: dq.len(),
            context: "try_compute_energy q/dq".into(),
        });
    }
    for &value in q
    {
        if !value.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "generalized coordinate",
                value,
            });
        }
    }
    for &value in dq
    {
        if !value.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "generalized velocity",
                value,
            });
        }
    }
    if !t.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "time",
            value: t,
        });
    }

    let eps = 1e-4;
    let n = q.len();
    let lagrangian_value = lagrangian(q, dq, t);
    if !lagrangian_value.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "Lagrangian",
            value: lagrangian_value,
        });
    }

    let mut energy = 0.0;
    for i in 0..n
    {
        let mut dq_plus = dq.to_vec();
        dq_plus[i] += eps;
        let lagrangian_plus = lagrangian(q, &dq_plus, t);
        if !lagrangian_plus.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "perturbed Lagrangian",
                value: lagrangian_plus,
            });
        }

        let mut dq_minus = dq.to_vec();
        dq_minus[i] -= eps;
        let lagrangian_minus = lagrangian(q, &dq_minus, t);
        if !lagrangian_minus.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "perturbed Lagrangian",
                value: lagrangian_minus,
            });
        }

        let d_l_ddq_i = (lagrangian_plus - lagrangian_minus) / (2.0 * eps);
        energy += d_l_ddq_i * dq[i];
    }
    energy -= lagrangian_value;

    if !energy.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "energy",
            value: energy,
        });
    }

    Ok(energy)
}

pub fn compute_energy_from_trajectory<F>(
    lagrangian: &F,
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
) -> Vec<f32>
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    try_compute_energy_from_trajectory(lagrangian, trajectory, ndim)
        .unwrap_or_else(|err| panic!("compute_energy_from_trajectory: {err}"))
}

/// Compute energy samples from a trajectory with exact phase-space dimension checks.
pub fn try_compute_energy_from_trajectory<F>(
    lagrangian: &F,
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
) -> Result<Vec<f32>>
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    let expected = ndim
        .checked_mul(2)
        .ok_or_else(|| VariationalError::UnsupportedOperation {
            details: "phase-space dimension overflow".into(),
        })?;

    let mut values = Vec::with_capacity(trajectory.len());
    for (index, (t, state)) in trajectory.iter().enumerate()
    {
        if state.len() != expected
        {
            return Err(VariationalError::DimensionMismatch {
                expected,
                got: state.len(),
                context: format!("try_compute_energy_from_trajectory state {index}"),
            });
        }
        let q = &state[..ndim];
        let dq = &state[ndim..expected];
        values.push(try_compute_energy(lagrangian, q, dq, *t)?);
    }

    Ok(values)
}

pub fn compute_invariant_diagnostics<F>(
    lagrangian: &F,
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
    label: &str,
) -> ConservationReport
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    try_compute_invariant_diagnostics(lagrangian, trajectory, ndim, label)
        .unwrap_or_else(|err| panic!("compute_invariant_diagnostics: {err}"))
}

/// Compute and print conservation diagnostics through the checked energy path.
pub fn try_compute_invariant_diagnostics<F>(
    lagrangian: &F,
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
    label: &str,
) -> Result<ConservationReport>
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    if trajectory.is_empty()
    {
        return Err(VariationalError::DimensionMismatch {
            expected: 1,
            got: 0,
            context: "try_compute_invariant_diagnostics trajectory".into(),
        });
    }

    let values = try_compute_energy_from_trajectory(lagrangian, trajectory, ndim)?;
    let report = ConservationReport::try_new(&values)?;
    println!(
        "Conservation diagnostics for {label}: \
         initial={:.6}, final={:.6}, max_drift={:.2e}, rel_drift={:.2e}, rms_drift={:.2e}",
        report.initial,
        report.final_value,
        report.max_abs_drift,
        report.relative_drift,
        report.rms_drift
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_oscillator_energy(q: &[f32], dq: &[f32], _t: f32) -> f32 {
        0.5 * dq[0] * dq[0] + 0.5 * q[0] * q[0]
    }

    #[test]
    fn test_energy_conservation_harmonic() {
        let traj = vec![
            (0.0, vec![1.0, 0.0]),
            (0.5, vec![0.87758, -0.47943]),
            (1.0, vec![0.54030, -0.84147]),
            (1.5, vec![0.07074, -0.99749]),
            (2.0, vec![-0.41615, -0.90930]),
        ];
        let report = compute_invariant_diagnostics(
            &harmonic_oscillator_energy,
            &traj,
            1,
            "harmonic_oscillator",
        );
        assert!(
            report.max_abs_drift < 1.0,
            "energy drift too large: {}",
            report.max_abs_drift
        );
    }

    #[test]
    fn test_conservation_report_all_constant() {
        let values = vec![10.0, 10.0, 10.0];
        let report = ConservationReport::new(&values);
        assert!(report.max_abs_drift < 1e-6);
        assert!(report.relative_drift < 1e-6);
        assert!(report.rms_drift < 1e-6);
    }

    #[test]
    fn test_conservation_report_with_drift() {
        let values = vec![10.0, 10.1, 10.2, 10.3];
        let report = ConservationReport::new(&values);
        assert!((report.max_abs_drift - 0.3).abs() < 1e-6);
    }

    #[test]
    fn try_report_rejects_empty_values() {
        let err = ConservationReport::try_new(&[]).unwrap_err();
        match err {
            VariationalError::DimensionMismatch {
                expected,
                got,
                context,
            } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
                assert_eq!(context, "ConservationReport::try_new");
            },
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn try_report_rejects_non_finite_values() {
        let err = ConservationReport::try_new(&[1.0, f32::NAN]).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "conservation value",
                ..
            }
        ));
    }

    #[test]
    fn try_energy_rejects_dimension_mismatch() {
        let err = try_compute_energy(&harmonic_oscillator_energy, &[1.0], &[], 0.0).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn try_trajectory_energy_rejects_wrong_state_dimension() {
        let trajectory = vec![(0.0, vec![1.0])];
        let err = try_compute_energy_from_trajectory(&harmonic_oscillator_energy, &trajectory, 1)
            .unwrap_err();
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
    fn try_invariant_diagnostics_rejects_empty_trajectory() {
        let err = try_compute_invariant_diagnostics(
            &harmonic_oscillator_energy,
            &[],
            1,
            "empty",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn try_energy_rejects_non_finite_lagrangian_output() {
        let nan_lagrangian = |_q: &[f32], _dq: &[f32], _t: f32| f32::NAN;
        let err = try_compute_energy(&nan_lagrangian, &[1.0], &[0.0], 0.0).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "Lagrangian",
                ..
            }
        ));
    }
}
