use scirust_core::nn::rng::PcgEngine;

use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct TrajectorySample {
    pub q: Vec<f32>,
    pub dq: Vec<f32>,
    pub ddq: Vec<f32>,
    pub t: f32,
}

#[derive(Debug, Clone)]
pub struct TrajectoryDataset {
    pub samples: Vec<TrajectorySample>,
    pub ndim: usize,
    pub indices: Vec<usize>,
}

impl TrajectoryDataset {
    pub fn new(samples: Vec<TrajectorySample>) -> Result<Self> {
        if samples.is_empty()
        {
            return Err(VariationalError::TrainingFailure {
                details: "empty trajectory dataset".into(),
            });
        }
        let ndim = samples[0].q.len();
        if ndim == 0
        {
            return Err(VariationalError::TrainingFailure {
                details: "zero-dimensional trajectory dataset".into(),
            });
        }
        for sample in &samples
        {
            validate_sample(sample, ndim, "TrajectoryDataset::new")?;
        }
        let indices: Vec<usize> = (0..samples.len()).collect();
        Ok(Self {
            samples,
            ndim,
            indices,
        })
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn shuffle(&mut self, rng: &mut PcgEngine) {
        let n = self.indices.len();
        for i in (1..n).rev()
        {
            let j = (rng.float() * (i as f32 + 1.0)) as usize;
            self.indices.swap(i, j.min(i));
        }
    }

    /// Returns one sample from the active dataset view after validating both index layers.
    pub fn try_get(&self, index: usize) -> Result<&TrajectorySample> {
        let sample_index = *self
            .indices
            .get(index)
            .ok_or_else(|| VariationalError::TrainingFailure {
                details: format!(
                    "TrajectoryDataset::try_get active index {index} is out of bounds for {} samples",
                    self.indices.len()
                ),
            })?;
        self.samples
            .get(sample_index)
            .ok_or_else(|| VariationalError::TrainingFailure {
                details: format!(
                    "TrajectoryDataset::try_get stored sample index {sample_index} is out of bounds for {} samples",
                    self.samples.len()
                ),
            })
    }

    pub fn get(&self, index: usize) -> &TrajectorySample {
        self.try_get(index)
            .expect("TrajectoryDataset::get requires a valid active sample index")
    }

    /// Splits the active dataset view while validating the requested fraction and stored indices.
    pub fn try_split(mut self, train_frac: f32, seed: u64) -> Result<(Self, Self)> {
        if !train_frac.is_finite() || !(0.0..=1.0).contains(&train_frac)
        {
            return Err(VariationalError::TrainingFailure {
                details: format!(
                    "TrajectoryDataset::try_split requires a finite train fraction in [0, 1], got {train_frac}"
                ),
            });
        }
        if let Some(&bad_index) = self
            .indices
            .iter()
            .find(|&&sample_index| sample_index >= self.samples.len())
        {
            return Err(VariationalError::TrainingFailure {
                details: format!(
                    "TrajectoryDataset::try_split contains out-of-range sample index {bad_index} for {} samples",
                    self.samples.len()
                ),
            });
        }

        let mut rng = PcgEngine::new(seed);
        self.shuffle(&mut rng);
        let split_point = (self.indices.len() as f32 * train_frac).floor() as usize;
        let train_indices = self.indices[..split_point].to_vec();
        let val_indices = self.indices[split_point..].to_vec();

        let train = Self {
            samples: self.samples.clone(),
            ndim: self.ndim,
            indices: train_indices,
        };
        let val = Self {
            samples: self.samples,
            ndim: self.ndim,
            indices: val_indices,
        };
        Ok((train, val))
    }

    pub fn split(self, train_frac: f32, seed: u64) -> (Self, Self) {
        self.try_split(train_frac, seed)
            .expect("TrajectoryDataset::split requires a valid split request")
    }
}

fn validate_sample(sample: &TrajectorySample, ndim: usize, context: &str) -> Result<()> {
    for (component, got) in [
        ("q", sample.q.len()),
        ("dq", sample.dq.len()),
        ("ddq", sample.ddq.len()),
    ]
    {
        if got != ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: ndim,
                got,
                context: format!("{context} {component}"),
            });
        }
    }
    ensure_finite_slice(&sample.q, "trajectory q")?;
    ensure_finite_slice(&sample.dq, "trajectory dq")?;
    ensure_finite_slice(&sample.ddq, "trajectory ddq")?;
    if !sample.t.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "trajectory time",
            value: sample.t,
        });
    }
    Ok(())
}

fn ensure_finite_slice(values: &[f32], component: &'static str) -> Result<()> {
    if let Some(&value) = values.iter().find(|value| !value.is_finite())
    {
        return Err(VariationalError::NonFiniteValue { component, value });
    }
    Ok(())
}

fn validate_dynamics_output(values: &[f32], stage: &'static str) -> Result<()> {
    ensure_finite_slice(values, stage)
}

pub fn trajectory_from_ode<F>(
    dynamics: F,
    q0: &[f32],
    dq0: &[f32],
    t_span: &[f32],
    ndim: usize,
) -> Result<TrajectoryDataset>
where
    F: Fn(f32, &[f32], &mut [f32]),
{
    if ndim == 0
    {
        return Err(VariationalError::TrainingFailure {
            details: "trajectory_from_ode requires ndim > 0".into(),
        });
    }
    if q0.len() != ndim
    {
        return Err(VariationalError::DimensionMismatch {
            expected: ndim,
            got: q0.len(),
            context: "trajectory_from_ode q0".into(),
        });
    }
    if dq0.len() != ndim
    {
        return Err(VariationalError::DimensionMismatch {
            expected: ndim,
            got: dq0.len(),
            context: "trajectory_from_ode dq0".into(),
        });
    }
    ensure_finite_slice(q0, "trajectory initial q")?;
    ensure_finite_slice(dq0, "trajectory initial dq")?;

    if t_span.len() < 2
    {
        return Err(VariationalError::InvalidInterval {
            start: t_span.first().copied().unwrap_or(0.0),
            end: t_span.last().copied().unwrap_or(0.0),
        });
    }
    for &t in t_span
    {
        if !t.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "trajectory time grid",
                value: t,
            });
        }
    }
    for interval in t_span.windows(2)
    {
        if interval[0] >= interval[1]
        {
            return Err(VariationalError::InvalidInterval {
                start: interval[0],
                end: interval[1],
            });
        }
    }

    let phase_dim = ndim
        .checked_mul(2)
        .ok_or_else(|| VariationalError::TrainingFailure {
            details: "trajectory_from_ode phase-space dimension overflow".into(),
        })?;

    let mut samples = Vec::with_capacity(t_span.len());
    let mut state: Vec<f32> = Vec::with_capacity(phase_dim);
    state.extend_from_slice(q0);
    state.extend_from_slice(dq0);

    let mut deriv = vec![0.0; phase_dim];

    for step in 0..t_span.len()
    {
        let t = t_span[step];
        ensure_finite_slice(&state, "trajectory state")?;

        deriv.fill(0.0);
        dynamics(t, &state, &mut deriv);
        validate_dynamics_output(&deriv, "trajectory dynamics")?;

        let q: Vec<f32> = state[..ndim].to_vec();
        let dq: Vec<f32> = state[ndim..].to_vec();
        let ddq: Vec<f32> = deriv[ndim..].to_vec();
        samples.push(TrajectorySample { q, dq, ddq, t });

        if step + 1 == t_span.len()
        {
            break;
        }

        let h = t_span[step + 1] - t;
        let k1 = deriv.clone();
        let mut k2 = vec![0.0; phase_dim];
        let mut k3 = vec![0.0; phase_dim];
        let mut k4 = vec![0.0; phase_dim];
        let mut tmp = vec![0.0; phase_dim];

        for i in 0..phase_dim
        {
            tmp[i] = state[i] + 0.5 * h * k1[i];
        }
        dynamics(t + 0.5 * h, &tmp, &mut k2);
        validate_dynamics_output(&k2, "trajectory RK4 k2")?;

        for i in 0..phase_dim
        {
            tmp[i] = state[i] + 0.5 * h * k2[i];
        }
        dynamics(t + 0.5 * h, &tmp, &mut k3);
        validate_dynamics_output(&k3, "trajectory RK4 k3")?;

        for i in 0..phase_dim
        {
            tmp[i] = state[i] + h * k3[i];
        }
        dynamics(t + h, &tmp, &mut k4);
        validate_dynamics_output(&k4, "trajectory RK4 k4")?;

        for i in 0..phase_dim
        {
            state[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        ensure_finite_slice(&state, "trajectory integrated state")?;
    }

    TrajectoryDataset::new(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_dynamics(_t: f32, state: &[f32], deriv: &mut [f32]) {
        deriv[0] = state[1];
        deriv[1] = -state[0];
    }

    #[test]
    fn test_trajectory_dataset() {
        let t_span: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.ndim, 1);
    }

    #[test]
    fn test_train_val_split() {
        let t_span: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        let n = dataset.len();
        let (train, val) = dataset.split(0.8, 42);
        assert_eq!(train.len() + val.len(), n);
        assert!(!train.is_empty());
        assert!(!val.is_empty());
    }

    #[test]
    fn empty_active_split_reports_empty() {
        let t_span = [0.0, 0.1];
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        let (train, val) = dataset.try_split(0.0, 42).unwrap();
        assert!(train.is_empty());
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn repeated_split_uses_active_view_length() {
        let t_span: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        let (train, _) = dataset.try_split(0.5, 42).unwrap();
        assert_eq!(train.len(), 5);
        let (subtrain, subval) = train.try_split(0.8, 7).unwrap();
        assert_eq!(subtrain.len(), 4);
        assert_eq!(subval.len(), 1);
    }

    #[test]
    fn split_rejects_invalid_fraction() {
        let t_span = [0.0, 0.1];
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        assert!(dataset.clone().try_split(f32::NAN, 42).is_err());
        assert!(dataset.try_split(1.1, 42).is_err());
    }

    #[test]
    fn trajectory_rejects_initial_dimension_mismatch() {
        let t_span = [0.0, 0.1];
        assert!(trajectory_from_ode(harmonic_dynamics, &[], &[0.0], &t_span, 1).is_err());
        assert!(trajectory_from_ode(harmonic_dynamics, &[1.0], &[], &t_span, 1).is_err());
    }

    #[test]
    fn trajectory_rejects_non_increasing_time_grid() {
        assert!(
            trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &[0.0, 0.1, 0.1], 1)
                .is_err()
        );
    }

    #[test]
    fn trajectory_uses_each_nonuniform_time_step() {
        fn unit_position_rate(_t: f32, _state: &[f32], deriv: &mut [f32]) {
            deriv[0] = 1.0;
            deriv[1] = 0.0;
        }

        let dataset = trajectory_from_ode(
            unit_position_rate,
            &[0.0],
            &[0.0],
            &[0.0, 0.1, 0.4],
            1,
        )
        .unwrap();

        assert!((dataset.samples[0].q[0] - 0.0).abs() < 1e-6);
        assert!((dataset.samples[1].q[0] - 0.1).abs() < 1e-6);
        assert!((dataset.samples[2].q[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn dataset_rejects_non_finite_sample_values() {
        let sample = TrajectorySample {
            q: vec![f32::NAN],
            dq: vec![0.0],
            ddq: vec![0.0],
            t: 0.0,
        };
        assert!(TrajectoryDataset::new(vec![sample]).is_err());
    }
}
