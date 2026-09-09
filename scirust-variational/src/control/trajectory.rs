use crate::error::{Result, VariationalError};

/// Sampled state trajectory with an optional control sequence.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Time associated with each sampled state.
    pub times: Vec<f32>,
    /// State vector at each time sample.
    pub states: Vec<Vec<f32>>,
    /// Optional control vector associated with each time sample.
    pub controls: Option<Vec<Vec<f32>>>,
    /// Number of components in each state vector as observed at construction.
    pub state_dim: usize,
}

impl Trajectory {
    /// Creates a non-empty trajectory with one state per time and uniform state dimension.
    pub fn new(times: Vec<f32>, states: Vec<Vec<f32>>) -> Result<Self> {
        if times.is_empty() || states.is_empty()
        {
            return Err(VariationalError::TrainingFailure {
                details: "empty trajectory".into(),
            });
        }
        if times.len() != states.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: times.len(),
                got: states.len(),
                context: "Trajectory::new".into(),
            });
        }
        let state_dim = states[0].len();
        for s in &states
        {
            if s.len() != state_dim
            {
                return Err(VariationalError::DimensionMismatch {
                    expected: state_dim,
                    got: s.len(),
                    context: "Trajectory::new".into(),
                });
            }
        }
        Ok(Self {
            times,
            states,
            controls: None,
            state_dim,
        })
    }

    /// Returns the number of time/state samples stored in the trajectory.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns whether the trajectory currently contains no time samples.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Attaches one control vector per time sample.
    pub fn with_controls(mut self, controls: Vec<Vec<f32>>) -> Result<Self> {
        if controls.len() != self.times.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.times.len(),
                got: controls.len(),
                context: "Trajectory::with_controls".into(),
            });
        }
        self.controls = Some(controls);
        Ok(self)
    }

    /// Returns the last stored state.
    ///
    /// A trajectory created through [`Trajectory::new`] is non-empty. Because the
    /// fields are public, callers that later clear `states` must not call this method.
    pub fn final_state(&self) -> &[f32] {
        &self.states[self.states.len() - 1]
    }
}
