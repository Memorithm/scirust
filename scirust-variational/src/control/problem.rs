use crate::error::{Result, VariationalError};

/// Component-wise lower and upper bounds used by optimal-control problems.
#[derive(Debug, Clone)]
pub struct ControlBounds {
    /// Inclusive lower bound for each component.
    pub lower: Vec<f32>,
    /// Inclusive upper bound for each component.
    pub upper: Vec<f32>,
}

impl ControlBounds {
    /// Creates component-wise bounds after validating equal lengths and `lower <= upper`.
    pub fn new(lower: Vec<f32>, upper: Vec<f32>) -> Result<Self> {
        if lower.len() != upper.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: lower.len(),
                got: upper.len(),
                context: "ControlBounds::new".into(),
            });
        }
        for i in 0..lower.len()
        {
            if lower[i] > upper[i]
            {
                return Err(VariationalError::InfeasibleControlProblem {
                    details: format!("control bound {i}: lower {} > upper {}", lower[i], upper[i]),
                });
            }
        }
        Ok(Self { lower, upper })
    }
}

/// Definition of a discretized continuous-time optimal-control problem.
#[derive(Debug, Clone)]
pub struct OptimalControlProblem<D, RC, TC> {
    /// Number of state components.
    pub state_dim: usize,
    /// Number of control components.
    pub control_dim: usize,
    /// State dynamics callback `(time, state, control, derivative_out)`.
    pub dynamics: D,
    /// Running-cost callback evaluated along the trajectory.
    pub running_cost: RC,
    /// Terminal-cost callback evaluated at the final state and time.
    pub terminal_cost: TC,
    /// State at the beginning of the horizon.
    pub initial_state: Vec<f32>,
    /// Inclusive time horizon represented as `(start, end)`.
    pub horizon: (f32, f32),
    /// Optional component-wise control bounds.
    pub control_bounds: Option<ControlBounds>,
    /// Optional component-wise state bounds.
    pub state_bounds: Option<ControlBounds>,
    /// Number of time-grid points used by discretizing solvers.
    pub num_time_steps: usize,
}

impl<D, RC, TC> OptimalControlProblem<D, RC, TC>
where
    D: Fn(f32, &[f32], &[f32], &mut [f32]),
    RC: Fn(f32, &[f32], &[f32]) -> f32,
    TC: Fn(&[f32], f32) -> f32,
{
    /// Creates an optimal-control problem and validates the state dimension, initial state, and horizon.
    pub fn new(
        state_dim: usize,
        control_dim: usize,
        dynamics: D,
        running_cost: RC,
        terminal_cost: TC,
        initial_state: Vec<f32>,
        horizon: (f32, f32),
    ) -> Result<Self> {
        if state_dim == 0
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                context: "OptimalControlProblem::new".into(),
            });
        }
        if initial_state.len() != state_dim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: state_dim,
                got: initial_state.len(),
                context: "OptimalControlProblem::new".into(),
            });
        }
        if horizon.0 >= horizon.1
        {
            return Err(VariationalError::InvalidInterval {
                start: horizon.0,
                end: horizon.1,
            });
        }
        Ok(Self {
            state_dim,
            control_dim,
            dynamics,
            running_cost,
            terminal_cost,
            initial_state,
            horizon,
            control_bounds: None,
            state_bounds: None,
            num_time_steps: 100,
        })
    }

    /// Attaches component-wise bounds for the control vector.
    pub fn with_control_bounds(mut self, bounds: ControlBounds) -> Self {
        self.control_bounds = Some(bounds);
        self
    }

    /// Attaches component-wise bounds for the state vector.
    pub fn with_state_bounds(mut self, bounds: ControlBounds) -> Self {
        self.state_bounds = Some(bounds);
        self
    }

    /// Sets the number of points in the solver time grid.
    pub fn with_time_steps(mut self, n: usize) -> Self {
        self.num_time_steps = n;
        self
    }
}

/// Trajectory, objective, feasibility, and convergence information returned by a control solver.
#[derive(Debug, Clone)]
pub struct ControlSolution {
    /// Time grid corresponding to the trajectory samples.
    pub times: Vec<f32>,
    /// State vector at each time-grid point.
    pub states: Vec<Vec<f32>>,
    /// Control vector at each stored control sample.
    pub controls: Vec<Vec<f32>>,
    /// Solver-reported objective value for the returned trajectory.
    pub objective: f32,
    /// Solver-reported residual associated with feasibility.
    pub feasibility_residual: f32,
    /// Whether the solver reports that its convergence criterion was met.
    pub converged: bool,
    /// Number of solver iterations reported for the result.
    pub iterations: usize,
}

impl ControlSolution {
    /// Creates a trajectory container with objective and residual unset and convergence false.
    pub fn new(times: Vec<f32>, states: Vec<Vec<f32>>, controls: Vec<Vec<f32>>) -> Self {
        Self {
            times,
            states,
            controls,
            objective: f32::NAN,
            feasibility_residual: f32::NAN,
            converged: false,
            iterations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_bounds_validation() {
        assert!(ControlBounds::new(vec![0.0], vec![1.0]).is_ok());
        assert!(ControlBounds::new(vec![1.0], vec![0.0]).is_err());
    }
}
