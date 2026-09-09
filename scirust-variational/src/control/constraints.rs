/// Classification of a path constraint in an optimal-control problem.
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Equality constraint whose residual should evaluate to zero.
    Equality,
    /// Inequality constraint considered feasible when its function is non-negative.
    Inequality,
    /// Box constraint handled through explicit lower and upper bounds.
    BoxBound,
}

/// Named path constraint evaluated from state, control, and time.
#[derive(Debug, Clone)]
pub struct PathConstraint {
    /// Human-readable constraint identifier.
    pub name: String,
    /// Interpretation applied to the raw constraint function value.
    pub constraint_type: ConstraintType,
    /// Constraint function evaluated as `(state, control, time)`.
    pub function: fn(&[f32], &[f32], f32) -> f32,
    /// Caller-provided tolerance associated with the constraint.
    pub tolerance: f32,
}

impl PathConstraint {
    /// Creates a named path constraint from its type, evaluation function, and tolerance.
    pub fn new(
        name: &str,
        constraint_type: ConstraintType,
        function: fn(&[f32], &[f32], f32) -> f32,
        tolerance: f32,
    ) -> Self {
        Self {
            name: name.to_string(),
            constraint_type,
            function,
            tolerance,
        }
    }

    /// Returns the non-negative violation magnitude for the supplied trajectory point.
    ///
    /// Equality constraints use the absolute residual, inequality constraints violate
    /// only when the function is negative, and box bounds are handled separately.
    pub fn violation(&self, state: &[f32], control: &[f32], time: f32) -> f32 {
        let val = (self.function)(state, control, time);
        match self.constraint_type
        {
            ConstraintType::Equality => val.abs(),
            ConstraintType::Inequality => (-val).max(0.0),
            ConstraintType::BoxBound => 0.0,
        }
    }
}

/// Clamps each component of `x` in place to its corresponding inclusive box bounds.
pub fn project_to_box_bounds(x: &mut [f32], lower: &[f32], upper: &[f32]) {
    for i in 0..x.len()
    {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_to_box_bounds() {
        let mut x = vec![-0.5, 1.5, 0.5];
        project_to_box_bounds(&mut x, &[0.0; 3], &[1.0; 3]);
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
        assert!((x[2] - 0.5).abs() < 1e-6);
    }
}
