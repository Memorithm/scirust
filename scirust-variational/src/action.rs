use crate::error::{Result, VariationalError};

/// Closed time interval used by variational action problems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeInterval {
    /// Interval start time.
    pub start: f32,
    /// Interval end time.
    pub end: f32,
}

impl TimeInterval {
    /// Creates a time interval with `start < end`.
    pub fn new(start: f32, end: f32) -> Result<Self> {
        if start >= end
        {
            return Err(VariationalError::InvalidInterval { start, end });
        }
        Ok(Self { start, end })
    }

    /// Returns the interval length `end - start`.
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Returns whether `t` lies inside the closed interval.
    pub fn contains(&self, t: f32) -> bool {
        t >= self.start && t <= self.end
    }
}

/// Boundary condition imposed on one generalized coordinate.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    /// Fixes the coordinate to the stored value.
    Fixed(f32),
    /// Leaves the coordinate unconstrained.
    Free,
    /// Uses a natural boundary condition.
    Natural,
}

impl BoundaryCondition {
    /// Returns whether this condition fixes the coordinate to a value.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// Returns the fixed value, or `None` for non-fixed conditions.
    pub fn fixed_value(&self) -> Option<f32> {
        match self
        {
            Self::Fixed(v) => Some(*v),
            _ => None,
        }
    }
}

/// Named generalized coordinate identified by its index.
#[derive(Debug, Clone)]
pub struct GeneralizedCoordinate {
    /// Coordinate name.
    pub name: String,
    /// Coordinate index in the problem state.
    pub index: usize,
}

impl GeneralizedCoordinate {
    /// Creates a generalized coordinate from a name and index.
    pub fn new(name: &str, index: usize) -> Self {
        Self {
            name: name.to_string(),
            index,
        }
    }
}

/// Generalized velocity associated with a generalized coordinate.
#[derive(Debug, Clone)]
pub struct GeneralizedVelocity {
    /// Velocity name, derived by prefixing the coordinate name with `d`.
    pub name: String,
    /// Index of the associated generalized coordinate.
    pub coord_index: usize,
}

impl GeneralizedVelocity {
    /// Creates the velocity associated with `coordinate`.
    pub fn new(coordinate: &GeneralizedCoordinate) -> Self {
        Self {
            name: format!("d{}", coordinate.name),
            coord_index: coordinate.index,
        }
    }
}

/// Boundary-value specification for a variational action problem.
#[derive(Debug, Clone)]
pub struct ActionProblem {
    /// Number of generalized coordinates in the problem.
    pub num_coordinates: usize,
    /// Time interval over which the action is defined.
    pub interval: TimeInterval,
    /// Boundary conditions at the start of the interval.
    pub initial_conditions: Vec<BoundaryCondition>,
    /// Boundary conditions at the end of the interval.
    pub terminal_conditions: Vec<BoundaryCondition>,
    /// Whether the problem contains explicit time dependence.
    pub has_explicit_time: bool,
}

impl ActionProblem {
    /// Creates an action problem with free endpoint conditions.
    pub fn new(num_coordinates: usize, interval: TimeInterval) -> Result<Self> {
        if num_coordinates == 0
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                context: "ActionProblem::new".into(),
            });
        }
        Ok(Self {
            num_coordinates,
            interval,
            initial_conditions: vec![BoundaryCondition::Free; num_coordinates],
            terminal_conditions: vec![BoundaryCondition::Free; num_coordinates],
            has_explicit_time: false,
        })
    }

    /// Sets the initial boundary condition for one coordinate.
    pub fn with_initial_condition(
        mut self,
        index: usize,
        condition: BoundaryCondition,
    ) -> Result<Self> {
        if index >= self.num_coordinates
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.num_coordinates,
                got: index + 1,
                context: "ActionProblem::with_initial_condition".into(),
            });
        }
        self.initial_conditions[index] = condition;
        Ok(self)
    }

    /// Sets the terminal boundary condition for one coordinate.
    pub fn with_terminal_condition(
        mut self,
        index: usize,
        condition: BoundaryCondition,
    ) -> Result<Self> {
        if index >= self.num_coordinates
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.num_coordinates,
                got: index + 1,
                context: "ActionProblem::with_terminal_condition".into(),
            });
        }
        self.terminal_conditions[index] = condition;
        Ok(self)
    }

    /// Marks the action problem as explicitly time-dependent.
    pub fn with_explicit_time(mut self) -> Self {
        self.has_explicit_time = true;
        self
    }
}
