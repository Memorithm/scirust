use crate::error::{Result, VariationalError};

/// Generalized coordinates, velocities, and time for a mechanical state.
#[derive(Debug, Clone)]
pub struct GeneralizedState {
    /// Generalized coordinate values.
    pub positions: Vec<f32>,
    /// Generalized velocity values in the same coordinate order.
    pub velocities: Vec<f32>,
    /// Time associated with the state.
    pub time: f32,
}

impl GeneralizedState {
    /// Creates a state after validating equal position and velocity dimensions.
    pub fn new(positions: Vec<f32>, velocities: Vec<f32>, time: f32) -> Result<Self> {
        if positions.len() != velocities.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: positions.len(),
                got: velocities.len(),
                context: "GeneralizedState::new".into(),
            });
        }
        Ok(Self {
            positions,
            velocities,
            time,
        })
    }

    /// Returns the number of generalized coordinates.
    pub fn ndim(&self) -> usize {
        self.positions.len()
    }

    /// Concatenates positions and velocities into `[q, q_dot]` phase-space order.
    pub fn to_phase_space(&self) -> Vec<f32> {
        let mut phase = Vec::with_capacity(2 * self.ndim());
        phase.extend_from_slice(&self.positions);
        phase.extend_from_slice(&self.velocities);
        phase
    }

    /// Returns the phase-space dimension `2 * ndim`.
    pub fn phase_dim(&self) -> usize {
        2 * self.ndim()
    }
}

/// Dense square generalized mass matrix.
#[derive(Debug, Clone)]
pub struct GeneralizedMassMatrix {
    /// Matrix entries in row-major nested-vector form.
    pub matrix: Vec<Vec<f32>>,
    /// Number of generalized coordinates represented by the matrix.
    pub ndim: usize,
}

impl GeneralizedMassMatrix {
    /// Creates a non-empty square mass matrix.
    pub fn new(matrix: Vec<Vec<f32>>) -> Result<Self> {
        let n = matrix.len();
        if n == 0
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                context: "GeneralizedMassMatrix::new".into(),
            });
        }
        for row in &matrix
        {
            if row.len() != n
            {
                return Err(VariationalError::DimensionMismatch {
                    expected: n,
                    got: row.len(),
                    context: "GeneralizedMassMatrix::new (non-square)".into(),
                });
            }
        }
        Ok(Self { matrix, ndim: n })
    }

    /// Checks pairwise matrix symmetry using the supplied absolute tolerance.
    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        for i in 0..self.ndim
        {
            for j in i + 1..self.ndim
            {
                if (self.matrix[i][j] - self.matrix[j][i]).abs() > tolerance
                {
                    return false;
                }
            }
        }
        true
    }

    /// Runs the current leading-principal-minor test for positive definiteness.
    pub fn is_positive_definite(&self) -> bool {
        let n = self.ndim;
        for i in 0..n
        {
            let mut det = self.matrix[0][0];
            if i > 0
            {
                let mut sub = vec![vec![0.0; i + 1]; i + 1];
                for r in 0..=i
                {
                    for c in 0..=i
                    {
                        sub[r][c] = self.matrix[r][c];
                    }
                }
                det = sub[0][0];
                for k in 1..=i
                {
                    let factor = sub[k][0] / sub[0][0];
                    for j in 1..=i
                    {
                        sub[k][j] -= factor * sub[0][j];
                    }
                }
                for k in 1..=i
                {
                    det *= sub[k][k];
                }
            }
            if det <= 0.0
            {
                return false;
            }
        }
        true
    }
}
