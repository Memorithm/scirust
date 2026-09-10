use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct GeneralizedState {
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub time: f32,
}

impl GeneralizedState {
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

    pub fn ndim(&self) -> usize {
        self.positions.len()
    }

    pub fn to_phase_space(&self) -> Vec<f32> {
        let mut phase = Vec::with_capacity(2 * self.ndim());
        phase.extend_from_slice(&self.positions);
        phase.extend_from_slice(&self.velocities);
        phase
    }

    pub fn phase_dim(&self) -> usize {
        2 * self.ndim()
    }
}

#[derive(Debug, Clone)]
pub struct GeneralizedMassMatrix {
    pub matrix: Vec<Vec<f32>>,
    pub ndim: usize,
}

impl GeneralizedMassMatrix {
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

    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        if !tolerance.is_finite() || tolerance < 0.0 || self.matrix.len() != self.ndim
        {
            return false;
        }

        for row in &self.matrix
        {
            if row.len() != self.ndim
            {
                return false;
            }
        }

        for i in 0..self.ndim
        {
            if !self.matrix[i][i].is_finite()
            {
                return false;
            }
            for j in i + 1..self.ndim
            {
                let a = self.matrix[i][j];
                let b = self.matrix[j][i];
                if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_positive_definite(&self) -> bool {
        let n = self.ndim;
        if n == 0 || self.matrix.len() != n || self.matrix.iter().any(|row| row.len() != n)
        {
            return false;
        }

        // Positive definiteness is defined for symmetric real matrices. Use a
        // scale-aware tolerance so harmless f32 roundoff does not reject an
        // otherwise symmetric matrix, while materially asymmetric matrices do.
        let symmetry_factor = 32.0 * f32::EPSILON;
        for i in 0..n
        {
            let diagonal = self.matrix[i][i];
            if !diagonal.is_finite()
            {
                return false;
            }
            for j in i + 1..n
            {
                let a = self.matrix[i][j];
                let b = self.matrix[j][i];
                if !a.is_finite() || !b.is_finite()
                {
                    return false;
                }
                let scale = a.abs().max(b.abs()).max(1.0);
                if (a - b).abs() > symmetry_factor * scale
                {
                    return false;
                }
            }
        }

        // Cholesky factorization provides a direct test for symmetric positive
        // definiteness. Accumulate in f64 to reduce cancellation from the f32
        // source matrix without changing the public representation.
        let mut lower = vec![vec![0.0_f64; n]; n];
        for i in 0..n
        {
            for j in 0..=i
            {
                let mut value = self.matrix[i][j] as f64;
                for k in 0..j
                {
                    value -= lower[i][k] * lower[j][k];
                }

                if i == j
                {
                    if !value.is_finite() || value <= 0.0
                    {
                        return false;
                    }
                    lower[i][j] = value.sqrt();
                }
                else
                {
                    let pivot = lower[j][j];
                    if !pivot.is_finite() || pivot <= 0.0
                    {
                        return false;
                    }
                    lower[i][j] = value / pivot;
                    if !lower[i][j].is_finite()
                    {
                        return false;
                    }
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_definite_accepts_spd_matrix() {
        let matrix = GeneralizedMassMatrix::new(vec![
            vec![4.0, 1.0, 1.0],
            vec![1.0, 3.0, 0.5],
            vec![1.0, 0.5, 2.0],
        ])
        .unwrap();

        assert!(matrix.is_positive_definite());
    }

    #[test]
    fn positive_definite_rejects_indefinite_matrix() {
        let matrix = GeneralizedMassMatrix::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap();

        assert!(!matrix.is_positive_definite());
    }

    #[test]
    fn positive_definite_rejects_positive_semidefinite_matrix() {
        let matrix = GeneralizedMassMatrix::new(vec![vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap();

        assert!(!matrix.is_positive_definite());
    }

    #[test]
    fn positive_definite_rejects_asymmetric_matrix() {
        // The previous leading-minor implementation incorrectly returned true
        // for this matrix because it never validated symmetry and only
        // eliminated the first column of each principal submatrix.
        let matrix = GeneralizedMassMatrix::new(vec![vec![2.0, 10.0], vec![0.0, 2.0]]).unwrap();

        assert!(!matrix.is_positive_definite());
    }

    #[test]
    fn positive_definite_rejects_non_finite_entries() {
        let matrix =
            GeneralizedMassMatrix::new(vec![vec![1.0, f32::NAN], vec![f32::NAN, 1.0]]).unwrap();

        assert!(!matrix.is_positive_definite());
        assert!(!matrix.is_symmetric(1e-6));
    }

    #[test]
    fn mass_matrix_checks_reject_mutated_shape_metadata() {
        let mut matrix = GeneralizedMassMatrix::new(vec![vec![1.0]]).unwrap();
        matrix.ndim = 2;

        assert!(!matrix.is_positive_definite());
        assert!(!matrix.is_symmetric(1e-6));
    }

    #[test]
    fn symmetric_rejects_invalid_tolerance() {
        let matrix = GeneralizedMassMatrix::new(vec![vec![1.0]]).unwrap();

        assert!(!matrix.is_symmetric(-1.0));
        assert!(!matrix.is_symmetric(f32::NAN));
    }
}
