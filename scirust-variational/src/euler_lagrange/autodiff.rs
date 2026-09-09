use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};

/// Finite-difference approximation of the Lagrangian Hessian with respect to velocity.
#[derive(Debug, Clone)]
pub struct VelocityHessian {
    /// Dense Hessian matrix in generalized-coordinate order.
    pub matrix: Vec<Vec<f32>>,
    /// Estimated conditioning ratio used by the solver's singularity heuristic.
    pub condition_number: f32,
    /// Whether the Hessian is classified as singular by the configured tolerance.
    pub is_singular: bool,
}

/// Result of solving the Euler-Lagrange system for generalized acceleration.
#[derive(Debug, Clone)]
pub struct AccelerationResult {
    /// Generalized acceleration vector.
    pub acceleration: Vec<f32>,
    /// Velocity Hessian used for the linear solve.
    pub hessian: VelocityHessian,
    /// Right-hand side of the solved Euler-Lagrange linear system.
    pub rhs: Vec<f32>,
    /// Euclidean norm of the linear-system residual for the returned acceleration.
    pub residual_norm: f32,
}

/// First derivatives of a Lagrangian with respect to coordinates and velocities.
#[derive(Debug, Clone)]
pub struct ELGradients {
    /// Partial derivatives of the Lagrangian with respect to generalized coordinates.
    #[allow(non_snake_case)]
    pub dL_dq: Vec<f32>,
    /// Partial derivatives of the Lagrangian with respect to generalized velocities.
    #[allow(non_snake_case)]
    pub dL_ddq: Vec<f32>,
}

/// Interface for a Lagrangian that can be evaluated on SciRust's N-dimensional autodiff tape.
pub trait DifferentiableLagrangian {
    /// Evaluates the Lagrangian from coordinate, velocity, and optional time variables.
    fn compute_lagrangian<'t>(
        &self,
        tape: &'t NdTape,
        q: NdVar<'t>,
        dq: NdVar<'t>,
        t: Option<NdVar<'t>>,
    ) -> NdVar<'t>;
}

/// Autodiff-backed numerical Euler-Lagrange evaluator for a fixed coordinate dimension.
pub struct AutodiffEulerLagrange {
    /// Number of generalized coordinates expected by the evaluator.
    pub ndim: usize,
    /// Finite-difference step used for second and mixed derivatives.
    pub epsilon: f32,
    /// Threshold used when classifying the velocity Hessian as singular.
    pub singularity_tol: f32,
}

impl AutodiffEulerLagrange {
    /// Creates an evaluator with default finite-difference and singularity tolerances.
    pub fn new(ndim: usize) -> Self {
        Self {
            ndim,
            epsilon: 1e-4,
            singularity_tol: 1e-8,
        }
    }

    /// Replaces the finite-difference step and Hessian singularity tolerance.
    pub fn with_tolerances(mut self, epsilon: f32, singularity_tol: f32) -> Self {
        self.epsilon = epsilon;
        self.singularity_tol = singularity_tol;
        self
    }

    /// Computes first derivatives of the Lagrangian at `(q, dq, t)` using reverse-mode autodiff.
    pub fn compute_gradients<F>(
        &self,
        lagrangian: &F,
        q: &[f32],
        dq: &[f32],
        t: Option<f32>,
    ) -> Result<ELGradients>
    where
        F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
    {
        if q.len() != self.ndim || dq.len() != self.ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.ndim,
                got: if q.len() != self.ndim
                {
                    q.len()
                }
                else
                {
                    dq.len()
                },
                context: "AutodiffEulerLagrange::compute_gradients".into(),
            });
        }

        let tape = NdTape::new();
        let q_vars: Vec<NdVar<'_>> = (0..self.ndim)
            .map(|i| tape.input(TensorND::new(vec![q[i]], vec![1, 1])))
            .collect();
        let dq_vars: Vec<NdVar<'_>> = (0..self.ndim)
            .map(|i| tape.input(TensorND::new(vec![dq[i]], vec![1, 1])))
            .collect();
        let t_var = t.map(|tv| tape.input(TensorND::new(vec![tv], vec![1, 1])));

        let L = lagrangian(&tape, &q_vars, &dq_vars, t_var);
        let grads = tape.backward(L);

        let dL_dq: Vec<f32> = grads[..self.ndim].iter().map(|g| g.data[0]).collect();
        let dL_ddq: Vec<f32> = grads[self.ndim..2 * self.ndim]
            .iter()
            .map(|g| g.data[0])
            .collect();

        for &v in dL_dq.iter().chain(dL_ddq.iter())
        {
            if !v.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "EL gradient",
                    value: v,
                });
            }
        }

        Ok(ELGradients { dL_dq, dL_ddq })
    }

    /// Approximates the velocity Hessian by centered finite differences of autodiff gradients.
    pub fn compute_velocity_hessian<F>(
        &self,
        lagrangian: &F,
        q: &[f32],
        dq: &[f32],
        t: Option<f32>,
    ) -> Result<VelocityHessian>
    where
        F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
    {
        let n = self.ndim;
        let mut hessian = vec![vec![0.0; n]; n];
        let eps = self.epsilon;

        let _base_grad = self.compute_gradients(lagrangian, q, dq, t)?;

        for j in 0..n
        {
            let mut dq_plus = dq.to_vec();
            dq_plus[j] += eps;
            let grad_plus = self.compute_gradients(lagrangian, q, &dq_plus, t)?;

            let mut dq_minus = dq.to_vec();
            dq_minus[j] -= eps;
            let grad_minus = self.compute_gradients(lagrangian, q, &dq_minus, t)?;

            for i in 0..n
            {
                hessian[i][j] = (grad_plus.dL_ddq[i] - grad_minus.dL_ddq[i]) / (2.0 * eps);
            }
        }

        self.check_hessian_symmetry(&hessian)?;

        let cond = self.estimate_condition_number(&hessian);
        let is_singular = !cond.is_finite() || cond > 1.0 / self.singularity_tol;

        Ok(VelocityHessian {
            matrix: hessian,
            condition_number: cond,
            is_singular,
        })
    }

    /// Solves the Euler-Lagrange equations for acceleration at `(q, dq, t)`.
    ///
    /// The method combines autodiff first derivatives, finite-difference second and
    /// mixed derivatives, a velocity-Hessian singularity check, and a dense linear solve.
    pub fn compute_acceleration<F>(
        &self,
        lagrangian: &F,
        q: &[f32],
        dq: &[f32],
        t: Option<f32>,
    ) -> Result<AccelerationResult>
    where
        F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
    {
        let n = self.ndim;
        let hessian = self.compute_velocity_hessian(lagrangian, q, dq, t)?;

        if hessian.is_singular
        {
            return Err(VariationalError::SingularVelocityHessian {
                condition_number: hessian.condition_number,
                tolerance: self.singularity_tol,
            });
        }

        let grad = self.compute_gradients(lagrangian, q, dq, t)?;

        let eps = self.epsilon;
        let mut d2L_dq_ddq = vec![vec![0.0; n]; n];

        for j in 0..n
        {
            let mut dq_plus = dq.to_vec();
            dq_plus[j] += eps;
            let grad_plus = self.compute_gradients(lagrangian, q, &dq_plus, t)?;

            for i in 0..n
            {
                d2L_dq_ddq[i][j] = (grad_plus.dL_dq[i] - grad.dL_dq[i]) / eps;
            }
        }

        let mut mixed_dtdq = vec![0.0; n];
        if let Some(t_val) = t
        {
            let dt = eps;
            let grad_t_plus = self.compute_gradients(lagrangian, q, dq, Some(t_val + dt))?;
            for i in 0..n
            {
                mixed_dtdq[i] = (grad_t_plus.dL_ddq[i] - grad.dL_ddq[i]) / dt;
            }
        }

        let mut rhs = vec![0.0; n];
        for i in 0..n
        {
            rhs[i] = grad.dL_dq[i];
            for j in 0..n
            {
                rhs[i] -= d2L_dq_ddq[i][j] * dq[j];
            }
            rhs[i] -= mixed_dtdq[i];
        }

        let acceleration = self.solve_linear_system(&hessian.matrix, &rhs)?;

        let mut residual_norm = 0.0;
        for i in 0..n
        {
            let mut res = 0.0;
            for j in 0..n
            {
                res += hessian.matrix[i][j] * acceleration[j];
            }
            res -= rhs[i];
            residual_norm += res * res;
        }
        residual_norm = residual_norm.sqrt();

        Ok(AccelerationResult {
            acceleration,
            hessian,
            rhs,
            residual_norm,
        })
    }

    fn check_hessian_symmetry(&self, h: &[Vec<f32>]) -> Result<()> {
        let n = h.len();
        for i in 0..n
        {
            for j in i + 1..n
            {
                let diff = (h[i][j] - h[j][i]).abs();
                let max_val = h[i][j].abs().max(h[j][i].abs()).max(1.0);
                if diff > 1e-3 * max_val
                {
                    return Err(VariationalError::IllConditionedSystem {
                        condition_number: diff / max_val,
                        tolerance: 1e-3,
                    });
                }
            }
        }
        Ok(())
    }

    fn estimate_condition_number(&self, matrix: &[Vec<f32>]) -> f32 {
        let n = matrix.len();
        let mut max_row_sum: f32 = 0.0;
        let mut min_row_sum = f32::INFINITY;
        for i in 0..n
        {
            let row_sum: f32 = matrix[i].iter().map(|v| v.abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
            min_row_sum = min_row_sum.min(row_sum);
        }
        if min_row_sum < 1e-16
        {
            return f32::INFINITY;
        }
        max_row_sum / min_row_sum
    }

    fn solve_linear_system(&self, a: &[Vec<f32>], b: &[f32]) -> Result<Vec<f32>> {
        let n = a.len();
        let mut augmented: Vec<Vec<f32>> = vec![vec![0.0; n + 1]; n];
        for i in 0..n
        {
            for j in 0..n
            {
                augmented[i][j] = a[i][j];
            }
            augmented[i][n] = b[i];
        }

        for col in 0..n
        {
            let mut max_row = col;
            let mut max_val = augmented[col][col].abs();
            for row in col + 1..n
            {
                let val = augmented[row][col].abs();
                if val > max_val
                {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-16
            {
                return Err(VariationalError::SingularVelocityHessian {
                    condition_number: f32::INFINITY,
                    tolerance: self.singularity_tol,
                });
            }

            if max_row != col
            {
                augmented.swap(col, max_row);
            }

            for row in col + 1..n
            {
                let factor = augmented[row][col] / augmented[col][col];
                for k in col..=n
                {
                    augmented[row][k] -= factor * augmented[col][k];
                }
            }
        }

        let mut x = vec![0.0; n];
        for i in (0..n).rev()
        {
            let mut sum = augmented[i][n];
            for j in i + 1..n
            {
                sum -= augmented[i][j] * x[j];
            }
            if augmented[i][i].abs() < 1e-16
            {
                return Err(VariationalError::LinearSolveFailure {
                    details: format!("zero pivot at row {i}"),
                });
            }
            x[i] = sum / augmented[i][i];
        }

        for &v in x.iter()
        {
            if !v.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "acceleration",
                    value: v,
                });
            }
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn harmonic_lagrangian<'a>(
        tape: &'a NdTape,
        q: &'a [NdVar<'a>],
        dq: &'a [NdVar<'a>],
        _t: Option<NdVar<'a>>,
    ) -> NdVar<'a> {
        let _one = tape.input(TensorND::new(vec![1.0], vec![1, 1]));
        let half = tape.input(TensorND::new(vec![0.5], vec![1, 1]));
        let ke = dq[0].mul(dq[0]).mul(half);
        let pe = q[0].mul(q[0]).mul(half);
        ke.sub(pe)
    }

    #[test]
    fn test_harmonic_acceleration() {
        let el = AutodiffEulerLagrange::new(1);
        let q = vec![1.0];
        let dq = vec![0.0];
        let result = el
            .compute_acceleration(&harmonic_lagrangian, &q, &dq, None)
            .unwrap();
        assert_relative_eq!(result.acceleration[0], -1.0, epsilon = 0.1);
    }

    fn singular_lagrangian<'a>(
        tape: &'a NdTape,
        q: &'a [NdVar<'a>],
        _dq: &'a [NdVar<'a>],
        _t: Option<NdVar<'a>>,
    ) -> NdVar<'a> {
        let one = tape.input(TensorND::new(vec![1.0], vec![1, 1]));
        q[0].mul(q[0]).mul(one)
    }

    #[test]
    fn test_singular_lagrangian_error() {
        let el = AutodiffEulerLagrange::new(1);
        let q = vec![1.0];
        let dq = vec![0.0];
        let result = el.compute_acceleration(&singular_lagrangian, &q, &dq, None);
        assert!(result.is_err());
    }
}
