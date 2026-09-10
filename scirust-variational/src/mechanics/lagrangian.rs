use scirust_core::autodiff::nd::{NdTape, NdVar};

use crate::error::{Result, VariationalError};
use crate::euler_lagrange::autodiff::AutodiffEulerLagrange;

#[derive(Debug, Clone)]
pub struct LagrangianDynamicsConfig {
    pub ndim: usize,
    pub singularity_tol: f32,
    pub epsilon: f32,
}

impl Default for LagrangianDynamicsConfig {
    fn default() -> Self {
        Self {
            ndim: 1,
            singularity_tol: 1e-8,
            epsilon: 1e-4,
        }
    }
}

pub struct LagrangianDynamics<F> {
    pub lagrangian: F,
    pub config: LagrangianDynamicsConfig,
}

impl<F> LagrangianDynamics<F>
where
    F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
{
    pub fn new(lagrangian: F, config: LagrangianDynamicsConfig) -> Self {
        Self { lagrangian, config }
    }

    pub fn acceleration(&self, q: &[f32], dq: &[f32], t: f32) -> Result<Vec<f32>> {
        if q.len() != self.config.ndim || dq.len() != self.config.ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.config.ndim,
                got: if q.len() != self.config.ndim
                {
                    q.len()
                }
                else
                {
                    dq.len()
                },
                context: "LagrangianDynamics::acceleration".into(),
            });
        }

        let el = AutodiffEulerLagrange::new(self.config.ndim)
            .with_tolerances(self.config.epsilon, self.config.singularity_tol);

        let result = el.compute_acceleration(&self.lagrangian, q, dq, Some(t))?;
        Ok(result.acceleration)
    }

    pub fn dynamics(&self, t: f32, state: &[f32], deriv: &mut [f32]) -> Result<()> {
        let n = self.config.ndim;
        if state.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: state.len(),
                context: "LagrangianDynamics::dynamics state".into(),
            });
        }
        if deriv.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: deriv.len(),
                context: "LagrangianDynamics::dynamics deriv".into(),
            });
        }

        let q = &state[..n];
        let dq = &state[n..];
        let acc = self.acceleration(q, dq, t)?;

        deriv[..n].copy_from_slice(dq);
        deriv[n..].copy_from_slice(&acc);
        Ok(())
    }

    pub fn compute_ode_rhs(&self, t: f32, state: &[f32], deriv: &mut [f32]) -> Result<()> {
        let n = self.config.ndim;
        if state.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: state.len(),
                context: "LagrangianDynamics::compute_ode_rhs state".into(),
            });
        }
        if deriv.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: deriv.len(),
                context: "LagrangianDynamics::compute_ode_rhs deriv".into(),
            });
        }
        let q = &state[..n];
        let dq = &state[n..];
        let acc = self.acceleration(q, dq, t)?;
        deriv[..n].copy_from_slice(dq);
        deriv[n..].copy_from_slice(&acc);
        Ok(())
    }
}

impl<F> std::fmt::Debug for LagrangianDynamics<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LagrangianDynamics")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::tensor::tensor_nd::TensorND;

    fn harmonic_lagrangian<'a>(
        tape: &'a NdTape,
        q: &'a [NdVar<'a>],
        dq: &'a [NdVar<'a>],
        _t: Option<NdVar<'a>>,
    ) -> NdVar<'a> {
        let half = tape.input(TensorND::new(vec![0.5], vec![1, 1]));
        dq[0].mul(dq[0]).mul(half).sub(q[0].mul(q[0]).mul(half))
    }

    #[test]
    fn acceleration_reports_velocity_dimension_mismatch() {
        let dynamics = LagrangianDynamics::new(
            harmonic_lagrangian,
            LagrangianDynamicsConfig {
                ndim: 1,
                ..Default::default()
            },
        );

        let err = dynamics.acceleration(&[0.0], &[], 0.0).unwrap_err();
        match err
        {
            VariationalError::DimensionMismatch {
                expected,
                got,
                context,
            } =>
            {
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
                assert_eq!(context, "LagrangianDynamics::acceleration");
            },
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn dynamics_rejects_wrong_derivative_buffer_length() {
        let dynamics = LagrangianDynamics::new(
            harmonic_lagrangian,
            LagrangianDynamicsConfig {
                ndim: 1,
                ..Default::default()
            },
        );
        let mut deriv = vec![0.0; 1];

        let err = dynamics.dynamics(0.0, &[1.0, 0.0], &mut deriv).unwrap_err();
        match err
        {
            VariationalError::DimensionMismatch {
                expected,
                got,
                context,
            } =>
            {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
                assert_eq!(context, "LagrangianDynamics::dynamics deriv");
            },
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn compute_ode_rhs_rejects_wrong_derivative_buffer_length() {
        let dynamics = LagrangianDynamics::new(
            harmonic_lagrangian,
            LagrangianDynamicsConfig {
                ndim: 1,
                ..Default::default()
            },
        );
        let mut deriv = vec![0.0; 3];

        let err = dynamics
            .compute_ode_rhs(0.0, &[1.0, 0.0], &mut deriv)
            .unwrap_err();
        match err
        {
            VariationalError::DimensionMismatch {
                expected,
                got,
                context,
            } =>
            {
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
                assert_eq!(context, "LagrangianDynamics::compute_ode_rhs deriv");
            },
            other => panic!("unexpected error: {other}"),
        }
    }
}
