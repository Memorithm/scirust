use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::nn::nd_layers::NdLinear;
use scirust_core::nn::nd_optim::NdParam;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};
use crate::learning::trainer::HasParameters;
use crate::util::nd_tanh;

pub struct HamiltonianNetwork {
    pub layers: Vec<NdLinear>,
    pub ndim: usize,
    pub hidden_dim: usize,
}

impl HamiltonianNetwork {
    pub fn new(ndim: usize, hidden_dim: usize, rng: &mut PcgEngine) -> Self {
        let in_dim = 2 * ndim + 1;
        let layers = vec![
            NdLinear::new(in_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, 1, rng),
        ];
        Self {
            layers,
            ndim,
            hidden_dim,
        }
    }

    pub fn forward<'t>(
        &mut self,
        tape: &'t NdTape,
        q: &[NdVar],
        p: &[NdVar],
        t: Option<NdVar>,
    ) -> NdVar<'t> {
        let batch_ndim = q[0].shape();
        let batch_size = if batch_ndim.len() >= 2
        {
            batch_ndim[0]
        }
        else
        {
            1
        };

        let q_data = tape.value(q[0]);
        let p_data = tape.value(p[0]);
        let mut cat_data = Vec::new();
        for i in 0..batch_size
        {
            let base = i * self.ndim;
            for j in 0..self.ndim
            {
                cat_data.push(q_data.data[base + j]);
            }
            for j in 0..self.ndim
            {
                cat_data.push(p_data.data[base + j]);
            }
            match t
            {
                Some(tv) =>
                {
                    let td = tape.value(tv);
                    cat_data.push(td.data[i.min(td.data.len().saturating_sub(1))]);
                },
                None => cat_data.push(0.0),
            }
        }

        let inputs = tape.input(TensorND::new(cat_data, vec![batch_size, 2 * self.ndim + 1]));

        let h = nd_tanh(tape, self.layers[0].forward(tape, inputs));
        let h = nd_tanh(tape, self.layers[1].forward(tape, h));
        self.layers[2].forward(tape, h)
    }

    pub fn vector_field(&mut self, q: &[f32], p: &[f32], t: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        self.validate_state_inputs(q, p, t, "HamiltonianNetwork::vector_field")?;

        let n = self.ndim;
        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q.to_vec(), vec![1, n]));
        let pv = tape.input(TensorND::new(p.to_vec(), vec![1, n]));
        let tv = tape.input(TensorND::new(vec![t], vec![1, 1]));
        let q_arr = [qv];
        let p_arr = [pv];
        let hamiltonian = self.forward(&tape, &q_arr, &p_arr, Some(tv));
        let grads = tape.backward(hamiltonian);
        let d_h_dq: Vec<f32> = grads[0].data[..n].to_vec();
        let d_h_dp: Vec<f32> = grads[1].data[..n].to_vec();

        for &value in d_h_dq.iter().chain(d_h_dp.iter())
        {
            if !value.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "HNN gradient",
                    value,
                });
            }
        }

        let dq = d_h_dp;
        let dp: Vec<f32> = d_h_dq.iter().map(|&value| -value).collect();
        Ok((dq, dp))
    }

    pub fn eval_hamiltonian(&mut self, q: &[f32], p: &[f32], t: f32) -> f32 {
        self.try_eval_hamiltonian(q, p, t)
            .unwrap_or_else(|err| panic!("HamiltonianNetwork::eval_hamiltonian: {err}"))
    }

    /// Evaluate the learned Hamiltonian after validating phase-space inputs.
    pub fn try_eval_hamiltonian(&mut self, q: &[f32], p: &[f32], t: f32) -> Result<f32> {
        self.validate_state_inputs(q, p, t, "HamiltonianNetwork::try_eval_hamiltonian")?;

        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q.to_vec(), vec![1, self.ndim]));
        let pv = tape.input(TensorND::new(p.to_vec(), vec![1, self.ndim]));
        let tv = tape.input(TensorND::new(vec![t], vec![1, 1]));
        let q_arr = [qv];
        let p_arr = [pv];
        let hamiltonian = self.forward(&tape, &q_arr, &p_arr, Some(tv));
        let value = tape.value(hamiltonian).data[0];
        if !value.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "HNN Hamiltonian",
                value,
            });
        }
        Ok(value)
    }

    fn validate_state_inputs(&self, q: &[f32], p: &[f32], t: f32, context: &str) -> Result<()> {
        if q.len() != self.ndim || p.len() != self.ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.ndim,
                got: if q.len() != self.ndim
                {
                    q.len()
                }
                else
                {
                    p.len()
                },
                context: context.into(),
            });
        }
        if !t.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "HNN time",
                value: t,
            });
        }
        for &value in q.iter().chain(p.iter())
        {
            if !value.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "HNN state",
                    value,
                });
            }
        }
        Ok(())
    }
}

impl HasParameters for HamiltonianNetwork {
    fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        for layer in &mut self.layers
        {
            params.extend(layer.parameters());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnn_creation() {
        let mut rng = PcgEngine::new(42);
        let hnn = HamiltonianNetwork::new(2, 32, &mut rng);
        assert_eq!(hnn.ndim, 2);
        assert_eq!(hnn.layers.len(), 3);
    }

    #[test]
    fn vector_field_rejects_phase_dimension_mismatch() {
        let mut rng = PcgEngine::new(42);
        let mut hnn = HamiltonianNetwork::new(2, 8, &mut rng);

        let err = hnn.vector_field(&[0.0, 1.0], &[0.0], 0.0).unwrap_err();
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
    fn vector_field_rejects_non_finite_state() {
        let mut rng = PcgEngine::new(42);
        let mut hnn = HamiltonianNetwork::new(1, 8, &mut rng);

        let err = hnn.vector_field(&[f32::NAN], &[0.0], 0.0).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "HNN state",
                ..
            }
        ));
    }

    #[test]
    fn checked_hamiltonian_rejects_non_finite_time() {
        let mut rng = PcgEngine::new(42);
        let mut hnn = HamiltonianNetwork::new(1, 8, &mut rng);

        let err = hnn
            .try_eval_hamiltonian(&[0.0], &[0.0], f32::INFINITY)
            .unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "HNN time",
                ..
            }
        ));
    }
}
