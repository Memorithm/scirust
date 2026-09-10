use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::nn::nd_layers::NdLinear;
use scirust_core::nn::nd_optim::{NdAdam, NdParam};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};
use crate::learning::trainer::{HasParameters, TrainingConfig, TrainingMetrics};
use crate::pinn::collocation::CollocationPoints;
use crate::pinn::conditions::ConditionConfig;
use crate::util::nd_tanh;

pub struct PinnNet {
    pub layers: Vec<NdLinear>,
    pub ndim: usize,
    pub hidden_dim: usize,
}

impl PinnNet {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut PcgEngine,
    ) -> Self {
        let layers = vec![
            NdLinear::new(input_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, output_dim, rng),
        ];
        Self {
            layers,
            ndim: input_dim,
            hidden_dim,
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let h = nd_tanh(tape, self.layers[0].forward(tape, x));
        let h = nd_tanh(tape, self.layers[1].forward(tape, h));
        self.layers[2].forward(tape, h)
    }
}

impl HasParameters for PinnNet {
    fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        for layer in &mut self.layers
        {
            params.extend(layer.parameters());
        }
        params
    }
}

pub struct ResidualFn {
    pub residual: Box<dyn for<'a> Fn(&'a NdTape, &'a mut PinnNet, NdVar<'a>) -> NdVar<'a>>,
}

pub struct PinnTrainer {
    pub config: TrainingConfig,
}

impl PinnTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn train<F>(
        &self,
        model: &mut PinnNet,
        interior_points: &CollocationPoints,
        interior_residual: F,
        bc_config: &ConditionConfig,
    ) -> Result<TrainingMetrics>
    where
        F: for<'a> Fn(&'a NdTape, &'a mut PinnNet, NdVar<'a>) -> NdVar<'a>,
    {
        validate_training_config(&self.config)?;
        validate_collocation_points(interior_points, model.ndim)?;
        validate_conditions(bc_config, model.ndim)?;

        let mut opt = NdAdam::with_lr(self.config.learning_rate);
        let mut metrics = TrainingMetrics::new();

        for epoch in 0..self.config.num_epochs
        {
            let tape = NdTape::new();
            let (x_flat, x_shape) = interior_points.to_batched_tensor();
            let xv = tape.input(TensorND::new(x_flat, x_shape));

            let mut bc_loss_terms = Vec::new();
            for condition in &bc_config.conditions
            {
                let c_flat: Vec<f32> = condition
                    .points
                    .iter()
                    .flat_map(|p| p.iter())
                    .copied()
                    .collect();
                let c_shape = vec![condition.points.len(), model.ndim];
                let cv = tape.input(TensorND::new(c_flat, c_shape));
                let u_bc = model.forward(&tape, cv);

                let targets: Vec<f32> = condition
                    .points
                    .iter()
                    .map(|point| (condition.target_fn)(point))
                    .collect();
                if let Some(&value) = targets.iter().find(|value| !value.is_finite())
                {
                    return Err(VariationalError::NonFiniteValue {
                        component: "PINN boundary target",
                        value,
                    });
                }

                let output_len = tape.value(u_bc).data.len();
                if output_len != targets.len()
                {
                    return Err(VariationalError::TrainingFailure {
                        details: format!(
                            "PINN boundary condition '{}' expects one scalar model output per point: got {output_len} outputs for {} points",
                            condition.name,
                            targets.len()
                        ),
                    });
                }

                let t_tensor = TensorND::new(targets, vec![condition.points.len(), 1]);
                let tv = tape.input(t_tensor);
                let diff = u_bc.sub(tv);
                let bc_loss = diff.mul(diff).sum();
                let w = tape.input(TensorND::new(vec![condition.weight], vec![1, 1]));
                bc_loss_terms.push(bc_loss.mul(w));
            }

            let u_pred = model.forward(&tape, xv);
            let interior_loss = interior_residual(&tape, model, u_pred);
            ensure_scalar_finite_loss(&tape, interior_loss, "PINN interior residual loss")?;

            let total_loss = if bc_loss_terms.is_empty()
            {
                interior_loss
            }
            else
            {
                let mut loss = interior_loss;
                for term in bc_loss_terms
                {
                    loss = loss.add(term);
                }
                loss
            };

            let loss_val = ensure_scalar_finite_loss(&tape, total_loss, "PINN total loss")?;
            let grads = tape.backward(total_loss);
            let mut params = model.parameters();
            let (grad_norm, parameter_grad_indices) =
                validate_parameter_gradients(&params, &grads)?;

            let mut optimizer_grads = grads.clone();
            if let Some(clip_norm) = self.config.gradient_clip_norm
            {
                if grad_norm > clip_norm
                {
                    let scale = clip_norm / grad_norm;
                    for grad_idx in parameter_grad_indices
                    {
                        for value in optimizer_grads[grad_idx].data_mut().iter_mut()
                        {
                            *value *= scale;
                        }
                    }
                }
            }

            opt.step(&mut params, &optimizer_grads);

            if epoch % 10 == 0 || epoch + 1 == self.config.num_epochs
            {
                metrics.record(epoch, loss_val, f32::NAN, grad_norm);
            }
        }

        Ok(metrics)
    }

    pub fn solve_poisson(
        &self,
        model: &mut PinnNet,
        interior: &CollocationPoints,
        source_fn: impl Fn(f32) -> f32,
        bc_config: &ConditionConfig,
    ) -> Result<TrainingMetrics> {
        if interior.ndim != 1 || model.ndim != 1
        {
            return Err(VariationalError::UnsupportedOperation {
                details: format!(
                    "PinnTrainer::solve_poisson currently supports one-dimensional scalar-coordinate problems only (collocation ndim={}, model input dim={})",
                    interior.ndim, model.ndim
                ),
            });
        }
        validate_collocation_points(interior, 1)?;

        let x = interior.to_flat();
        let source_vals: Vec<f32> = x.iter().map(|&xi| source_fn(xi)).collect();
        if let Some(&value) = source_vals.iter().find(|value| !value.is_finite())
        {
            return Err(VariationalError::NonFiniteValue {
                component: "PINN Poisson source",
                value,
            });
        }

        {
            let tape = NdTape::new();
            let xv = tape.input(TensorND::new(x.clone(), vec![x.len(), 1]));
            let u = model.forward(&tape, xv);
            if tape.value(u).data.len() != x.len()
            {
                return Err(VariationalError::UnsupportedOperation {
                    details: "PinnTrainer::solve_poisson requires one scalar model output per collocation point"
                        .into(),
                });
            }
        }

        self.train(
            model,
            interior,
            move |tape, net, u| {
                let eps = 1e-3;
                let mut lap_terms = Vec::new();
                for i in 0..x.len()
                {
                    let mut xp = x.clone();
                    xp[i] += eps;
                    let xp_var = tape.input(TensorND::new(xp, vec![x.len(), 1]));
                    let up = net.forward(tape, xp_var);

                    let mut xm = x.clone();
                    xm[i] -= eps;
                    let xm_var = tape.input(TensorND::new(xm, vec![x.len(), 1]));
                    let um = net.forward(tape, xm_var);

                    let eps_sq = tape.input(TensorND::new(vec![eps * eps], vec![1, 1]));
                    let two = tape.input(TensorND::new(vec![2.0], vec![1, 1]));
                    let d2u = up.add(um).sub(u.mul(two)).div(eps_sq);
                    lap_terms.push(d2u);
                }

                let mut laplacian = tape.input(TensorND::zeros(&[x.len(), 1]));
                for term in lap_terms
                {
                    laplacian = laplacian.add(term);
                }

                let source = tape.input(TensorND::new(source_vals.clone(), vec![x.len(), 1]));
                let residual = laplacian.add(source);
                residual.mul(residual).sum()
            },
            bc_config,
        )
    }
}

fn validate_training_config(config: &TrainingConfig) -> Result<()> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0
    {
        return Err(VariationalError::TrainingFailure {
            details: format!(
                "PINN learning rate must be finite and positive, got {}",
                config.learning_rate
            ),
        });
    }
    if let Some(clip_norm) = config.gradient_clip_norm
    {
        if !clip_norm.is_finite() || clip_norm <= 0.0
        {
            return Err(VariationalError::TrainingFailure {
                details: format!(
                    "PINN gradient clip norm must be finite and positive, got {clip_norm}"
                ),
            });
        }
    }
    Ok(())
}

fn validate_collocation_points(points: &CollocationPoints, expected_dim: usize) -> Result<()> {
    if expected_dim == 0
    {
        return Err(VariationalError::TrainingFailure {
            details: "PINN model input dimension must be greater than zero".into(),
        });
    }
    if points.is_empty()
    {
        return Err(VariationalError::TrainingFailure {
            details: "PINN requires at least one interior collocation point".into(),
        });
    }
    if points.ndim != expected_dim
    {
        return Err(VariationalError::DimensionMismatch {
            expected: expected_dim,
            got: points.ndim,
            context: "PinnTrainer collocation dimension".into(),
        });
    }
    for point in &points.points
    {
        if point.len() != expected_dim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: expected_dim,
                got: point.len(),
                context: "PinnTrainer collocation point".into(),
            });
        }
        if let Some(&value) = point.iter().find(|value| !value.is_finite())
        {
            return Err(VariationalError::NonFiniteValue {
                component: "PINN collocation point",
                value,
            });
        }
    }
    Ok(())
}

fn validate_conditions(config: &ConditionConfig, expected_dim: usize) -> Result<()> {
    for condition in &config.conditions
    {
        if !condition.weight.is_finite() || condition.weight < 0.0
        {
            return Err(VariationalError::InvalidBoundaryCondition {
                details: format!(
                    "condition '{}' has invalid weight {}",
                    condition.name, condition.weight
                ),
            });
        }
        if condition.points.is_empty()
        {
            return Err(VariationalError::InvalidBoundaryCondition {
                details: format!("condition '{}' has no points", condition.name),
            });
        }
        for point in &condition.points
        {
            if point.len() != expected_dim
            {
                return Err(VariationalError::DimensionMismatch {
                    expected: expected_dim,
                    got: point.len(),
                    context: format!("PINN boundary condition '{}' point", condition.name),
                });
            }
            if let Some(&value) = point.iter().find(|value| !value.is_finite())
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN boundary point",
                    value,
                });
            }
        }
    }
    Ok(())
}

fn ensure_scalar_finite_loss(
    tape: &NdTape,
    loss: NdVar<'_>,
    component: &str,
) -> Result<f32> {
    let value = tape.value(loss);
    if value.data.len() != 1
    {
        return Err(VariationalError::TrainingFailure {
            details: format!(
                "{component} must be scalar, got {} values",
                value.data.len()
            ),
        });
    }
    let scalar = value.data[0];
    if !scalar.is_finite()
    {
        return Err(VariationalError::TrainingFailure {
            details: format!("{component} is non-finite: {scalar}"),
        });
    }
    Ok(scalar)
}

fn validate_parameter_gradients(
    params: &[NdParam<'_>],
    grads: &[TensorND],
) -> Result<(f32, Vec<usize>)> {
    let mut grad_norm_sq = 0.0f64;
    let mut grad_indices = Vec::with_capacity(params.len());

    for (param_index, param) in params.iter().enumerate()
    {
        if grad_indices.contains(&param.grad_idx)
        {
            return Err(VariationalError::TrainingFailure {
                details: format!(
                    "PINN parameter {param_index} reuses gradient index {}",
                    param.grad_idx
                ),
            });
        }
        let grad = grads
            .get(param.grad_idx)
            .ok_or_else(|| VariationalError::TrainingFailure {
                details: format!(
                    "PINN parameter {param_index} references missing gradient index {} ({} gradients available)",
                    param.grad_idx,
                    grads.len()
                ),
            })?;
        if grad.data.len() != param.value.data.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: param.value.data.len(),
                got: grad.data.len(),
                context: format!("PinnTrainer parameter {param_index} gradient"),
            });
        }
        for &value in grad.data.iter()
        {
            if !value.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN parameter gradient",
                    value,
                });
            }
            let value = value as f64;
            grad_norm_sq += value * value;
        }
        grad_indices.push(param.grad_idx);
    }

    let grad_norm = grad_norm_sq.sqrt() as f32;
    if !grad_norm.is_finite()
    {
        return Err(VariationalError::TrainingFailure {
            details: "PINN parameter gradient norm overflowed".into(),
        });
    }
    Ok((grad_norm, grad_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_dim_points() -> CollocationPoints {
        CollocationPoints {
            points: vec![vec![0.0], vec![0.5], vec![1.0]],
            ndim: 1,
        }
    }

    fn one_epoch_config() -> TrainingConfig {
        TrainingConfig {
            num_epochs: 1,
            ..TrainingConfig::default()
        }
    }

    #[test]
    fn train_rejects_vector_residual_instead_of_using_first_value() {
        let mut rng = PcgEngine::new(1);
        let mut model = PinnNet::new(1, 4, 1, &mut rng);
        let trainer = PinnTrainer::new(one_epoch_config());
        let result = trainer.train(
            &mut model,
            &one_dim_points(),
            |_tape, _net, u| u,
            &ConditionConfig::new(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn train_records_parameter_gradient_norm() {
        let mut rng = PcgEngine::new(2);
        let mut model = PinnNet::new(1, 4, 1, &mut rng);
        let trainer = PinnTrainer::new(one_epoch_config());
        let metrics = trainer
            .train(
                &mut model,
                &one_dim_points(),
                |_tape, _net, u| u.sum(),
                &ConditionConfig::new(),
            )
            .unwrap();
        assert_eq!(metrics.grad_norm.len(), 1);
        assert!(metrics.grad_norm[0].is_finite());
        assert!(metrics.grad_norm[0] > 0.0);
    }

    #[test]
    fn train_rejects_mutated_boundary_point_dimension() {
        let mut rng = PcgEngine::new(3);
        let mut model = PinnNet::new(1, 4, 1, &mut rng);
        let trainer = PinnTrainer::new(one_epoch_config());
        let condition = crate::pinn::conditions::Condition {
            kind: crate::pinn::conditions::ConditionKind::Dirichlet,
            points: vec![vec![0.0, 1.0]],
            target_fn: |_| 0.0,
            weight: 1.0,
            name: "bad".into(),
        };
        let config = ConditionConfig::new().with_condition(condition);
        let result = trainer.train(
            &mut model,
            &one_dim_points(),
            |_tape, _net, u| u.mul(u).sum(),
            &config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn solve_poisson_rejects_multidimensional_input() {
        let mut rng = PcgEngine::new(4);
        let mut model = PinnNet::new(2, 4, 1, &mut rng);
        let trainer = PinnTrainer::new(one_epoch_config());
        let points = CollocationPoints {
            points: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            ndim: 2,
        };
        let result = trainer.solve_poisson(&mut model, &points, |_| 0.0, &ConditionConfig::new());
        assert!(result.is_err());
    }

    #[test]
    fn solve_poisson_reduces_residual_to_scalar_loss() {
        let mut rng = PcgEngine::new(5);
        let mut model = PinnNet::new(1, 4, 1, &mut rng);
        let trainer = PinnTrainer::new(one_epoch_config());
        let metrics = trainer
            .solve_poisson(
                &mut model,
                &one_dim_points(),
                |_| 0.0,
                &ConditionConfig::new(),
            )
            .unwrap();
        assert_eq!(metrics.train_loss.len(), 1);
        assert!(metrics.train_loss[0].is_finite());
    }
}
