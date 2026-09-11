use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::nn::nd_optim::{NdAdam, NdParam};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};
use crate::learning::dataset::TrajectoryDataset;
use crate::learning::losses::LossValue;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub gradient_clip_norm: Option<f32>,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: usize,
    pub regularization_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 100,
            batch_size: 64,
            seed: 42,
            gradient_clip_norm: Some(10.0),
            checkpoint_dir: None,
            checkpoint_interval: 50,
            regularization_weight: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: Vec<usize>,
    pub train_loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub grad_norm: Vec<f32>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            epoch: Vec::new(),
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            grad_norm: Vec::new(),
        }
    }

    pub fn record(&mut self, epoch: usize, train: f32, val: f32, grad_norm: f32) {
        self.epoch.push(epoch);
        self.train_loss.push(train);
        self.val_loss.push(val);
        self.grad_norm.push(grad_norm);
    }

    pub fn best_epoch(&self) -> usize {
        self.epoch
            .iter()
            .copied()
            .zip(self.val_loss.iter().copied())
            .filter(|(_, loss)| loss.is_finite())
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(epoch, _)| epoch)
            .unwrap_or(0)
    }

    pub fn final_train_loss(&self) -> f32 {
        self.train_loss.last().copied().unwrap_or(f32::NAN)
    }

    pub fn final_val_loss(&self) -> f32 {
        self.val_loss.last().copied().unwrap_or(f32::NAN)
    }
}

pub struct PhysicsTrainer;

impl PhysicsTrainer {
    pub fn train_lnn<M, L>(
        model: &mut M,
        dataset: &TrajectoryDataset,
        config: &TrainingConfig,
        val_dataset: Option<&TrajectoryDataset>,
        compute_loss: L,
    ) -> Result<TrainingMetrics>
    where
        M: HasParameters,
        L: for<'a> Fn(
            &'a mut M,
            &'a NdTape,
            &'a [NdVar<'a>],
            &'a [NdVar<'a>],
            &'a [f32],
            &'a [f32],
        ) -> (NdVar<'a>, LossValue),
    {
        validate_training_config(config)?;
        validate_dataset(dataset, dataset.ndim, "training")?;
        if let Some(validation) = val_dataset
        {
            validate_dataset(validation, dataset.ndim, "validation")?;
        }

        let mut opt = NdAdam::with_lr(config.learning_rate);
        let mut rng = PcgEngine::new(config.seed);
        let mut order: Vec<usize> = (0..dataset.len()).collect();
        let mut metrics = TrainingMetrics::new();
        let ndim = dataset.ndim;

        for epoch in 0..config.num_epochs
        {
            shuffle_order(&mut order, &mut rng);

            let mut total_train_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut n_train = 0usize;

            for batch_start in (0..order.len()).step_by(config.batch_size)
            {
                let batch_end = (batch_start + config.batch_size).min(order.len());
                let batch_size = batch_end - batch_start;

                let mut q_batch = Vec::with_capacity(batch_size * ndim);
                let mut dq_batch = Vec::with_capacity(batch_size * ndim);
                let mut ddq_batch = Vec::with_capacity(batch_size * ndim);

                for &position in &order[batch_start..batch_end]
                {
                    let sample = dataset.get(position);
                    q_batch.extend_from_slice(&sample.q);
                    dq_batch.extend_from_slice(&sample.dq);
                    ddq_batch.extend_from_slice(&sample.ddq);
                }

                let tape = NdTape::new();
                let qv = tape.input(TensorND::new(q_batch.clone(), vec![batch_size, ndim]));
                let dqv = tape.input(TensorND::new(dq_batch.clone(), vec![batch_size, ndim]));
                let q_arr = [qv];
                let dq_arr = [dqv];

                let grads = {
                    let (loss_var, _) =
                        compute_loss(model, &tape, &q_arr, &dq_arr, &ddq_batch, &q_batch);
                    let loss_tensor = tape.value(loss_var);
                    let loss_val =
                        scalar_loss(&loss_tensor, "PhysicsTrainer::train_lnn training loss")?;
                    total_train_loss += loss_val * batch_size as f32;
                    tape.backward(loss_var)
                };

                let mut params = model.parameters();
                let (batch_grad_norm, parameter_grad_indices) =
                    parameter_gradient_norm(&params, &grads)?;
                total_grad_norm += batch_grad_norm * batch_size as f32;

                let mut clipped_grads = grads.clone();
                if let Some(clip_norm) = config.gradient_clip_norm
                {
                    if batch_grad_norm > clip_norm
                    {
                        let scale = if batch_grad_norm > 0.0
                        {
                            clip_norm / batch_grad_norm
                        }
                        else
                        {
                            1.0
                        };
                        for grad_index in parameter_grad_indices
                        {
                            for value in clipped_grads[grad_index].data_mut()
                            {
                                *value *= scale;
                            }
                        }
                    }
                }

                opt.step(&mut params, &clipped_grads);
                n_train += batch_size;
            }

            let avg_train_loss = total_train_loss / n_train as f32;
            let avg_grad_norm = total_grad_norm / n_train as f32;

            let avg_val_loss = if let Some(validation) = val_dataset
            {
                let mut val_loss = 0.0;
                let mut val_samples = 0usize;

                for batch_start in (0..validation.len()).step_by(config.batch_size)
                {
                    let batch_end = (batch_start + config.batch_size).min(validation.len());
                    let batch_size = batch_end - batch_start;

                    let mut q_batch = Vec::with_capacity(batch_size * ndim);
                    let mut dq_batch = Vec::with_capacity(batch_size * ndim);
                    let mut ddq_batch = Vec::with_capacity(batch_size * ndim);

                    for i in batch_start..batch_end
                    {
                        let sample = validation.get(i);
                        q_batch.extend_from_slice(&sample.q);
                        dq_batch.extend_from_slice(&sample.dq);
                        ddq_batch.extend_from_slice(&sample.ddq);
                    }

                    let tape = NdTape::new();
                    let qv = tape.input(TensorND::new(q_batch.clone(), vec![batch_size, ndim]));
                    let dqv = tape.input(TensorND::new(dq_batch.clone(), vec![batch_size, ndim]));
                    let q_arr = [qv];
                    let dq_arr = [dqv];

                    let (loss_var, _) =
                        compute_loss(model, &tape, &q_arr, &dq_arr, &ddq_batch, &q_batch);
                    let loss_tensor = tape.value(loss_var);
                    let loss_val =
                        scalar_loss(&loss_tensor, "PhysicsTrainer::train_lnn validation loss")?;
                    val_loss += loss_val * batch_size as f32;
                    val_samples += batch_size;
                }

                val_loss / val_samples as f32
            }
            else
            {
                f32::NAN
            };

            metrics.record(epoch, avg_train_loss, avg_val_loss, avg_grad_norm);

            if let Some(ref checkpoint_dir) = config.checkpoint_dir
            {
                if epoch % config.checkpoint_interval == 0
                {
                    let dir = format!("{checkpoint_dir}/epoch_{epoch}");
                    std::fs::create_dir_all(&dir).map_err(|err| {
                        VariationalError::TrainingFailure {
                            details: format!("failed to create checkpoint directory {dir}: {err}"),
                        }
                    })?;
                }
            }
        }

        Ok(metrics)
    }
}

pub trait HasParameters {
    fn parameters(&mut self) -> Vec<NdParam<'_>>;
}

fn validate_training_config(config: &TrainingConfig) -> Result<()> {
    if !config.learning_rate.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "training learning rate",
            value: config.learning_rate,
        });
    }
    if config.learning_rate <= 0.0
    {
        return Err(VariationalError::TrainingFailure {
            details: "learning rate must be strictly positive".into(),
        });
    }
    if config.batch_size == 0
    {
        return Err(VariationalError::TrainingFailure {
            details: "batch size must be greater than zero".into(),
        });
    }
    if let Some(clip_norm) = config.gradient_clip_norm
    {
        if !clip_norm.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "gradient clip norm",
                value: clip_norm,
            });
        }
        if clip_norm < 0.0
        {
            return Err(VariationalError::TrainingFailure {
                details: "gradient clip norm must be non-negative".into(),
            });
        }
    }
    if config.checkpoint_dir.is_some() && config.checkpoint_interval == 0
    {
        return Err(VariationalError::TrainingFailure {
            details: "checkpoint interval must be greater than zero when checkpointing is enabled"
                .into(),
        });
    }
    Ok(())
}

fn validate_dataset(dataset: &TrajectoryDataset, expected_ndim: usize, label: &str) -> Result<()> {
    if expected_ndim == 0 || dataset.ndim != expected_ndim
    {
        return Err(VariationalError::DimensionMismatch {
            expected: expected_ndim.max(1),
            got: dataset.ndim,
            context: format!("PhysicsTrainer::{label} dataset ndim"),
        });
    }
    if dataset.is_empty()
    {
        return Err(VariationalError::TrainingFailure {
            details: format!("{label} dataset has no active samples"),
        });
    }

    for (position, &sample_index) in dataset.indices.iter().enumerate()
    {
        let sample =
            dataset
                .samples
                .get(sample_index)
                .ok_or_else(|| VariationalError::TrainingFailure {
                    details: format!("{label} dataset index {position} points outside samples"),
                })?;
        for (component, len) in [
            ("q", sample.q.len()),
            ("dq", sample.dq.len()),
            ("ddq", sample.ddq.len()),
        ]
        {
            if len != expected_ndim
            {
                return Err(VariationalError::DimensionMismatch {
                    expected: expected_ndim,
                    got: len,
                    context: format!("PhysicsTrainer::{label} sample {position} {component}"),
                });
            }
        }
        for &value in sample
            .q
            .iter()
            .chain(sample.dq.iter())
            .chain(sample.ddq.iter())
        {
            if !value.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "training sample state",
                    value,
                });
            }
        }
        if !sample.t.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "training sample time",
                value: sample.t,
            });
        }
    }
    Ok(())
}

fn shuffle_order(order: &mut [usize], rng: &mut PcgEngine) {
    for i in (1..order.len()).rev()
    {
        let j = (rng.float() * (i as f32 + 1.0)) as usize;
        order.swap(i, j.min(i));
    }
}

fn scalar_loss(loss: &TensorND, context: &str) -> Result<f32> {
    if loss.data.len() != 1
    {
        return Err(VariationalError::DimensionMismatch {
            expected: 1,
            got: loss.data.len(),
            context: context.into(),
        });
    }
    let value = loss.data[0];
    if !value.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "training loss",
            value,
        });
    }
    Ok(value)
}

fn parameter_gradient_norm(
    params: &[NdParam<'_>],
    grads: &[TensorND],
) -> Result<(f32, Vec<usize>)> {
    let mut sum_sq = 0.0f32;
    let mut grad_indices = Vec::new();

    for (parameter_index, param) in params.iter().enumerate()
    {
        let grad = grads
            .get(param.grad_idx)
            .ok_or_else(|| VariationalError::TrainingFailure {
                details: format!(
                    "parameter {parameter_index} references missing gradient index {}",
                    param.grad_idx
                ),
            })?;
        if grad.data.len() != param.value.data.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: param.value.data.len(),
                got: grad.data.len(),
                context: format!("PhysicsTrainer parameter gradient {parameter_index}"),
            });
        }
        for &value in grad.data.iter()
        {
            if !value.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "parameter gradient",
                    value,
                });
            }
            sum_sq += value * value;
        }
        if !grad_indices.contains(&param.grad_idx)
        {
            grad_indices.push(param.grad_idx);
        }
    }

    if !sum_sq.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "parameter gradient norm squared",
            value: sum_sq,
        });
    }
    Ok((sum_sq.sqrt(), grad_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::dataset::TrajectorySample;

    struct DummyModel;

    impl HasParameters for DummyModel {
        fn parameters(&mut self) -> Vec<NdParam<'_>> {
            Vec::new()
        }
    }

    fn dummy_loss<'a>(
        _model: &'a mut DummyModel,
        _tape: &'a NdTape,
        q: &'a [NdVar<'a>],
        _dq: &'a [NdVar<'a>],
        _ddq: &'a [f32],
        _q_batch: &'a [f32],
    ) -> (NdVar<'a>, LossValue) {
        (q[0].sum(), LossValue::new(0.0))
    }

    fn dataset(values: &[f32]) -> TrajectoryDataset {
        let samples = values
            .iter()
            .map(|&q| TrajectorySample {
                q: vec![q],
                dq: vec![0.0],
                ddq: vec![0.0],
                t: 0.0,
            })
            .collect();
        TrajectoryDataset::new(samples).unwrap()
    }

    #[test]
    fn test_training_config_default() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.learning_rate, 1e-3);
        assert_eq!(cfg.num_epochs, 100);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut metrics = TrainingMetrics::new();
        metrics.record(0, 1.0, 2.0, 0.5);
        metrics.record(1, 0.5, 1.0, 0.3);
        assert_eq!(metrics.best_epoch(), 1);
    }

    #[test]
    fn best_epoch_ignores_non_finite_validation_losses() {
        let mut metrics = TrainingMetrics::new();
        metrics.record(0, 3.0, f32::NAN, 0.0);
        metrics.record(1, 2.0, 2.0, 0.0);
        metrics.record(2, 1.0, 1.0, 0.0);
        assert_eq!(metrics.best_epoch(), 2);
    }

    #[test]
    fn training_rejects_zero_batch_size() {
        let training = dataset(&[1.0]);
        let mut model = DummyModel;
        let config = TrainingConfig {
            batch_size: 0,
            num_epochs: 1,
            ..Default::default()
        };

        let err = PhysicsTrainer::train_lnn(&mut model, &training, &config, None, dummy_loss)
            .unwrap_err();
        assert!(matches!(err, VariationalError::TrainingFailure { .. }));
    }

    #[test]
    fn validation_denominator_resets_each_epoch() {
        let training = dataset(&[1.0]);
        let validation = dataset(&[2.0, 4.0]);
        let mut model = DummyModel;
        let config = TrainingConfig {
            num_epochs: 2,
            batch_size: 1,
            gradient_clip_norm: None,
            ..Default::default()
        };

        let metrics = PhysicsTrainer::train_lnn(
            &mut model,
            &training,
            &config,
            Some(&validation),
            dummy_loss,
        )
        .unwrap();

        assert_eq!(metrics.val_loss.len(), 2);
        assert!((metrics.val_loss[0] - 3.0).abs() < 1e-6);
        assert!((metrics.val_loss[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn validation_dataset_dimension_must_match_training() {
        let training = dataset(&[1.0]);
        let validation = TrajectoryDataset::new(vec![TrajectorySample {
            q: vec![1.0, 2.0],
            dq: vec![0.0, 0.0],
            ddq: vec![0.0, 0.0],
            t: 0.0,
        }])
        .unwrap();
        let mut model = DummyModel;
        let config = TrainingConfig {
            num_epochs: 1,
            ..Default::default()
        };

        let err = PhysicsTrainer::train_lnn(
            &mut model,
            &training,
            &config,
            Some(&validation),
            dummy_loss,
        )
        .unwrap_err();
        assert!(matches!(err, VariationalError::DimensionMismatch { .. }));
    }
}
