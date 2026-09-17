use crate::{NeuralOperatorError, OperatorDataset1d, Result};
use scirust_core::autodiff::nd::NdTape;
use scirust_core::nn::fno::NdFno;
use scirust_core::nn::nd_optim::NdAdam;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

/// Uniform inference contract for learned function-to-function maps.
pub trait LearnedOperator {
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>>;
}

/// Configuration of SciRust's trainable 1-D Fourier Neural Operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fno1dConfig {
    pub points: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub modes: usize,
    pub seed: u64,
}

impl Fno1dConfig {
    /// Validate dimensions and the strictly one-sided retained Fourier modes.
    pub fn validate(self) -> Result<Self> {
        if self.points == 0
        {
            return Err(NeuralOperatorError::Empty { what: "FNO grid" });
        }
        for channels in [self.in_channels, self.out_channels, self.hidden_channels]
        {
            if channels == 0
            {
                return Err(NeuralOperatorError::InvalidChannels { channels });
            }
        }
        // The wrapped inverse DFT reconstructs negative frequencies by doubling
        // each retained nonzero positive-frequency mode.  For even grids the
        // Nyquist bin is self-conjugate and therefore must not enter that path.
        // Keep the public wrapper on the strictly one-sided spectrum until the
        // low-level inverse handles Nyquist separately.
        let max_modes = self.points / 2 + self.points % 2;
        if self.modes == 0 || self.modes > max_modes
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FNO modes",
                expected: max_modes,
                got: self.modes,
            });
        }
        Ok(self)
    }
}

/// Summary of a deterministic sample-wise training run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitReport {
    pub epochs: usize,
    pub optimizer_steps: u64,
    pub first_mse: f64,
    pub last_mse: f64,
}

/// High-level trainable wrapper around SciRust's native [`NdFno`].
pub struct Fno1dOperator {
    cfg: Fno1dConfig,
    model: NdFno,
}

impl Fno1dOperator {
    /// Construct a deterministically seeded trainable FNO from validated configuration.
    pub fn new(cfg: Fno1dConfig) -> Result<Self> {
        let cfg = cfg.validate()?;
        let mut rng = PcgEngine::new(cfg.seed);
        let model = NdFno::new(
            cfg.points,
            cfg.in_channels,
            cfg.out_channels,
            cfg.hidden_channels,
            cfg.modes,
            &mut rng,
        );
        Ok(Self { cfg, model })
    }

    /// Return the validated configuration used to construct this operator.
    pub fn config(&self) -> Fno1dConfig {
        self.cfg
    }

    /// Deterministic sample-wise Adam training. MSE is reported per scalar output.
    pub fn fit(
        &mut self,
        dataset: &OperatorDataset1d,
        epochs: usize,
        lr: f32,
    ) -> Result<FitReport> {
        if dataset.points() != self.cfg.points
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FNO dataset points",
                expected: self.cfg.points,
                got: dataset.points(),
            });
        }
        if dataset.in_channels() != self.cfg.in_channels
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FNO input channels",
                expected: self.cfg.in_channels,
                got: dataset.in_channels(),
            });
        }
        if dataset.out_channels() != self.cfg.out_channels
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FNO output channels",
                expected: self.cfg.out_channels,
                got: dataset.out_channels(),
            });
        }
        if !lr.is_finite()
        {
            return Err(NeuralOperatorError::InvalidLearningRate { lr });
        }
        let mut opt = NdAdam::with_lr(lr);
        let denom = (self.cfg.points * self.cfg.out_channels) as f64;
        let mut first_mse = f64::NAN;
        let mut last_mse = f64::NAN;

        for epoch in 0..epochs
        {
            let mut epoch_loss = 0.0f64;
            for sample in dataset.samples()
            {
                let tape = NdTape::new();
                let x = tape.input(TensorND::new(
                    sample.input.clone(),
                    vec![self.cfg.points, self.cfg.in_channels],
                ));
                let target = tape.input(TensorND::new(
                    sample.target.clone(),
                    vec![self.cfg.points, self.cfg.out_channels],
                ));
                let pred = self.model.forward(&tape, x);
                let diff = pred.sub(target);
                let loss = diff.mul(diff).sum();
                epoch_loss += tape.value(loss).data[0] as f64 / denom;
                let grads = tape.backward(loss);
                opt.step(&mut self.model.parameters(), &grads);
            }
            let mse = epoch_loss / dataset.len() as f64;
            if epoch == 0
            {
                first_mse = mse;
            }
            last_mse = mse;
        }
        Ok(FitReport {
            epochs,
            optimizer_steps: opt.step_count(),
            first_mse,
            last_mse,
        })
    }
}

impl LearnedOperator for Fno1dOperator {
    fn input_len(&self) -> usize {
        self.cfg.points * self.cfg.in_channels
    }
    fn output_len(&self) -> usize {
        self.cfg.points * self.cfg.out_channels
    }

    fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_len()
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FNO input",
                expected: self.input_len(),
                got: input.len(),
            });
        }
        if let Some((index, _)) = input.iter().enumerate().find(|(_, x)| !x.is_finite())
        {
            return Err(NeuralOperatorError::NonFinite {
                what: "FNO input",
                index,
            });
        }
        let tape = NdTape::new();
        let x = tape.input(TensorND::new(
            input.to_vec(),
            vec![self.cfg.points, self.cfg.in_channels],
        ));
        let y = self.model.forward(&tape, x);
        Ok(tape.value(y).data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperatorSample1d, relative_l2};
    use std::f32::consts::TAU;

    fn derivative_dataset(phases: &[f32], n: usize) -> OperatorDataset1d {
        let samples = phases
            .iter()
            .map(|&phase| {
                let input: Vec<f32> = (0..n)
                    .map(|j| (TAU * j as f32 / n as f32 + phase).sin())
                    .collect();
                let target: Vec<f32> = (0..n)
                    .map(|j| (TAU * j as f32 / n as f32 + phase).cos())
                    .collect();
                OperatorSample1d::new(input, target).unwrap()
            })
            .collect();
        OperatorDataset1d::new(samples, n, 1, 1).unwrap()
    }

    #[test]
    fn fno_config_rejects_nyquist_and_negative_frequency_bins() {
        let even = Fno1dConfig {
            points: 8,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 4,
            modes: 4,
            seed: 7,
        };
        assert!(even.validate().is_ok());
        let err = Fno1dConfig { modes: 5, ..even }.validate().unwrap_err();
        assert_eq!(
            err,
            NeuralOperatorError::ShapeMismatch {
                what: "FNO modes",
                expected: 4,
                got: 5,
            }
        );

        let odd = Fno1dConfig {
            points: 7,
            modes: 4,
            ..even
        };
        assert!(odd.validate().is_ok());
    }

    #[test]
    fn fno_wrapper_predicts_correct_shape() {
        let cfg = Fno1dConfig {
            points: 8,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 4,
            modes: 3,
            seed: 7,
        };
        let mut op = Fno1dOperator::new(cfg).unwrap();
        assert_eq!(op.predict(&[0.0; 8]).unwrap().len(), 8);
    }

    #[test]
    fn fno_training_rejects_non_finite_learning_rates() {
        let n = 8;
        let train = derivative_dataset(&[0.0], n);
        let cfg = Fno1dConfig {
            points: n,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 4,
            modes: 3,
            seed: 11,
        };
        for lr in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
        {
            let mut op = Fno1dOperator::new(cfg).unwrap();
            assert!(matches!(
                op.fit(&train, 1, lr),
                Err(NeuralOperatorError::InvalidLearningRate { lr: got }) if got.is_nan() == lr.is_nan() && (lr.is_nan() || got == lr)
            ));
        }
    }

    #[test]
    fn fno_training_reduces_operator_error() {
        let n = 16;
        let train = derivative_dataset(&[0.0, 0.4, 0.8, 1.2], n);
        let cfg = Fno1dConfig {
            points: n,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 6,
            modes: 4,
            seed: 3,
        };
        let mut op = Fno1dOperator::new(cfg).unwrap();
        let probe = derivative_dataset(&[0.25], n);
        let before = relative_l2(
            &op.predict(&probe.samples()[0].input).unwrap(),
            &probe.samples()[0].target,
        )
        .unwrap();
        let report = op.fit(&train, 120, 0.01).unwrap();
        let after = relative_l2(
            &op.predict(&probe.samples()[0].input).unwrap(),
            &probe.samples()[0].target,
        )
        .unwrap();
        assert!(
            report.last_mse < report.first_mse,
            "training MSE did not decrease: {:?}",
            report
        );
        assert!(
            after < before,
            "operator error did not improve: {before} -> {after}"
        );
    }
}
