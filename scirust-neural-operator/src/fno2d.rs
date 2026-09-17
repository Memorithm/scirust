//! High-level trainable 2-D Fourier Neural Operator.

use crate::{FitReport, LearnedOperator, NeuralOperatorError, OperatorDataset2d, Result};
use scirust_core::autodiff::nd::NdTape;
use scirust_core::nn::fno::{FourierMode2d, NdFno2d, low_frequency_modes_2d};
use scirust_core::nn::nd_optim::NdAdam;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

/// Configuration of SciRust's trainable 2-D Fourier Neural Operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fno2dConfig {
    pub rows: usize,
    pub cols: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    /// Largest wrapped absolute Fourier index retained on the row axis.
    pub max_abs_ky: usize,
    /// Largest wrapped absolute Fourier index retained on the column axis.
    pub max_abs_kx: usize,
    pub seed: u64,
}

impl Fno2dConfig {
    /// Validate grid, channels and Nyquist-bounded low-frequency rectangle.
    pub fn validate(self) -> Result<Self> {
        if self.rows == 0 || self.cols == 0
        {
            return Err(NeuralOperatorError::Empty {
                what: "FNO 2-D grid",
            });
        }
        for channels in [self.in_channels, self.out_channels, self.hidden_channels]
        {
            if channels == 0
            {
                return Err(NeuralOperatorError::InvalidChannels { channels });
            }
        }
        if self.max_abs_ky > self.rows / 2
        {
            return Err(NeuralOperatorError::InvalidFourierModeBound {
                axis: "y",
                requested: self.max_abs_ky,
                nyquist: self.rows / 2,
            });
        }
        if self.max_abs_kx > self.cols / 2
        {
            return Err(NeuralOperatorError::InvalidFourierModeBound {
                axis: "x",
                requested: self.max_abs_kx,
                nyquist: self.cols / 2,
            });
        }
        Ok(self)
    }

    /// Canonical conjugate representatives retained by this configuration.
    pub fn modes(self) -> Result<Vec<FourierMode2d>> {
        let cfg = self.validate()?;
        Ok(low_frequency_modes_2d(
            cfg.rows,
            cfg.cols,
            cfg.max_abs_ky,
            cfg.max_abs_kx,
        ))
    }
}

/// High-level deterministic 2-D FNO wrapper.
pub struct Fno2dOperator {
    cfg: Fno2dConfig,
    modes: Vec<FourierMode2d>,
    model: NdFno2d,
}

impl Fno2dOperator {
    /// Construct a seeded 2-D FNO using canonical conjugate mode representatives.
    pub fn new(cfg: Fno2dConfig) -> Result<Self> {
        let cfg = cfg.validate()?;
        let modes = cfg.modes()?;
        let mut rng = PcgEngine::new(cfg.seed);
        let model = NdFno2d::new(
            cfg.rows,
            cfg.cols,
            cfg.in_channels,
            cfg.out_channels,
            cfg.hidden_channels,
            modes.clone(),
            &mut rng,
        );
        Ok(Self { cfg, modes, model })
    }

    /// Return the validated configuration used to construct this operator.
    pub const fn config(&self) -> Fno2dConfig {
        self.cfg
    }
    /// Return the canonical trainable Fourier-mode representatives.
    pub fn modes(&self) -> &[FourierMode2d] {
        &self.modes
    }

    /// Deterministic sample-wise Adam training; MSE is per scalar output.
    pub fn fit(
        &mut self,
        dataset: &OperatorDataset2d,
        epochs: usize,
        lr: f32,
    ) -> Result<FitReport> {
        for (what, expected, got) in [
            ("FNO 2-D dataset rows", self.cfg.rows, dataset.rows()),
            ("FNO 2-D dataset cols", self.cfg.cols, dataset.cols()),
            (
                "FNO 2-D input channels",
                self.cfg.in_channels,
                dataset.in_channels(),
            ),
            (
                "FNO 2-D output channels",
                self.cfg.out_channels,
                dataset.out_channels(),
            ),
        ]
        {
            if expected != got
            {
                return Err(NeuralOperatorError::ShapeMismatch {
                    what,
                    expected,
                    got,
                });
            }
        }
        if !lr.is_finite()
        {
            return Err(NeuralOperatorError::InvalidLearningRate { lr });
        }
        let mut optimizer = NdAdam::with_lr(lr);
        let denom = (self.cfg.rows * self.cfg.cols * self.cfg.out_channels) as f64;
        let mut first_mse = f64::NAN;
        let mut last_mse = f64::NAN;
        for epoch in 0..epochs
        {
            let mut epoch_loss = 0.0f64;
            for sample in dataset.samples()
            {
                let tape = NdTape::new();
                let input = tape.input(TensorND::new(
                    sample.input.clone(),
                    vec![self.cfg.rows, self.cfg.cols, self.cfg.in_channels],
                ));
                let target = tape.input(TensorND::new(
                    sample.target.clone(),
                    vec![self.cfg.rows, self.cfg.cols, self.cfg.out_channels],
                ));
                let prediction = self.model.forward(&tape, input);
                let residual = prediction.sub(target);
                let loss = residual.mul(residual).sum();
                epoch_loss += tape.value(loss).data[0] as f64 / denom;
                let gradients = tape.backward(loss);
                optimizer.step(&mut self.model.parameters(), &gradients);
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
            optimizer_steps: optimizer.step_count(),
            first_mse,
            last_mse,
        })
    }

    fn validate_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.input_len()
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FNO 2-D input",
                expected: self.input_len(),
                got: input.len(),
            });
        }
        if let Some((index, _)) = input
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NeuralOperatorError::NonFinite {
                what: "FNO 2-D input",
                index,
            });
        }
        Ok(())
    }
}

impl LearnedOperator for Fno2dOperator {
    fn input_len(&self) -> usize {
        self.cfg.rows * self.cfg.cols * self.cfg.in_channels
    }

    fn output_len(&self) -> usize {
        self.cfg.rows * self.cfg.cols * self.cfg.out_channels
    }

    fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.validate_input(input)?;
        let tape = NdTape::new();
        let x = tape.input(TensorND::new(
            input.to_vec(),
            vec![self.cfg.rows, self.cfg.cols, self.cfg.in_channels],
        ));
        let y = self.model.forward(&tape, x);
        Ok(tape.value(y).data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperatorSample2d, relative_l2};
    use std::f32::consts::TAU;

    fn identity_dataset(phases: &[f32], rows: usize, cols: usize) -> OperatorDataset2d {
        let samples = phases
            .iter()
            .map(|&phase| {
                let field = (0..rows)
                    .flat_map(|y| {
                        (0..cols).map(move |x| {
                            let yy = TAU * y as f32 / rows as f32;
                            let xx = TAU * x as f32 / cols as f32;
                            (xx + phase).sin() + 0.35 * (yy - 0.5 * phase).cos()
                        })
                    })
                    .collect::<Vec<_>>();
                OperatorSample2d::new(field.clone(), field).unwrap()
            })
            .collect();
        OperatorDataset2d::new(samples, rows, cols, 1, 1).unwrap()
    }

    #[test]
    fn fno2d_wrapper_predicts_expected_shape() {
        let cfg = Fno2dConfig {
            rows: 5,
            cols: 5,
            in_channels: 1,
            out_channels: 2,
            hidden_channels: 4,
            max_abs_ky: 1,
            max_abs_kx: 1,
            seed: 7,
        };
        let mut operator = Fno2dOperator::new(cfg).unwrap();
        assert_eq!(operator.modes().len(), 5);
        assert_eq!(operator.predict(&[0.0; 25]).unwrap().len(), 50);
    }

    #[test]
    fn fno2d_training_reduces_identity_operator_error() {
        let (rows, cols) = (5usize, 5usize);
        let cfg = Fno2dConfig {
            rows,
            cols,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 6,
            max_abs_ky: 1,
            max_abs_kx: 1,
            seed: 23,
        };
        let train = identity_dataset(&[0.0, 0.4, 0.8, 1.2], rows, cols);
        let probe = identity_dataset(&[0.23], rows, cols);
        let mut operator = Fno2dOperator::new(cfg).unwrap();
        let before = relative_l2(
            &operator.predict(&probe.samples()[0].input).unwrap(),
            &probe.samples()[0].target,
        )
        .unwrap();
        let report = operator.fit(&train, 160, 0.01).unwrap();
        let after = relative_l2(
            &operator.predict(&probe.samples()[0].input).unwrap(),
            &probe.samples()[0].target,
        )
        .unwrap();
        assert!(
            report.last_mse < report.first_mse,
            "MSE did not decrease: {report:?}"
        );
        assert!(
            after < before,
            "2-D operator error did not improve: {before} -> {after}"
        );
    }
}
