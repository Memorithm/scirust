//! Radix-2 FFT inference runtime for trained SciRust FNO 1-D snapshots.
//!
//! Training remains on the differentiable dense-DFT path in `scirust-core`.
//! This module consumes an immutable parameter snapshot and replaces only the
//! inference-time Fourier transform/reconstruction with SciRust's O(N log N)
//! radix-2 FFT. It therefore provides an independently testable optimization
//! surface without changing the training semantics.

use crate::{LearnedOperator, NeuralOperatorError, Result, SpectralModePlan};
use scirust_core::nn::fno::Fno1dInferenceSnapshot;
use scirust_core::tensor::tensor_nd::TensorND;
use scirust_signal::{
    Complex,
    fft::{fft, fft_portable, ifft, ifft_portable},
};

/// FFT implementation used for FNO inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftInferenceMode {
    /// Platform-libm twiddles; intended as the faster CPU inference path.
    Fast,
    /// Portable twiddles; cross-platform bit-reproducible FFT reference path.
    Portable,
}

impl FftInferenceMode {
    fn forward(self, values: &mut [Complex]) {
        match self
        {
            Self::Fast => fft(values),
            Self::Portable => fft_portable(values),
        }
    }

    fn inverse(self, values: &mut [Complex]) {
        match self
        {
            Self::Fast => ifft(values),
            Self::Portable => ifft_portable(values),
        }
    }
}

/// Reusable allocation-stable CPU inference runtime for one FNO 1-D snapshot.
pub struct Fno1dFftInference {
    snapshot: Fno1dInferenceSnapshot,
    mode: FftInferenceMode,
    hidden: Vec<f32>,
    local: Vec<f32>,
    spectral: Vec<f32>,
    activated: Vec<f32>,
    input_spectrum: Vec<Complex>,
    output_spectrum: Vec<Complex>,
    full_plan: SpectralModePlan,
}

impl Fno1dFftInference {
    /// Construct and validate an FFT inference runtime from trained parameters.
    pub fn new(snapshot: Fno1dInferenceSnapshot, mode: FftInferenceMode) -> Result<Self> {
        validate_snapshot(&snapshot)?;
        if !snapshot.n.is_power_of_two()
        {
            return Err(NeuralOperatorError::NonRadix2 { len: snapshot.n });
        }
        let hidden_len = snapshot.n.saturating_mul(snapshot.width);
        let full_plan = SpectralModePlan::new(snapshot.modes, (0..snapshot.modes).collect())?;
        Ok(Self {
            hidden: vec![0.0; hidden_len],
            local: vec![0.0; hidden_len],
            spectral: vec![0.0; hidden_len],
            activated: vec![0.0; hidden_len],
            input_spectrum: vec![Complex::zero(); hidden_len],
            output_spectrum: vec![Complex::zero(); hidden_len],
            snapshot,
            mode,
            full_plan,
        })
    }

    /// FFT backend selected for this runtime.
    pub const fn mode(&self) -> FftInferenceMode {
        self.mode
    }

    /// Immutable trained parameter snapshot used by this runtime.
    pub const fn snapshot(&self) -> &Fno1dInferenceSnapshot {
        &self.snapshot
    }

    /// Predict while evaluating only the selected spectral modes.
    ///
    /// The local branch is always executed. Rejected Fourier modes are absent
    /// from complex channel mixing and remain zero in the inverse spectrum.
    pub fn predict_with_mode_plan(
        &mut self,
        input: &[f32],
        plan: &SpectralModePlan,
    ) -> Result<Vec<f32>> {
        self.validate_input(input)?;
        if plan.total_modes() != self.snapshot.modes
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FFT inference spectral plan modes",
                expected: self.snapshot.modes,
                got: plan.total_modes(),
            });
        }

        linear_into(
            input,
            self.snapshot.n,
            self.snapshot.in_channels(),
            self.snapshot.width,
            &self.snapshot.lift_weight,
            &self.snapshot.lift_bias,
            &mut self.hidden,
        );
        linear_into(
            &self.hidden,
            self.snapshot.n,
            self.snapshot.width,
            self.snapshot.width,
            &self.snapshot.local_weight,
            &self.snapshot.local_bias,
            &mut self.local,
        );

        self.forward_hidden_fft();
        self.mix_selected_modes(plan.active_modes());
        self.inverse_hidden_fft();

        for index in 0..self.activated.len()
        {
            self.activated[index] = (self.spectral[index] + self.local[index]).max(0.0);
        }

        let mut output = vec![0.0; self.output_len()];
        linear_into(
            &self.activated,
            self.snapshot.n,
            self.snapshot.width,
            self.snapshot.out_channels(),
            &self.snapshot.projection_weight,
            &self.snapshot.projection_bias,
            &mut output,
        );
        Ok(output)
    }

    fn validate_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.input_len()
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "FFT FNO input",
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
                what: "FFT FNO input",
                index,
            });
        }
        Ok(())
    }

    fn forward_hidden_fft(&mut self) {
        let n = self.snapshot.n;
        let width = self.snapshot.width;
        for channel in 0..width
        {
            let spectrum = &mut self.input_spectrum[channel * n..(channel + 1) * n];
            for (point, value) in spectrum.iter_mut().enumerate()
            {
                *value = Complex::new(self.hidden[point * width + channel] as f64, 0.0);
            }
            self.mode.forward(spectrum);
        }
    }

    fn mix_selected_modes(&mut self, selected_modes: &[usize]) {
        let n = self.snapshot.n;
        let width = self.snapshot.width;
        self.output_spectrum.fill(Complex::zero());
        let ar = &self.snapshot.spectral_real.data;
        let ai = &self.snapshot.spectral_imag.data;

        for &frequency in selected_modes
        {
            for out_channel in 0..width
            {
                let mut real = 0.0f64;
                let mut imag = 0.0f64;
                for in_channel in 0..width
                {
                    let input = self.input_spectrum[in_channel * n + frequency];
                    let weight_index = frequency * width * width + out_channel * width + in_channel;
                    let wr = ar[weight_index] as f64;
                    let wi = ai[weight_index] as f64;
                    real += wr * input.re - wi * input.im;
                    imag += wr * input.im + wi * input.re;
                }
                let mixed = Complex::new(real, imag);
                self.output_spectrum[out_channel * n + frequency] = mixed;
                if frequency != 0
                {
                    let conjugate = n - frequency;
                    self.output_spectrum[out_channel * n + conjugate] = mixed.conj();
                }
            }
        }
    }

    fn inverse_hidden_fft(&mut self) {
        let n = self.snapshot.n;
        let width = self.snapshot.width;
        for channel in 0..width
        {
            let spectrum = &mut self.output_spectrum[channel * n..(channel + 1) * n];
            self.mode.inverse(spectrum);
            for (point, value) in spectrum.iter().enumerate()
            {
                self.spectral[point * width + channel] = value.re as f32;
            }
        }
    }
}

impl LearnedOperator for Fno1dFftInference {
    fn input_len(&self) -> usize {
        self.snapshot.n * self.snapshot.in_channels()
    }

    fn output_len(&self) -> usize {
        self.snapshot.n * self.snapshot.out_channels()
    }

    fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let plan = self.full_plan.clone();
        self.predict_with_mode_plan(input, &plan)
    }
}

fn validate_snapshot(snapshot: &Fno1dInferenceSnapshot) -> Result<()> {
    if snapshot.n == 0
    {
        return Err(NeuralOperatorError::Empty {
            what: "FFT FNO grid",
        });
    }
    if snapshot.width == 0
    {
        return Err(NeuralOperatorError::InvalidChannels { channels: 0 });
    }
    let max_modes = snapshot.n / 2 + snapshot.n % 2;
    if snapshot.modes == 0 || snapshot.modes > max_modes
    {
        return Err(NeuralOperatorError::ShapeMismatch {
            what: "FFT FNO modes",
            expected: max_modes,
            got: snapshot.modes,
        });
    }
    validate_tensor_rank("FFT lift weight rank", &snapshot.lift_weight, 2)?;
    validate_tensor_rank("FFT projection weight rank", &snapshot.projection_weight, 2)?;
    let in_channels = snapshot.lift_weight.shape[0];
    let out_channels = snapshot.projection_weight.shape[1];
    validate_tensor_shape(
        "FFT lift weight",
        &snapshot.lift_weight,
        &[in_channels, snapshot.width],
    )?;
    validate_tensor_shape("FFT lift bias", &snapshot.lift_bias, &[1, snapshot.width])?;
    validate_tensor_shape(
        "FFT spectral real weights",
        &snapshot.spectral_real,
        &[snapshot.modes, snapshot.width, snapshot.width],
    )?;
    validate_tensor_shape(
        "FFT spectral imaginary weights",
        &snapshot.spectral_imag,
        &[snapshot.modes, snapshot.width, snapshot.width],
    )?;
    validate_tensor_shape(
        "FFT local weight",
        &snapshot.local_weight,
        &[snapshot.width, snapshot.width],
    )?;
    validate_tensor_shape("FFT local bias", &snapshot.local_bias, &[1, snapshot.width])?;
    validate_tensor_shape(
        "FFT projection weight",
        &snapshot.projection_weight,
        &[snapshot.width, out_channels],
    )?;
    validate_tensor_shape(
        "FFT projection bias",
        &snapshot.projection_bias,
        &[1, out_channels],
    )?;
    Ok(())
}

fn validate_tensor_rank(what: &'static str, tensor: &TensorND, expected_rank: usize) -> Result<()> {
    if tensor.shape.len() != expected_rank
    {
        return Err(NeuralOperatorError::ShapeMismatch {
            what,
            expected: expected_rank,
            got: tensor.shape.len(),
        });
    }
    Ok(())
}

fn validate_tensor_shape(what: &'static str, tensor: &TensorND, expected: &[usize]) -> Result<()> {
    if tensor.shape != expected
    {
        return Err(NeuralOperatorError::ShapeMismatch {
            what,
            expected: expected.iter().product(),
            got: tensor.numel(),
        });
    }
    if let Some((index, _)) = tensor
        .data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(NeuralOperatorError::NonFinite { what, index });
    }
    Ok(())
}

fn linear_into(
    input: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    weight: &TensorND,
    bias: &TensorND,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), rows * in_features);
    debug_assert_eq!(output.len(), rows * out_features);
    for row in 0..rows
    {
        for out_feature in 0..out_features
        {
            let mut value = bias.data[out_feature];
            for in_feature in 0..in_features
            {
                value += input[row * in_features + in_feature]
                    * weight.data[in_feature * out_features + out_feature];
            }
            output[row * out_features + out_feature] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fno1dConfig, Fno1dOperator, relative_l2};

    #[test]
    fn fft_inference_matches_dense_reference_path() {
        let cfg = Fno1dConfig {
            points: 16,
            in_channels: 2,
            out_channels: 2,
            hidden_channels: 5,
            modes: 6,
            seed: 77,
        };
        let mut dense = Fno1dOperator::new(cfg).unwrap();
        let input: Vec<f32> = (0..dense.input_len())
            .map(|index| ((index as f32) * 0.173 - 0.4).sin())
            .collect();
        let expected = dense.predict(&input).unwrap();
        let snapshot = dense.inference_snapshot();
        for mode in [FftInferenceMode::Fast, FftInferenceMode::Portable]
        {
            let mut runtime = Fno1dFftInference::new(snapshot.clone(), mode).unwrap();
            let observed = runtime.predict(&input).unwrap();
            let error = relative_l2(&observed, &expected).unwrap();
            assert!(error < 2e-5, "{mode:?} relative error {error}");
        }
    }

    #[test]
    fn fft_mode_plan_matches_dense_selected_mode_path() {
        let cfg = Fno1dConfig {
            points: 16,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 4,
            modes: 6,
            seed: 91,
        };
        let mut dense = Fno1dOperator::new(cfg).unwrap();
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.41).cos()).collect();
        let plan = SpectralModePlan::new(6, vec![0, 1, 3, 5]).unwrap();
        let expected = dense.predict_with_mode_plan(&input, &plan).unwrap();
        let mut runtime =
            Fno1dFftInference::new(dense.inference_snapshot(), FftInferenceMode::Portable).unwrap();
        let observed = runtime.predict_with_mode_plan(&input, &plan).unwrap();
        let error = relative_l2(&observed, &expected).unwrap();
        assert!(error < 2e-5, "selected-mode relative error {error}");
    }

    #[test]
    fn fft_inference_rejects_malformed_snapshot_ranks_without_panicking() {
        let dense = Fno1dOperator::new(Fno1dConfig {
            points: 16,
            in_channels: 2,
            out_channels: 3,
            hidden_channels: 4,
            modes: 6,
            seed: 19,
        })
        .unwrap();

        let mut missing_lift_rank = dense.inference_snapshot();
        missing_lift_rank.lift_weight.shape.clear();
        assert!(matches!(
            Fno1dFftInference::new(missing_lift_rank, FftInferenceMode::Fast),
            Err(NeuralOperatorError::ShapeMismatch {
                what: "FFT lift weight rank",
                expected: 2,
                got: 0,
            })
        ));

        let mut truncated_projection_rank = dense.inference_snapshot();
        truncated_projection_rank
            .projection_weight
            .shape
            .truncate(1);
        assert!(matches!(
            Fno1dFftInference::new(truncated_projection_rank, FftInferenceMode::Portable),
            Err(NeuralOperatorError::ShapeMismatch {
                what: "FFT projection weight rank",
                expected: 2,
                got: 1,
            })
        ));
    }

    #[test]
    fn fft_inference_rejects_non_radix2_grid() {
        let dense = Fno1dOperator::new(Fno1dConfig {
            points: 15,
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 4,
            modes: 6,
            seed: 5,
        })
        .unwrap();
        assert!(matches!(
            Fno1dFftInference::new(dense.inference_snapshot(), FftInferenceMode::Fast),
            Err(NeuralOperatorError::NonRadix2 { len: 15 })
        ));
    }
}
