//! Mixed-precision training bookkeeping using FP32 master weights and
//! binary16-rounded forward weights.
//!
//! `Tensor` stores `f32`, so `fp16_weights` are values rounded through IEEE
//! binary16 and converted back to `f32`; this module does not provide physical
//! 16-bit tensor storage. Gradient application remains the responsibility of an
//! external optimizer.

use crate::autodiff::reverse::Tensor;
use crate::error::{Result, SciRustError};

/// Tracks FP32 master parameters, FP16-rounded forward copies, and loss scaling.
///
/// # Examples
///
/// ```
/// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
/// let params = vec![Tensor::from_vec(vec![0.1], 1, 1)];
/// let trainer = MixedPrecisionTrainer::new(&params, 1024.0);
/// assert_eq!(trainer.master_weights[0].data[0], 0.1);
/// assert_eq!(trainer.loss_scale, 1024.0);
/// ```
///
/// ```
/// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
/// let params = vec![Tensor::from_vec(vec![0.1], 1, 1)];
/// let trainer = MixedPrecisionTrainer::new(&params, 1.0);
/// assert_ne!(trainer.fp16_weights[0].data[0], trainer.master_weights[0].data[0]);
/// ```
pub struct MixedPrecisionTrainer {
    /// FP32 source-of-truth parameters updated by the external optimizer.
    pub master_weights: Vec<Tensor>,
    /// Forward parameters rounded through binary16, represented in `f32` tensors.
    pub fp16_weights: Vec<Tensor>,
    /// Current loss-scale value.
    pub loss_scale: f32,
    scale_growth_factor: f32,
    scale_backoff_factor: f32,
    growth_interval: usize,
    step_counter: usize,
    max_scale: f32,
}

impl MixedPrecisionTrainer {
    /// Creates master and FP16-rounded copies of `model_params`.
    ///
    /// `initial_scale` is stored verbatim; this constructor currently does not
    /// validate that it is finite and positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(2, 3)], 512.0);
    /// assert_eq!(trainer.master_weights.len(), 1);
    /// assert_eq!(trainer.fp16_weights[0].data.len(), 6);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let trainer = MixedPrecisionTrainer::new(&[Tensor::from_vec(vec![1.0001], 1, 1)], 2.0);
    /// assert_eq!(trainer.master_weights[0].data[0], 1.0001);
    /// assert_ne!(trainer.fp16_weights[0].data[0], 1.0001);
    /// ```
    #[must_use]
    pub fn new(model_params: &[Tensor], initial_scale: f32) -> Self {
        let master_weights = model_params.to_vec();
        let fp16_weights = master_weights.iter().map(cast_to_fp16).collect();
        Self {
            master_weights,
            fp16_weights,
            loss_scale: initial_scale,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            growth_interval: 2000,
            step_counter: 0,
            max_scale: 65536.0,
        }
    }

    /// Refreshes forward weights from the current FP32 masters without changing
    /// the master copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::from_vec(vec![0.1], 1, 1)], 1.0);
    /// let master = trainer.master_weights[0].data.clone();
    /// trainer.before_forward();
    /// assert_eq!(trainer.master_weights[0].data, master);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 1)], 1.0);
    /// trainer.master_weights[0].data[0] = 1.5;
    /// trainer.before_forward();
    /// assert_eq!(trainer.fp16_weights[0].data[0], 1.5);
    /// ```
    pub fn before_forward(&mut self) {
        self.update_fp16_from_master();
    }

    /// Unscales gradients for overflow detection and updates loss-scale state.
    ///
    /// Every call increments the internal total-step counter. This method does
    /// not expose the unscaled gradient buffers and does not update
    /// `master_weights`.
    ///
    /// # Errors
    ///
    /// Returns [`SciRustError::GradientOverflow`] when an unscaled gradient is
    /// non-finite or when a finite source gradient is outside binary16's finite
    /// range. The loss scale is multiplied by `0.5` before that error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 2)], 8.0);
    /// assert_eq!(trainer.after_backward(&[Tensor::from_vec(vec![2.0, 4.0], 1, 2)]).unwrap(), 8.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::{autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor}, error::SciRustError};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 1)], 8.0);
    /// let result = trainer.after_backward(&[Tensor::from_vec(vec![f32::NAN], 1, 1)]);
    /// assert!(matches!(result, Err(SciRustError::GradientOverflow { loss_scale }) if loss_scale == 4.0));
    /// ```
    pub fn after_backward(&mut self, grads: &[Tensor]) -> Result<f32> {
        self.step_counter += 1;
        let mut any_overflow = false;
        let mut scaled_grads = Vec::with_capacity(grads.len());

        for grad in grads
        {
            let unscaled = unscale_tensor(grad, 1.0 / self.loss_scale);
            if has_nan_or_inf(&unscaled)
            {
                any_overflow = true;
                break;
            }
            scaled_grads.push(unscaled);
        }

        if any_overflow
        {
            self.loss_scale *= self.scale_backoff_factor;
            return Err(SciRustError::GradientOverflow {
                loss_scale: self.loss_scale,
            });
        }

        Ok(self.loss_scale)
    }

    /// Grows the scale every `growth_interval`-th total backward call, capped at
    /// 65536.
    ///
    /// The counter is not reset on overflow. Also, step zero is a multiple of
    /// the interval, so calling this before any backward pass grows the scale.
    /// This differs from the common policy based on consecutive successful steps.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 1)], 2.0);
    /// trainer.maybe_grow_scale();
    /// assert_eq!(trainer.loss_scale, 4.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 1)], 2.0);
    /// trainer.after_backward(&[Tensor::zeros(1, 1)]).unwrap();
    /// trainer.maybe_grow_scale();
    /// assert_eq!(trainer.loss_scale, 2.0);
    /// ```
    pub fn maybe_grow_scale(&mut self) {
        if self.step_counter.is_multiple_of(self.growth_interval)
        {
            self.loss_scale = (self.loss_scale * self.scale_growth_factor).min(self.max_scale);
        }
    }

    /// Rebuilds each paired forward tensor from its FP32 master.
    ///
    /// Because the two vectors are public, callers can make their lengths differ;
    /// this method updates only the pairs produced by `zip`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 1)], 1.0);
    /// trainer.master_weights[0].data[0] = 2.25;
    /// trainer.update_fp16_from_master();
    /// assert_eq!(trainer.fp16_weights[0].data[0], 2.25);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::from_vec(vec![0.1], 1, 1)], 1.0);
    /// let master = trainer.master_weights[0].data.clone();
    /// trainer.update_fp16_from_master();
    /// assert_eq!(trainer.master_weights[0].data, master);
    /// assert_ne!(trainer.fp16_weights[0].data, master);
    /// ```
    pub fn update_fp16_from_master(&mut self) {
        for (fp16, master) in self.fp16_weights.iter_mut().zip(&self.master_weights)
        {
            *fp16 = cast_to_fp16(master);
        }
    }
}

fn cast_to_fp16(t: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(t.rows, t.cols);
    for (dst, &src) in out.data.iter_mut().zip(&t.data)
    {
        let half = half::f16::from_f32(src);
        *dst = half.to_f32();
    }
    out
}

fn unscale_tensor(t: &Tensor, scale: f32) -> Tensor {
    let mut out = Tensor::zeros(t.rows, t.cols);
    for (dst, &src) in out.data.iter_mut().zip(&t.data)
    {
        let f16_overflow = src.is_finite() && half::f16::from_f32(src).is_infinite();
        *dst = if f16_overflow
        {
            f32::INFINITY
        }
        else
        {
            src * scale
        };
    }
    out
}

fn has_nan_or_inf(t: &Tensor) -> bool {
    t.data.iter().any(|&x| x.is_nan() || x.is_infinite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let params = vec![Tensor::zeros(10, 20), Tensor::zeros(5, 5)];
        let trainer = MixedPrecisionTrainer::new(&params, 1.0);
        assert_eq!(trainer.master_weights.len(), 2);
        assert_eq!(trainer.fp16_weights.len(), 2);
    }

    #[test]
    fn test_loss_scale_backoff() {
        let params = vec![Tensor::zeros(10, 20)];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        let grads = vec![Tensor {
            data: vec![f32::NAN; 200],
            rows: 10,
            cols: 20,
        }];
        let result = trainer.after_backward(&grads);
        assert!(result.is_err(), "Should fail on NaN");
        assert!(trainer.loss_scale < 1.0, "Loss scale should decrease");
    }

    #[test]
    fn test_before_forward_preserves_master_precision() {
        let master = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], 2, 2);
        let params = vec![master.clone()];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        trainer.before_forward();
        assert_eq!(trainer.master_weights[0].data, master.data);
        let expected_fp16: Vec<f32> = master
            .data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect();
        assert_eq!(trainer.fp16_weights[0].data, expected_fp16);
        assert_ne!(trainer.fp16_weights[0].data, master.data);
    }

    #[test]
    fn test_before_forward_tracks_master_updates() {
        let params = vec![Tensor::from_vec(vec![0.1, 0.2], 1, 2)];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        trainer.before_forward();
        trainer.master_weights[0].data = vec![1.5, 2.5];
        trainer.before_forward();
        let expected: Vec<f32> = [1.5_f32, 2.5]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect();
        assert_eq!(trainer.fp16_weights[0].data, expected);
        assert_eq!(trainer.master_weights[0].data, vec![1.5, 2.5]);
    }

    #[test]
    fn test_scale_growth() {
        let params = vec![Tensor::zeros(10, 20)];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        trainer.growth_interval = 5;
        for _ in 0..5
        {
            let grads = vec![Tensor::zeros(10, 20)];
            let _ = trainer.after_backward(&grads);
        }
        trainer.maybe_grow_scale();
        assert!(trainer.loss_scale > 1.0, "Loss scale should grow");
    }
}
