//! Mixed-precision training support with FP32 master weights and FP16-rounded
//! forward weights.
//!
//! This module stores values in the existing `Tensor<f32>` representation. The
//! `fp16_weights` buffers therefore contain values rounded through IEEE binary16
//! and converted back to `f32`; they are not physically stored as 16-bit tensor
//! elements. An external optimizer remains responsible for applying gradients to
//! `master_weights`.

use crate::autodiff::reverse::Tensor;
use crate::error::{Result, SciRustError};

/// Maintains FP32 master parameters, FP16-rounded forward parameters, and a
/// dynamic loss-scale value.
///
/// `after_backward` validates/unscales gradients for overflow detection but does
/// not apply them to `master_weights`. `maybe_grow_scale` uses the total number
/// of backward calls, including overflowed calls; it is therefore not the
/// textbook "consecutive successful steps" policy unless the caller adds that
/// policy externally.
///
/// # Examples
///
/// ```
/// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
/// let params = vec![Tensor::from_vec(vec![0.1, 0.2], 1, 2)];
/// let trainer = MixedPrecisionTrainer::new(&params, 1024.0);
/// assert_eq!(trainer.master_weights[0].data, params[0].data);
/// assert_eq!(trainer.loss_scale, 1024.0);
/// ```
///
/// The forward copy is binary16-rounded while the master remains FP32:
///
/// ```
/// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
/// let params = vec![Tensor::from_vec(vec![0.1], 1, 1)];
/// let trainer = MixedPrecisionTrainer::new(&params, 1.0);
/// assert_eq!(trainer.master_weights[0].data[0], 0.1);
/// assert_ne!(trainer.fp16_weights[0].data[0], 0.1);
/// ```
pub struct MixedPrecisionTrainer {
    /// FP32 source-of-truth parameters updated by the external optimizer.
    pub master_weights: Vec<Tensor>,
    /// Forward parameters rounded through binary16 and stored back in `f32` tensors.
    pub fp16_weights: Vec<Tensor>,
    /// Current dynamic loss scale.
    pub loss_scale: f32,
    scale_growth_factor: f32,
    scale_backoff_factor: f32,
    growth_interval: usize,
    step_counter: usize,
    max_scale: f32,
}

impl MixedPrecisionTrainer {
    /// Creates a trainer by cloning the FP32 parameters and constructing their
    /// FP16-rounded forward copies.
    ///
    /// `initial_scale` is stored verbatim. This constructor currently performs
    /// no positivity/finite-value validation, so callers should supply a finite,
    /// strictly positive scale before using [`after_backward`](Self::after_backward).
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let params = vec![Tensor::zeros(2, 3)];
    /// let trainer = MixedPrecisionTrainer::new(&params, 512.0);
    /// assert_eq!(trainer.master_weights.len(), 1);
    /// assert_eq!(trainer.fp16_weights[0].data.len(), 6);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let params = vec![Tensor::from_vec(vec![1.0001], 1, 1)];
    /// let trainer = MixedPrecisionTrainer::new(&params, 2.0);
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

    /// Refreshes all FP16-rounded forward weights from the current FP32 masters.
    ///
    /// The master tensors are not modified. Call this after an external optimizer
    /// updates `master_weights` and before a forward pass that consumes
    /// `fp16_weights`.
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

    /// Unscales gradients and updates overflow/loss-scale bookkeeping.
    ///
    /// Each call increments the internal total-step counter. Gradients are
    /// multiplied by `1 / loss_scale` in `f32`. A finite source gradient that is
    /// outside binary16's finite range is deliberately treated as an overflow,
    /// matching the assumption that it originated from an FP16 pass.
    ///
    /// The returned unscaled gradient buffers are not exposed and no optimizer
    /// update is performed by this method.
    ///
    /// # Errors
    ///
    /// Returns [`SciRustError::GradientOverflow`] when an unscaled gradient is
    /// `NaN`/infinite or the source finite gradient lies outside binary16's
    /// finite range. Before returning, `loss_scale` is multiplied by the 0.5
    /// backoff factor; the error contains that new scale.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 2)], 8.0);
    /// let scale = trainer.after_backward(&[Tensor::from_vec(vec![2.0, 4.0], 1, 2)]).unwrap();
    /// assert_eq!(scale, 8.0);
    /// assert_eq!(trainer.loss_scale, 8.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::{autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor}, error::SciRustError};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::zeros(1, 1)], 8.0);
    /// let result = trainer.after_backward(&[Tensor::from_vec(vec![f32::NAN], 1, 1)]);
    /// assert!(matches!(result, Err(SciRustError::GradientOverflow { loss_scale }) if loss_scale == 4.0));
    /// assert_eq!(trainer.loss_scale, 4.0);
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

    /// Grows the loss scale on every `growth_interval`-th total backward call.
    ///
    /// Growth multiplies the scale by 2 and caps it at 65536. The internal
    /// counter is not reset by overflow; callers requiring growth after
    /// consecutive overflow-free steps must enforce that policy externally.
    /// Calling this method before any backward pass sees step zero, which is a
    /// multiple of the interval and therefore grows the scale immediately.
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
    /// After one backward call the default interval is not reached:
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

    /// Rebuilds `fp16_weights` by binary16-rounding the corresponding masters.
    ///
    /// The method updates pairs produced by `zip`; if callers mutate the two
    /// public vectors to different lengths, unmatched entries are left unchanged.
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
    /// The master remains the FP32 source of truth:
    ///
    /// ```
    /// use scirust_core::autodiff::{mixed_precision::MixedPrecisionTrainer, reverse::Tensor};
    /// let mut trainer = MixedPrecisionTrainer::new(&[Tensor::from_vec(vec![0.1], 1, 1)], 1.0);
    /// let before = trainer.master_weights[0].data.clone();
    /// trainer.update_fp16_from_master();
    /// assert_eq!(trainer.master_weights[0].data, before);
    /// assert_ne!(trainer.fp16_weights[0].data, before);
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
