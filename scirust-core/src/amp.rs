//! Automatic Mixed Precision (AMP) simulation utilities.
//!
//! This experimental module has no workspace consumers at the time of this
//! review. It operates on `f32` slices: FP16/BF16 conversions return `f32`
//! values rounded/truncated to the target format rather than allocating physical
//! 16-bit buffers. It does not own a model, run autodiff, or apply optimizer
//! updates.
//!
//! A separate trainer-oriented implementation lives in
//! [`crate::autodiff::mixed_precision`]. The two APIs have different scale-growth
//! semantics and must not be treated as interchangeable without an adapter.

/// Precision format simulated by [`MixedPrecisionEnv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixedPrecisionKind {
    /// IEEE-754 binary16, rounded through `half::f16`.
    FP16,
    /// BF16-like truncation retaining the upper 16 bits of an `f32`.
    BF16,
}

/// Loss-scaling strategy.
#[derive(Debug, Clone, Copy)]
pub enum LossScale {
    /// Fixed scale stored verbatim and never changed by `update_scale`.
    Fixed(f32),
    /// Dynamic scale initialized to 65536, grown after 2000 successful updates.
    Dynamic,
}

/// Experimental AMP helper state.
///
/// # Examples
///
/// ```
/// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
/// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(8.0));
/// assert_eq!(amp.current_scale, 8.0);
/// ```
///
/// ```
/// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
/// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Dynamic);
/// assert_eq!(amp.current_scale, 65_536.0);
/// ```
#[derive(Debug, Clone)]
pub struct MixedPrecisionEnv {
    /// Target precision simulation.
    pub kind: MixedPrecisionKind,
    /// Configured loss-scaling strategy.
    pub loss_scale: LossScale,
    /// Current effective loss scale.
    pub current_scale: f32,
    growth_steps: u32,
    growth_interval: u32,
    growth_factor: f32,
    backoff_factor: f32,
    max_scale: f32,
}

impl MixedPrecisionEnv {
    /// Creates an AMP helper environment.
    ///
    /// `LossScale::Fixed(s)` stores `s` verbatim; no finite/positive validation
    /// is currently performed. `Dynamic` starts at 2^16.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(1024.0));
    /// assert_eq!(amp.current_scale, 1024.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Dynamic);
    /// assert_eq!(amp.current_scale, 65_536.0);
    /// ```
    #[must_use]
    pub fn new(kind: MixedPrecisionKind, loss_scale: LossScale) -> Self {
        let init_scale = match loss_scale {
            LossScale::Fixed(s) => s,
            LossScale::Dynamic => 2.0f32.powi(16),
        };
        Self {
            kind,
            loss_scale,
            current_scale: init_scale,
            growth_steps: 0,
            growth_interval: 2000,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            max_scale: 2.0f32.powi(24),
        }
    }

    /// Returns a new `f32` vector whose values are quantized to the configured
    /// target precision.
    ///
    /// FP16 uses round-to-nearest-even through `half::f16`; finite magnitudes
    /// outside binary16 range become infinity. BF16 currently truncates the low
    /// 16 bits rather than performing BF16 round-to-nearest-even.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(1.0));
    /// assert_ne!(amp.cast_to(&[0.1])[0], 0.1);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(1.0));
    /// assert!(amp.cast_to(&[70_000.0])[0].is_infinite());
    /// ```
    #[must_use]
    pub fn cast_to(&self, data: &[f32]) -> Vec<f32> {
        match self.kind {
            MixedPrecisionKind::FP16 => data.iter().map(|&x| fp32_to_fp16(x)).collect(),
            MixedPrecisionKind::BF16 => data.iter().map(|&x| fp32_to_bf16(x)).collect(),
        }
    }

    /// Copies already-simulated target-precision values into a new `f32` vector.
    ///
    /// Because target values are represented as `f32`, this operation is
    /// numerically an identity copy; it cannot recover precision discarded by
    /// [`cast_to`](Self::cast_to).
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(1.0));
    /// let q = amp.cast_to(&[0.1]);
    /// assert_eq!(amp.cast_from(&q), q);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Fixed(1.0));
    /// assert_eq!(amp.cast_from(&[]), Vec::<f32>::new());
    /// ```
    #[must_use]
    pub fn cast_from(&self, data: &[f32]) -> Vec<f32> {
        match self.kind {
            MixedPrecisionKind::FP16 => data.iter().map(|&x| fp16_to_fp32(x)).collect(),
            MixedPrecisionKind::BF16 => data.iter().map(|&x| bf16_to_fp32(x)).collect(),
        }
    }

    /// Multiplies `loss` by the current scale.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(8.0));
    /// assert_eq!(amp.scale_loss(0.125), 1.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(0.0));
    /// assert_eq!(amp.scale_loss(2.0), 0.0);
    /// ```
    #[must_use]
    pub fn scale_loss(&self, loss: f32) -> f32 { loss * self.current_scale }

    /// Multiplies every gradient by `1 / current_scale`.
    ///
    /// No scale validation is performed. In particular, a zero fixed scale can
    /// therefore produce infinities or NaNs.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(8.0));
    /// assert_eq!(amp.unscale_gradients(&[8.0, 4.0]), vec![1.0, 0.5]);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(0.0));
    /// assert!(amp.unscale_gradients(&[1.0])[0].is_infinite());
    /// ```
    #[must_use]
    pub fn unscale_gradients(&self, grads: &[f32]) -> Vec<f32> {
        let inv_scale = 1.0 / self.current_scale;
        grads.iter().map(|&g| g * inv_scale).collect()
    }

    /// Returns `true` when any gradient is NaN or positive/negative infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::MixedPrecisionEnv;
    /// assert!(MixedPrecisionEnv::has_overflow(&[1.0, f32::INFINITY]));
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::MixedPrecisionEnv;
    /// assert!(!MixedPrecisionEnv::has_overflow(&[1.0, 2.0]));
    /// ```
    #[must_use]
    pub fn has_overflow(grads: &[f32]) -> bool {
        grads.iter().any(|&g| g.is_nan() || g.is_infinite())
    }

    /// Updates dynamic loss-scale state after one training step.
    ///
    /// Fixed mode never changes. In dynamic mode an overflow halves the scale
    /// and resets the consecutive-good-step counter; 2000 consecutive successful
    /// calls double the scale, capped at 2^24.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Dynamic);
    /// amp.update_scale(true);
    /// assert_eq!(amp.current_scale, 32_768.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(8.0));
    /// amp.update_scale(true);
    /// assert_eq!(amp.current_scale, 8.0);
    /// ```
    pub fn update_scale(&mut self, had_overflow: bool) {
        match self.loss_scale {
            LossScale::Fixed(_) => {},
            LossScale::Dynamic => {
                if had_overflow {
                    self.current_scale *= self.backoff_factor;
                    self.growth_steps = 0;
                } else {
                    self.growth_steps += 1;
                    if self.growth_steps >= self.growth_interval {
                        self.current_scale = (self.current_scale * self.growth_factor).min(self.max_scale);
                        self.growth_steps = 0;
                    }
                }
            },
        }
    }
}

/// Quantizes one `f32` through IEEE-754 binary16 and returns the rounded value as `f32`.
///
/// # Examples
///
/// ```
/// use scirust_core::amp::fp32_to_fp16;
/// assert_eq!(fp32_to_fp16(1.5), 1.5);
/// ```
///
/// ```
/// use scirust_core::amp::fp32_to_fp16;
/// assert!(fp32_to_fp16(70_000.0).is_infinite());
/// ```
#[inline]
#[must_use]
pub fn fp32_to_fp16(x: f32) -> f32 { half::f16::from_f32(x).to_f32() }

/// Returns an already binary16-rounded value unchanged as `f32`.
///
/// # Examples
///
/// ```
/// use scirust_core::amp::{fp16_to_fp32, fp32_to_fp16};
/// let q = fp32_to_fp16(0.1);
/// assert_eq!(fp16_to_fp32(q), q);
/// ```
///
/// ```
/// use scirust_core::amp::fp16_to_fp32;
/// assert!(fp16_to_fp32(f32::INFINITY).is_infinite());
/// ```
#[inline]
#[must_use]
pub fn fp16_to_fp32(x: f32) -> f32 { x }

/// Truncates an `f32` to a BF16-like value by clearing its low 16 bits.
///
/// This is truncation, not BF16 round-to-nearest-even.
///
/// # Examples
///
/// ```
/// use scirust_core::amp::fp32_to_bf16;
/// assert_eq!(fp32_to_bf16(1.0), 1.0);
/// ```
///
/// ```
/// use scirust_core::amp::fp32_to_bf16;
/// assert_ne!(fp32_to_bf16(1.2345679), 1.2345679);
/// ```
#[inline]
#[must_use]
pub fn fp32_to_bf16(x: f32) -> f32 { f32::from_bits(x.to_bits() & 0xFFFF_0000) }

/// Returns an already BF16-truncated value unchanged as `f32`.
///
/// # Examples
///
/// ```
/// use scirust_core::amp::{bf16_to_fp32, fp32_to_bf16};
/// let q = fp32_to_bf16(0.1);
/// assert_eq!(bf16_to_fp32(q), q);
/// ```
///
/// ```
/// use scirust_core::amp::bf16_to_fp32;
/// assert_eq!(bf16_to_fp32(-2.0), -2.0);
/// ```
#[inline]
#[must_use]
pub fn bf16_to_fp32(x: f32) -> f32 { x }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion() {
        let x = 1.234_567_9f32;
        let half = fp32_to_fp16(x);
        assert!((fp16_to_fp32(half) - x).abs() > 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        let x = 1.234_567_9f32;
        let half = fp32_to_bf16(x);
        assert!((bf16_to_fp32(half) - x).abs() > 1e-5);
    }

    #[test]
    fn test_overflow_detection() {
        assert!(MixedPrecisionEnv::has_overflow(&[f32::NAN, 1.0]));
        assert!(MixedPrecisionEnv::has_overflow(&[1.0, f32::INFINITY]));
        assert!(!MixedPrecisionEnv::has_overflow(&[1.0, 2.0]));
    }

    #[test]
    fn fp32_to_fp16_satur