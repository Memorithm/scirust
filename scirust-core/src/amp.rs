//! Experimental Automatic Mixed Precision (AMP) simulation helpers.
//!
//! Values remain stored as `f32`: FP16/BF16 operations return `f32` values
//! quantized to the target format. This module does not allocate physical 16-bit
//! tensors, own a model, run autodiff, or perform optimizer updates. The separate
//! [`crate::autodiff::mixed_precision`] trainer has different scale semantics.

/// Precision format simulated by [`MixedPrecisionEnv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixedPrecisionKind { FP16, BF16 }

/// Loss-scaling strategy.
#[derive(Debug, Clone, Copy)]
pub enum LossScale { Fixed(f32), Dynamic }

/// AMP conversion and loss-scaling state.
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
    /// Configured scale strategy.
    pub loss_scale: LossScale,
    /// Current effective scale.
    pub current_scale: f32,
    growth_steps: u32,
    growth_interval: u32,
    growth_factor: f32,
    backoff_factor: f32,
    max_scale: f32,
}

impl MixedPrecisionEnv {
    /// Creates an environment. Fixed scales are stored verbatim; `Dynamic`
    /// starts at 2^16. Fixed values are not validated for positivity/finiteness.
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// assert_eq!(MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(2.0)).current_scale, 2.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// assert_eq!(MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Dynamic).current_scale, 65_536.0);
    /// ```
    #[must_use]
    pub fn new(kind: MixedPrecisionKind, loss_scale: LossScale) -> Self {
        let current_scale = match loss_scale { LossScale::Fixed(s) => s, LossScale::Dynamic => 65_536.0 };
        Self { kind, loss_scale, current_scale, growth_steps: 0, growth_interval: 2000,
            growth_factor: 2.0, backoff_factor: 0.5, max_scale: 16_777_216.0 }
    }

    /// Quantizes `data` to the configured precision and returns `f32` storage.
    /// FP16 uses binary16 round-to-nearest-even; BF16 clears the low 16 bits.
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
        match self.kind { MixedPrecisionKind::FP16 => data.iter().map(|&x| fp32_to_fp16(x)).collect(),
            MixedPrecisionKind::BF16 => data.iter().map(|&x| fp32_to_bf16(x)).collect() }
    }

    /// Copies already-quantized values back into a new `f32` vector. This is
    /// numerically an identity and cannot restore discarded precision.
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(1.0));
    /// let q = amp.cast_to(&[0.1]); assert_eq!(amp.cast_from(&q), q);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Fixed(1.0));
    /// assert!(amp.cast_from(&[]).is_empty());
    /// ```
    #[must_use]
    pub fn cast_from(&self, data: &[f32]) -> Vec<f32> { data.to_vec() }

    /// Multiplies a loss by the current scale.
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

    /// Multiplies gradients by `1/current_scale`. A zero fixed scale can produce
    /// infinities or NaNs because scale validation is not performed.
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
        let inv = 1.0 / self.current_scale; grads.iter().map(|&g| g * inv).collect()
    }

    /// Reports whether any gradient is NaN or infinite.
    ///
    /// ```
    /// use scirust_core::amp::MixedPrecisionEnv;
    /// assert!(MixedPrecisionEnv::has_overflow(&[f32::NAN]));
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::MixedPrecisionEnv;
    /// assert!(!MixedPrecisionEnv::has_overflow(&[1.0, 2.0]));
    /// ```
    #[must_use]
    pub fn has_overflow(grads: &[f32]) -> bool { grads.iter().any(|x| !x.is_finite()) }

    /// Updates scale bookkeeping. Fixed mode is unchanged. Dynamic overflow
    /// halves the scale and resets the good-step counter; 2000 consecutive good
    /// calls double it, capped at 2^24.
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Dynamic);
    /// amp.update_scale(true); assert_eq!(amp.current_scale, 32_768.0);
    /// ```
    ///
    /// ```
    /// use scirust_core::amp::{LossScale, MixedPrecisionEnv, MixedPrecisionKind};
    /// let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(8.0));
    /// amp.update_scale(true); assert_eq!(amp.current_scale, 8.0);
    /// ```
    pub fn update_scale(&mut self, overflow: bool) {
        if let LossScale::Dynamic = self.loss_scale {
            if overflow { self.current_scale *= self.backoff_factor; self.growth_steps = 0; }
            else { self.growth_steps += 1; if self.growth_steps >= self.growth_interval {
                self.current_scale = (self.current_scale * self.growth_factor).min(self.max_scale); self.growth_steps = 0; } }
        }
    }
}

/// Quantizes one value through IEEE-754 binary16 and returns it as `f32`.
///
/// ```
/// use scirust_core::amp::fp32_to_fp16; assert_eq!(fp32_to_fp16(1.5), 1.5);
/// ```
///
/// ```
/// use scirust_core::amp::fp32_to_fp16; assert!(fp32_to_fp16(70_000.0).is_infinite());
/// ```
#[inline] #[must_use]
pub fn fp32_to_fp16(x: f32) -> f32 { half::f16::from_f32(x).to_f32() }

/// Returns an already FP16-rounded `f32` value unchanged.
///
/// ```
/// use scirust_core::amp::{fp16_to_fp32, fp32_to_fp16}; let q=fp32_to_fp16(0.1); assert_eq!(fp16_to_fp32(q),q);
/// ```
///
/// ```
/// use scirust_core::amp::fp16_to_fp32; assert!(fp16_to_fp32(f32::INFINITY).is_infinite());
/// ```
#[inline] #[must_use]
pub fn fp16_to_fp32(x: f32) -> f32 { x }

/// Produces a BF16-like value by clearing the low 16 bits of an `f32`.
/// This is truncation, not BF16 round-to-nearest-even.
///
/// ```
/// use scirust_core::amp::fp32_to_bf16; assert_eq!(fp32_to_bf16(1.0),1.0);
/// ```
///
/// ```
/// use scirust_core::amp::fp32_to_bf16; assert_ne!(fp32_to_bf16(1.2345679),1.2345679);
/// ```
#[inline] #[must_use]
pub fn fp32_to_bf16(x: f32) -> f32 { f32::from_bits(x.to_bits() & 0xFFFF_0000) }

/// Returns an already BF16-truncated `f32` value unchanged.
///
/// ```
/// use scirust_core::amp::{bf16_to_fp32,fp32_to_bf16}; let q=fp32_to_bf16(0.1); assert_eq!(bf16_to_fp32(q),q);
/// ```
///
/// ```
/// use scirust_core::amp::bf16_to_fp32; assert_eq!(bf16_to_fp32(-2.0),-2.0);
/// ```
#[inline] #[must_use]
pub fn bf16_to_fp32(x: f32) -> f32 { x }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn fp16_overflow_is_observable(){ assert!(fp32_to_fp16(70_000.0).is_infinite()); }
    #[test] fn dynamic_backoff(){ let mut a=MixedPrecisionEnv::new(MixedPrecisionKind::FP16,LossScale::Dynamic); a.update_scale(true); assert_eq!(a.current_scale,32_768.0); }
    #[test] fn fixed_scale_is_stable(){ let mut a=MixedPrecisionEnv::new(MixedPrecisionKind::FP16,LossScale::Fixed(8.0)); a.update_scale(true); assert_eq!(a.current_scale,8.0); }
}
