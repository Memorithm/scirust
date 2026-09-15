//! Verified CPU convolution backend.
//!
//! This module provides the convolution backend that `scirust-core` can execute
//! directly. GPU matrix acceleration lives in `scirust-gpu` and is deliberately
//! not advertised as an interchangeable implementation of this trait.
//!
//! # Numerical safety
//!
//! Input samples and kernel coefficients are rejected when they contain `NaN`
//! or infinity. Products are accumulated in `f64`; a result that is non-finite
//! or outside the finite `f32` range is returned as [`BackendError::Overflow`]
//! before the output conversion.

use thiserror::Error;

/// Error returned by the convolution backend.
#[derive(Debug, Error)]
pub enum BackendError {
    /// A kernel or data element is not finite.
    #[error("NaN or Inf detected in input data (index {idx}, value {value:.3e})")]
    NanDetected { idx: usize, value: f32 },

    /// A convolution output cannot be represented as a finite `f32`.
    #[error("overflow detected during convolution at index {idx}: value {value:.3e})")]
    Overflow { idx: usize, value: f32 },

    /// The request violates an internal backend precondition.
    #[error("internal compute error: {0}")]
    Internal(String),
}

/// Result type used by convolution backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

fn check_finite_slice(data: &[f32], _label: &str) -> BackendResult<()> {
    for (i, &v) in data.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(BackendError::NanDetected { idx: i, value: v });
        }
    }
    Ok(())
}

/// Common interface for SciRust convolution backends.
///
/// [`execute_kernel`](ComputeBackend::execute_kernel) performs a centered,
/// one-dimensional, zero-padded convolution. The returned vector has exactly
/// the same length as `data`.
///
/// # Examples
///
/// ```
/// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
/// let backend = CpuFallback;
/// assert!(backend.is_available());
/// ```
///
/// ```
/// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
/// let backend = CpuFallback;
/// let output = backend.execute_kernel(&[1.0], &[2.0, 3.0]).unwrap();
/// assert_eq!(output, vec![2.0, 3.0]);
/// ```
pub trait ComputeBackend {
    /// Reports whether this backend can execute in the current process.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
    /// assert!(CpuFallback.is_available());
    /// ```
    ///
    /// ```
    /// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
    /// let backend = CpuFallback;
    /// assert_eq!(backend.is_available(), backend.is_available());
    /// ```
    fn is_available(&self) -> bool;

    /// Applies a centered one-dimensional kernel with zero padding.
    ///
    /// For an input of length `n`, the result also has length `n`. Samples whose
    /// kernel footprint falls outside the input are treated as zero. Accumulation
    /// is performed in `f64` and converted to `f32` only after range checking.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::NanDetected`] if either slice contains a non-finite
    /// value, [`BackendError::Internal`] when `kernel` is empty, and
    /// [`BackendError::Overflow`] when an output is not representable as finite
    /// `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
    /// let output = CpuFallback.execute_kernel(&[1.0], &[1.0, -2.0, 3.0]).unwrap();
    /// assert_eq!(output, vec![1.0, -2.0, 3.0]);
    /// ```
    ///
    /// ```
    /// use scirust_core::compute_backend::{BackendError, ComputeBackend, CpuFallback};
    /// let error = match CpuFallback.execute_kernel(&[1.0, f32::NAN], &[1.0]) {
    ///     Err(error) => error,
    ///     Ok(_) => panic!("non-finite kernel unexpectedly accepted"),
    /// };
    /// assert!(matches!(error, BackendError::NanDetected { .. }));
    /// ```
    fn execute_kernel(&self, kernel: &[f32], data: &[f32]) -> BackendResult<Vec<f32>>;
}

/// Always-available CPU implementation of [`ComputeBackend`].
///
/// # Examples
///
/// ```
/// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
/// let backend = CpuFallback;
/// assert!(backend.is_available());
/// ```
///
/// ```
/// use scirust_core::compute_backend::{ComputeBackend, CpuFallback};
/// assert_eq!(CpuFallback.execute_kernel(&[2.0], &[3.0]).unwrap(), vec![6.0]);
/// ```
pub struct CpuFallback;

impl ComputeBackend for CpuFallback {
    fn is_available(&self) -> bool {
        true
    }

    fn execute_kernel(&self, kernel: &[f32], data: &[f32]) -> BackendResult<Vec<f32>> {
        check_finite_slice(kernel, "kernel")?;
        check_finite_slice(data, "data")?;
        if kernel.is_empty()
        {
            return Err(BackendError::Internal("empty kernel".into()));
        }
        let mut out = vec![0.0f32; data.len()];
        let half_k = kernel.len() / 2;
        #[allow(clippy::needless_range_loop)]
        for i in 0..data.len()
        {
            let mut sum = 0.0f64;
            for (j, &k) in kernel.iter().enumerate()
            {
                let idx = i as isize + j as isize - half_k as isize;
                if idx >= 0 && (idx as usize) < data.len()
                {
                    sum += data[idx as usize] as f64 * k as f64;
                }
            }
            if !sum.is_finite() || sum.abs() > f32::MAX as f64
            {
                return Err(BackendError::Overflow { idx: i, value: sum as f32 });
            }
            out[i] = sum as f32;
        }
        Ok(out)
    }
}

/// Returns the convolution backend implemented directly by `scirust-core`.
///
/// This function currently returns [`CpuFallback`]. GPU backends expose a
/// different matrix-oriented API in `scirust-gpu` and are not silently selected
/// here.
///
/// # Errors
///
/// The current CPU implementation has no initialization failure, so this
/// function presently returns `Ok`.
///
/// # Examples
///
/// ```
/// use scirust_core::compute_backend::get_backend;
/// let backend = get_backend().unwrap();
/// assert!(backend.is_available());
/// ```
///
/// ```
/// use scirust_core::compute_backend::get_backend;
/// let backend = get_backend().unwrap();
/// assert_eq!(backend.execute_kernel(&[1.0], &[4.0, 5.0]).unwrap(), vec![4.0, 5.0]);
/// ```
pub fn get_backend() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(CpuFallback))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback() {
        let backend = CpuFallback;
        assert!(backend.is_available());
        let kernel = vec![1.0f32, 0.0, -1.0];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = backend.execute_kernel(&kernel, &data).unwrap();
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_get_backend() {
        let backend = get_backend().unwrap();
        assert!(backend.is_available());
    }

    #[test]
    fn test_nan_detected() {
        let backend = CpuFallback;
        let kernel = vec![1.0f32, f32::NAN];
        let data = vec![1.0, 2.0, 3.0];
        let result = backend.execute_kernel(&kernel, &data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BackendError::NanDetected { .. }));
    }
}
