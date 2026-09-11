//! Spectral surrogate generators for controlled time-series null models.
//!
//! A phase-randomized surrogate preserves the discrete Fourier magnitudes of a
//! real signal while replacing the non-DC/non-Nyquist phases with deterministic
//! pseudo-random phases.  It is useful when an experiment needs a null that
//! preserves second-order spectral structure but destroys the original phase
//! relationships and much of the higher-order temporal organization.
//!
//! This module deliberately makes no claim that a phase-randomized surrogate is
//! a complete null for every scientific question.  It is one controlled null
//! whose preserved and destroyed structure are explicit.

use core::f64::consts::{PI, TAU};
use core::fmt;

use scirust_stats::SplitMix64;

use crate::complex::Complex;
use crate::fft::{fft_portable, ifft_portable};

/// Errors returned by [`phase_randomized_surrogate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurrogateError {
    /// The input is too short to contain a non-trivial conjugate frequency pair.
    TooShort {
        /// Observed input length.
        len: usize,
    },
    /// The radix-2 portable FFT requires a power-of-two input length.
    LengthNotPowerOfTwo {
        /// Observed input length.
        len: usize,
    },
    /// At least one input sample is NaN or infinite.
    NonFiniteSample {
        /// Index of the first non-finite sample.
        index: usize,
    },
}

impl fmt::Display for SurrogateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooShort { len } => write!(
                f,
                "surrogate input must contain at least 4 samples, got {len}"
            ),
            Self::LengthNotPowerOfTwo { len } =>
            {
                write!(
                    f,
                    "surrogate input length must be a power of two, got {len}"
                )
            },
            Self::NonFiniteSample { index } =>
            {
                write!(
                    f,
                    "surrogate input contains a non-finite sample at index {index}"
                )
            },
        }
    }
}

impl std::error::Error for SurrogateError {}

/// Construct a deterministic phase-randomized surrogate of a real-valued signal.
///
/// The input length must be a power of two and at least four samples.  The
/// routine computes SciRust's portable FFT, preserves the DC and Nyquist bins,
/// replaces every positive-frequency phase with a phase drawn from a seeded
/// [`SplitMix64`], mirrors the corresponding negative-frequency bin by complex
/// conjugation, and reconstructs the real signal with the portable inverse FFT.
///
/// Therefore the target null preserves the Fourier magnitudes (up to floating
/// roundoff from the inverse/forward transforms) while randomizing phase
/// relationships.  The same `seed` and input produce the same result.  A
/// different seed is not guaranteed to change signals whose randomized bins all
/// have zero magnitude.
///
/// # Errors
///
/// Returns [`SurrogateError`] when the input is shorter than four samples, its
/// length is not a power of two, or any sample is non-finite.
pub fn phase_randomized_surrogate(signal: &[f64], seed: u64) -> Result<Vec<f64>, SurrogateError> {
    let n = signal.len();
    if n < 4
    {
        return Err(SurrogateError::TooShort { len: n });
    }
    if !n.is_power_of_two()
    {
        return Err(SurrogateError::LengthNotPowerOfTwo { len: n });
    }
    if let Some((index, _)) = signal
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SurrogateError::NonFiniteSample { index });
    }

    let mut spectrum: Vec<Complex> = signal
        .iter()
        .copied()
        .map(|value| Complex::new(value, 0.0))
        .collect();
    fft_portable(&mut spectrum);

    let mut rng = SplitMix64::new(seed);
    let nyquist = n / 2;

    for k in 1..nyquist
    {
        let magnitude = spectrum[k].mag();
        let phase = TAU * rng.next_f64() - PI;
        let (sin_phase, cos_phase) = scirust_core::portable_f32::sincos_small_f64(phase);
        let randomized = Complex::new(magnitude * cos_phase, magnitude * sin_phase);
        spectrum[k] = randomized;
        spectrum[n - k] = randomized.conj();
    }

    ifft_portable(&mut spectrum);
    Ok(spectrum.into_iter().map(|value| value.re).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spectrum_magnitude_squared(signal: &[f64]) -> Vec<f64> {
        let mut spectrum: Vec<Complex> = signal
            .iter()
            .copied()
            .map(|value| Complex::new(value, 0.0))
            .collect();
        fft_portable(&mut spectrum);
        spectrum.into_iter().map(|value| value.mag_sq()).collect()
    }

    fn mean(signal: &[f64]) -> f64 {
        signal.iter().sum::<f64>() / signal.len() as f64
    }

    #[test]
    fn same_seed_is_bit_reproducible() {
        let signal: Vec<f64> = (0..64)
            .map(|i| {
                let x = i as f64;
                (0.17 * x).sin() + 0.35 * (0.43 * x).cos() + 0.01 * x
            })
            .collect();
        let a = phase_randomized_surrogate(&signal, 42).unwrap();
        let b = phase_randomized_surrogate(&signal, 42).unwrap();
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn different_seeds_change_nontrivial_signal() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (0.21 * i as f64).sin() + 0.4 * (0.37 * i as f64).cos())
            .collect();
        let a = phase_randomized_surrogate(&signal, 1).unwrap();
        let b = phase_randomized_surrogate(&signal, 2).unwrap();
        assert!(
            a.iter()
                .zip(&b)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        );
    }

    #[test]
    fn preserves_fourier_magnitudes_and_mean() {
        let signal: Vec<f64> = (0..128)
            .map(|i| {
                let x = i as f64;
                2.5 + (0.09 * x).sin() + 0.2 * (0.31 * x).cos() + 0.03 * (0.71 * x).sin()
            })
            .collect();
        let surrogate = phase_randomized_surrogate(&signal, 0x5eed).unwrap();
        let before = spectrum_magnitude_squared(&signal);
        let after = spectrum_magnitude_squared(&surrogate);

        for (index, (expected, observed)) in before.iter().zip(&after).enumerate()
        {
            let scale = expected.abs().max(1.0);
            assert!(
                (expected - observed).abs() <= 2.0e-10 * scale,
                "bin {index}: expected {expected}, observed {observed}"
            );
        }
        assert!((mean(&signal) - mean(&surrogate)).abs() <= 1.0e-12);
    }

    #[test]
    fn constant_signal_remains_constant_within_roundoff() {
        let signal = vec![3.25; 32];
        let surrogate = phase_randomized_surrogate(&signal, 99).unwrap();
        for value in surrogate
        {
            assert!((value - 3.25).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn rejects_malformed_inputs() {
        assert_eq!(
            phase_randomized_surrogate(&[1.0, 2.0], 0),
            Err(SurrogateError::TooShort { len: 2 })
        );
        assert_eq!(
            phase_randomized_surrogate(&[0.0; 6], 0),
            Err(SurrogateError::LengthNotPowerOfTwo { len: 6 })
        );
        let mut non_finite = [0.0; 8];
        non_finite[3] = f64::NAN;
        assert_eq!(
            phase_randomized_surrogate(&non_finite, 0),
            Err(SurrogateError::NonFiniteSample { index: 3 })
        );
    }
}
