//! Bounded partial Walsh projection of multinomial residuals over physical
//! u32 bit contexts, for SciRust signal research and SML-GENIUS A6-E5.
//!
//! The existing scirust-modalg::boolean module already provides exact whole
//! truth-table Walsh-Hadamard and GF(2) Mobius. This complementary numerical
//! primitive stores ONLY requested masks: no 2^32 table and no extra FFT.
//! The caller must provide out-of-fold model probabilities from TRAIN only.

use std::collections::BTreeSet;

pub const MAX_PARTIAL_WALSH_MASKS: usize = 96;
pub const MAX_PARTIAL_WALSH_CLASSES: usize = 256;
pub const MAX_PARTIAL_WALSH_ROWS: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialWalshError {
    InvalidBudget,
    DuplicateMask,
    InvalidDistribution,
    InvalidLabel,
    Full,
    Empty,
}

impl std::fmt::Display for PartialWalshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PartialWalshError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalshProjection {
    pub mask: u32,
    pub class_residual_norm: f64,
}

#[derive(Clone, Debug)]
pub struct PartialWalsh {
    masks: Vec<u32>,
    classes: usize,
    max_rows: usize,
    rows: usize,
    sum: Vec<f64>,
}

impl PartialWalsh {
    pub fn new(
        masks: &[u32],
        classes: usize,
        max_rows: usize,
    ) -> Result<Self, PartialWalshError> {
        if masks.is_empty()
            || masks.len() > MAX_PARTIAL_WALSH_MASKS
            || masks.contains(&0)
            || !(2..=MAX_PARTIAL_WALSH_CLASSES).contains(&classes)
            || max_rows == 0
            || max_rows > MAX_PARTIAL_WALSH_ROWS
        {
            return Err(PartialWalshError::InvalidBudget);
        }
        let mut unique = BTreeSet::new();
        if masks.iter().any(|mask| !unique.insert(*mask)) {
            return Err(PartialWalshError::DuplicateMask);
        }
        Ok(Self {
            masks: masks.to_vec(),
            classes,
            max_rows,
            rows: 0,
            sum: vec![0.0; masks.len() * classes],
        })
    }

    /// r_y = onehot(y = truth) - p_ng(y), the NEGATIVE of the
    /// cross-entropy logit gradient; no classification-error filtering.
    pub fn observe(
        &mut self,
        context: u32,
        truth: u8,
        baseline: &[f64],
    ) -> Result<(), PartialWalshError> {
        if self.rows >= self.max_rows {
            return Err(PartialWalshError::Full);
        }
        if usize::from(truth) >= self.classes {
            return Err(PartialWalshError::InvalidLabel);
        }
        if baseline.len() != self.classes
            || baseline
                .iter()
                .any(|&p| !p.is_finite() || !(0.0..=1.0).contains(&p))
            || (baseline.iter().sum::<f64>() - 1.0).abs() > 1.0e-9
        {
            return Err(PartialWalshError::InvalidDistribution);
        }

        for (mask_index, &mask) in self.masks.iter().enumerate() {
            let sign = if (mask & context).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            let sums = &mut self.sum[mask_index * self.classes..(mask_index + 1) * self.classes];
            for (y, (slot, &p)) in sums.iter_mut().zip(baseline).enumerate() {
                *slot += sign * (f64::from(u8::from(y == usize::from(truth))) - p);
            }
        }
        self.rows += 1;
        Ok(())
    }

    pub fn samples(&self) -> usize {
        self.rows
    }

    pub fn class_coefficient(
        &self,
        mask_index: usize,
        y: usize,
    ) -> Result<f64, PartialWalshError> {
        if self.rows == 0 {
            return Err(PartialWalshError::Empty);
        }
        if mask_index >= self.masks.len() || y >= self.classes {
            return Err(PartialWalshError::InvalidBudget);
        }
        Ok(self.sum[mask_index * self.classes + y] / self.rows as f64)
    }

    /// Magnitude is descriptive and NOT a calibrated p-value after selection.
    pub fn ranked(&self) -> Result<Vec<WalshProjection>, PartialWalshError> {
        if self.rows == 0 {
            return Err(PartialWalshError::Empty);
        }
        let mut out = self
            .masks
            .iter()
            .enumerate()
            .map(|(index, &mask)| {
                let magnitude = self.sum[index * self.classes..(index + 1) * self.classes]
                    .iter()
                    .map(|&v| (v / self.rows as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                WalshProjection {
                    mask,
                    class_residual_norm: magnitude,
                }
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| {
            b.class_residual_norm
                .total_cmp(&a.class_residual_norm)
                .then_with(|| a.mask.cmp(&b.mask))
        });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_xor_is_detected_without_marginal_bit_correlation() {
        let mut projection = PartialWalsh::new(&[1, 2, 3], 2, 128).unwrap();
        for _ in 0..16 {
            for context in 0..4u32 {
                let parity = (context & 3).count_ones() % 2;
                projection
                    .observe(context, parity as u8, &[0.5, 0.5])
                    .unwrap();
            }
        }
        assert_eq!(projection.samples(), 64);
        let ranked = projection.ranked().unwrap();
        assert_eq!(ranked[0].mask, 3);
        assert!(ranked[0].class_residual_norm > 0.7);
        assert!(ranked[1].class_residual_norm.abs() < 1.0e-12);
        // Signed coefficient agrees with the explicit 2^2 parity oracle.
        assert_eq!(projection.class_coefficient(2, 0).unwrap(), 0.5);
        assert_eq!(projection.class_coefficient(2, 1).unwrap(), -0.5);
    }

    #[test]
    fn null_residual_must_not_create_a_walsh_peak() {
        let mut projection = PartialWalsh::new(&[1, 3], 2, 64).unwrap();
        for context in 0..4u32 {
            for label in [0u8, 1u8] {
                projection.observe(context, label, &[0.5, 0.5]).unwrap();
            }
        }
        assert!(projection.ranked().unwrap().iter().all(|s| s.class_residual_norm == 0.0));
    }

    #[test]
    fn fail_closed_on_bad_mass_duplicate_masks_and_overflow() {
        assert_eq!(
            PartialWalsh::new(&[1, 1], 2, 1).err(),
            Some(PartialWalshError::DuplicateMask)
        );
        let mut projection = PartialWalsh::new(&[1], 2, 1).unwrap();
        assert_eq!(
            projection.observe(0, 0, &[0.4, 0.4]),
            Err(PartialWalshError::InvalidDistribution)
        );
        assert_eq!(projection.samples(), 0);
        projection.observe(0, 0, &[0.5, 0.5]).unwrap();
        assert_eq!(projection.observe(0, 0, &[0.5, 0.5]), Err(PartialWalshError::Full));
    }
}
