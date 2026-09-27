//! Exact packed Boolean bitplanes backed by 64-bit words.
//!
//! This module provides a reusable storage primitive only. It assigns no
//! runtime policy, cache, routing, or model semantics to the bits.

use core::fmt;

/// Versioned identity of the first packed bitplane contract.
pub const PACKED_BITPLANE_V1: &str = "scirust.packed-bitplane@1.0.0";

/// Exact packed Boolean plane with zero-only unused tail bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBitPlane {
    len_bits: usize,
    words: Vec<u64>,
}

impl PackedBitPlane {
    /// Construct a zero-filled plane with an exact logical bit length.
    #[must_use]
    pub fn zeroed(len_bits: usize) -> Self {
        Self {
            len_bits,
            words: vec![0; word_count(len_bits)],
        }
    }

    /// Construct a plane from exact backing words.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing word count is not exact or when unused
    /// bits in the final word are non-zero.
    pub fn from_words(len_bits: usize, words: Vec<u64>) -> Result<Self, PackedBitPlaneError> {
        let expected = word_count(len_bits);
        if words.len() != expected
        {
            return Err(PackedBitPlaneError::WordCountMismatch {
                expected,
                actual: words.len(),
            });
        }
        validate_tail(len_bits, &words)?;
        Ok(Self { len_bits, words })
    }

    /// Construct a plane by evaluating one predicate for every logical index.
    #[must_use]
    pub fn from_fn(len_bits: usize, mut predicate: impl FnMut(usize) -> bool) -> Self {
        let mut plane = Self::zeroed(len_bits);
        for index in 0..len_bits
        {
            if predicate(index)
            {
                plane.set_unchecked(index, true);
            }
        }
        plane
    }

    /// Logical bit length.
    #[must_use]
    pub const fn len_bits(&self) -> usize {
        self.len_bits
    }

    /// Whether the logical plane is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len_bits == 0
    }

    /// Exact backing words.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Number of backing words.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Exact backing payload bits, including zero tail capacity.
    #[must_use]
    pub fn backing_bits(&self) -> usize {
        self.words.len() * 64
    }

    /// Read one logical bit or return None outside the logical domain.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len_bits
        {
            return None;
        }
        let word = index / 64;
        let bit = index % 64;
        Some(self.words[word] & (1_u64 << bit) != 0)
    }

    /// Set one logical bit.
    ///
    /// # Errors
    ///
    /// Returns an error when the index lies outside the logical domain.
    pub fn set(&mut self, index: usize, value: bool) -> Result<(), PackedBitPlaneError> {
        if index >= self.len_bits
        {
            return Err(PackedBitPlaneError::IndexOutOfBounds {
                index,
                len_bits: self.len_bits,
            });
        }
        self.set_unchecked(index, value);
        Ok(())
    }

    /// Clear every logical bit while retaining backing capacity.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Count set logical bits.
    #[must_use]
    pub fn count_ones(&self) -> u64 {
        self.words
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum()
    }

    /// Return whether every logical bit is zero.
    #[must_use]
    pub fn is_all_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Compute exact bitwise AND.
    ///
    /// # Errors
    ///
    /// Returns an error when logical lengths differ.
    pub fn bitand(&self, rhs: &Self) -> Result<Self, PackedBitPlaneError> {
        self.binary(rhs, |left, right| left & right)
    }

    /// Compute exact bitwise OR.
    ///
    /// # Errors
    ///
    /// Returns an error when logical lengths differ.
    pub fn bitor(&self, rhs: &Self) -> Result<Self, PackedBitPlaneError> {
        self.binary(rhs, |left, right| left | right)
    }

    /// Compute exact bitwise XOR.
    ///
    /// # Errors
    ///
    /// Returns an error when logical lengths differ.
    pub fn bitxor(&self, rhs: &Self) -> Result<Self, PackedBitPlaneError> {
        self.binary(rhs, |left, right| left ^ right)
    }

    fn binary(
        &self,
        rhs: &Self,
        operation: impl Fn(u64, u64) -> u64,
    ) -> Result<Self, PackedBitPlaneError> {
        if self.len_bits != rhs.len_bits
        {
            return Err(PackedBitPlaneError::LengthMismatch {
                left: self.len_bits,
                right: rhs.len_bits,
            });
        }
        let words = self
            .words
            .iter()
            .copied()
            .zip(rhs.words.iter().copied())
            .map(|(left, right)| operation(left, right))
            .collect::<Vec<_>>();
        Self::from_words(self.len_bits, words)
    }

    fn set_unchecked(&mut self, index: usize, value: bool) {
        let word = index / 64;
        let bit = index % 64;
        let mask = 1_u64 << bit;
        if value
        {
            self.words[word] |= mask;
        }
        else
        {
            self.words[word] &= !mask;
        }
    }
}

/// Fail-closed packed bitplane errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedBitPlaneError {
    IndexOutOfBounds { index: usize, len_bits: usize },
    WordCountMismatch { expected: usize, actual: usize },
    NonZeroTailBits { value: u64, valid_mask: u64 },
    LengthMismatch { left: usize, right: usize },
}

impl fmt::Display for PackedBitPlaneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::IndexOutOfBounds { index, len_bits } => write!(
                formatter,
                "bit index {index} is outside packed bitplane length {len_bits}"
            ),
            Self::WordCountMismatch { expected, actual } => write!(
                formatter,
                "packed bitplane requires {expected} backing words, observed {actual}"
            ),
            Self::NonZeroTailBits { value, valid_mask } => write!(
                formatter,
                "packed bitplane final word {value:#x} contains bits outside valid mask {valid_mask:#x}"
            ),
            Self::LengthMismatch { left, right } => write!(
                formatter,
                "packed bitplane logical length mismatch: left={left}, right={right}"
            ),
        }
    }
}

impl std::error::Error for PackedBitPlaneError {}

const fn word_count(len_bits: usize) -> usize {
    len_bits.div_ceil(64)
}

fn validate_tail(len_bits: usize, words: &[u64]) -> Result<(), PackedBitPlaneError> {
    let remainder = len_bits % 64;
    if remainder == 0 || words.is_empty()
    {
        return Ok(());
    }
    let valid_mask = (1_u64 << remainder) - 1;
    let value = words[words.len() - 1];
    if value & !valid_mask != 0
    {
        return Err(PackedBitPlaneError::NonZeroTailBits { value, valid_mask });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_word_count_and_tail_invariant_cover_boundaries() {
        for (bits, words) in [(0, 0), (1, 1), (63, 1), (64, 1), (65, 2), (128, 2)]
        {
            let plane = PackedBitPlane::zeroed(bits);
            assert_eq!(plane.word_count(), words);
            assert_eq!(plane.len_bits(), bits);
        }

        assert_eq!(
            PackedBitPlane::from_words(65, vec![0, 2]),
            Err(PackedBitPlaneError::NonZeroTailBits {
                value: 2,
                valid_mask: 1,
            })
        );
    }

    #[test]
    fn set_get_clear_and_count_are_exact_across_word_boundaries() {
        let mut plane = PackedBitPlane::zeroed(130);
        for index in [0, 1, 63, 64, 65, 127, 128, 129]
        {
            plane.set(index, true).unwrap();
            assert_eq!(plane.get(index), Some(true));
        }
        assert_eq!(plane.count_ones(), 8);
        assert_eq!(plane.get(130), None);

        plane.set(64, false).unwrap();
        assert_eq!(plane.get(64), Some(false));
        assert_eq!(plane.count_ones(), 7);

        plane.clear();
        assert!(plane.is_all_zero());
        assert_eq!(plane.count_ones(), 0);
    }

    #[test]
    fn from_fn_and_boolean_operations_match_scalar_oracle() {
        let left = PackedBitPlane::from_fn(137, |index| index % 2 == 0);
        let right = PackedBitPlane::from_fn(137, |index| index % 3 == 0);

        let and = left.bitand(&right).unwrap();
        let or = left.bitor(&right).unwrap();
        let xor = left.bitxor(&right).unwrap();

        for index in 0..137
        {
            let a = index % 2 == 0;
            let b = index % 3 == 0;
            assert_eq!(and.get(index), Some(a & b));
            assert_eq!(or.get(index), Some(a | b));
            assert_eq!(xor.get(index), Some(a ^ b));
        }
    }

    #[test]
    fn mismatched_lengths_fail_closed() {
        let left = PackedBitPlane::zeroed(64);
        let right = PackedBitPlane::zeroed(65);
        assert_eq!(
            left.bitand(&right),
            Err(PackedBitPlaneError::LengthMismatch {
                left: 64,
                right: 65,
            })
        );
    }

    #[test]
    fn backing_words_round_trip_exactly() {
        let words = vec![0x0123_4567_89ab_cdef, 0x1];
        let plane = PackedBitPlane::from_words(65, words.clone()).unwrap();
        assert_eq!(plane.words(), words);
        assert_eq!(plane.backing_bits(), 128);
    }
}
