//! Portable 2-bit bitplanes, GF(4) operations, and optional binary SECDED admission.
//!
//! Four *classical* values occupy two 64-bit words for 64 lanes (2 bits/lane).
//! The arithmetic GF(4) interpretation is optional: callers may instead assign
//! application-specific meanings to the four labels. GF(4) uses
//! alpha^2 = alpha + 1 and the encoding 0, 1, alpha, alpha+1.
//!
//! SECDED (8,4) applies to an independently protected 4-bit payload; it is not
//! a zero-overhead property of the packed representation. It corrects one
//! flipped bit and detects any two flipped bits. It does not promise reliable
//! detection/correction of three or more flips or correlated errors.
//!
//! No model routing policy is embedded here. Callers provide admission masks.
//! All operations are deterministic, CPU-portable, allocation-free references.

/// Invalid index, value, batch size, or a detected uncorrectable codeword.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactLogicError {
    /// One of the 64 lanes was addressed out of range.
    LaneOutOfBounds,
    /// A quaternary value was not in 0..=3.
    InvalidSymbol,
    /// A four-bit payload was not in 0..=15.
    InvalidNibble,
    /// Selection batches support at most 64 codewords and require enough output.
    InvalidBatch,
    /// A two-bit error was detected; more errors may escape detection.
    Uncorrectable,
}

/// Sixty-four quaternary lanes represented in two u64 bitplanes.
///
/// Lane i has value (low >> i & 1) | ((high >> i & 1) << 1).
/// This is 128 bits / 64 lanes = exactly 2 bits per lane excluding the struct
/// itself. No padding/valid-length is stored: mask unused tail lanes explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuadBitplanes {
    /// Least-significant bit of each lane.
    pub low: u64,
    /// Most-significant bit of each lane.
    pub high: u64,
}

impl QuadBitplanes {
    /// Read one quaternary lane.
    pub fn get(self, lane: u32) -> Result<u8, CompactLogicError> {
        if lane >= 64 {
            return Err(CompactLogicError::LaneOutOfBounds);
        }
        Ok(((self.low >> lane) & 1) as u8 | ((((self.high >> lane) & 1) as u8) << 1))
    }

    /// Replace one lane; rejects values outside 0..=3 without mutation.
    pub fn set(&mut self, lane: u32, value: u8) -> Result<(), CompactLogicError> {
        if lane >= 64 {
            return Err(CompactLogicError::LaneOutOfBounds);
        }
        if value > 3 {
            return Err(CompactLogicError::InvalidSymbol);
        }
        let bit = 1u64 << lane;
        self.low = (self.low & !bit) | (u64::from(value & 1) << lane);
        self.high = (self.high & !bit) | (u64::from((value >> 1) & 1) << lane);
        Ok(())
    }

    /// Return one bit per lane whose value equals `symbol`.
    pub fn matches(self, symbol: u8) -> Result<u64, CompactLogicError> {
        if symbol > 3 {
            return Err(CompactLogicError::InvalidSymbol);
        }
        let low = if symbol & 1 != 0 { self.low } else { !self.low };
        let high = if symbol & 2 != 0 { self.high } else { !self.high };
        Ok(low & high)
    }

    /// Retain only admitted lanes; rejected lanes become the value zero.
    #[must_use]
    pub fn select(self, admitted: u64) -> Self {
        Self {
            low: self.low & admitted,
            high: self.high & admitted,
        }
    }

    /// Replace admitted lanes from `replacement`; preserve all others exactly.
    #[must_use]
    pub fn replace_selected(self, replacement: Self, admitted: u64) -> Self {
        Self {
            low: (self.low & !admitted) | (replacement.low & admitted),
            high: (self.high & !admitted) | (replacement.high & admitted),
        }
    }

    /// Lane-wise addition in GF(4), equivalently bitwise XOR of both planes.
    ///
    /// This is not integer addition modulo four.
    #[must_use]
    pub fn gf4_add(self, rhs: Self) -> Self {
        Self {
            low: self.low ^ rhs.low,
            high: self.high ^ rhs.high,
        }
    }

    /// Lane-wise multiplication in GF(4), with alpha^2 = alpha + 1.
    #[must_use]
    pub fn gf4_mul(self, rhs: Self) -> Self {
        let cross = (self.low & rhs.high) ^ (self.high & rhs.low);
        let high_product = self.high & rhs.high;
        Self {
            low: (self.low & rhs.low) ^ high_product,
            high: cross ^ high_product,
        }
    }

    /// XOR-sum of admitted GF(4) values, encoded in two bits.
    ///
    /// Parity is the GF(4) sum because the field has characteristic two.
    #[must_use]
    pub fn gf4_sum_selected(self, admitted: u64) -> u8 {
        ((self.low & admitted).count_ones() as u8 & 1)
            | (((self.high & admitted).count_ones() as u8 & 1) << 1)
    }
}

/// Encode a four-bit payload as an extended Hamming (8,4) SECDED byte.
///
/// Positions 1,2,4 carry Hamming parity; 3,5,6,7 carry payload bits 0..3;
/// position 8 carries overall even parity. Positions are one-based.
pub fn secded_encode(nibble: u8) -> Result<u8, CompactLogicError> {
    if nibble > 15 {
        return Err(CompactLogicError::InvalidNibble);
    }
    let mut word = 0u8;
    for (payload_bit, position) in [3u8, 5, 6, 7].into_iter().enumerate() {
        word |= ((nibble >> payload_bit) & 1) << (position - 1);
    }
    for parity_position in [1u8, 2, 4] {
        let mut parity = 0u8;
        for position in 1u8..=7 {
            if position & parity_position != 0 {
                parity ^= (word >> (position - 1)) & 1;
            }
        }
        word |= parity << (parity_position - 1);
    }
    word |= ((word.count_ones() & 1) as u8) << 7;
    Ok(word)
}

/// SECDED decoding outcome. Corrected bit positions are one-based, 1..=8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodedNibble {
    /// Codeword passed both parity checks.
    Clean(u8),
    /// A single-bit fault was corrected; the identified bit is reported.
    Corrected {
        /// Four-bit payload after correction.
        value: u8,
        /// Corrected one-based bit position.
        bit_position: u8,
    },
}

/// Decode an extended Hamming (8,4) byte.
///
/// One-bit flips are corrected; every two-bit flip is rejected. More than two
/// corruptions may be miscorrected or remain undetected, as for ordinary SECDED.
pub fn secded_decode(mut word: u8) -> Result<DecodedNibble, CompactLogicError> {
    let mut syndrome = 0u8;
    for position in 1u8..=7 {
        if (word >> (position - 1)) & 1 != 0 {
            syndrome ^= position;
        }
    }
    let odd_parity = word.count_ones() & 1 != 0;
    let corrected = match (syndrome, odd_parity) {
        (0, false) => None,
        (0, true) => Some(8),
        (position, true) => Some(position),
        (_, false) => return Err(CompactLogicError::Uncorrectable),
    };
    if let Some(position) = corrected {
        word ^= 1u8 << (position - 1);
    }
    let mut value = 0u8;
    for (payload_bit, position) in [3u8, 5, 6, 7].into_iter().enumerate() {
        value |= ((word >> (position - 1)) & 1) << payload_bit;
    }
    Ok(match corrected {
        Some(bit_position) => DecodedNibble::Corrected {
            value,
            bit_position,
        },
        None => DecodedNibble::Clean(value),
    })
}

/// Bit masks describing decisions made by a selected SECDED batch.
///
/// `accepted` includes corrected payloads, while `corrected` is a subset.
/// `rejected` contains codewords with a detected multi-bit fault.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelectedSecded {
    /// Selected lanes that were decoded successfully.
    pub accepted: u64,
    /// Selected lanes requiring a one-bit correction.
    pub corrected: u64,
    /// Selected lanes rejected for a detected uncorrectable fault.
    pub rejected: u64,
}

/// Decode only admitted codewords, without changing non-admitted output lanes.
///
/// Both buffers are caller-owned. `output` must have at least `codewords.len()`
/// elements. Batches are limited to 64 codewords. The admission mask cannot
/// reference non-existent lanes. Rejected lanes leave their output untouched.
/// This helper does not provide authentication or protection against adversarial
/// faults; it provides only the SECDED error model documented above.
pub fn decode_selected_secded(
    codewords: &[u8],
    admitted: u64,
    output: &mut [u8],
) -> Result<SelectedSecded, CompactLogicError> {
    if codewords.len() > 64
        || output.len() < codewords.len()
        || (codewords.len() < 64 && (admitted >> codewords.len()) != 0)
    {
        return Err(CompactLogicError::InvalidBatch);
    }
    let mut decisions = SelectedSecded::default();
    for (index, &codeword) in codewords.iter().enumerate() {
        let bit = 1u64 << index;
        if admitted & bit == 0 {
            continue;
        }
        match secded_decode(codeword) {
            Ok(DecodedNibble::Clean(value)) => {
                output[index] = value;
                decisions.accepted |= bit;
            }
            Ok(DecodedNibble::Corrected { value, .. }) => {
                output[index] = value;
                decisions.accepted |= bit;
                decisions.corrected |= bit;
            }
            Err(CompactLogicError::Uncorrectable) => {
                decisions.rejected |= bit;
            }
            Err(err) => return Err(err),
        }
    }
    Ok(decisions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_roundtrip_and_exact_selection_masks() {
        let mut packed = QuadBitplanes::default();
        for lane in 0..64 {
            packed.set(lane, (lane % 4) as u8).unwrap();
        }
        for symbol in 0..4 {
            let mut oracle_mask = 0u64;
            for lane in 0..64 {
                assert_eq!(packed.get(lane).unwrap(), (lane % 4) as u8);
                if lane % 4 == symbol {
                    oracle_mask |= 1u64 << lane;
                }
            }
            let selected = packed.matches(symbol as u8).unwrap();
            assert_eq!(selected, oracle_mask);
            assert_eq!(packed.select(selected).matches(symbol as u8).unwrap() & selected, selected);
        }
        assert_eq!(packed.get(64), Err(CompactLogicError::LaneOutOfBounds));
        assert_eq!(packed.set(64, 0), Err(CompactLogicError::LaneOutOfBounds));
        assert_eq!(packed.set(0, 4), Err(CompactLogicError::InvalidSymbol));
        assert_eq!(packed.matches(4), Err(CompactLogicError::InvalidSymbol));
    }

    #[test]
    fn gf4_arithmetic_matches_independent_truth_tables() {
        let multiply = [
            [0u8, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
        ];
        for left in 0..4u8 {
            for right in 0..4u8 {
                let a = QuadBitplanes {
                    low: if left & 1 != 0 { u64::MAX } else { 0 },
                    high: if left & 2 != 0 { u64::MAX } else { 0 },
                };
                let b = QuadBitplanes {
                    low: if right & 1 != 0 { u64::MAX } else { 0 },
                    high: if right & 2 != 0 { u64::MAX } else { 0 },
                };
                assert_eq!(a.gf4_add(b).get(63).unwrap(), left ^ right);
                let product = a.gf4_mul(b);
                for lane in 0..64 {
                    assert_eq!(product.get(lane).unwrap(), multiply[left as usize][right as usize]);
                }
            }
        }
    }

    #[test]
    fn gf4_masked_parity_and_replacement_preserve_unselected_lanes() {
        let mut source = QuadBitplanes::default();
        let mut replacement = QuadBitplanes::default();
        for i in 0..64 {
            source.set(i, (i % 4) as u8).unwrap();
            replacement.set(i, ((i + 1) % 4) as u8).unwrap();
        }
        let mask = 0x0f0f_0f0f_0f0f_0f0fu64;
        let combined = source.replace_selected(replacement, mask);
        let mut expected_sum = 0u8;
        for i in 0..64 {
            let is_selected = (mask >> i) & 1 != 0;
            assert_eq!(
                combined.get(i).unwrap(),
                if is_selected { replacement.get(i).unwrap() } else { source.get(i).unwrap() }
            );
            if is_selected {
                expected_sum ^= source.get(i).unwrap();
            }
        }
        assert_eq!(source.gf4_sum_selected(mask), expected_sum);
    }

    #[test]
    fn secded_exhaustive_clean_single_and_double_faults() {
        for nibble in 0..16u8 {
            let encoded = secded_encode(nibble).unwrap();
            assert_eq!(secded_decode(encoded), Ok(DecodedNibble::Clean(nibble)));
            for bit in 0..8u8 {
                assert_eq!(
                    secded_decode(encoded ^ (1 << bit)),
                    Ok(DecodedNibble::Corrected { value: nibble, bit_position: bit + 1 })
                );
                for second in (bit + 1)..8 {
                    assert_eq!(
                        secded_decode(encoded ^ (1 << bit) ^ (1 << second)),
                        Err(CompactLogicError::Uncorrectable)
                    );
                }
            }
        }
        assert_eq!(secded_encode(16), Err(CompactLogicError::InvalidNibble));
    }

    #[test]
    fn only_selected_verified_payloads_are_written() {
        let clean = secded_encode(3).unwrap();
        let correctable = secded_encode(11).unwrap() ^ (1 << 4);
        let rejected = secded_encode(9).unwrap() ^ 1 ^ (1 << 3);
        let unselected = secded_encode(7).unwrap();
        let mut output = [0xAA; 4];
        let result = decode_selected_secded(
            &[clean, correctable, rejected, unselected],
            0b0111,
            &mut output,
        )
        .unwrap();
        assert_eq!(result.accepted, 0b0011);
        assert_eq!(result.corrected, 0b0010);
        assert_eq!(result.rejected, 0b0100);
        assert_eq!(output, [3, 11, 0xAA, 0xAA]);
        assert_eq!(
            decode_selected_secded(&[clean], 0b10, &mut output),
            Err(CompactLogicError::InvalidBatch)
        );
        assert_eq!(
            decode_selected_secded(&[clean], 1, &mut []),
            Err(CompactLogicError::InvalidBatch)
        );
    }
}
