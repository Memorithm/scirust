//! Space-efficient SECDED for 64 packed quaternary symbols (128 payload bits).
//!
//! A shortened, extended Hamming code protects the two `u64` bitplanes with
//! eight Hamming syndrome bits and one overall parity bit. The payload is
//! systematic: no reordering or expansion of the two data words is necessary.
//! The 9 control bits must be stored separately. A densely bit-packed stream
//! would consume 137 bits per 128-bit payload (7.03125% overhead); a Rust
//! `u16` per block or an aligned struct uses more memory. The representation
//! used by a caller must account for this overhead accurately.
//!
//! This is binary SECDED applied to the classical GF(4) storage encoding.
//! It corrects one arbitrary bit flip across the 137-bit protected codeword
//! and detects any two flips; 3+ flips can be miscorrected or undetected.
//! It does not authenticate data and must not be treated as an adversarial
//! tamper detector or a quantum error-correction routine.

use super::{CompactLogicError, QuadBitplanes, SelectedSecded};

/// Data-bit mask contributing to each of the eight Hamming parity positions.
///
/// Hamming positions 1,2,4,8,16,32,64,128 are held separately in the low
/// eight tag bits. The remaining 128 positions within 1..=136 are payload.
const fn make_masks() -> [(u64, u64); 8] {
    let mut masks = [(0u64, 0u64); 8];
    let mut position = 1u16;
    let mut payload_bit = 0usize;
    while position <= 136 {
        if position & (position - 1) != 0 {
            let mut check = 0usize;
            while check < 8 {
                if position & (1u16 << check) != 0 {
                    if payload_bit < 64 {
                        masks[check].0 |= 1u64 << payload_bit;
                    } else {
                        masks[check].1 |= 1u64 << (payload_bit - 64);
                    }
                }
                check += 1;
            }
            payload_bit += 1;
        }
        position += 1;
    }
    masks
}

const PARITY_MASKS: [(u64, u64); 8] = make_masks();

fn payload_syndrome(data: QuadBitplanes) -> u8 {
    let mut syndrome = 0u8;
    for (check, &(low_mask, high_mask)) in PARITY_MASKS.iter().enumerate() {
        let parity = ((data.low & low_mask).count_ones()
            ^ (data.high & high_mask).count_ones()) & 1;
        syndrome |= (parity as u8) << check;
    }
    syndrome
}

/// Return the 9-bit extended-Hamming tag for 128 payload bits.
///
/// Bits 0..=7 carry Hamming parities and bit 8 overall even parity.
/// The caller is responsible for persisting the tag together with the data.
#[must_use]
pub fn encode_tag(data: QuadBitplanes) -> u16 {
    let hamming = payload_syndrome(data);
    let overall = (data.low.count_ones()
        ^ data.high.count_ones()
        ^ hamming.count_ones()) & 1;
    u16::from(hamming) | ((overall as u16) << 8)
}

/// SECDED result for one protected 128-bit block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockCheck {
    /// The codeword is consistent with all parity constraints.
    Clean,
    /// A single bit was corrected at the 1-based codeword position 1..=137.
    ///
    /// A multi-bit corruption outside the SECDED guarantee may still mimic
    /// this outcome; do not treat it as cryptographic integrity.
    Corrected {
        /// Corrected Hamming data/parity bit, or overall parity position 137.
        bit_position: u16,
    },
}

/// Check and correct a data block and its separately stored tag in place.
///
/// Every one-bit error is corrected, including a corrupted tag bit.
/// Every two-bit error is detected and rejected without modifying the inputs.
/// If the 9-bit tag has out-of-range bits, neither argument is changed.
/// Three or more bit errors are outside the code's guarantee.
pub fn verify_and_correct(
    data: &mut QuadBitplanes,
    tag: &mut u16,
) -> Result<BlockCheck, CompactLogicError> {
    if *tag > 0x1ff {
        return Err(CompactLogicError::InvalidTag);
    }

    let syndrome = payload_syndrome(*data) ^ (*tag as u8);
    let odd = (data.low.count_ones()
        ^ data.high.count_ones()
        ^ tag.count_ones()) & 1 != 0;

    match (syndrome, odd) {
        (0, false) => Ok(BlockCheck::Clean),
        (0, true) => {
            *tag ^= 1 << 8;
            Ok(BlockCheck::Corrected { bit_position: 137 })
        }
        (_, false) => Err(CompactLogicError::Uncorrectable),
        (position, true) if position > 136 => Err(CompactLogicError::Uncorrectable),
        (position, true) => {
            if position.is_power_of_two() {
                *tag ^= 1u16 << position.trailing_zeros();
            } else {
                let mut parity_slots = 0usize;
                for check in 0..8 {
                    if (1u16 << check) <= u16::from(position) {
                        parity_slots += 1;
                    }
                }
                // Subtract the preceding Hamming parity positions.
                let bit = position as usize - 1 - parity_slots;
                if bit < 64 {
                    data.low ^= 1u64 << bit;
                } else {
                    data.high ^= 1u64 << (bit - 64);
                }
            }
            Ok(BlockCheck::Corrected {
                bit_position: position as u16,
            })
        }
    }
}

/// Check up to 64 data blocks admitted by a caller-supplied Boolean mask.
///
/// Inputs and outputs are disjoint caller-managed slices. Rejected blocks
/// and non-admitted blocks leave outputs untouched. A malformed 9-bit tag
/// in an admitted lane fails the whole call before modifying any outputs.
/// Other detected uncorrectable blocks are reported in the rejected mask.
/// Clean and corrected blocks are copied to the corresponding output slots.
pub fn decode_selected_blocks(
    blocks: &[QuadBitplanes],
    tags: &[u16],
    admitted: u64,
    output: &mut [QuadBitplanes],
    output_tags: &mut [u16],
) -> Result<SelectedSecded, CompactLogicError> {
    if blocks.len() > 64
        || tags.len() != blocks.len()
        || output.len() < blocks.len()
        || output_tags.len() < blocks.len()
        || (blocks.len() < 64 && (admitted >> blocks.len()) != 0)
    {
        return Err(CompactLogicError::InvalidBatch);
    }
    for (index, &tag) in tags.iter().enumerate() {
        if admitted & (1u64 << index) != 0 && tag > 0x1ff {
            return Err(CompactLogicError::InvalidTag);
        }
    }

    let mut decisions = SelectedSecded::default();
    for (index, (&block, &tag)) in blocks.iter().zip(tags).enumerate() {
        let lane = 1u64 << index;
        if admitted & lane == 0 {
            continue;
        }
        let mut repaired = block;
        let mut repaired_tag = tag;
        match verify_and_correct(&mut repaired, &mut repaired_tag) {
            Ok(status) => {
                output[index] = repaired;
                output_tags[index] = repaired_tag;
                decisions.accepted |= lane;
                if matches!(status, BlockCheck::Corrected { .. }) {
                    decisions.corrected |= lane;
                }
            }
            Err(CompactLogicError::Uncorrectable) => decisions.rejected |= lane,
            Err(err) => return Err(err),
        }
    }
    Ok(decisions)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent oracle: construct the 128 payload codeword positions with a
    // straightforward loop, rather than using the encoder's parity masks.
    fn oracle_tag(data: QuadBitplanes) -> u16 {
        let all = u128::from(data.low) | (u128::from(data.high) << 64);
        let mut parity = 0u16;
        let mut payload_index = 0u32;
        for position in 1u16..=136 {
            if !position.is_power_of_two() {
                if (all >> payload_index) & 1 != 0 {
                    parity ^= position;
                }
                payload_index += 1;
            }
        }
        assert_eq!(payload_index, 128);
        let overall = ((all.count_ones() ^ parity.count_ones()) & 1) as u16;
        parity | (overall << 8)
    }

    fn flip(data: &mut QuadBitplanes, tag: &mut u16, position: u16) {
        if position == 137 {
            *tag ^= 1 << 8;
        } else if position.is_power_of_two() {
            *tag ^= 1u16 << position.trailing_zeros();
        } else {
            let mut index = 0usize;
            for p in 1..position {
                if !(p as u16).is_power_of_two() {
                    index += 1;
                }
            }
            if index < 64 {
                data.low ^= 1u64 << index;
            } else {
                data.high ^= 1u64 << (index - 64);
            }
        }
    }

    #[test]
    fn independent_oracle_and_exhaustive_one_two_bit_faults() {
        let cases = [
            QuadBitplanes { low: 0, high: 0 },
            QuadBitplanes { low: u64::MAX, high: u64::MAX },
            QuadBitplanes {
                low: 0xaaaa_5555_01fe_dcbau64,
                high: 0x1357_9bdf_2468_ace0u64,
            },
        ];
        for original in cases {
            let original_tag = encode_tag(original);
            assert_eq!(original_tag, oracle_tag(original));
            let mut clean_data = original;
            let mut clean_tag = original_tag;
            assert_eq!(verify_and_correct(&mut clean_data, &mut clean_tag), Ok(BlockCheck::Clean));
            assert_eq!((clean_data, clean_tag), (original, original_tag));

            for first in 1..=137u16 {
                let mut single_data = original;
                let mut single_tag = original_tag;
                flip(&mut single_data, &mut single_tag, first);
                assert_eq!(
                    verify_and_correct(&mut single_data, &mut single_tag),
                    Ok(BlockCheck::Corrected { bit_position: first })
                );
                assert_eq!((single_data, single_tag), (original, original_tag));
                for second in (first + 1)..=137u16 {
                    let mut double_data = original;
                    let mut double_tag = original_tag;
                    flip(&mut double_data, &mut double_tag, first);
                    flip(&mut double_data, &mut double_tag, second);
                    let before = (double_data, double_tag);
                    assert_eq!(
                        verify_and_correct(&mut double_data, &mut double_tag),
                        Err(CompactLogicError::Uncorrectable),
                    );
                    assert_eq!((double_data, double_tag), before);
                }
            }
        }
    }

    #[test]
    fn admitted_blocks_correct_reject_and_preserve_outputs() {
        let base = [
            QuadBitplanes { low: 0xabcd, high: 0x1234 },
            QuadBitplanes { low: 0x1357, high: 0x2468 },
            QuadBitplanes { low: 0x1111, high: 0x2222 },
            QuadBitplanes { low: 0x3333, high: 0x4444 },
        ];
        let mut blocks = base;
        let tags = base.map(encode_tag);
        // Each tag lives in caller-owned metadata, not inside the bitplanes.
        let mut corrupted_tags = tags;
        flip(&mut blocks[1], &mut corrupted_tags[1], 3);
        flip(&mut blocks[2], &mut corrupted_tags[2], 5);
        flip(&mut blocks[2], &mut corrupted_tags[2], 6);
        let marker = QuadBitplanes {
            low: u64::MAX,
            high: 0xdead_beef,
        };
        let mut output = [marker; 4];
        let mut output_tags = [0xbeef; 4];
        let decisions = decode_selected_blocks(
            &blocks,
            &corrupted_tags,
            0b0111,
            &mut output,
            &mut output_tags,
        )
        .unwrap();
        assert_eq!(decisions.accepted, 0b0011);
        assert_eq!(decisions.corrected, 0b0010);
        assert_eq!(decisions.rejected, 0b0100);
        assert_eq!(output, [base[0], base[1], marker, marker]);
        assert_eq!(output_tags, [tags[0], tags[1], 0xbeef, 0xbeef]);

        let mut bad_tags = corrupted_tags;
        bad_tags[1] = 0x200;
        let prior = output;
        let prior_tags = output_tags;
        assert_eq!(
            decode_selected_blocks(&blocks, &bad_tags, 0b0011, &mut output, &mut output_tags),
            Err(CompactLogicError::InvalidTag)
        );
        assert_eq!(output, prior);
        assert_eq!(output_tags, prior_tags);
        assert_eq!(
            decode_selected_blocks(&blocks, &tags, 1 << 4, &mut output, &mut output_tags),
            Err(CompactLogicError::InvalidBatch)
        );
    }
}
