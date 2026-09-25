//! Portable reference codec for block-16 FP4 KV experiments.
//!
//! Scalar correctness/reference implementation for SciRust roadmap item #81.
//! Sixteen consecutive values share one positive FP8 E4M3 scale; each element
//! is stored as FP4 E2M1; no second/global scale is present in this reference.
//! Two E2M1 nibbles are packed per byte, low nibble first.
//!
//! This byte layout and scale-selection rule are a SciRust research contract,
//! not a claim of bit-identical compatibility with any external implementation.
//! E2M1 conversion follows the OCP FP4 value set and round-to-nearest,
//! ties-to-even semantics. E4M3 scale conversion uses the finite positive E4M3
//! value set with saturating conversion. Exact packed footprint is 9 bytes:
//! 8 payload bytes plus one E4M3 scale byte.

use core::fmt;

/// Number of scalar values sharing one E4M3 scale.
pub const FP4_KV_BLOCK_SIZE: usize = 16;
/// Maximum finite magnitude representable by E2M1.
pub const E2M1_MAX: f32 = 6.0;
/// Maximum positive finite E4M3 scale value.
pub const E4M3_MAX: f32 = 448.0;
/// Smallest positive E4M3 subnormal scale value.
pub const E4M3_MIN_POSITIVE: f32 = 1.0 / 512.0;
/// Exact serialized bytes of one block: eight payload bytes plus one scale byte.
pub const FP4_KV_BLOCK_PACKED_BYTES: usize = 9;

const E2M1_POSITIVE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp4KvError {
    NonFinite,
    NegativeScale,
    NanScaleEncoding,
    NegativeScaleEncoding,
}

impl fmt::Display for Fp4KvError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonFinite => write!(output, "FP4 KV input must be finite"),
            Self::NegativeScale => write!(output, "FP4 KV scale must be non-negative"),
            Self::NanScaleEncoding => write!(output, "E4M3 NaN scale encoding is invalid"),
            Self::NegativeScaleEncoding => write!(output, "E4M3 KV scale must be positive-sign"),
        }
    }
}

impl std::error::Error for Fp4KvError {}

#[must_use]
pub fn decode_e2m1(code: u8) -> f32 {
    let nibble = code & 0x0f;
    let sign = if nibble & 0x08 == 0 { 1.0 } else { -1.0 };
    sign * E2M1_POSITIVE[(nibble & 0x07) as usize]
}

/// Encode one finite scalar to E2M1 with saturating round-to-nearest, ties-to-even.
pub fn encode_e2m1_sat(value: f32) -> Result<u8, Fp4KvError> {
    if !value.is_finite()
    {
        return Err(Fp4KvError::NonFinite);
    }
    let sign = value.is_sign_negative();
    let magnitude = value.abs().min(E2M1_MAX);

    let mut best_code = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=7
    {
        let distance = (magnitude - E2M1_POSITIVE[code as usize]).abs();
        if distance < best_distance
            || (distance == best_distance && code.is_multiple_of(2) && !best_code.is_multiple_of(2))
        {
            best_code = code;
            best_distance = distance;
        }
    }

    Ok(best_code | if sign { 0x08 } else { 0 })
}

/// Decode one positive-sign finite E4M3 byte used as the block scale.
pub fn decode_e4m3_scale(code: u8) -> Result<f32, Fp4KvError> {
    if code & 0x80 != 0
    {
        return Err(Fp4KvError::NegativeScaleEncoding);
    }
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0x0f && mantissa == 0x07
    {
        return Err(Fp4KvError::NanScaleEncoding);
    }

    if exponent == 0
    {
        return Ok(2.0f32.powi(-6) * (mantissa as f32 / 8.0));
    }

    Ok(2.0f32.powi(i32::from(exponent) - 7) * (1.0 + mantissa as f32 / 8.0))
}

/// Encode a finite non-negative scale to positive E4M3 with saturation and RNE.
pub fn encode_e4m3_scale_sat(value: f32) -> Result<u8, Fp4KvError> {
    if !value.is_finite()
    {
        return Err(Fp4KvError::NonFinite);
    }
    if value.is_sign_negative()
    {
        return Err(Fp4KvError::NegativeScale);
    }

    let target = value.min(E4M3_MAX);
    let mut best_code = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=0x7e
    {
        let candidate = decode_e4m3_scale(code)?;
        let distance = (target - candidate).abs();
        if distance < best_distance
            || (distance == best_distance && code.is_multiple_of(2) && !best_code.is_multiple_of(2))
        {
            best_code = code;
            best_distance = distance;
        }
    }
    Ok(best_code)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp4KvBlock16 {
    payload: [u8; 8],
    scale_e4m3: u8,
}

impl Fp4KvBlock16 {
    pub fn from_f32(values: [f32; FP4_KV_BLOCK_SIZE]) -> Result<Self, Fp4KvError> {
        let mut amax = 0.0f32;
        for &value in &values
        {
            if !value.is_finite()
            {
                return Err(Fp4KvError::NonFinite);
            }
            amax = amax.max(value.abs());
        }

        if amax == 0.0
        {
            return Ok(Self {
                payload: [0; 8],
                scale_e4m3: 0,
            });
        }

        let raw_scale = amax / E2M1_MAX;
        let mut scale_e4m3 = encode_e4m3_scale_sat(raw_scale)?;
        if scale_e4m3 == 0
        {
            scale_e4m3 = 1;
        }
        let scale = decode_e4m3_scale(scale_e4m3)?;

        let mut payload = [0u8; 8];
        for (index, &value) in values.iter().enumerate()
        {
            let code = encode_e2m1_sat(value / scale)?;
            let byte = index / 2;
            if index % 2 == 0
            {
                payload[byte] = code;
            }
            else
            {
                payload[byte] |= code << 4;
            }
        }

        Ok(Self {
            payload,
            scale_e4m3,
        })
    }

    /// Reconstruct a block from the exact 9-byte reference layout after scale validation.
    pub fn from_packed_bytes(bytes: [u8; FP4_KV_BLOCK_PACKED_BYTES]) -> Result<Self, Fp4KvError> {
        decode_e4m3_scale(bytes[8])?;
        Ok(Self {
            payload: bytes[..8].try_into().expect("fixed eight-byte slice"),
            scale_e4m3: bytes[8],
        })
    }

    #[must_use]
    pub fn packed_bytes(self) -> [u8; FP4_KV_BLOCK_PACKED_BYTES] {
        let mut bytes = [0u8; FP4_KV_BLOCK_PACKED_BYTES];
        bytes[..8].copy_from_slice(&self.payload);
        bytes[8] = self.scale_e4m3;
        bytes
    }

    #[must_use]
    pub const fn scale_code(self) -> u8 {
        self.scale_e4m3
    }

    /// Decode all 16 reconstructed values into `f32`.
    pub fn decode(self) -> Result<[f32; FP4_KV_BLOCK_SIZE], Fp4KvError> {
        let scale = decode_e4m3_scale(self.scale_e4m3)?;
        let mut output = [0.0f32; FP4_KV_BLOCK_SIZE];
        for (index, value) in output.iter_mut().enumerate()
        {
            let byte = self.payload[index / 2];
            let code = if index % 2 == 0
            {
                byte & 0x0f
            }
            else
            {
                byte >> 4
            };
            *value = decode_e2m1(code) * scale;
        }
        Ok(output)
    }

    #[must_use]
    pub const fn packed_len(self) -> usize {
        FP4_KV_BLOCK_PACKED_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_decodes_complete_positive_value_set() {
        let expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, &value) in expected.iter().enumerate()
        {
            assert_eq!(decode_e2m1(code as u8), value);
            assert_eq!(decode_e2m1(code as u8 | 0x08), -value);
        }
    }

    #[test]
    fn e2m1_rounds_ties_to_even_and_saturates() {
        assert_eq!(encode_e2m1_sat(0.25).unwrap(), 0);
        assert_eq!(encode_e2m1_sat(0.75).unwrap(), 2);
        assert_eq!(encode_e2m1_sat(1.25).unwrap(), 2);
        assert_eq!(encode_e2m1_sat(1.75).unwrap(), 4);
        assert_eq!(encode_e2m1_sat(2.5).unwrap(), 4);
        assert_eq!(encode_e2m1_sat(3.5).unwrap(), 6);
        assert_eq!(encode_e2m1_sat(5.0).unwrap(), 6);
        assert_eq!(encode_e2m1_sat(99.0).unwrap(), 7);
        assert_eq!(encode_e2m1_sat(-99.0).unwrap(), 15);
    }

    #[test]
    fn e4m3_known_endpoints_decode_exactly() {
        assert_eq!(decode_e4m3_scale(0).unwrap(), 0.0);
        assert_eq!(decode_e4m3_scale(1).unwrap(), E4M3_MIN_POSITIVE);
        assert_eq!(decode_e4m3_scale(0x38).unwrap(), 1.0);
        assert_eq!(decode_e4m3_scale(0x7e).unwrap(), E4M3_MAX);
        assert_eq!(decode_e4m3_scale(0x7f), Err(Fp4KvError::NanScaleEncoding));
    }

    #[test]
    fn e4m3_encoder_is_deterministic_and_saturating() {
        assert_eq!(encode_e4m3_scale_sat(0.0).unwrap(), 0);
        assert_eq!(encode_e4m3_scale_sat(1.0).unwrap(), 0x38);
        assert_eq!(encode_e4m3_scale_sat(E4M3_MAX).unwrap(), 0x7e);
        assert_eq!(encode_e4m3_scale_sat(10_000.0).unwrap(), 0x7e);
    }

    #[test]
    fn block_round_trip_is_exact_for_representable_values_at_unit_scale() {
        let input = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
        ];
        let block = Fp4KvBlock16::from_f32(input).unwrap();
        assert_eq!(block.scale_code(), 0x38);
        assert_eq!(block.packed_len(), 9);
        assert_eq!(block.decode().unwrap(), input);
        assert_eq!(
            Fp4KvBlock16::from_packed_bytes(block.packed_bytes()).unwrap(),
            block
        );
    }

    #[test]
    fn zero_block_uses_zero_scale_and_zero_payload() {
        let block = Fp4KvBlock16::from_f32([0.0; 16]).unwrap();
        assert_eq!(block.packed_bytes(), [0; 9]);
        assert_eq!(block.decode().unwrap(), [0.0; 16]);
    }

    #[test]
    fn rejects_non_finite_inputs_and_invalid_scales() {
        let mut values = [0.0; 16];
        values[3] = f32::NAN;
        assert_eq!(Fp4KvBlock16::from_f32(values), Err(Fp4KvError::NonFinite));
        assert_eq!(encode_e4m3_scale_sat(-1.0), Err(Fp4KvError::NegativeScale));
        assert_eq!(
            decode_e4m3_scale(0x80),
            Err(Fp4KvError::NegativeScaleEncoding)
        );
    }
}
