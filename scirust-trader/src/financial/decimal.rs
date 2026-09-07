use core::cmp::Ordering;
use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MAX_DECIMAL_SCALE: u32 = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ExactDecimal {
    mantissa: i128,
    scale: u32,
}

impl ExactDecimal {
    pub(super) const ZERO: Self = Self {
        mantissa: 0,
        scale: 0,
    };

    fn parse(input: &str) -> Result<Self, FinancialError> {
        if input.is_empty() || input.trim() != input
        {
            return Err(FinancialError::InvalidDecimal);
        }

        let (negative, body) = match input.as_bytes().first()
        {
            Some(b'-') => (true, &input[1..]),
            Some(b'+') => (false, &input[1..]),
            _ => (false, input),
        };
        if body.is_empty()
        {
            return Err(FinancialError::InvalidDecimal);
        }

        let mut pieces = body.split('.');
        let integer = pieces.next().ok_or(FinancialError::InvalidDecimal)?;
        let fraction = pieces.next();
        if pieces.next().is_some()
            || integer.is_empty()
            || !integer.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(FinancialError::InvalidDecimal);
        }

        let mut fraction = fraction.unwrap_or("");
        if !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(FinancialError::InvalidDecimal);
        }
        fraction = fraction.trim_end_matches('0');
        if fraction.len() > MAX_DECIMAL_SCALE as usize
        {
            return Err(FinancialError::ExcessPrecision);
        }

        let digits = if fraction.is_empty()
        {
            integer.to_string()
        }
        else
        {
            format!("{integer}{fraction}")
        };
        let digits = digits.trim_start_matches('0');
        let unsigned_text = if digits.is_empty() { "0" } else { digits };
        let signed_text = if negative && unsigned_text != "0"
        {
            format!("-{unsigned_text}")
        }
        else
        {
            unsigned_text.to_string()
        };
        let mantissa = signed_text
            .parse::<i128>()
            .map_err(|_| FinancialError::ArithmeticOverflow)?;
        Ok(Self::normalized(mantissa, fraction.len() as u32))
    }

    const fn normalized(mut mantissa: i128, mut scale: u32) -> Self {
        while scale > 0 && mantissa % 10 == 0
        {
            mantissa /= 10;
            scale -= 1;
        }
        if mantissa == 0
        {
            scale = 0;
        }
        Self { mantissa, scale }
    }

    pub(super) fn as_string(self) -> String {
        if self.scale == 0
        {
            return self.mantissa.to_string();
        }
        let negative = self.mantissa < 0;
        let digits = self.mantissa.unsigned_abs().to_string();
        let scale = self.scale as usize;
        let rendered = if digits.len() <= scale
        {
            format!("0.{}{}", "0".repeat(scale - digits.len()), digits)
        }
        else
        {
            let split = digits.len() - scale;
            format!("{}.{}", &digits[..split], &digits[split..])
        };
        if negative
        {
            format!("-{rendered}")
        }
        else
        {
            rendered
        }
    }

    pub(super) const fn is_positive(self) -> bool {
        self.mantissa > 0
    }

    const fn is_negative(self) -> bool {
        self.mantissa < 0
    }

    const fn is_zero(self) -> bool {
        self.mantissa == 0
    }

    pub(super) fn checked_add(self, rhs: Self) -> Result<Self, FinancialError> {
        let (left, right, scale) = common_scale(self, rhs)?;
        let sum = left
            .checked_add(right)
            .ok_or(FinancialError::ArithmeticOverflow)?;
        Ok(Self::normalized(sum, scale))
    }

    pub(super) fn checked_sub(self, rhs: Self) -> Result<Self, FinancialError> {
        let (left, right, scale) = common_scale(self, rhs)?;
        let difference = left
            .checked_sub(right)
            .ok_or(FinancialError::ArithmeticOverflow)?;
        Ok(Self::normalized(difference, scale))
    }

    pub(super) const fn parts(self) -> (i128, u32) {
        (self.mantissa, self.scale)
    }
}

impl Ord for ExactDecimal {
    fn cmp(&self, other: &Self) -> Ordering {
        decimal_cmp_parts(self.mantissa, self.scale, other.mantissa, other.scale)
    }
}

impl PartialOrd for ExactDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn decimal_cmp_parts(
    left_mantissa: i128,
    left_scale: u32,
    right_mantissa: i128,
    right_scale: u32,
) -> Ordering {
    match (left_mantissa.signum(), right_mantissa.signum())
    {
        (left, right) if left != right => return left.cmp(&right),
        (0, 0) => return Ordering::Equal,
        _ => {},
    }

    let negative = left_mantissa < 0;
    let left = left_mantissa.unsigned_abs().to_string();
    let right = right_mantissa.unsigned_abs().to_string();
    let left_exponent = left.len() as i64 - i64::from(left_scale);
    let right_exponent = right.len() as i64 - i64::from(right_scale);

    let magnitude = if left_exponent != right_exponent
    {
        left_exponent.cmp(&right_exponent)
    }
    else
    {
        let width = left.len().max(right.len());
        let left_padded = format!("{left}{}", "0".repeat(width - left.len()));
        let right_padded = format!("{right}{}", "0".repeat(width - right.len()));
        left_padded.cmp(&right_padded)
    };
    if negative
    {
        magnitude.reverse()
    }
    else
    {
        magnitude
    }
}

fn scale_factor(diff: u32) -> Result<i128, FinancialError> {
    10_i128
        .checked_pow(diff)
        .ok_or(FinancialError::ArithmeticOverflow)
}

pub(super) fn common_scale(
    left: ExactDecimal,
    right: ExactDecimal,
) -> Result<(i128, i128, u32), FinancialError> {
    let scale = left.scale.max(right.scale);
    let left_factor = scale_factor(scale - left.scale)?;
    let right_factor = scale_factor(scale - right.scale)?;
    let left = left
        .mantissa
        .checked_mul(left_factor)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    let right = right
        .mantissa
        .checked_mul(right_factor)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    Ok((left, right, scale))
}

pub(super) fn aligned(value: ExactDecimal, step: ExactDecimal) -> Result<bool, FinancialError> {
    let (value, step, _) = common_scale(value, step)?;
    if step <= 0
    {
        return Err(FinancialError::InvalidStep);
    }
    Ok(value % step == 0)
}

pub(super) fn round_down(
    value: ExactDecimal,
    step: ExactDecimal,
) -> Result<ExactDecimal, FinancialError> {
    let (value, step, scale) = common_scale(value, step)?;
    if value <= 0 || step <= 0
    {
        return Err(FinancialError::InvalidStep);
    }
    Ok(ExactDecimal::normalized(value - value % step, scale))
}

pub(super) fn round_up(
    value: ExactDecimal,
    step: ExactDecimal,
) -> Result<ExactDecimal, FinancialError> {
    let (value, step, scale) = common_scale(value, step)?;
    if value <= 0 || step <= 0
    {
        return Err(FinancialError::InvalidStep);
    }
    let remainder = value % step;
    if remainder == 0
    {
        return Ok(ExactDecimal::normalized(value, scale));
    }
    let effective = value
        .checked_add(step - remainder)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    Ok(ExactDecimal::normalized(effective, scale))
}

macro_rules! exact_positive_type {
    ($name:ident, $non_positive:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(ExactDecimal);

        impl $name {
            pub fn parse(value: &str) -> Result<Self, FinancialError> {
                let value = ExactDecimal::parse(value)?;
                if !value.is_positive()
                {
                    return Err(FinancialError::$non_positive);
                }
                Ok(Self(value))
            }

            pub fn as_decimal_string(self) -> String {
                self.0.as_string()
            }

            pub(super) const fn exact(self) -> ExactDecimal {
                self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.0.as_string())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let text = String::deserialize(deserializer)?;
                Self::parse(&text).map_err(serde::de::Error::custom)
            }
        }
    };
}

exact_positive_type!(Price, NonPositivePrice);
exact_positive_type!(Quantity, NonPositiveQuantity);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonNegativeMoney(ExactDecimal);

impl NonNegativeMoney {
    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        let value = ExactDecimal::parse(value)?;
        if value.is_negative()
        {
            return Err(FinancialError::NegativeMoney);
        }
        Ok(Self(value))
    }

    pub fn as_decimal_string(self) -> String {
        self.0.as_string()
    }

    pub(super) const fn exact(self) -> ExactDecimal {
        self.0
    }
}

impl Serialize for NonNegativeMoney {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.as_string())
    }
}

impl<'de> Deserialize<'de> for NonNegativeMoney {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Exact signed base-10 amount for fees, rebates and ledger deltas.
///
/// Zero is representable, unlike `Price` and `Quantity`. Values serialize as
/// decimal strings and checked arithmetic rejects range overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedAmount(ExactDecimal);

impl SignedAmount {
    pub const ZERO: Self = Self(ExactDecimal::ZERO);

    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        ExactDecimal::parse(value).map(Self)
    }

    pub fn as_decimal_string(self) -> String {
        self.0.as_string()
    }

    pub const fn is_zero(self) -> bool {
        self.0.is_zero()
    }

    pub const fn is_negative(self) -> bool {
        self.0.is_negative()
    }

    pub fn checked_add(self, rhs: Self) -> Result<Self, FinancialError> {
        self.0.checked_add(rhs.0).map(Self)
    }

    pub fn checked_sub(self, rhs: Self) -> Result<Self, FinancialError> {
        self.0.checked_sub(rhs.0).map(Self)
    }

    pub fn checked_add_quantity(self, rhs: Quantity) -> Result<Self, FinancialError> {
        self.0.checked_add(rhs.0).map(Self)
    }

    pub fn checked_sub_quantity(self, rhs: Quantity) -> Result<Self, FinancialError> {
        self.0.checked_sub(rhs.0).map(Self)
    }

    pub(super) const fn exact(self) -> ExactDecimal {
        self.0
    }
}

impl From<Quantity> for SignedAmount {
    fn from(value: Quantity) -> Self {
        Self(value.0)
    }
}

impl Serialize for SignedAmount {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.as_string())
    }
}

impl<'de> Deserialize<'de> for SignedAmount {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinancialError {
    InvalidDecimal,
    ExcessPrecision,
    NonPositivePrice,
    NonPositiveQuantity,
    NegativeMoney,
    InvalidStep,
    InvalidInstrumentIdentity,
    InvalidQuantityBounds,
    InvalidNotionalBounds,
    InstrumentMismatch,
    RulesVersionMismatch,
    QuantityBelowMinimum,
    QuantityAboveMaximum,
    QuantityNotAligned,
    PriceNotAligned,
    StopPriceNotAligned,
    MissingReferencePrice,
    InvalidReferenceWindow,
    StaleReferencePrice,
    NotionalBelowMinimum,
    NotionalAboveMaximum,
    ArithmeticOverflow,
    InvalidOrderCombination,
}

impl fmt::Display for FinancialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for FinancialError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_decimal_roundtrip_never_uses_json_number() {
        let price = Price::parse("100000.010000").unwrap();
        assert_eq!(price.as_decimal_string(), "100000.01");
        let json = serde_json::to_string(&price).unwrap();
        assert_eq!(json, "\"100000.01\"");
        assert_eq!(serde_json::from_str::<Price>(&json).unwrap(), price);
    }

    #[test]
    fn parser_rejects_scientific_notation_whitespace_and_excess_precision() {
        assert_eq!(Price::parse("1e-8"), Err(FinancialError::InvalidDecimal));
        assert_eq!(Price::parse(" 1"), Err(FinancialError::InvalidDecimal));
        assert_eq!(
            Price::parse("0.1234567890123456789"),
            Err(FinancialError::ExcessPrecision)
        );
    }

    #[test]
    fn signed_amount_supports_exact_fee_and_rebate_arithmetic() {
        let fees = SignedAmount::parse("0.001").unwrap();
        let rebate = SignedAmount::parse("-0.0002").unwrap();
        assert_eq!(
            fees.checked_add(rebate).unwrap().as_decimal_string(),
            "0.0008"
        );
        assert_eq!(
            SignedAmount::ZERO
                .checked_add_quantity(Quantity::parse("0.1").unwrap())
                .unwrap()
                .as_decimal_string(),
            "0.1"
        );
    }

    #[test]
    fn comparisons_are_exact_across_scales() {
        assert!(Price::parse("1.2").unwrap() > Price::parse("1.11").unwrap());
        assert!(
            SignedAmount::parse("-1.2").unwrap() < SignedAmount::parse("-1.11").unwrap()
        );
        assert_eq!(
            NonNegativeMoney::parse("0.0200").unwrap(),
            NonNegativeMoney::parse("0.02").unwrap()
        );
    }
}
