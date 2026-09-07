use core::cmp::Ordering;
use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::orders::{Side, TimeInForce};

const MAX_DECIMAL_SCALE: u32 = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Decimal {
    mantissa: i128,
    scale: u32,
}

impl Decimal {
    const ZERO: Self = Self {
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

        let mut parts = body.split('.');
        let integer = parts.next().ok_or(FinancialError::InvalidDecimal)?;
        let fraction = parts.next().unwrap_or("");
        if parts.next().is_some()
            || integer.is_empty()
            || !integer.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(FinancialError::InvalidDecimal);
        }

        let fraction = fraction.trim_end_matches('0');
        if fraction.len() > MAX_DECIMAL_SCALE as usize
        {
            return Err(FinancialError::ExcessPrecision);
        }

        let digits = format!("{integer}{fraction}");
        let digits = digits.trim_start_matches('0');
        let unsigned = if digits.is_empty() { "0" } else { digits };
        let signed = if negative && unsigned != "0"
        {
            format!("-{unsigned}")
        }
        else
        {
            unsigned.to_string()
        };
        let mantissa = signed
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

    fn render(self) -> String {
        if self.scale == 0
        {
            return self.mantissa.to_string();
        }

        let negative = self.mantissa < 0;
        let digits = self.mantissa.unsigned_abs().to_string();
        let scale = self.scale as usize;
        let text = if digits.len() <= scale
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
            format!("-{text}")
        }
        else
        {
            text
        }
    }

    fn checked_add(self, rhs: Self) -> Result<Self, FinancialError> {
        let (left, right, scale) = common_scale(self, rhs)?;
        let value = left
            .checked_add(right)
            .ok_or(FinancialError::ArithmeticOverflow)?;
        Ok(Self::normalized(value, scale))
    }

    fn checked_sub(self, rhs: Self) -> Result<Self, FinancialError> {
        let (left, right, scale) = common_scale(self, rhs)?;
        let value = left
            .checked_sub(right)
            .ok_or(FinancialError::ArithmeticOverflow)?;
        Ok(Self::normalized(value, scale))
    }
}

impl Ord for Decimal {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_parts(self.mantissa, self.scale, other.mantissa, other.scale)
    }
}

impl PartialOrd for Decimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn compare_parts(
    left_mantissa: i128,
    left_scale: u32,
    right_mantissa: i128,
    right_scale: u32,
) -> Ordering {
    match (left_mantissa.signum(), right_mantissa.signum())
    {
        (left, right) if left != right => return left.cmp(&right),
        (0, 0) => return Ordering::Equal,
        _ =>
        {},
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
        let left = format!("{left}{}", "0".repeat(width - left.len()));
        let right = format!("{right}{}", "0".repeat(width - right.len()));
        left.cmp(&right)
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

fn common_scale(left: Decimal, right: Decimal) -> Result<(i128, i128, u32), FinancialError> {
    let scale = left.scale.max(right.scale);
    let left_factor = 10_i128
        .checked_pow(scale - left.scale)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    let right_factor = 10_i128
        .checked_pow(scale - right.scale)
        .ok_or(FinancialError::ArithmeticOverflow)?;
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

fn aligned(value: Decimal, step: Decimal) -> Result<bool, FinancialError> {
    let (value, step, _) = common_scale(value, step)?;
    if step <= 0
    {
        return Err(FinancialError::InvalidStep);
    }
    Ok(value % step == 0)
}

fn round_down(value: Decimal, step: Decimal) -> Result<Decimal, FinancialError> {
    let (value, step, scale) = common_scale(value, step)?;
    if value <= 0 || step <= 0
    {
        return Err(FinancialError::InvalidStep);
    }
    Ok(Decimal::normalized(value - value % step, scale))
}

fn round_up(value: Decimal, step: Decimal) -> Result<Decimal, FinancialError> {
    let (value, step, scale) = common_scale(value, step)?;
    if value <= 0 || step <= 0
    {
        return Err(FinancialError::InvalidStep);
    }
    let remainder = value % step;
    if remainder == 0
    {
        return Ok(Decimal::normalized(value, scale));
    }
    let value = value
        .checked_add(step - remainder)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    Ok(Decimal::normalized(value, scale))
}

macro_rules! positive_decimal_type {
    ($name:ident, $error:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Decimal);

        impl $name {
            pub fn parse(value: &str) -> Result<Self, FinancialError> {
                let value = Decimal::parse(value)?;
                if value.mantissa <= 0
                {
                    return Err(FinancialError::$error);
                }
                Ok(Self(value))
            }

            pub fn as_decimal_string(self) -> String {
                self.0.render()
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.0.render())
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

positive_decimal_type!(Price, NonPositivePrice);
positive_decimal_type!(Quantity, NonPositiveQuantity);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonNegativeMoney(Decimal);

impl NonNegativeMoney {
    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        let value = Decimal::parse(value)?;
        if value.mantissa < 0
        {
            return Err(FinancialError::NegativeMoney);
        }
        Ok(Self(value))
    }

    pub fn as_decimal_string(self) -> String {
        self.0.render()
    }
}

impl Serialize for NonNegativeMoney {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.render())
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedAmount(Decimal);

impl SignedAmount {
    pub const ZERO: Self = Self(Decimal::ZERO);

    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        Decimal::parse(value).map(Self)
    }

    pub fn as_decimal_string(self) -> String {
        self.0.render()
    }

    pub const fn is_zero(self) -> bool {
        self.0.mantissa == 0
    }

    pub const fn is_negative(self) -> bool {
        self.0.mantissa < 0
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
        serializer.serialize_str(&self.0.render())
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactInstrumentRules {
    pub venue: String,
    pub instrument_id: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub rules_version: String,
    pub price_tick: Price,
    pub quantity_step: Quantity,
    pub min_quantity: Quantity,
    pub max_quantity: Option<Quantity>,
    pub min_notional: NonNegativeMoney,
    pub max_notional: Option<NonNegativeMoney>,
}

impl ExactInstrumentRules {
    pub fn validate(&self) -> Result<(), FinancialError> {
        if self.venue.is_empty()
            || self.instrument_id.is_empty()
            || self.base_asset.is_empty()
            || self.quote_asset.is_empty()
            || self.rules_version.is_empty()
            || self.base_asset == self.quote_asset
        {
            return Err(FinancialError::InvalidInstrumentIdentity);
        }
        if self
            .max_quantity
            .is_some_and(|maximum| maximum < self.min_quantity)
        {
            return Err(FinancialError::InvalidQuantityBounds);
        }
        if self
            .max_notional
            .is_some_and(|maximum| maximum < self.min_notional)
        {
            return Err(FinancialError::InvalidNotionalBounds);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactOrderType {
    Market,
    Limit { price: Price },
    StopMarket { stop: Price },
    StopLimit { stop: Price, limit: Price },
    TakeProfit { price: Price },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactOrderRequest {
    pub instrument_id: String,
    pub side: Side,
    pub order_type: ExactOrderType,
    pub quantity: Quantity,
    pub tif: TimeInForce,
    pub reduce_only: bool,
    pub post_only: bool,
    pub rules_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferencePrice {
    pub price: Price,
    pub observed_at_ms: i64,
    pub valid_until_ms: i64,
}

impl ReferencePrice {
    pub fn validate_at(self, now_ms: i64) -> Result<(), FinancialError> {
        if self.valid_until_ms < self.observed_at_ms
        {
            return Err(FinancialError::InvalidReferenceWindow);
        }
        if self.observed_at_ms > now_ms || now_ms > self.valid_until_ms
        {
            return Err(FinancialError::StaleReferencePrice);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationField {
    Quantity,
    LimitPrice,
    StopPrice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationDirection {
    Down,
    Up,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationChange {
    pub field: NormalizationField,
    pub requested: String,
    pub effective: String,
    pub direction: NormalizationDirection,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedOrder {
    pub order: ExactOrderRequest,
    pub changes: Vec<NormalizationChange>,
}

fn normalize_limit_price(
    price: Price,
    tick: Price,
    side: Side,
) -> Result<(Price, Option<NormalizationChange>), FinancialError> {
    let effective = match side
    {
        Side::Buy => round_down(price.0, tick.0)?,
        Side::Sell => round_up(price.0, tick.0)?,
    };
    let effective = Price(effective);
    let change = (effective != price).then(|| NormalizationChange {
        field: NormalizationField::LimitPrice,
        requested: price.as_decimal_string(),
        effective: effective.as_decimal_string(),
        direction: match side
        {
            Side::Buy => NormalizationDirection::Down,
            Side::Sell => NormalizationDirection::Up,
        },
        reason: "align limit-like price to venue tick without making the order more aggressive"
            .to_string(),
    });
    Ok((effective, change))
}

fn normalize_stop_price(
    stop: Price,
    tick: Price,
    side: Side,
) -> Result<(Price, Option<NormalizationChange>), FinancialError> {
    let effective = match side
    {
        Side::Buy => round_up(stop.0, tick.0)?,
        Side::Sell => round_down(stop.0, tick.0)?,
    };
    let effective = Price(effective);
    let change = (effective != stop).then(|| NormalizationChange {
        field: NormalizationField::StopPrice,
        requested: stop.as_decimal_string(),
        effective: effective.as_decimal_string(),
        direction: match side
        {
            Side::Buy => NormalizationDirection::Up,
            Side::Sell => NormalizationDirection::Down,
        },
        reason: "align stop trigger to venue tick without triggering earlier than requested"
            .to_string(),
    });
    Ok((effective, change))
}

pub fn normalize_order(
    request: &ExactOrderRequest,
    rules: &ExactInstrumentRules,
    reference: Option<ReferencePrice>,
    now_ms: i64,
) -> Result<NormalizedOrder, FinancialError> {
    rules.validate()?;
    ensure_identity(request, rules)?;
    ensure_order_combination(request)?;

    let effective_quantity = Quantity(round_down(request.quantity.0, rules.quantity_step.0)?);
    let mut changes = Vec::new();
    if effective_quantity != request.quantity
    {
        changes.push(NormalizationChange {
            field: NormalizationField::Quantity,
            requested: request.quantity.as_decimal_string(),
            effective: effective_quantity.as_decimal_string(),
            direction: NormalizationDirection::Down,
            reason: "align quantity to venue lot step".to_string(),
        });
    }

    let order_type = match request.order_type
    {
        ExactOrderType::Market => ExactOrderType::Market,
        ExactOrderType::Limit { price } =>
        {
            let (price, change) = normalize_limit_price(price, rules.price_tick, request.side)?;
            changes.extend(change);
            ExactOrderType::Limit { price }
        },
        ExactOrderType::StopMarket { stop } =>
        {
            let (stop, change) = normalize_stop_price(stop, rules.price_tick, request.side)?;
            changes.extend(change);
            ExactOrderType::StopMarket { stop }
        },
        ExactOrderType::StopLimit { stop, limit } =>
        {
            let (stop, stop_change) = normalize_stop_price(stop, rules.price_tick, request.side)?;
            let (limit, limit_change) =
                normalize_limit_price(limit, rules.price_tick, request.side)?;
            changes.extend(stop_change);
            changes.extend(limit_change);
            ExactOrderType::StopLimit { stop, limit }
        },
        ExactOrderType::TakeProfit { price } =>
        {
            let (price, change) = normalize_limit_price(price, rules.price_tick, request.side)?;
            changes.extend(change);
            ExactOrderType::TakeProfit { price }
        },
    };

    let order = ExactOrderRequest {
        instrument_id: request.instrument_id.clone(),
        side: request.side,
        order_type,
        quantity: effective_quantity,
        tif: request.tif,
        reduce_only: request.reduce_only,
        post_only: request.post_only,
        rules_version: request.rules_version.clone(),
    };
    validate_order(&order, rules, reference, now_ms)?;
    Ok(NormalizedOrder { order, changes })
}

pub fn validate_order(
    request: &ExactOrderRequest,
    rules: &ExactInstrumentRules,
    reference: Option<ReferencePrice>,
    now_ms: i64,
) -> Result<(), FinancialError> {
    rules.validate()?;
    ensure_identity(request, rules)?;
    ensure_order_combination(request)?;

    if request.quantity < rules.min_quantity
    {
        return Err(FinancialError::QuantityBelowMinimum);
    }
    if rules
        .max_quantity
        .is_some_and(|maximum| request.quantity > maximum)
    {
        return Err(FinancialError::QuantityAboveMaximum);
    }
    if !aligned(request.quantity.0, rules.quantity_step.0)?
    {
        return Err(FinancialError::QuantityNotAligned);
    }

    match request.order_type
    {
        ExactOrderType::Limit { price } | ExactOrderType::TakeProfit { price } =>
        {
            validate_price_alignment(price, rules.price_tick)?;
        },
        ExactOrderType::StopMarket { stop } =>
        {
            validate_stop_alignment(stop, rules.price_tick)?;
        },
        ExactOrderType::StopLimit { stop, limit } =>
        {
            validate_stop_alignment(stop, rules.price_tick)?;
            validate_price_alignment(limit, rules.price_tick)?;
        },
        ExactOrderType::Market =>
        {},
    }

    let notional_price = match request.order_type
    {
        ExactOrderType::Limit { price }
        | ExactOrderType::TakeProfit { price }
        | ExactOrderType::StopLimit { limit: price, .. } => price,
        ExactOrderType::Market | ExactOrderType::StopMarket { .. } =>
        {
            let reference = reference.ok_or(FinancialError::MissingReferencePrice)?;
            reference.validate_at(now_ms)?;
            reference.price
        },
    };
    validate_notional(notional_price, request.quantity, rules)
}

fn ensure_identity(
    request: &ExactOrderRequest,
    rules: &ExactInstrumentRules,
) -> Result<(), FinancialError> {
    if request.instrument_id != rules.instrument_id
    {
        return Err(FinancialError::InstrumentMismatch);
    }
    if request.rules_version != rules.rules_version
    {
        return Err(FinancialError::RulesVersionMismatch);
    }
    Ok(())
}

fn ensure_order_combination(request: &ExactOrderRequest) -> Result<(), FinancialError> {
    if request.post_only && !matches!(request.order_type, ExactOrderType::Limit { .. })
    {
        return Err(FinancialError::InvalidOrderCombination);
    }
    Ok(())
}

fn validate_price_alignment(price: Price, tick: Price) -> Result<(), FinancialError> {
    if !aligned(price.0, tick.0)?
    {
        return Err(FinancialError::PriceNotAligned);
    }
    Ok(())
}

fn validate_stop_alignment(stop: Price, tick: Price) -> Result<(), FinancialError> {
    if !aligned(stop.0, tick.0)?
    {
        return Err(FinancialError::StopPriceNotAligned);
    }
    Ok(())
}

fn validate_notional(
    price: Price,
    quantity: Quantity,
    rules: &ExactInstrumentRules,
) -> Result<(), FinancialError> {
    let product = price
        .0
        .mantissa
        .checked_mul(quantity.0.mantissa)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    let product_scale = price
        .0
        .scale
        .checked_add(quantity.0.scale)
        .ok_or(FinancialError::ArithmeticOverflow)?;

    if compare_parts(
        product,
        product_scale,
        rules.min_notional.0.mantissa,
        rules.min_notional.0.scale,
    ) == Ordering::Less
    {
        return Err(FinancialError::NotionalBelowMinimum);
    }
    if let Some(maximum) = rules.max_notional
    {
        if compare_parts(product, product_scale, maximum.0.mantissa, maximum.0.scale)
            == Ordering::Greater
        {
            return Err(FinancialError::NotionalAboveMaximum);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(value: &str) -> Price {
        Price::parse(value).unwrap()
    }

    fn q(value: &str) -> Quantity {
        Quantity::parse(value).unwrap()
    }

    fn money(value: &str) -> NonNegativeMoney {
        NonNegativeMoney::parse(value).unwrap()
    }

    fn rules() -> ExactInstrumentRules {
        ExactInstrumentRules {
            venue: "test".to_string(),
            instrument_id: "BTC-USDT".to_string(),
            base_asset: "BTC".to_string(),
            quote_asset: "USDT".to_string(),
            rules_version: "v7".to_string(),
            price_tick: p("0.05"),
            quantity_step: q("0.0001"),
            min_quantity: q("0.0001"),
            max_quantity: Some(q("100")),
            min_notional: money("5"),
            max_notional: Some(money("10000000")),
        }
    }

    fn request(side: Side, order_type: ExactOrderType, quantity: &str) -> ExactOrderRequest {
        ExactOrderRequest {
            instrument_id: "BTC-USDT".to_string(),
            side,
            order_type,
            quantity: q(quantity),
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: false,
            rules_version: "v7".to_string(),
        }
    }

    #[test]
    fn canonical_decimal_roundtrip_never_uses_json_number() {
        let price = p("100000.010000");
        assert_eq!(price.as_decimal_string(), "100000.01");
        let json = serde_json::to_string(&price).unwrap();
        assert_eq!(json, "\"100000.01\"");
        assert_eq!(serde_json::from_str::<Price>(&json).unwrap(), price);
    }

    #[test]
    fn parser_rejects_non_decimal_input_and_excess_precision() {
        assert_eq!(Price::parse("1e-8"), Err(FinancialError::InvalidDecimal));
        assert_eq!(Price::parse(" 1"), Err(FinancialError::InvalidDecimal));
        assert_eq!(
            Price::parse("0.1234567890123456789"),
            Err(FinancialError::ExcessPrecision)
        );
    }

    #[test]
    fn signed_amount_uses_exact_checked_arithmetic() {
        let fees = SignedAmount::parse("0.001").unwrap();
        let rebate = SignedAmount::parse("-0.0002").unwrap();
        assert_eq!(
            fees.checked_add(rebate).unwrap().as_decimal_string(),
            "0.0008"
        );
        assert_eq!(
            SignedAmount::ZERO
                .checked_add_quantity(q("0.1"))
                .unwrap()
                .as_decimal_string(),
            "0.1"
        );
    }

    #[test]
    fn comparison_is_exact_across_decimal_scales() {
        assert!(p("1.2") > p("1.11"));
        assert!(SignedAmount::parse("-1.2").unwrap() < SignedAmount::parse("-1.11").unwrap());
        assert_eq!(money("0.0200"), money("0.02"));
    }

    #[test]
    fn strict_validation_never_rounds_unaligned_values() {
        let order = request(
            Side::Buy,
            ExactOrderType::Limit { price: p("100.03") },
            "0.00015",
        );
        assert_eq!(
            validate_order(&order, &rules(), None, 0),
            Err(FinancialError::QuantityNotAligned)
        );
    }

    #[test]
    fn normalization_is_side_aware_and_explicit() {
        let buy = request(
            Side::Buy,
            ExactOrderType::Limit { price: p("100.03") },
            "0.05009",
        );
        let buy = normalize_order(&buy, &rules(), None, 0).unwrap();
        assert_eq!(buy.order.quantity.as_decimal_string(), "0.05");
        assert_eq!(
            match buy.order.order_type
            {
                ExactOrderType::Limit { price } => price.as_decimal_string(),
                _ => unreachable!(),
            },
            "100"
        );
        assert_eq!(buy.changes.len(), 2);

        let sell = request(
            Side::Sell,
            ExactOrderType::Limit { price: p("100.03") },
            "0.05",
        );
        let sell = normalize_order(&sell, &rules(), None, 0).unwrap();
        assert_eq!(
            match sell.order.order_type
            {
                ExactOrderType::Limit { price } => price.as_decimal_string(),
                _ => unreachable!(),
            },
            "100.05"
        );
    }

    #[test]
    fn stop_rounding_never_triggers_earlier_than_requested() {
        let order = request(
            Side::Buy,
            ExactOrderType::StopMarket { stop: p("100.03") },
            "0.05",
        );
        let reference = ReferencePrice {
            price: p("101"),
            observed_at_ms: 100,
            valid_until_ms: 200,
        };
        let normalized = normalize_order(&order, &rules(), Some(reference), 150).unwrap();
        assert_eq!(
            match normalized.order.order_type
            {
                ExactOrderType::StopMarket { stop } => stop.as_decimal_string(),
                _ => unreachable!(),
            },
            "100.05"
        );
    }

    #[test]
    fn market_notional_requires_fresh_reference_evidence() {
        let market = request(Side::Buy, ExactOrderType::Market, "0.05");
        assert_eq!(
            validate_order(&market, &rules(), None, 150),
            Err(FinancialError::MissingReferencePrice)
        );
        let stale = ReferencePrice {
            price: p("100"),
            observed_at_ms: 100,
            valid_until_ms: 120,
        };
        assert_eq!(
            validate_order(&market, &rules(), Some(stale), 150),
            Err(FinancialError::StaleReferencePrice)
        );
    }

    #[test]
    fn notional_boundary_is_exact_for_decimal_fractions() {
        let mut exact_rules = rules();
        exact_rules.price_tick = p("0.1");
        exact_rules.quantity_step = q("0.1");
        exact_rules.min_quantity = q("0.1");
        exact_rules.min_notional = money("0.02");
        let order = request(Side::Buy, ExactOrderType::Limit { price: p("0.1") }, "0.2");
        assert!(validate_order(&order, &exact_rules, None, 0).is_ok());
        exact_rules.min_notional = money("0.020000000000000001");
        assert_eq!(
            validate_order(&order, &exact_rules, None, 0),
            Err(FinancialError::NotionalBelowMinimum)
        );
    }

    #[test]
    fn version_drift_and_invalid_order_combinations_are_rejected() {
        let mut order = request(Side::Buy, ExactOrderType::Limit { price: p("100") }, "0.05");
        order.rules_version = "v6".to_string();
        assert_eq!(
            validate_order(&order, &rules(), None, 0),
            Err(FinancialError::RulesVersionMismatch)
        );

        let mut market = request(Side::Buy, ExactOrderType::Market, "0.05");
        market.post_only = true;
        assert_eq!(
            validate_order(&market, &rules(), None, 0),
            Err(FinancialError::InvalidOrderCombination)
        );
    }
}
