//! Decimal-exact contracts for venue-facing economic values.
//!
//! The existing `f32` trading primitives remain useful for indicators,
//! statistics and historical simulation. They are not an exact venue boundary.
//! This module is the exact contract for prices, quantities, instrument filters
//! and order normalization used by execution runtimes/adapters.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::orders::{Side, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Price(#[serde(with = "rust_decimal::serde::str")] pub Decimal);

impl Price {
    pub fn new(value: Decimal) -> Result<Self, FinancialError> {
        if value <= Decimal::ZERO
        {
            return Err(FinancialError::NonPositivePrice);
        }
        Ok(Self(value))
    }

    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        let value = Decimal::from_str_exact(value).map_err(|_| FinancialError::InvalidDecimal)?;
        Self::new(value)
    }

    pub fn value(self) -> Decimal {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Quantity(#[serde(with = "rust_decimal::serde::str")] pub Decimal);

impl Quantity {
    pub fn new(value: Decimal) -> Result<Self, FinancialError> {
        if value <= Decimal::ZERO
        {
            return Err(FinancialError::NonPositiveQuantity);
        }
        Ok(Self(value))
    }

    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        let value = Decimal::from_str_exact(value).map_err(|_| FinancialError::InvalidDecimal)?;
        Self::new(value)
    }

    pub fn value(self) -> Decimal {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NonNegativeMoney(#[serde(with = "rust_decimal::serde::str")] pub Decimal);

impl NonNegativeMoney {
    pub fn new(value: Decimal) -> Result<Self, FinancialError> {
        if value < Decimal::ZERO
        {
            return Err(FinancialError::NegativeMoney);
        }
        Ok(Self(value))
    }

    pub fn parse(value: &str) -> Result<Self, FinancialError> {
        let value = Decimal::from_str_exact(value).map_err(|_| FinancialError::InvalidDecimal)?;
        Self::new(value)
    }

    pub fn value(self) -> Decimal {
        self.0
    }
}

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
    #[serde(with = "rust_decimal::serde::str")]
    pub requested: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub effective: Decimal,
    pub direction: NormalizationDirection,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedOrder {
    pub order: ExactOrderRequest,
    pub changes: Vec<NormalizationChange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinancialError {
    InvalidDecimal,
    NonPositivePrice,
    NonPositiveQuantity,
    NegativeMoney,
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

fn aligned(value: Decimal, step: Decimal) -> Result<bool, FinancialError> {
    Ok(value
        .checked_rem(step)
        .ok_or(FinancialError::ArithmeticOverflow)?
        == Decimal::ZERO)
}

fn round_down(value: Decimal, step: Decimal) -> Result<Decimal, FinancialError> {
    let remainder = value
        .checked_rem(step)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    value
        .checked_sub(remainder)
        .ok_or(FinancialError::ArithmeticOverflow)
}

fn round_up(value: Decimal, step: Decimal) -> Result<Decimal, FinancialError> {
    let remainder = value
        .checked_rem(step)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    if remainder == Decimal::ZERO
    {
        return Ok(value);
    }
    let delta = step
        .checked_sub(remainder)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    value
        .checked_add(delta)
        .ok_or(FinancialError::ArithmeticOverflow)
}

fn normalize_price(
    price: Price,
    tick: Price,
    side: Side,
) -> Result<(Price, Option<NormalizationChange>), FinancialError> {
    let requested = price.value();
    let effective = match side
    {
        Side::Buy => round_down(requested, tick.value())?,
        Side::Sell => round_up(requested, tick.value())?,
    };
    let effective = Price::new(effective)?;
    let change = if effective == price
    {
        None
    }
    else
    {
        Some(NormalizationChange {
            field: NormalizationField::LimitPrice,
            requested,
            effective: effective.value(),
            direction: match side
            {
                Side::Buy => NormalizationDirection::Down,
                Side::Sell => NormalizationDirection::Up,
            },
            reason: "align limit-like price to venue tick without making the order more aggressive"
                .to_string(),
        })
    };
    Ok((effective, change))
}

fn normalize_stop(
    stop: Price,
    tick: Price,
    side: Side,
) -> Result<(Price, Option<NormalizationChange>), FinancialError> {
    let requested = stop.value();
    let effective = match side
    {
        Side::Buy => round_up(requested, tick.value())?,
        Side::Sell => round_down(requested, tick.value())?,
    };
    let effective = Price::new(effective)?;
    let change = if effective == stop
    {
        None
    }
    else
    {
        Some(NormalizationChange {
            field: NormalizationField::StopPrice,
            requested,
            effective: effective.value(),
            direction: match side
            {
                Side::Buy => NormalizationDirection::Up,
                Side::Sell => NormalizationDirection::Down,
            },
            reason: "align stop trigger to venue tick without triggering earlier than requested"
                .to_string(),
        })
    };
    Ok((effective, change))
}

pub fn normalize_order(
    request: &ExactOrderRequest,
    rules: &ExactInstrumentRules,
    reference: Option<ReferencePrice>,
    now_ms: i64,
) -> Result<NormalizedOrder, FinancialError> {
    rules.validate()?;
    if request.instrument_id != rules.instrument_id
    {
        return Err(FinancialError::InstrumentMismatch);
    }
    if request.rules_version != rules.rules_version
    {
        return Err(FinancialError::RulesVersionMismatch);
    }
    if request.post_only && !matches!(request.order_type, ExactOrderType::Limit { .. })
    {
        return Err(FinancialError::InvalidOrderCombination);
    }

    let mut changes = Vec::new();
    let requested_quantity = request.quantity.value();
    let effective_quantity = round_down(requested_quantity, rules.quantity_step.value())?;
    let effective_quantity = Quantity::new(effective_quantity)?;
    if effective_quantity != request.quantity
    {
        changes.push(NormalizationChange {
            field: NormalizationField::Quantity,
            requested: requested_quantity,
            effective: effective_quantity.value(),
            direction: NormalizationDirection::Down,
            reason: "align quantity to venue lot step".to_string(),
        });
    }

    let order_type = match request.order_type
    {
        ExactOrderType::Market => ExactOrderType::Market,
        ExactOrderType::Limit { price } =>
        {
            let (price, change) = normalize_price(price, rules.price_tick, request.side)?;
            changes.extend(change);
            ExactOrderType::Limit { price }
        },
        ExactOrderType::StopMarket { stop } =>
        {
            let (stop, change) = normalize_stop(stop, rules.price_tick, request.side)?;
            changes.extend(change);
            ExactOrderType::StopMarket { stop }
        },
        ExactOrderType::StopLimit { stop, limit } =>
        {
            let (stop, stop_change) = normalize_stop(stop, rules.price_tick, request.side)?;
            let (limit, limit_change) = normalize_price(limit, rules.price_tick, request.side)?;
            changes.extend(stop_change);
            changes.extend(limit_change);
            ExactOrderType::StopLimit { stop, limit }
        },
        ExactOrderType::TakeProfit { price } =>
        {
            let (price, change) = normalize_price(price, rules.price_tick, request.side)?;
            changes.extend(change);
            ExactOrderType::TakeProfit { price }
        },
    };

    let normalized = ExactOrderRequest {
        instrument_id: request.instrument_id.clone(),
        side: request.side,
        order_type,
        quantity: effective_quantity,
        tif: request.tif,
        reduce_only: request.reduce_only,
        post_only: request.post_only,
        rules_version: request.rules_version.clone(),
    };
    validate_order(&normalized, rules, reference, now_ms)?;
    Ok(NormalizedOrder {
        order: normalized,
        changes,
    })
}

pub fn validate_order(
    request: &ExactOrderRequest,
    rules: &ExactInstrumentRules,
    reference: Option<ReferencePrice>,
    now_ms: i64,
) -> Result<(), FinancialError> {
    rules.validate()?;
    if request.instrument_id != rules.instrument_id
    {
        return Err(FinancialError::InstrumentMismatch);
    }
    if request.rules_version != rules.rules_version
    {
        return Err(FinancialError::RulesVersionMismatch);
    }
    if request.post_only && !matches!(request.order_type, ExactOrderType::Limit { .. })
    {
        return Err(FinancialError::InvalidOrderCombination);
    }

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
    if !aligned(request.quantity.value(), rules.quantity_step.value())?
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
            if !aligned(stop.value(), rules.price_tick.value())?
            {
                return Err(FinancialError::StopPriceNotAligned);
            }
        },
        ExactOrderType::StopLimit { stop, limit } =>
        {
            if !aligned(stop.value(), rules.price_tick.value())?
            {
                return Err(FinancialError::StopPriceNotAligned);
            }
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
    let notional = notional_price
        .value()
        .checked_mul(request.quantity.value())
        .ok_or(FinancialError::ArithmeticOverflow)?;
    if notional < rules.min_notional.value()
    {
        return Err(FinancialError::NotionalBelowMinimum);
    }
    if rules
        .max_notional
        .is_some_and(|maximum| notional > maximum.value())
    {
        return Err(FinancialError::NotionalAboveMaximum);
    }
    Ok(())
}

fn validate_price_alignment(price: Price, tick: Price) -> Result<(), FinancialError> {
    if !aligned(price.value(), tick.value())?
    {
        return Err(FinancialError::PriceNotAligned);
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
            venue: "qualification".into(),
            instrument_id: "BTC-USDT".into(),
            base_asset: "BTC".into(),
            quote_asset: "USDT".into(),
            rules_version: "rules-1".into(),
            price_tick: p("0.05"),
            quantity_step: q("0.003"),
            min_quantity: q("0.006"),
            max_quantity: Some(q("100")),
            min_notional: money("10"),
            max_notional: Some(money("1000000")),
        }
    }

    fn limit(side: Side, quantity: &str, price: &str) -> ExactOrderRequest {
        ExactOrderRequest {
            instrument_id: "BTC-USDT".into(),
            side,
            order_type: ExactOrderType::Limit { price: p(price) },
            quantity: q(quantity),
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: false,
            rules_version: "rules-1".into(),
        }
    }

    #[test]
    fn decimal_roundtrip_preserves_non_binary_precision() {
        let price = p("100000.01");
        let json = serde_json::to_string(&price).unwrap();
        assert_eq!(json, "\"100000.01\"");
        let restored: Price = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, price);
        assert_eq!(restored.value().to_string(), "100000.01");
    }

    #[test]
    fn strict_validation_rejects_unaligned_quantity_and_price() {
        let request = limit(Side::Buy, "1.004", "100.03");
        assert_eq!(
            validate_order(&request, &rules(), None, 0),
            Err(FinancialError::QuantityNotAligned)
        );

        let request = limit(Side::Buy, "1.002", "100.03");
        assert_eq!(
            validate_order(&request, &rules(), None, 0),
            Err(FinancialError::PriceNotAligned)
        );
    }

    #[test]
    fn normalization_reports_every_effective_change() {
        let buy = normalize_order(&limit(Side::Buy, "1.004", "100.03"), &rules(), None, 0)
            .unwrap();
        assert_eq!(buy.order.quantity.value().to_string(), "1.002");
        let ExactOrderType::Limit { price } = buy.order.order_type
        else
        {
            panic!("limit expected");
        };
        assert_eq!(price.value().to_string(), "100.00");
        assert_eq!(buy.changes.len(), 2);

        let sell = normalize_order(&limit(Side::Sell, "1.004", "100.03"), &rules(), None, 0)
            .unwrap();
        let ExactOrderType::Limit { price } = sell.order.order_type
        else
        {
            panic!("limit expected");
        };
        assert_eq!(price.value().to_string(), "100.05");
    }

    #[test]
    fn stop_trigger_normalization_does_not_trigger_earlier() {
        let buy_request = ExactOrderRequest {
            instrument_id: "BTC-USDT".into(),
            side: Side::Buy,
            order_type: ExactOrderType::StopMarket { stop: p("100.03") },
            quantity: q("1.002"),
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: false,
            rules_version: "rules-1".into(),
        };
        let reference = ReferencePrice {
            price: p("100"),
            observed_at_ms: 90,
            valid_until_ms: 110,
        };
        let normalized = normalize_order(&buy_request, &rules(), Some(reference), 100).unwrap();
        let ExactOrderType::StopMarket { stop } = normalized.order.order_type
        else
        {
            panic!("stop market expected");
        };
        assert_eq!(stop.value().to_string(), "100.05");
    }

    #[test]
    fn market_notional_requires_a_fresh_reference_price() {
        let request = ExactOrderRequest {
            instrument_id: "BTC-USDT".into(),
            side: Side::Buy,
            order_type: ExactOrderType::Market,
            quantity: q("0.102"),
            tif: TimeInForce::Ioc,
            reduce_only: false,
            post_only: false,
            rules_version: "rules-1".into(),
        };
        assert_eq!(
            validate_order(&request, &rules(), None, 100),
            Err(FinancialError::MissingReferencePrice)
        );
        let stale = ReferencePrice {
            price: p("100"),
            observed_at_ms: 0,
            valid_until_ms: 99,
        };
        assert_eq!(
            validate_order(&request, &rules(), Some(stale), 100),
            Err(FinancialError::StaleReferencePrice)
        );
        let fresh = ReferencePrice {
            price: p("100"),
            observed_at_ms: 90,
            valid_until_ms: 110,
        };
        assert!(validate_order(&request, &rules(), Some(fresh), 100).is_ok());
    }

    #[test]
    fn rules_version_cannot_change_between_plan_and_validation() {
        let mut request = limit(Side::Buy, "1.002", "100.00");
        request.rules_version = "old-rules".into();
        assert_eq!(
            validate_order(&request, &rules(), None, 0),
            Err(FinancialError::RulesVersionMismatch)
        );
    }

    #[test]
    fn direct_deserialization_still_hits_domain_validation() {
        let raw = r#"{
            "instrument_id":"BTC-USDT",
            "side":"Buy",
            "order_type":{"Limit":{"price":"100.03"}},
            "quantity":"1.004",
            "tif":"Gtc",
            "reduce_only":false,
            "post_only":false,
            "rules_version":"rules-1"
        }"#;
        let request: ExactOrderRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(
            validate_order(&request, &rules(), None, 0),
            Err(FinancialError::QuantityNotAligned)
        );
    }
}
