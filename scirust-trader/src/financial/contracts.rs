use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

use super::decimal::{
    FinancialError, NonNegativeMoney, Price, Quantity, aligned, decimal_cmp_parts, round_down,
    round_up,
};
use crate::orders::{Side, TimeInForce};

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

fn price_from_exact(value: super::decimal::ExactDecimal) -> Result<Price, FinancialError> {
    Price::parse(&value.as_string())
}

fn quantity_from_exact(value: super::decimal::ExactDecimal) -> Result<Quantity, FinancialError> {
    Quantity::parse(&value.as_string())
}

fn normalize_limit_price(
    price: Price,
    tick: Price,
    side: Side,
) -> Result<(Price, Option<NormalizationChange>), FinancialError> {
    let effective = match side
    {
        Side::Buy => round_down(price.exact(), tick.exact())?,
        Side::Sell => round_up(price.exact(), tick.exact())?,
    };
    let effective = price_from_exact(effective)?;
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
        Side::Buy => round_up(stop.exact(), tick.exact())?,
        Side::Sell => round_down(stop.exact(), tick.exact())?,
    };
    let effective = price_from_exact(effective)?;
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

    let effective_quantity = quantity_from_exact(round_down(
        request.quantity.exact(),
        rules.quantity_step.exact(),
    )?)?;
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
    if !aligned(request.quantity.exact(), rules.quantity_step.exact())?
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
        ExactOrderType::Market => {},
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
    if !aligned(price.exact(), tick.exact())?
    {
        return Err(FinancialError::PriceNotAligned);
    }
    Ok(())
}

fn validate_stop_alignment(stop: Price, tick: Price) -> Result<(), FinancialError> {
    if !aligned(stop.exact(), tick.exact())?
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
    let (price_mantissa, price_scale) = price.exact().parts();
    let (quantity_mantissa, quantity_scale) = quantity.exact().parts();
    let product = price_mantissa
        .checked_mul(quantity_mantissa)
        .ok_or(FinancialError::ArithmeticOverflow)?;
    let product_scale = price_scale
        .checked_add(quantity_scale)
        .ok_or(FinancialError::ArithmeticOverflow)?;

    let (minimum_mantissa, minimum_scale) = rules.min_notional.exact().parts();
    if decimal_cmp_parts(product, product_scale, minimum_mantissa, minimum_scale) == Ordering::Less
    {
        return Err(FinancialError::NotionalBelowMinimum);
    }
    if let Some(maximum) = rules.max_notional
    {
        let (maximum_mantissa, maximum_scale) = maximum.exact().parts();
        if decimal_cmp_parts(product, product_scale, maximum_mantissa, maximum_scale)
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
        let buy = request(
            Side::Buy,
            ExactOrderType::StopMarket { stop: p("100.03") },
            "0.05",
        );
        let reference = ReferencePrice {
            price: p("101"),
            observed_at_ms: 100,
            valid_until_ms: 200,
        };
        let normalized = normalize_order(&buy, &rules(), Some(reference), 150).unwrap();
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
    fn rules_version_and_invalid_order_combinations_are_rejected() {
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
