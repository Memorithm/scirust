//! Decimal-exact contracts for venue-facing economic values.
//!
//! Simulation/statistical code may continue to use floating point where its
//! semantics allow it. Venue-facing prices, quantities, fees and notionals use
//! the exact base-10 types exported here and serialize as JSON strings.

mod contracts;
mod decimal;

pub use contracts::{
    ExactInstrumentRules, ExactOrderRequest, ExactOrderType, NormalizationChange,
    NormalizationDirection, NormalizationField, NormalizedOrder, ReferencePrice, normalize_order,
    validate_order,
};
pub use decimal::{FinancialError, NonNegativeMoney, Price, Quantity, SignedAmount};
