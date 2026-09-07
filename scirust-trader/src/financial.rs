//! Decimal-exact contracts for venue-facing economic values.
//!
//! Simulation/statistical code may continue to use floating point where its
//! semantics allow it. Venue-facing prices, quantities, fees and notionals use
//! the exact base-10 types exported here and serialize as JSON strings.

mod exact;

pub use exact::{
    ExactInstrumentRules, ExactOrderRequest, ExactOrderType, FinancialError, NonNegativeMoney,
    NormalizationChange, NormalizationDirection, NormalizationField, NormalizedOrder, Price,
    Quantity, ReferencePrice, SignedAmount, normalize_order, validate_order,
};
