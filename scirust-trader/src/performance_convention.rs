//! Explicit conventions for annualised trading-performance metrics.
//!
//! This module separates three quantities that are easy to conflate when they
//! are passed around as bare `f32` values:
//!
//! - the number of observation periods in one 24/7 crypto year;
//! - the per-period benchmark used by Sharpe;
//! - the per-period minimum acceptable return used by Sortino.
//!
//! Parsing is deliberately fallible. Unknown or malformed interval strings are
//! never silently reinterpreted as daily observations.

use core::fmt;

/// Validation failures for performance-metric conventions.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceConventionError {
    /// The interval was empty after trimming whitespace.
    EmptyInterval,
    /// The interval omitted its numeric magnitude, for example `"h"`.
    MissingIntervalMagnitude,
    /// The interval omitted its unit, for example `"15"`.
    MissingIntervalUnit,
    /// The interval magnitude could not be parsed as a finite positive number.
    InvalidIntervalMagnitude(String),
    /// The interval unit is not one of the explicitly supported 24/7 units.
    UnsupportedIntervalUnit(String),
    /// Periods-per-year must be finite and strictly positive.
    InvalidPeriodsPerYear(f32),
    /// The Sharpe benchmark must be finite.
    NonFiniteSharpeBenchmark(f32),
    /// The Sortino target must be finite.
    NonFiniteSortinoTarget(f32),
}

impl fmt::Display for PerformanceConventionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyInterval => f.write_str("interval must not be empty"),
            Self::MissingIntervalMagnitude =>
            {
                f.write_str("interval must start with a positive numeric magnitude")
            },
            Self::MissingIntervalUnit => f.write_str("interval must include an explicit unit"),
            Self::InvalidIntervalMagnitude(value) =>
            {
                write!(
                    f,
                    "invalid interval magnitude `{value}`; expected a finite value > 0"
                )
            },
            Self::UnsupportedIntervalUnit(unit) => write!(
                f,
                "unsupported interval unit `{unit}`; expected s/sec, m/min, h/hr, d/day, or w/week"
            ),
            Self::InvalidPeriodsPerYear(value) =>
            {
                write!(f, "periods_per_year must be finite and > 0, got {value}")
            },
            Self::NonFiniteSharpeBenchmark(value) =>
            {
                write!(f, "Sharpe benchmark per period must be finite, got {value}")
            },
            Self::NonFiniteSortinoTarget(value) =>
            {
                write!(f, "Sortino target per period must be finite, got {value}")
            },
        }
    }
}

impl std::error::Error for PerformanceConventionError {}

/// Parse a bar interval into periods per 365-day year for a 24/7 market.
///
/// Accepted units are seconds (`s`, `sec`), minutes (`m`, `min`), hours (`h`,
/// `hr`), days (`d`, `day`), and weeks (`w`, `week`). Unit matching is ASCII
/// case-insensitive and surrounding whitespace is ignored. The numeric
/// magnitude may be fractional, but it must be finite and strictly positive.
///
/// Unlike the historical `metrics::periods_per_year` helper, this function has
/// no fallback for unknown input.
pub fn try_crypto_periods_per_year(interval: &str) -> Result<f32, PerformanceConventionError> {
    let normalized = interval.trim().to_ascii_lowercase();
    if normalized.is_empty()
    {
        return Err(PerformanceConventionError::EmptyInterval);
    }

    let unit_start = normalized
        .find(|c: char| c.is_ascii_alphabetic())
        .ok_or(PerformanceConventionError::MissingIntervalUnit)?;
    let (magnitude_text, unit) = normalized.split_at(unit_start);
    if magnitude_text.is_empty()
    {
        return Err(PerformanceConventionError::MissingIntervalMagnitude);
    }

    let magnitude: f32 = magnitude_text.parse().map_err(|_| {
        PerformanceConventionError::InvalidIntervalMagnitude(magnitude_text.to_string())
    })?;
    if !magnitude.is_finite() || magnitude <= 0.0
    {
        return Err(PerformanceConventionError::InvalidIntervalMagnitude(
            magnitude_text.to_string(),
        ));
    }

    let periods_per_day = match unit
    {
        "s" | "sec" => 86_400.0 / magnitude,
        "m" | "min" => 1_440.0 / magnitude,
        "h" | "hr" => 24.0 / magnitude,
        "d" | "day" => 1.0 / magnitude,
        "w" | "week" => 1.0 / (7.0 * magnitude),
        _ =>
        {
            return Err(PerformanceConventionError::UnsupportedIntervalUnit(
                unit.to_string(),
            ));
        },
    };
    let periods_per_year = periods_per_day * 365.0;
    if !periods_per_year.is_finite() || periods_per_year <= 0.0
    {
        return Err(PerformanceConventionError::InvalidPeriodsPerYear(
            periods_per_year,
        ));
    }
    Ok(periods_per_year)
}

/// Validated convention used to construct annualised performance metrics.
///
/// Sharpe and Sortino deliberately carry separate per-period reference values:
/// a risk-free/benchmark return is not necessarily the same quantity as a
/// minimum acceptable return.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceConvention {
    periods_per_year: f32,
    sharpe_benchmark_per_period: f32,
    sortino_target_per_period: f32,
}

impl PerformanceConvention {
    /// Build a convention from already-resolved annualisation and per-period
    /// reference rates.
    pub fn new(
        periods_per_year: f32,
        sharpe_benchmark_per_period: f32,
        sortino_target_per_period: f32,
    ) -> Result<Self, PerformanceConventionError> {
        if !periods_per_year.is_finite() || periods_per_year <= 0.0
        {
            return Err(PerformanceConventionError::InvalidPeriodsPerYear(
                periods_per_year,
            ));
        }
        if !sharpe_benchmark_per_period.is_finite()
        {
            return Err(PerformanceConventionError::NonFiniteSharpeBenchmark(
                sharpe_benchmark_per_period,
            ));
        }
        if !sortino_target_per_period.is_finite()
        {
            return Err(PerformanceConventionError::NonFiniteSortinoTarget(
                sortino_target_per_period,
            ));
        }
        Ok(Self {
            periods_per_year,
            sharpe_benchmark_per_period,
            sortino_target_per_period,
        })
    }

    /// Resolve a 24/7 crypto interval and build a validated convention.
    pub fn for_crypto_interval(
        interval: &str,
        sharpe_benchmark_per_period: f32,
        sortino_target_per_period: f32,
    ) -> Result<Self, PerformanceConventionError> {
        Self::new(
            try_crypto_periods_per_year(interval)?,
            sharpe_benchmark_per_period,
            sortino_target_per_period,
        )
    }

    /// Number of observation periods in one year.
    pub const fn periods_per_year(self) -> f32 {
        self.periods_per_year
    }

    /// Per-period benchmark subtracted from returns for Sharpe.
    pub const fn sharpe_benchmark_per_period(self) -> f32 {
        self.sharpe_benchmark_per_period
    }

    /// Per-period minimum acceptable return used for Sortino downside risk.
    pub const fn sortino_target_per_period(self) -> f32 {
        self.sortino_target_per_period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn strict_crypto_intervals_match_supported_reference_values() {
        assert!(approx(
            try_crypto_periods_per_year("1d").unwrap(),
            365.0,
            1.0
        ));
        assert!(approx(
            try_crypto_periods_per_year("1h").unwrap(),
            24.0 * 365.0,
            1.0
        ));
        assert!(approx(
            try_crypto_periods_per_year("15m").unwrap(),
            96.0 * 365.0,
            1.0
        ));
        assert!(approx(
            try_crypto_periods_per_year(" 4HR ").unwrap(),
            6.0 * 365.0,
            1.0
        ));
        assert!(approx(
            try_crypto_periods_per_year("0.5day").unwrap(),
            2.0 * 365.0,
            1.0
        ));
    }

    #[test]
    fn malformed_intervals_are_rejected_instead_of_becoming_daily() {
        assert_eq!(
            try_crypto_periods_per_year(""),
            Err(PerformanceConventionError::EmptyInterval)
        );
        assert_eq!(
            try_crypto_periods_per_year("15"),
            Err(PerformanceConventionError::MissingIntervalUnit)
        );
        assert_eq!(
            try_crypto_periods_per_year("h"),
            Err(PerformanceConventionError::MissingIntervalMagnitude)
        );
        assert_eq!(
            try_crypto_periods_per_year("1fortnight"),
            Err(PerformanceConventionError::UnsupportedIntervalUnit(
                "fortnight".to_string()
            ))
        );
        assert!(matches!(
            try_crypto_periods_per_year("0h"),
            Err(PerformanceConventionError::InvalidIntervalMagnitude(_))
        ));
        assert!(matches!(
            try_crypto_periods_per_year("-1h"),
            Err(PerformanceConventionError::InvalidIntervalMagnitude(_))
        ));
    }

    #[test]
    fn convention_keeps_sharpe_and_sortino_references_distinct() {
        let convention = PerformanceConvention::for_crypto_interval("1h", 0.0001, 0.001)
            .expect("valid convention");
        assert!(approx(convention.periods_per_year(), 24.0 * 365.0, 1.0));
        assert_eq!(convention.sharpe_benchmark_per_period(), 0.0001);
        assert_eq!(convention.sortino_target_per_period(), 0.001);
    }

    #[test]
    fn convention_rejects_non_finite_or_non_positive_inputs() {
        assert!(matches!(
            PerformanceConvention::new(0.0, 0.0, 0.0),
            Err(PerformanceConventionError::InvalidPeriodsPerYear(_))
        ));
        assert!(matches!(
            PerformanceConvention::new(f32::NAN, 0.0, 0.0),
            Err(PerformanceConventionError::InvalidPeriodsPerYear(_))
        ));
        assert!(matches!(
            PerformanceConvention::new(365.0, f32::INFINITY, 0.0),
            Err(PerformanceConventionError::NonFiniteSharpeBenchmark(_))
        ));
        assert!(matches!(
            PerformanceConvention::new(365.0, 0.0, f32::NEG_INFINITY),
            Err(PerformanceConventionError::NonFiniteSortinoTarget(_))
        ));
    }
}
