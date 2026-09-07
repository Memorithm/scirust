//! Deterministic execution-realism models.
//!
//! These models make latency, queue position and venue back-pressure explicit
//! inputs to simulation. They do not claim to infer a live venue's hidden queue
//! or matching-engine latency. Callers supply the assumptions or measured
//! values, and replay receives the same result for the same event sequence.

use serde::{Deserialize, Serialize};

/// Fixed one-way and processing latencies for a normalized venue path.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LatencyProfile {
    pub outbound_ms: u64,
    pub venue_processing_ms: u64,
    pub inbound_ms: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LatencyTimeline {
    pub local_submit_ts_ms: i64,
    pub venue_arrival_ts_ms: i64,
    pub venue_effective_ts_ms: i64,
    pub local_ack_ts_ms: i64,
}

impl LatencyProfile {
    /// Deterministically project a local submit timestamp through the configured
    /// transport and venue-processing path. Returns `None` if any configured
    /// duration cannot be represented as `i64` milliseconds or if a timestamp
    /// addition crosses the `i64` range.
    pub fn timeline(&self, local_submit_ts_ms: i64) -> Option<LatencyTimeline> {
        let outbound_ms = i64::try_from(self.outbound_ms).ok()?;
        let venue_processing_ms = i64::try_from(self.venue_processing_ms).ok()?;
        let inbound_ms = i64::try_from(self.inbound_ms).ok()?;
        let arrival = local_submit_ts_ms.checked_add(outbound_ms)?;
        let effective = arrival.checked_add(venue_processing_ms)?;
        let ack = effective.checked_add(inbound_ms)?;
        Some(LatencyTimeline {
            local_submit_ts_ms,
            venue_arrival_ts_ms: arrival,
            venue_effective_ts_ms: effective,
            local_ack_ts_ms: ack,
        })
    }
}

/// FIFO queue-position estimate for one resting order at one price level.
///
/// `ahead_qty` is caller-supplied visible/estimated quantity ahead of the order.
/// Trade flow consumes that quantity first and only then fills `own_qty`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct QueuePosition {
    pub ahead_qty: f32,
    pub own_qty: f32,
    pub filled_qty: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct QueueConsumption {
    pub traded_qty: f32,
    pub consumed_ahead_qty: f32,
    pub own_fill_qty: f32,
    pub ahead_qty_after: f32,
    pub remaining_own_qty: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueError {
    InvalidQuantity,
}

impl QueuePosition {
    pub fn new(ahead_qty: f32, own_qty: f32) -> Result<Self, QueueError> {
        if !ahead_qty.is_finite() || ahead_qty < 0.0 || !own_qty.is_finite() || own_qty <= 0.0
        {
            return Err(QueueError::InvalidQuantity);
        }
        Ok(Self {
            ahead_qty,
            own_qty,
            filled_qty: 0.0,
        })
    }

    pub fn remaining_own_qty(&self) -> f32 {
        (self.own_qty - self.filled_qty).max(0.0)
    }

    /// Consume matched quantity at this exact price level.
    pub fn consume_trade(&mut self, traded_qty: f32) -> Result<QueueConsumption, QueueError> {
        if !traded_qty.is_finite() || traded_qty < 0.0
        {
            return Err(QueueError::InvalidQuantity);
        }
        let consumed_ahead_qty = traded_qty.min(self.ahead_qty);
        self.ahead_qty -= consumed_ahead_qty;
        let after_ahead = traded_qty - consumed_ahead_qty;
        let own_fill_qty = after_ahead.min(self.remaining_own_qty());
        self.filled_qty += own_fill_qty;
        Ok(QueueConsumption {
            traded_qty,
            consumed_ahead_qty,
            own_fill_qty,
            ahead_qty_after: self.ahead_qty,
            remaining_own_qty: self.remaining_own_qty(),
        })
    }

    /// Reduce estimated quantity ahead because of cancellations or amendments at
    /// the same price. This never increases the order's own fill quantity.
    pub fn cancel_ahead(&mut self, qty: f32) -> Result<f32, QueueError> {
        if !qty.is_finite() || qty < 0.0
        {
            return Err(QueueError::InvalidQuantity);
        }
        let removed = qty.min(self.ahead_qty);
        self.ahead_qty -= removed;
        Ok(removed)
    }
}

/// Deterministic token-bucket-like limiter using integer refill intervals.
///
/// Each completed refill interval restores `tokens_per_interval` up to
/// `capacity`. No wall clock is read internally; callers provide timestamps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RateLimitBucket {
    pub capacity: u32,
    pub tokens_per_interval: u32,
    pub refill_interval_ms: u64,
    pub available: u32,
    pub last_refill_ts_ms: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AcquireDecision {
    Allowed {
        remaining: u32,
    },
    Backpressured {
        available: u32,
        required: u32,
        retry_after_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitError {
    InvalidConfig,
    NonMonotonicClock,
    RequestExceedsCapacity,
    TimestampOverflow,
}

impl RateLimitBucket {
    pub fn new(
        capacity: u32,
        tokens_per_interval: u32,
        refill_interval_ms: u64,
        start_ts_ms: i64,
    ) -> Result<Self, RateLimitError> {
        if capacity == 0 || tokens_per_interval == 0 || refill_interval_ms == 0
        {
            return Err(RateLimitError::InvalidConfig);
        }
        Ok(Self {
            capacity,
            tokens_per_interval,
            refill_interval_ms,
            available: capacity,
            last_refill_ts_ms: start_ts_ms,
        })
    }

    fn elapsed_since_refill(&self, now_ts_ms: i64) -> Result<u64, RateLimitError> {
        if now_ts_ms < self.last_refill_ts_ms
        {
            return Err(RateLimitError::NonMonotonicClock);
        }
        let elapsed = now_ts_ms
            .checked_sub(self.last_refill_ts_ms)
            .ok_or(RateLimitError::TimestampOverflow)?;
        u64::try_from(elapsed).map_err(|_| RateLimitError::TimestampOverflow)
    }

    fn refill(&mut self, now_ts_ms: i64) -> Result<(), RateLimitError> {
        let elapsed = self.elapsed_since_refill(now_ts_ms)?;
        let intervals = elapsed / self.refill_interval_ms;
        if intervals == 0
        {
            return Ok(());
        }
        let restored = intervals
            .checked_mul(u64::from(self.tokens_per_interval))
            .ok_or(RateLimitError::TimestampOverflow)?;
        let available = u64::from(self.available)
            .checked_add(restored)
            .ok_or(RateLimitError::TimestampOverflow)?;
        self.available = available.min(u64::from(self.capacity)) as u32;

        let advanced = intervals
            .checked_mul(self.refill_interval_ms)
            .ok_or(RateLimitError::TimestampOverflow)?;
        let advanced_i64 =
            i64::try_from(advanced).map_err(|_| RateLimitError::TimestampOverflow)?;
        self.last_refill_ts_ms = self
            .last_refill_ts_ms
            .checked_add(advanced_i64)
            .ok_or(RateLimitError::TimestampOverflow)?;
        Ok(())
    }

    /// Attempt to consume request weight. Back-pressure is reported rather than
    /// sleeping, so scheduling policy remains outside the model.
    pub fn try_acquire(
        &mut self,
        now_ts_ms: i64,
        required: u32,
    ) -> Result<AcquireDecision, RateLimitError> {
        if required > self.capacity
        {
            return Err(RateLimitError::RequestExceedsCapacity);
        }
        if required == 0
        {
            self.refill(now_ts_ms)?;
            return Ok(AcquireDecision::Allowed {
                remaining: self.available,
            });
        }
        self.refill(now_ts_ms)?;
        if self.available >= required
        {
            self.available -= required;
            return Ok(AcquireDecision::Allowed {
                remaining: self.available,
            });
        }

        let missing = required - self.available;
        let intervals_needed = u64::from(missing).div_ceil(u64::from(self.tokens_per_interval));
        let since_refill = self.elapsed_since_refill(now_ts_ms)?;
        let until_next = self
            .refill_interval_ms
            .checked_sub(since_refill)
            .ok_or(RateLimitError::TimestampOverflow)?;
        let remaining_intervals = intervals_needed.saturating_sub(1);
        let additional_wait = remaining_intervals
            .checked_mul(self.refill_interval_ms)
            .ok_or(RateLimitError::TimestampOverflow)?;
        let retry_after_ms = until_next
            .checked_add(additional_wait)
            .ok_or(RateLimitError::TimestampOverflow)?;
        Ok(AcquireDecision::Backpressured {
            available: self.available,
            required,
            retry_after_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_timeline_is_explicit_and_deterministic() {
        let p = LatencyProfile {
            outbound_ms: 10,
            venue_processing_ms: 3,
            inbound_ms: 7,
        };
        let a = p.timeline(1_000).unwrap();
        let b = p.timeline(1_000).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.venue_arrival_ts_ms, 1_010);
        assert_eq!(a.venue_effective_ts_ms, 1_013);
        assert_eq!(a.local_ack_ts_ms, 1_020);
    }

    #[test]
    fn latency_timeline_rejects_unsigned_cast_and_timestamp_overflow() {
        let too_large_duration = LatencyProfile {
            outbound_ms: u64::MAX,
            venue_processing_ms: 0,
            inbound_ms: 0,
        };
        assert!(too_large_duration.timeline(0).is_none());

        let addition_overflow = LatencyProfile {
            outbound_ms: 1,
            venue_processing_ms: 0,
            inbound_ms: 0,
        };
        assert!(addition_overflow.timeline(i64::MAX).is_none());
    }

    #[test]
    fn fifo_queue_consumes_ahead_before_own_order() {
        let mut q = QueuePosition::new(3.0, 2.0).unwrap();
        let first = q.consume_trade(2.0).unwrap();
        assert_eq!(first.own_fill_qty, 0.0);
        assert!((q.ahead_qty - 1.0).abs() < 1e-6);

        let second = q.consume_trade(2.5).unwrap();
        assert!((second.consumed_ahead_qty - 1.0).abs() < 1e-6);
        assert!((second.own_fill_qty - 1.5).abs() < 1e-6);
        assert!((q.remaining_own_qty() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cancellation_ahead_improves_position_without_fake_fill() {
        let mut q = QueuePosition::new(5.0, 1.0).unwrap();
        assert!((q.cancel_ahead(2.0).unwrap() - 2.0).abs() < 1e-6);
        assert!((q.ahead_qty - 3.0).abs() < 1e-6);
        assert_eq!(q.filled_qty, 0.0);
    }

    #[test]
    fn limiter_reports_backpressure_and_exact_retry() {
        let mut b = RateLimitBucket::new(10, 2, 1_000, 0).unwrap();
        assert_eq!(
            b.try_acquire(0, 8).unwrap(),
            AcquireDecision::Allowed { remaining: 2 }
        );
        assert_eq!(
            b.try_acquire(100, 5).unwrap(),
            AcquireDecision::Backpressured {
                available: 2,
                required: 5,
                retry_after_ms: 1_900,
            }
        );
        assert_eq!(
            b.try_acquire(2_000, 5).unwrap(),
            AcquireDecision::Allowed { remaining: 1 }
        );
    }

    #[test]
    fn limiter_rejects_time_travel() {
        let mut b = RateLimitBucket::new(10, 2, 1_000, 100).unwrap();
        assert_eq!(
            b.try_acquire(99, 1),
            Err(RateLimitError::NonMonotonicClock)
        );
    }

    #[test]
    fn limiter_rejects_elapsed_timestamp_overflow() {
        let mut b = RateLimitBucket::new(10, 2, 1_000, i64::MIN).unwrap();
        assert_eq!(
            b.try_acquire(i64::MAX, 1),
            Err(RateLimitError::TimestampOverflow)
        );
    }

    #[test]
    fn limiter_rejects_refill_arithmetic_overflow() {
        let mut b = RateLimitBucket::new(u32::MAX, u32::MAX, 1, 0).unwrap();
        b.available = 0;
        assert_eq!(
            b.try_acquire(i64::MAX, 1),
            Err(RateLimitError::TimestampOverflow)
        );
    }
}
