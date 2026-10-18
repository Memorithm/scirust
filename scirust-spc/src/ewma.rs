//! EWMA control chart — sensitive to small sustained mean shifts.

use serde::{Deserialize, Serialize};

/// Exponentially-Weighted Moving-Average control chart.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EwmaChart {
    center: f64,
    sigma: f64,
    lambda: f64,
    l: f64,
    z: f64,
    t: u32,
}

impl EwmaChart {
    /// New chart for an in-control mean/σ, smoothing `lambda ∈ (0,1]` and limit
    /// width `l` (≈ 2.7–3).
    pub fn new(center: f64, sigma: f64, lambda: f64, l: f64) -> Self {
        Self {
            center,
            sigma,
            lambda: lambda.clamp(1e-3, 1.0),
            l,
            z: center,
            t: 0,
        }
    }

    /// Current EWMA statistic.
    pub fn value(&self) -> f64 {
        self.z
    }

    /// Half-width of the (time-varying) control limit at the current step `t`:
    /// `L·σ·√(λ/(2−λ)·(1 − (1−λ)^(2t)))`, which widens towards the steady-state
    /// `L·σ·√(λ/(2−λ))`.
    pub fn limit(&self) -> f64 {
        let lam = self.lambda;
        // The exponent `2t` is formed in `f64`: as `2 * (t as i32)` it
        // overflowed from t = 2^30 (a debug-build panic; a negative exponent,
        // hence a `NaN` limit, in release).
        let decay = (1.0 - lam).powf(2.0 * f64::from(self.t));
        let factor = lam / (2.0 - lam) * (1.0 - decay);
        self.l * self.sigma * factor.sqrt()
    }

    /// Feed a new observation; returns `true` if it is out of control.
    ///
    /// The step counter saturates at `u32::MAX`; by then `(1−λ)^(2t)` is `0`
    /// for every admissible `λ ≥ 1e-3`, so the limit is already the
    /// steady-state one.
    pub fn update(&mut self, x: f64) -> bool {
        self.t = self.t.saturating_add(1);
        self.z = self.lambda * x + (1.0 - self.lambda) * self.z;
        (self.z - self.center).abs() > self.limit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_small_sustained_shift_shewhart_would_miss() {
        // 1.5σ mean shift — never trips a 3σ Shewhart rule, but EWMA accumulates.
        let mut chart = EwmaChart::new(0.0, 1.0, 0.2, 2.7);
        let shift = 1.5;
        let mut tripped = false;
        // Deterministic small zig-zag around the shifted mean.
        for k in 0..60
        {
            let noise = if k % 2 == 0 { 0.1 } else { -0.1 };
            if chart.update(shift + noise)
            {
                tripped = true;
                break;
            }
            // A single sample never exceeds 3σ, so a Shewhart 3σ test stays quiet.
            assert!((shift + noise).abs() < 3.0);
        }
        assert!(tripped, "EWMA failed to catch the 0.5σ shift");
    }

    /// Asymptotic half-width `L·σ·√(λ/(2−λ))` the time-varying limit tends to.
    fn steady_state_limit(chart: &EwmaChart) -> f64 {
        chart.l * chart.sigma * (chart.lambda / (2.0 - chart.lambda)).sqrt()
    }

    #[test]
    fn limit_stays_finite_past_two_to_the_thirtieth_steps() {
        // The exponent `2t` used to be formed as `2 * (t as i32)`: from
        // t = 2^30 that product overflows i32 (a panic in debug builds) and in
        // release it wraps negative, so `(1−λ)^(2t)` became `+∞`, the limit
        // `NaN`, and the chart silently stopped signalling. 2^30 steps is
        // about 12.4 days of a 1 kHz stream.
        let mut chart = EwmaChart::new(0.0, 1.0, 0.2, 3.0);
        let expected = steady_state_limit(&chart);
        for t in [1u32 << 30, (1u32 << 31) - 1, 1u32 << 31, u32::MAX]
        {
            chart.t = t;
            let limit = chart.limit();
            assert!(
                (limit - expected).abs() <= 1e-12 * expected,
                "t = {t}: limit {limit}, expected {expected}"
            );
        }
    }

    #[test]
    fn keeps_signalling_after_the_step_counter_saturates() {
        // At the end of the counter range the next update must neither
        // overflow `t` nor lose the ability to flag a gross outlier.
        let mut chart = EwmaChart::new(0.0, 1.0, 0.2, 3.0);
        chart.t = u32::MAX - 1;
        let expected = steady_state_limit(&chart);
        assert!(!chart.update(0.0));
        assert!(!chart.update(0.0));
        // A wrapping counter would restart at t = 0, collapsing the limit to
        // its start-up width instead of keeping the steady-state one.
        assert!(
            (chart.limit() - expected).abs() <= 1e-12 * expected,
            "limit {} after saturation, expected {expected}",
            chart.limit()
        );
        assert!(
            chart.update(100.0),
            "z = {}, limit = {}",
            chart.value(),
            chart.limit()
        );
        assert!(chart.limit().is_finite());
    }

    #[test]
    fn stays_in_control_on_centered_data() {
        let mut chart = EwmaChart::new(0.0, 1.0, 0.2, 3.0);
        for k in 0..200
        {
            let x = if k % 2 == 0 { 0.3 } else { -0.3 };
            assert!(
                !chart.update(x),
                "false alarm at step {k}, z={}",
                chart.value()
            );
        }
    }
}
