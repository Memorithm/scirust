//! Measurements taken *from* a finished trajectory, as opposed to
//! [`crate::execute_support`], which is about producing one.
//!
//! The generic sampled-signal implementation now lives in `scirust-signal` so
//! the Studio runtime, NoiseLab, and other consumers share one measurement
//! contract rather than maintaining subtly different crossing logic.

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn crossing_times(t: &[f64], values: &[f64], level: f64) -> Vec<f64> {
    scirust_signal::crossing_times(t, values, level)
}

pub(crate) fn period_from_second_half(t: &[f64], values: &[f64], level: f64) -> Option<f64> {
    scirust_signal::period_from_second_half(t, values, level)
}

pub(crate) fn upward_crossing_times(t: &[f64], values: &[f64], level: f64) -> Vec<f64> {
    scirust_signal::upward_crossing_times(t, values, level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_crossings_preserve_interpolation_contract() {
        let crossings = crossing_times(&[0.0, 2.0], &[-1.0, 1.0], 0.0);
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn studio_period_uses_shared_one_direction_measurement() {
        let period = 0.7;
        let omega = 2.0 * std::f64::consts::PI / period;
        let h = period / 37.0;
        let n = 400;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        let values: Vec<f64> = t
            .iter()
            .map(|time| 0.3 + (omega * time).sin())
            .collect();
        let measured = period_from_second_half(&t, &values, 0.0).expect("many crossings");
        assert!((measured - period).abs() / period < 1e-4);
        assert!(!upward_crossing_times(&t, &values, 0.0).is_empty());
    }

    #[test]
    fn studio_empty_period_measurement_is_none() {
        assert_eq!(period_from_second_half(&[], &[], 0.0), None);
    }
}
