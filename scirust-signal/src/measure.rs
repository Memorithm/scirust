//! Time-domain measurements extracted from sampled oscillatory signals.
//!
//! These helpers deliberately measure full periods using crossings in one
//! direction. Using every crossing as a half-period is biased for asymmetric
//! waveforms whose alternating crossing intervals are not equal.

/// Return times at which `values` crosses `level`, using linear interpolation
/// between adjacent samples.
///
/// The shorter of `t` and `values` bounds the scan. Samples equal to `level`
/// are accepted as the right endpoint of a crossing when the previous sample
/// is strictly on the opposite side.
#[must_use]
pub fn crossing_times(t: &[f64], values: &[f64], level: f64) -> Vec<f64> {
    let mut crossings = Vec::new();
    for i in 1..values.len().min(t.len())
    {
        let (a, b) = (values[i - 1] - level, values[i] - level);
        if (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0)
        {
            let span = b - a;
            let fraction = if span != 0.0 { -a / span } else { 0.0 };
            crossings.push(t[i - 1] + fraction * (t[i] - t[i - 1]));
        }
    }
    crossings
}

/// Return only rising crossings of `level`, with linear interpolation.
#[must_use]
pub fn upward_crossing_times(t: &[f64], values: &[f64], level: f64) -> Vec<f64> {
    let mut crossings = Vec::new();
    for i in 1..values.len().min(t.len())
    {
        let (a, b) = (values[i - 1] - level, values[i] - level);
        if a < 0.0 && b >= 0.0
        {
            let span = b - a;
            let fraction = if span != 0.0 { -a / span } else { 0.0 };
            crossings.push(t[i - 1] + fraction * (t[i] - t[i - 1]));
        }
    }
    crossings
}

/// Estimate an oscillation period around `level` from upward crossings in the
/// second half of the sampled time span.
///
/// Restricting to one direction makes successive retained crossings one full
/// period apart even for asymmetric waveforms. Restricting to the second half
/// is useful for decaying transients whose late-time dynamics are closer to a
/// small-signal linearization.
///
/// Returns `None` when the input is empty or when fewer than three upward
/// crossings lie in the second half. Three crossings provide two measured
/// periods and therefore at least one average rather than a single interval.
#[must_use]
pub fn period_from_second_half(t: &[f64], values: &[f64], level: f64) -> Option<f64> {
    let (first, last) = (*t.first()?, *t.last()?);
    let midpoint = first + (last - first) / 2.0;
    let late: Vec<f64> = upward_crossing_times(t, values, level)
        .into_iter()
        .filter(|crossing| *crossing >= midpoint)
        .collect();
    if late.len() < 3
    {
        return None;
    }
    let periods = (late.len() - 1) as f64;
    Some((late[late.len() - 1] - late[0]) / periods)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_is_interpolated_between_samples() {
        let crossings = crossing_times(&[0.0, 2.0], &[-1.0, 1.0], 0.0);
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn nonzero_level_is_honoured() {
        let crossings = crossing_times(&[0.0, 1.0], &[4.0, 6.0], 5.0);
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn sampled_sine_recovers_its_period() {
        let period = 0.7;
        let omega = 2.0 * std::f64::consts::PI / period;
        let h = period / 37.0;
        let n = 400;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        let values: Vec<f64> = t.iter().map(|time| (omega * time).sin()).collect();
        let measured = period_from_second_half(&t, &values, 0.0).expect("many crossings");
        let relative_error = (measured - period).abs() / period;
        assert!(relative_error < 1e-4, "measured {measured}, error {relative_error:e}");
    }

    #[test]
    fn asymmetric_crossing_intervals_do_not_bias_full_period() {
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
        assert!((measured - period).abs() / period < 1e-4, "measured {measured}");
    }

    #[test]
    fn too_few_late_crossings_returns_none() {
        let t: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let values = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        assert_eq!(period_from_second_half(&t, &values, 0.0), None);
        assert_eq!(period_from_second_half(&[], &[], 0.0), None);
    }
}
