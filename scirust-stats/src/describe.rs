//! Descriptive statistics over a slice of samples. All functions are pure and
//! deterministic; variance/standard deviation use the unbiased `n − 1` divisor.

/// Arithmetic mean. Returns `NaN` for an empty slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty()
    {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Unbiased sample variance (divisor `n − 1`). Returns `NaN` for fewer than two
/// samples. Uses a two-pass formula for numerical stability.
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2
    {
        return f64::NAN;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0)
}

/// Unbiased sample standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Standard error of the mean, `s / √n`.
pub fn std_error(data: &[f64]) -> f64 {
    std_dev(data) / (data.len() as f64).sqrt()
}

/// Linear-interpolated quantile (the common "type 7" rule).
///
/// Probabilities outside `[0, 1]` are clamped, including infinite probabilities.
/// Returns `NaN` for an empty slice, a `NaN` probability, or any `NaN` sample.
/// Exact order statistics are returned without interpolation, including infinite
/// endpoints. An interior interpolation between opposite infinities is `NaN`;
/// otherwise an infinite bound determines the interior result. Interpolation
/// between finite bounds avoids overflowing their difference.
///
/// The input is not mutated. Use [`quantiles`] to share one sort across several
/// probabilities.
pub fn quantile(data: &[f64], p: f64) -> f64 {
    if p.is_nan()
    {
        return f64::NAN;
    }
    match sorted_samples(data)
    {
        Some(sorted) => quantile_sorted(&sorted, p),
        None => f64::NAN,
    }
}

/// Several type-7 quantiles with one sort, preserving probability order.
///
/// Uses the same clamping, `NaN`, and infinity policy as [`quantile`]. A `NaN`
/// probability affects only its own output; a `NaN` sample affects all outputs.
/// Empty probabilities produce an empty vector. Neither input is mutated.
/// For `n` samples and `q` probabilities, this uses one `O(n log n)` sort plus
/// `O(q)` interpolation work instead of sorting the samples `q` times.
///
/// ```
/// use scirust_stats::describe::quantiles;
/// assert_eq!(
///     quantiles(&[4.0, 1.0, 3.0, 2.0], &[0.0, 0.5, 1.0]),
///     vec![1.0, 2.5, 4.0],
/// );
/// ```
pub fn quantiles(data: &[f64], probabilities: &[f64]) -> Vec<f64> {
    if probabilities.is_empty()
    {
        return Vec::new();
    }
    match sorted_samples(data)
    {
        Some(sorted) => probabilities
            .iter()
            .map(|&p| quantile_sorted(&sorted, p))
            .collect(),
        None => vec![f64::NAN; probabilities.len()],
    }
}

fn sorted_samples(data: &[f64]) -> Option<Vec<f64>> {
    if data.is_empty() || data.iter().any(|x| x.is_nan())
    {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted)
}

fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if p.is_nan()
    {
        return f64::NAN;
    }
    let p = p.clamp(0.0, 1.0);
    let h = (sorted.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let a = sorted[lo];
    let b = sorted[hi];
    // In particular, never compute 0 * infinity or infinity - infinity
    // when the requested order statistic is already known exactly.
    if lo == hi || a == b
    {
        return a;
    }
    if a.is_infinite() || b.is_infinite()
    {
        return if a.is_infinite() && b.is_infinite()
        {
            f64::NAN
        }
        else if a.is_infinite()
        {
            a
        }
        else
        {
            b
        };
    }
    let weight = h - lo as f64;
    if a.is_sign_negative() != b.is_sign_negative()
    {
        // Opposite finite signs can overflow b - a even when the quantile
        // is representable. Each weighted term is bounded by its endpoint.
        (1.0 - weight) * a + weight * b
    }
    else
    {
        a + weight * (b - a)
    }
}

/// Median (the 0.5 quantile); follows the [`quantile`] non-finite-input policy.
pub fn median(data: &[f64]) -> f64 {
    quantile(data, 0.5)
}

/// Minimum, ignoring `NaN`. `NaN` if empty or every sample is `NaN`.
pub fn min(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::NAN, f64::min)
}

/// Maximum, ignoring `NaN`. `NaN` if empty or every sample is `NaN`.
pub fn max(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::NAN, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&d) - 5.0).abs() < 1e-12);
        // Unbiased variance of this classic set is 32/7.
        assert!((variance(&d) - 32.0 / 7.0).abs() < 1e-12);
        assert!((median(&d) - 4.5).abs() < 1e-12);
        assert_eq!(min(&d), 2.0);
        assert_eq!(max(&d), 9.0);
    }

    #[test]
    fn quantile_endpoints_and_interpolation() {
        let d = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&d, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&d, 1.0) - 4.0).abs() < 1e-12);
        assert!((quantile(&d, 0.5) - 2.5).abs() < 1e-12);
        assert_eq!(quantile(&d, 0.25), 1.75);
        assert_eq!(quantile(&d, 0.75), 3.25);
    }

    #[test]
    fn degenerate_inputs_are_nan_not_panics() {
        assert!(mean(&[]).is_nan());
        assert!(variance(&[1.0]).is_nan());
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn audit_extrema_empty_and_all_nan_are_nan() {
        for data in [&[][..], &[f64::NAN][..], &[f64::NAN, f64::NAN][..]]
        {
            assert!(min(data).is_nan());
            assert!(max(data).is_nan());
        }
    }

    #[test]
    fn audit_extrema_ignore_nan_without_discarding_infinities() {
        let data = [f64::NAN, 3.0, -2.0, f64::NAN];
        assert_eq!(min(&data), -2.0);
        assert_eq!(max(&data), 3.0);
        assert_eq!(min(&[f64::NAN, f64::INFINITY]), f64::INFINITY);
        assert_eq!(max(&[f64::NAN, f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }

    #[test]
    fn audit_quantile_finite_extremes_do_not_overflow() {
        let data = [-f64::MAX, f64::MAX];
        assert_eq!(quantile(&data, 0.0), -f64::MAX);
        assert_eq!(quantile(&data, 1.0), f64::MAX);
        assert_eq!(quantile(&data, 0.5), 0.0);
        assert_eq!(quantile(&data, 0.25), -f64::MAX / 2.0);
        assert_eq!(quantile(&data, 0.75), f64::MAX / 2.0);
    }

    #[test]
    fn audit_quantile_infinite_order_statistics_are_preserved() {
        for value in [f64::NEG_INFINITY, f64::INFINITY]
        {
            for p in [0.0, 0.25, 0.5, 1.0]
            {
                assert_eq!(quantile(&[value], p), value);
                assert_eq!(quantile(&[value, value], p), value);
            }
        }
        assert_eq!(quantile(&[1.0, f64::INFINITY], 0.0), 1.0);
        assert_eq!(quantile(&[1.0, f64::INFINITY], 0.5), f64::INFINITY);
        assert_eq!(quantile(&[f64::NEG_INFINITY, 1.0], 0.5), f64::NEG_INFINITY);
        assert_eq!(quantile(&[f64::NEG_INFINITY, 1.0], 1.0), 1.0);
        assert!(quantile(&[f64::NEG_INFINITY, f64::INFINITY], 0.5).is_nan());
    }

    #[test]
    fn audit_quantile_nan_policy_is_independent_of_sample_order() {
        for data in [
            [f64::NAN, 1.0, 2.0],
            [1.0, f64::NAN, 2.0],
            [1.0, 2.0, f64::NAN],
        ]
        {
            for p in [0.0, 0.5, 1.0]
            {
                assert!(quantile(&data, p).is_nan());
            }
        }
        assert!(quantile(&[1.0, 2.0], f64::NAN).is_nan());
    }

    #[test]
    fn audit_quantile_keeps_legacy_probability_clamping() {
        for p in [f64::NEG_INFINITY, -1.0, 0.0]
        {
            assert_eq!(quantile(&[2.0, 8.0], p), 2.0);
        }
        for p in [1.0, 2.0, f64::INFINITY]
        {
            assert_eq!(quantile(&[2.0, 8.0], p), 8.0);
        }
    }

    #[test]
    fn audit_quantiles_share_scalar_semantics_and_preserve_order() {
        let data = [4.0, -f64::MAX, 0.0, f64::MAX, 2.0];
        let probabilities = [0.95, 0.0, 0.5, f64::NAN, 1.0, -1.0, 2.0];
        let outputs = quantiles(&data, &probabilities);
        assert_eq!(outputs.len(), probabilities.len());
        for (&p, &actual) in probabilities.iter().zip(&outputs)
        {
            let expected = quantile(&data, p);
            if expected.is_nan()
            {
                assert!(actual.is_nan());
            }
            else
            {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        assert_eq!(data[0], 4.0);
        assert_eq!(data[1], -f64::MAX);
        assert_eq!(probabilities[0], 0.95);
    }

    #[test]
    fn audit_quantiles_empty_and_nan_samples_have_explicit_results() {
        assert!(quantiles(&[1.0, 2.0], &[]).is_empty());
        assert!(quantiles(&[], &[]).is_empty());
        for data in [&[][..], &[1.0, f64::NAN][..]]
        {
            let results = quantiles(data, &[0.0, 0.5, 1.0]);
            assert_eq!(results.len(), 3);
            assert!(results.iter().all(|x| x.is_nan()));
        }
    }

    #[test]
    fn audit_quantiles_are_monotone_and_bounded_on_finite_samples() {
        for data in [
            vec![-f64::MAX, f64::MAX],
            vec![f64::MAX / 2.0, f64::MAX],
            vec![-f64::MAX, -f64::MAX / 2.0],
            vec![7.0, -3.0, 0.0, 7.0, 2.0],
        ]
        {
            let probabilities: Vec<_> = (0..=100).map(|i| f64::from(i) / 100.0).collect();
            let results = quantiles(&data, &probabilities);
            assert!(results.iter().all(|x| x.is_finite()));
            assert!(results.windows(2).all(|pair| pair[0] <= pair[1]));
            assert!(results.iter().all(|&x| x >= min(&data) && x <= max(&data)));
        }
    }
}
