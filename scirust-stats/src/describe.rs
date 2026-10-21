//! Descriptive statistics over a slice of samples. The input is not mutated;
//! variance and standard deviation use the unbiased `n − 1` divisor.

/// Arithmetic mean of the samples in `data`.
///
/// Returns `NaN` for an empty slice. Non-finite samples follow floating-point
/// sum/division semantics; a `NaN` sample propagates. Does not mutate the input.
/// Runs in `O(n)` time with `O(1)` auxiliary storage.
///
/// # Numerical limitations
///
/// The implementation sums before dividing and does not compensate rounding,
/// so the result depends on sample order and is not a correctly rounded mean.
/// When every sample is finite but the plain sum overflows, the samples are
/// rescaled by a power of two before summing, so a representable mean such as
/// that of two `f64::MAX` samples is returned instead of an infinity.
///
/// ```
/// use scirust_stats::describe::mean;
/// assert_eq!(mean(&[f64::MAX, f64::MAX]), f64::MAX);
/// assert_eq!(mean(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX / 3.0);
/// ```
///
/// # Examples
///
/// Compute a mean without changing the samples:
///
/// ```
/// use scirust_stats::describe::mean;
/// let samples = [2.0, 4.0, 6.0];
/// assert_eq!(mean(&samples), 4.0);
/// assert_eq!(samples, [2.0, 4.0, 6.0]);
/// ```
///
/// Handle empty or invalid samples explicitly:
///
/// ```
/// use scirust_stats::describe::mean;
/// assert!(mean(&[]).is_nan());
/// assert!(mean(&[1.0, f64::NAN]).is_nan());
/// ```
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty()
    {
        return f64::NAN;
    }
    let n = data.len() as f64;
    let sum = data.iter().sum::<f64>();
    if sum.is_finite() || data.iter().any(|x| !x.is_finite())
    {
        return sum / n;
    }
    // Every sample is finite, so the mean is representable, but the plain sum
    // overflowed. Dividing by a power of two is exact for these magnitudes, and
    // each rescaled sample is below 2 in magnitude, so the rescaled sum cannot
    // overflow; `(sum / n) * scale` is bounded by the largest sample.
    let scale = pow2_floor(max_abs(data));
    (data.iter().map(|x| x / scale).sum::<f64>() / n) * scale
}

/// Largest absolute value of finite samples (`0` for an empty slice).
fn max_abs(data: &[f64]) -> f64 {
    data.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

/// Largest power of two not above a finite, positive `x`, clamped below at the
/// smallest normal power of two so that dividing by it never overflows.
fn pow2_floor(x: f64) -> f64 {
    if x < f64::MIN_POSITIVE
    {
        return f64::MIN_POSITIVE;
    }
    f64::from_bits(x.to_bits() & 0x7ff0_0000_0000_0000)
}

/// Unbiased variance as `(scaled, scale)`, meaning `variance = scaled * scale²`
/// and `std_dev = sqrt(scaled) * scale`. Returns `(NaN, 1)` for fewer than two
/// samples or any non-finite sample.
fn scaled_variance(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 2 || data.iter().any(|x| !x.is_finite())
    {
        return (f64::NAN, 1.0);
    }
    let denom = n as f64 - 1.0;
    let m = mean(data);
    let ss = data.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
    if ss.is_finite() && (ss >= f64::MIN_POSITIVE || data.iter().all(|&x| x == m))
    {
        return (ss / denom, 1.0);
    }
    // The deviations or their squares overflowed, or the squares underflowed
    // into the subnormal range and lost digits. Rescale by a power of two so
    // the largest sample lies in [1, 2): deviations are then at most 4, and
    // nonzero deviations are at least about 2^-52, so squares neither
    // overflow nor underflow.
    let scale = pow2_floor(max_abs(data));
    let ys = data.iter().map(|x| x / scale);
    let my = ys.clone().sum::<f64>() / n as f64;
    let ssy = ys.map(|y| (y - my) * (y - my)).sum::<f64>();
    (ssy / denom, scale)
}

/// Unbiased sample variance using the `n - 1` divisor.
///
/// Returns `NaN` for fewer than two samples or a non-finite sample. Uses a
/// two-pass calculation with [`mean`]; the input is not modified. Runs in
/// `O(n)` time with `O(1)` auxiliary storage.
///
/// # Numerical limitations
///
/// When the deviations from the mean or their squares overflow, or the
/// squares underflow into the subnormal range, the samples are rescaled by a
/// power of two and the two passes are repeated, so the result is finite and
/// nonzero whenever the true variance is representable. A variance above
/// `f64::MAX` still returns infinity; [`std_dev`] and [`std_error`] avoid
/// forming the variance and stay finite in that case.
///
/// ```
/// use scirust_stats::describe::variance;
/// // One large outlier among many zeros: the squared deviation overflows,
/// // but the variance (about 1.96e307) is representable.
/// let mut data = vec![0.0; 1000];
/// data.push(1.4e155);
/// let v = variance(&data);
/// assert!(v.is_finite() && (v / 1.958e307 - 1.0).abs() < 1e-3);
/// // A variance above `f64::MAX` is still infinite.
/// assert_eq!(variance(&[1e200, -1e200]), f64::INFINITY);
/// ```
///
/// # Examples
///
/// The squared deviations of `[2, 4, 6]` sum to eight; dividing by two gives four:
///
/// ```
/// use scirust_stats::describe::variance;
/// assert_eq!(variance(&[2.0, 4.0, 6.0]), 4.0);
/// assert_eq!(variance(&[7.0, 7.0]), 0.0);
/// ```
///
/// An unbiased sample variance needs at least two samples:
///
/// ```
/// use scirust_stats::describe::variance;
/// assert!(variance(&[]).is_nan());
/// assert!(variance(&[7.0]).is_nan());
/// assert!(variance(&[1.0, f64::INFINITY]).is_nan());
/// ```
pub fn variance(data: &[f64]) -> f64 {
    let (scaled, scale) = scaled_variance(data);
    scaled * scale * scale
}

/// Sample standard deviation: the square root of [`variance`].
///
/// Uses the unbiased sample variance, not the population variance. Returns
/// `NaN` for fewer than two samples or non-finite samples. The square root is
/// taken before undoing the power-of-two rescaling described in [`variance`],
/// so the result stays finite for finite samples even when the variance itself
/// would overflow. Runs in `O(n)` time with `O(1)` auxiliary storage and leaves
/// `data` unchanged.
///
/// ```
/// use scirust_stats::describe::std_dev;
/// let s = std_dev(&[1e200, -1e200]);
/// assert!((s / (2.0_f64.sqrt() * 1e200) - 1.0).abs() < 1e-14);
/// ```
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::std_dev;
/// assert_eq!(std_dev(&[2.0, 4.0, 6.0]), 2.0);
/// assert_eq!(std_dev(&[7.0, 7.0]), 0.0);
/// ```
///
/// ```
/// use scirust_stats::describe::std_dev;
/// assert!(std_dev(&[]).is_nan());
/// assert!(std_dev(&[1.0]).is_nan());
/// assert!(std_dev(&[1.0, f64::NAN]).is_nan());
/// ```
pub fn std_dev(data: &[f64]) -> f64 {
    let (scaled, scale) = scaled_variance(data);
    scaled.sqrt() * scale
}

/// Estimated standard error of the mean, `s / sqrt(n)`.
///
/// Here `s` is [`std_dev`] and `n` is the sample count. This computes the usual
/// independent-sample estimator; it does not correct dependence between samples.
/// Returns `NaN` for fewer than two samples or non-finite samples and, like
/// [`std_dev`], stays finite for finite samples. Runs in `O(n)` time, uses `O(1)` auxiliary
/// storage and leaves `data` unchanged.
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::std_error;
/// let expected = 2.0 / 3.0_f64.sqrt();
/// assert!((std_error(&[2.0, 4.0, 6.0]) - expected).abs() < 1e-12);
/// assert_eq!(std_error(&[7.0, 7.0]), 0.0);
/// ```
///
/// ```
/// use scirust_stats::describe::std_error;
/// assert!(std_error(&[]).is_nan());
/// assert!(std_error(&[1.0]).is_nan());
/// assert!(std_error(&[1.0, f64::NAN]).is_nan());
/// ```
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
/// The input is not mutated. Uses `O(n log n)` sorting work and `O(n)` auxiliary
/// storage. Use [`quantiles`] to share one sort across several probabilities.
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::quantile;
/// let data = [4.0, 1.0, 3.0, 2.0];
/// assert_eq!(quantile(&data, 0.25), 1.75);
/// assert_eq!(quantile(&data, 0.5), 2.5);
/// assert_eq!(data, [4.0, 1.0, 3.0, 2.0]);
/// ```
///
/// Clamping is separate from the policy for invalid samples:
///
/// ```
/// use scirust_stats::describe::quantile;
/// assert_eq!(quantile(&[2.0, 8.0], -1.0), 2.0);
/// assert_eq!(quantile(&[f64::INFINITY], 0.5), f64::INFINITY);
/// assert!(quantile(&[1.0, f64::NAN], 0.5).is_nan());
/// assert!(quantile(&[1.0], f64::NAN).is_nan());
/// ```
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
/// `O(q)` interpolation work and `O(n + q)` storage, instead of sorting the
/// samples `q` times. No measured speedup is implied by this structural cost.
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::quantiles;
/// assert_eq!(
///     quantiles(&[4.0, 1.0, 3.0, 2.0], &[0.0, 0.5, 1.0]),
///     vec![1.0, 2.5, 4.0],
/// );
/// ```
///
/// One invalid probability does not invalidate its neighbors:
///
/// ```
/// use scirust_stats::describe::quantiles;
/// let results = quantiles(&[2.0, 8.0], &[1.0, f64::NAN, 0.0]);
/// assert_eq!(results[0], 8.0);
/// assert!(results[1].is_nan());
/// assert_eq!(results[2], 2.0);
/// assert!(quantiles(&[2.0, 8.0], &[]).is_empty());
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

/// Median (the 0.5 quantile), following [`quantile`]'s non-finite policy.
///
/// For an even sample count this interpolates the two central order statistics.
/// An empty input or any `NaN` sample produces `NaN`. Does not mutate `data`.
/// Uses `O(n log n)` sorting work and `O(n)` auxiliary storage.
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::median;
/// assert_eq!(median(&[7.0, 1.0, 3.0]), 3.0);
/// assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
/// ```
///
/// ```
/// use scirust_stats::describe::median;
/// assert_eq!(median(&[-f64::MAX, f64::MAX]), 0.0);
/// assert!(median(&[]).is_nan());
/// assert!(median(&[1.0, f64::NAN]).is_nan());
/// ```
pub fn median(data: &[f64]) -> f64 {
    quantile(data, 0.5)
}

/// Minimum sample, ignoring individual `NaN` values.
///
/// Returns `NaN` if `data` is empty or every sample is `NaN`. Infinite samples
/// remain eligible extrema. Unlike [`quantile`], a `NaN` does not invalidate
/// other samples. Runs in `O(n)` time with `O(1)` auxiliary storage and does not
/// mutate the input.
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::min;
/// assert_eq!(min(&[3.0, -2.0, 7.0]), -2.0);
/// assert_eq!(min(&[f64::NAN, 3.0, -2.0]), -2.0);
/// ```
///
/// ```
/// use scirust_stats::describe::min;
/// assert!(min(&[]).is_nan());
/// assert!(min(&[f64::NAN]).is_nan());
/// assert_eq!(min(&[f64::NAN, f64::INFINITY]), f64::INFINITY);
/// ```
pub fn min(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::NAN, f64::min)
}

/// Maximum sample, ignoring individual `NaN` values.
///
/// Returns `NaN` if `data` is empty or every sample is `NaN`. Infinite samples
/// remain eligible extrema. Unlike [`quantile`], a `NaN` does not invalidate
/// other samples. Runs in `O(n)` time with `O(1)` auxiliary storage and does not
/// mutate the input.
///
/// # Examples
///
/// ```
/// use scirust_stats::describe::max;
/// assert_eq!(max(&[3.0, -2.0, 7.0]), 7.0);
/// assert_eq!(max(&[f64::NAN, 3.0, -2.0]), 3.0);
/// ```
///
/// ```
/// use scirust_stats::describe::max;
/// assert!(max(&[]).is_nan());
/// assert!(max(&[f64::NAN]).is_nan());
/// assert_eq!(max(&[f64::NAN, f64::NEG_INFINITY]), f64::NEG_INFINITY);
/// ```
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

    fn rel(actual: f64, expected: f64) -> f64 {
        ((actual - expected) / expected).abs()
    }

    #[test]
    fn regression_mean_of_finite_samples_does_not_overflow() {
        // Before the fix the plain sum overflowed to +inf (or inf - inf = NaN).
        assert_eq!(mean(&[f64::MAX, f64::MAX]), f64::MAX);
        assert_eq!(mean(&[-f64::MAX, -f64::MAX]), -f64::MAX);
        assert_eq!(mean(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX / 3.0);
        let m = mean(&[1e308, 1e308, 1e308, -1e308]);
        assert!(rel(m, 5e307) < 1e-15, "{m}");
        // Non-finite inputs keep plain floating-point semantics.
        assert_eq!(mean(&[f64::MAX, f64::INFINITY]), f64::INFINITY);
        assert!(mean(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    }

    #[test]
    fn regression_variance_large_outlier_is_finite() {
        // The squared deviation (~1.96e310) overflowed although the variance
        // (exact rational value 1.9580419580419583e307) is representable.
        let mut data = vec![0.0; 1000];
        data.push(1.4e155);
        let v = variance(&data);
        assert!(rel(v, 1.958_041_958_041_958_3e307) < 1e-12, "{v}");
        let s = std_dev(&data);
        assert!(
            rel(s, 1.958_041_958_041_958_3e307_f64.sqrt()) < 1e-12,
            "{s}"
        );
    }

    #[test]
    fn regression_std_dev_of_huge_samples_is_finite() {
        // Variance 2e400 overflows, but the standard deviation is ~1.414e200.
        let s = std_dev(&[1e200, -1e200]);
        assert!(rel(s, 2.0_f64.sqrt() * 1e200) < 1e-14, "{s}");
        assert!(variance(&[1e200, -1e200]).is_infinite());
        let se = std_error(&[1e200, -1e200]);
        assert!(rel(se, 1e200) < 1e-14, "{se}");
        // Opposite extremes: deviations themselves overflowed before the fix.
        // sqrt(2) * f64::MAX is not representable, so infinity is correct here.
        assert!(std_dev(&[f64::MAX, -f64::MAX]).is_infinite());
        let s = std_dev(&[f64::MAX / 2.0, -f64::MAX / 2.0]);
        assert!(rel(s, f64::MAX / 2.0_f64.sqrt()) < 1e-14, "{s}");
    }

    #[test]
    fn regression_std_dev_of_tiny_samples_keeps_precision() {
        // Squared deviations (1e-320) were subnormal and lost most digits.
        let s = std_dev(&[1e-160, 3e-160]);
        assert!(rel(s, 2.0_f64.sqrt() * 1e-160) < 1e-14, "{s}");
        let s = std_dev(&[2e-300, 4e-300, 6e-300]);
        assert!(rel(s, 2e-300) < 1e-14, "{s}");
        assert_eq!(std_dev(&[5e-320, 5e-320]), 0.0);
    }

    #[test]
    fn rescaling_keeps_ordinary_results_bit_identical() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = d.iter().sum::<f64>() / 8.0;
        let v = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 7.0;
        assert_eq!(mean(&d).to_bits(), m.to_bits());
        assert_eq!(variance(&d).to_bits(), v.to_bits());
        assert_eq!(std_dev(&d).to_bits(), v.sqrt().to_bits());
        assert!(variance(&[1.0, f64::INFINITY]).is_nan());
        assert!(std_dev(&[f64::NAN, 1.0]).is_nan());
        assert!(std_dev(&[f64::INFINITY, f64::INFINITY]).is_nan());
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
        let scale = f64::MAX / 2.0;
        assert!((quantile(&data, 0.25) / scale + 1.0).abs() <= f64::EPSILON);
        assert!((quantile(&data, 0.75) / scale - 1.0).abs() <= f64::EPSILON);
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
            assert!(
                results
                    .iter()
                    .all(|x| (min(&data)..=max(&data)).contains(x))
            );
        }
    }
}
