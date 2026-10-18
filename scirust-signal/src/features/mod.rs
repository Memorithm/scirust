pub mod spectral;

/// Root Mean Square of a signal.
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
    f64::sqrt(sum_sq / signal.len() as f64)
}

/// Peak value (maximum absolute value).
pub fn peak(signal: &[f64]) -> f64 {
    signal.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()))
}

/// Peak-to-peak amplitude.
pub fn peak_to_peak(signal: &[f64]) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    let min = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    max - min
}

/// Crest factor: peak / RMS.
/// Indicates impulsiveness; high values (> 4-5) suggest bearing defects.
///
/// The amplitude cancels in the ratio, so the result does not depend on the
/// signal's units: it is evaluated as `1 / rms(x / peak)`. An empty or
/// all-zero signal returns `0`.
pub fn crest_factor(signal: &[f64]) -> f64 {
    let p = peak(signal);
    if p == 0.0
    {
        // Empty or all-zero signal; an all-NaN one (whose peak also folds to
        // 0) still propagates NaN.
        return if signal.iter().any(|x| x.is_nan())
        {
            f64::NAN
        }
        else
        {
            0.0
        };
    }
    let mean_sq = signal.iter().map(|&x| (x / p) * (x / p)).sum::<f64>() / signal.len() as f64;
    1.0 / mean_sq.sqrt()
}

/// Population moments `(m2, m3, m4)` of the deviations from the mean, each
/// divided by the largest absolute deviation, or `None` for a signal with
/// fewer than two samples or whose samples are all equal (no shape).
///
/// Dividing by the largest deviation keeps the moments in `[0, 1]` whatever
/// the signal's units, so the standardized ratios built from them neither
/// underflow nor overflow. A non-finite sample yields NaN moments.
fn scaled_central_moments(signal: &[f64]) -> Option<(f64, f64, f64)> {
    if signal.len() < 2
    {
        return None;
    }
    if signal.iter().any(|x| !x.is_finite())
    {
        return Some((f64::NAN, f64::NAN, f64::NAN));
    }
    if signal.iter().all(|&x| x == signal[0])
    {
        return None;
    }
    let n = signal.len() as f64;
    let mean0 = signal.iter().sum::<f64>() / n;
    // One correction pass removes most of the rounding error of the mean.
    let mean = mean0 + signal.iter().map(|&x| x - mean0).sum::<f64>() / n;
    let scale = signal
        .iter()
        .fold(0.0f64, |acc, &x| acc.max((x - mean).abs()));
    if scale == 0.0
    {
        return None;
    }
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for &x in signal
    {
        let u = (x - mean) / scale;
        let u2 = u * u;
        m2 += u2;
        m3 += u2 * u;
        m4 += u2 * u2;
    }
    Some((m2 / n, m3 / n, m4 / n))
}

/// Kurtosis (excess kurtosis: normal = 0), the biased (population) estimator
/// `m4 / m2² − 3`, as `scipy.stats.kurtosis(x)`.
/// High kurtosis (> 3) indicates impulsive signals typical of bearing faults.
///
/// The ratio does not depend on the signal's units. A signal with fewer than
/// two samples, or whose samples are all equal, returns `0`; a non-finite
/// sample returns NaN.
pub fn kurtosis(signal: &[f64]) -> f64 {
    match scaled_central_moments(signal)
    {
        Some((m2, _, m4)) => m4 / (m2 * m2) - 3.0, // excess kurtosis
        None => 0.0,
    }
}

/// Skewness: asymmetry of the distribution, the biased (population)
/// estimator `m3 / m2^{3/2}`, as `scipy.stats.skew(x)`.
/// Positive = right-skewed, negative = left-skewed.
///
/// The ratio does not depend on the signal's units. A signal with fewer than
/// two samples, or whose samples are all equal, returns `0`; a non-finite
/// sample returns NaN.
pub fn skewness(signal: &[f64]) -> f64 {
    match scaled_central_moments(signal)
    {
        Some((m2, m3, _)) => m3 / m2.powf(1.5),
        None => 0.0,
    }
}

/// Zero-crossing rate: number of sign changes / (N-1).
/// Useful for voice/signal activity detection.
pub fn zero_crossing_rate(signal: &[f64]) -> f64 {
    if signal.len() < 2
    {
        return 0.0;
    }
    let count = signal
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    count as f64 / (signal.len() - 1) as f64
}

/// Autocorrelation at lag `k`.
pub fn autocorrelation(signal: &[f64], lag: usize) -> f64 {
    if lag >= signal.len()
    {
        return 0.0;
    }
    let n = signal.len() - lag;
    if n == 0
    {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n
    {
        sum += signal[i] * signal[i + lag];
    }
    sum / n as f64
}

/// Autocorrelation for all lags 0..max_lag.
pub fn autocorrelation_full(signal: &[f64], max_lag: usize) -> Vec<f64> {
    (0..=max_lag.min(signal.len().saturating_sub(1)))
        .map(|lag| autocorrelation(signal, lag))
        .collect()
}

/// Signal energy: sum of squared samples.
pub fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|&x| x * x).sum()
}

/// Shannon entropy estimate via histogram (64 bins).
pub fn entropy(signal: &[f64]) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    let n_bins = 64usize;
    let min = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    // Only a constant signal has a single bin. An absolute cut-off here
    // (formerly `range < f64::EPSILON`) reported any small-amplitude record
    // as constant, although the binning is independent of the units.
    if range.is_nan() || range <= 0.0
    {
        return 0.0;
    }
    let mut bins = vec![0usize; n_bins];
    for &x in signal
    {
        let mut idx = ((x - min) / range * n_bins as f64) as usize;
        if idx >= n_bins
        {
            idx = n_bins - 1;
        }
        bins[idx] += 1;
    }
    let n = signal.len() as f64;
    let mut h = 0.0;
    for &count in &bins
    {
        if count > 0
        {
            let p = count as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

/// Returns a vector of standard time-domain features.
/// Order: [rms, crest_factor, kurtosis, skewness, zcr, entropy, peak_to_peak, energy]
pub fn time_features(signal: &[f64]) -> Vec<f64> {
    vec![
        rms(signal),
        crest_factor(signal),
        kurtosis(signal),
        skewness(signal),
        zero_crossing_rate(signal),
        entropy(signal),
        peak_to_peak(signal),
        energy(signal),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_rms() {
        let s = vec![1.0, 2.0, 3.0];
        // sqrt((1+4+9)/3) = sqrt(14/3) = sqrt(4.6667) ≈ 2.1602
        let expected = f64::sqrt(14.0 / 3.0);
        assert!((rms(&s) - expected).abs() < EPS);
    }

    #[test]
    fn test_zcr() {
        let s = vec![-1.0, 1.0, -1.0, 1.0];
        assert!((zero_crossing_rate(&s) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_autocorrelation() {
        let s = vec![1.0, 1.0, 1.0, 1.0];
        assert!((autocorrelation(&s, 0) - 1.0).abs() < EPS);
        assert!((autocorrelation(&s, 1) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_kurtosis_normalish() {
        // A simple sine wave has negative excess kurtosis
        let n = 128;
        let s: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let k = kurtosis(&s);
        // Sine waves have excess kurtosis around -1.5
        assert!(k < 0.0);
    }

    #[test]
    fn test_empty_inputs() {
        let empty: Vec<f64> = vec![];
        assert!((rms(&empty) - 0.0).abs() < EPS);
        assert!((crest_factor(&empty) - 0.0).abs() < EPS);
        assert!((kurtosis(&empty) - 0.0).abs() < EPS);
        assert!((entropy(&empty) - 0.0).abs() < EPS);
    }

    /// Impulsive test record: a slow sine plus a spike every 16 samples.
    fn impulsive_record() -> Vec<f64> {
        (0..64)
            .map(|i| (0.3 * i as f64).sin() + if i % 16 == 3 { 4.0 } else { 0.0 })
            .collect()
    }

    #[test]
    fn shape_features_do_not_depend_on_signal_units() {
        // Regression: kurtosis/skewness returned 0 whenever the variance was
        // below f64::EPSILON (std < 1.5e-8), crest_factor whenever the RMS was
        // below f64::EPSILON, and entropy whenever the range was below
        // f64::EPSILON. All four are ratios in which the amplitude cancels, so
        // a record in small units (e.g. displacement in metres) must give the
        // same values as the same record in larger units.
        //
        // References for the unscaled record: scipy.stats.kurtosis(x) =
        // 4.396864597639196, scipy.stats.skew(x) = 1.873656292059915,
        // max|x|/rms(x) = 3.9591891296906208, 64-bin histogram entropy (numpy,
        // same binning) = 4.2682330072378285 bits.
        let x = impulsive_record();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        // Powers of two: the scaled records are exact copies of `x` in other
        // units (no rounding), so every feature must match.
        for scale in [1.0, 2f64.powi(-30), 2f64.powi(-60), 2f64.powi(-200)]
        {
            let y: Vec<f64> = x.iter().map(|v| v * scale).collect();
            assert!(
                close(kurtosis(&y), 4.396_864_597_639_196),
                "kurtosis at {scale:e}: {}",
                kurtosis(&y)
            );
            assert!(
                close(skewness(&y), 1.873_656_292_059_915),
                "skewness at {scale:e}: {}",
                skewness(&y)
            );
            assert!(
                close(crest_factor(&y), 3.959_189_129_690_620_8),
                "crest at {scale:e}: {}",
                crest_factor(&y)
            );
            assert!(
                close(entropy(&y), 4.268_233_007_237_828_5),
                "entropy at {scale:e}: {}",
                entropy(&y)
            );
        }
    }

    #[test]
    fn shape_features_of_constant_records_stay_zero() {
        // A constant record has no shape: every feature keeps returning 0,
        // including for a value (0.1) whose mean does not round-trip exactly.
        for c in [0.0, 0.1, -3.7e-12, 1.0e6]
        {
            let s = vec![c; 1000];
            assert_eq!(kurtosis(&s), 0.0, "kurtosis of constant {c}");
            assert_eq!(skewness(&s), 0.0, "skewness of constant {c}");
            assert_eq!(entropy(&s), 0.0, "entropy of constant {c}");
        }
        assert_eq!(crest_factor(&[0.0; 16]), 0.0);
        assert_eq!(kurtosis(&[1.0]), 0.0);
        assert_eq!(skewness(&[1.0]), 0.0);
        assert!(kurtosis(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(skewness(&[1.0, f64::INFINITY, 2.0]).is_nan());
        assert!(crest_factor(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(crest_factor(&[f64::NAN, f64::NAN]).is_nan());
    }

    #[test]
    fn test_crest_factor_impulse() {
        // A signal with a large spike should have high crest factor
        let mut s = vec![0.1; 100];
        s[50] = 10.0;
        let cf = crest_factor(&s);
        assert!(cf > 3.0, "crest_factor too low: {}", cf);
    }
}
