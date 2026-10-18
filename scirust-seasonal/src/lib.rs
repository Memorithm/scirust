//! Seasonal and cyclic pattern detection algorithms.
//!
//! Provides STL decomposition, autocorrelation-based period detection, Fourier
//! analysis, wavelet-like windowed transforms, seasonal adjustment, trend tests,
//! and change-point detection — all operating on `f64` time-series data.
//!
//! # Modules
//! - **stl** — Seasonal-Trend decomposition using Loess
//! - **detection** — ACF, PACF, periodogram, dominant frequency
//! - **cyclic** — Fourier analysis, windowed FFT, cycle length, phase
//! - **adjustment** — Moving-average decomposition, X-11 style, deseasonalize
//! - **trend** — Mann-Kendall test, Sen's slope, linear trend with CI
//! - **changepoint** — Seasonal CUSUM and seasonal break detection

use serde::{Deserialize, Serialize};

// Modules are defined inline below.

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Next power of two ≥ `n`.
#[inline]
fn next_pow2(n: usize) -> usize {
    if n <= 1
    {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Mean of a slice.
#[inline]
fn mean(data: &[f64]) -> f64 {
    if data.is_empty()
    {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Trimmed mean: discard `trim` elements from each end, then average.
#[inline]
fn trimmed_mean(data: &[f64], trim: usize) -> f64 {
    if data.len() <= 2 * trim
    {
        return mean(data);
    }
    let trimmed = &data[trim..data.len() - trim];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Variance of a slice.
#[inline]
fn variance(data: &[f64]) -> f64 {
    if data.len() < 2
    {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Pad a centered array (shorter than n) to full length by repeating edge values.
/// Subtract the sample mean of `data`, then zero-pad to `target_len`.
///
/// The mean must be removed before padding: subtracting it afterwards would
/// fill the padded tail with `-mean`, a rectangular pulse whose low-frequency
/// leakage dominates the spectrum of any series with a non-zero level.
fn demean_and_zero_pad(data: &[f64], target_len: usize) -> Vec<f64> {
    let m = mean(data);
    let mut out: Vec<f64> = data.iter().map(|&x| x - m).collect();
    if out.len() < target_len
    {
        out.resize(target_len, 0.0);
    }
    out
}

fn pad_centered(data: &mut Vec<f64>, target_len: usize) {
    let current_len = data.len();
    if current_len >= target_len
    {
        data.truncate(target_len);
        return;
    }
    let pad_each = (target_len - current_len) / 2;
    let mut left = Vec::with_capacity(pad_each);
    for _ in 0..pad_each
    {
        left.push(data[0]);
    }
    let mut right = Vec::with_capacity(target_len - current_len - pad_each);
    for _ in 0..target_len - current_len - pad_each
    {
        right.push(data[data.len() - 1]);
    }
    let mut result = left;
    result.append(data);
    result.extend(right);
    *data = result;
}

/// Inverse of the standard normal CDF (quantile / probit), via Acklam's
/// rational approximation. Accurate to ~1e-9 over the open interval (0, 1).
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0
    {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0
    {
        return f64::INFINITY;
    }
    // Coefficients for the rational approximation.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW
    {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
    else if p <= P_HIGH
    {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
    else
    {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Two-sided Student-t critical value for confidence level `conf` (e.g. 0.95)
/// with `df` degrees of freedom, i.e. the quantile `t_{(1 + conf)/2, df}`.
///
/// `df = 1` (Cauchy) and `df = 2` use their exact closed forms; any other
/// positive finite `df` (integer or not) inverts the Student-t CDF built on the
/// regularized incomplete beta (`scirust_core::nn::intervals::student_t_ppf`),
/// which agrees with `scipy.stats.t.ppf` to better than 1e-9 relative for
/// `conf` up to 0.999. A non-positive or non-finite `df` falls back to the
/// normal quantile (the `df → ∞` limit).
fn t_critical(conf: f64, df: f64) -> f64 {
    // Upper-tail probability of the two-sided interval: 1 - alpha/2.
    let p = 0.5 + conf / 2.0;
    if df <= 0.0 || !df.is_finite() || !(p > 0.0 && p < 1.0)
    {
        return inv_normal_cdf(p);
    }
    if df == 1.0
    {
        // Cauchy quantile.
        return (std::f64::consts::PI * (p - 0.5)).tan();
    }
    if df == 2.0
    {
        return (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
    }
    scirust_core::nn::intervals::student_t_ppf(p, df)
}

/// Quantile of a sorted slice (linear interpolation).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty()
    {
        return 0.0;
    }
    if sorted.len() == 1
    {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Running (simple) moving average of `window` size.
fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty()
    {
        return Vec::new();
    }
    let n = data.len();
    if window > n
    {
        return vec![mean(data)];
    }
    let mut result = Vec::with_capacity(n - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..n
    {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Centered moving average: for even `window`, average two successive MAs.
fn centered_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty()
    {
        return Vec::new();
    }
    let n = data.len();
    if window > n
    {
        return vec![mean(data)];
    }
    if window % 2 == 1
    {
        return moving_average(data, window);
    }
    // Even window: compute MA(window), then MA(2) of that
    let ma = moving_average(data, window);
    moving_average(&ma, 2)
}

/// Weighted running moving average of `window` size. Each window's value is the
/// weighted mean of the points it covers; a window whose weights sum to zero
/// falls back to the unweighted mean of that window.
fn weighted_moving_average(data: &[f64], weights: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty()
    {
        return Vec::new();
    }
    let n = data.len();
    if window > n
    {
        return vec![mean(data)];
    }
    let mut result = Vec::with_capacity(n - window + 1);
    for start in 0..=n - window
    {
        let mut wsum = 0.0;
        let mut vsum = 0.0;
        let win = &data[start..start + window];
        for (offset, &x) in win.iter().enumerate()
        {
            let w = weights.get(start + offset).copied().unwrap_or(1.0);
            wsum += w;
            vsum += w * x;
        }
        if wsum > f64::EPSILON
        {
            result.push(vsum / wsum);
        }
        else
        {
            result.push(win.iter().sum::<f64>() / window as f64);
        }
    }
    result
}

/// Weighted centered moving average. For odd `window` this is a single weighted
/// pass; for even `window` it averages two successive weighted passes to recenter.
fn weighted_centered_moving_average(data: &[f64], weights: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty()
    {
        return Vec::new();
    }
    let n = data.len();
    if window > n
    {
        return vec![mean(data)];
    }
    if window % 2 == 1
    {
        return weighted_moving_average(data, weights, window);
    }
    let ma = weighted_moving_average(data, weights, window);
    moving_average(&ma, 2)
}

/// Bisquare (Tukey) robustness weights of a residual series:
/// `w_i = (1 - (r_i / h)^2)^2` for `|r_i| < h`, else 0, with `h = 6 · MAD`.
/// When all residuals are ~0 (`h ≈ 0`) every weight is 1.0.
fn bisquare_weights(remainder: &[f64]) -> Vec<f64> {
    let n = remainder.len();
    let mut abs: Vec<f64> = remainder.iter().map(|r| r.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = quantile(&abs, 0.5);
    let h = 6.0 * mad;
    if h <= f64::EPSILON
    {
        return vec![1.0; n];
    }
    remainder
        .iter()
        .map(|&r| {
            let u = (r / h).abs();
            if u < 1.0
            {
                let t = 1.0 - u * u;
                t * t
            }
            else
            {
                0.0
            }
        })
        .collect()
}

/// Linear regression: returns (slope, intercept).
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    let n = x.len() as f64;
    if n < 2.0
    {
        return (0.0, if n == 1.0 { y[0] } else { 0.0 });
    }
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&v| v * v).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON
    {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

// ─── Result types ────────────────────────────────────────────────────────────

/// Result of STL decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STLResult {
    /// Original time series length.
    pub length: usize,
    /// Seasonal period.
    pub period: usize,
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder (original - trend - seasonal).
    pub remainder: Vec<f64>,
}

/// Result of a seasonal detection analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodDetection {
    /// Detected dominant period (in samples).
    pub period: usize,
    /// Strength of the detected period (0.0 – 1.0).
    pub strength: f64,
    /// All candidate periods with their strengths.
    pub candidates: Vec<PeriodCandidate>,
}

/// A single candidate period from detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodCandidate {
    /// Period in samples.
    pub period: usize,
    /// Strength / confidence score.
    pub strength: f64,
}

/// Cyclic pattern information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclicPattern {
    /// Estimated cycle length in samples.
    pub cycle_length: f64,
    /// Phase offset in radians.
    pub phase: f64,
    /// Amplitude of the dominant cycle.
    pub amplitude: f64,
    /// All detected harmonics.
    pub harmonics: Vec<Harmonic>,
}

/// A detected harmonic component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Harmonic {
    /// Frequency index.
    pub frequency_index: usize,
    /// Frequency in cycles per sample.
    pub frequency: f64,
    /// Amplitude.
    pub amplitude: f64,
    /// Phase in radians.
    pub phase: f64,
}

/// Linear trend result with confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendResult {
    /// Slope (change per time unit).
    pub slope: f64,
    /// Intercept.
    pub intercept: f64,
    /// Mann-Kendall p-value.
    pub p_value: f64,
    /// Mann-Kendall S statistic.
    pub s_statistic: f64,
    /// Mann-Kendall Z statistic.
    pub z_statistic: f64,
    /// Sen's slope estimate.
    pub sens_slope: f64,
    /// 95% CI for slope [lower, upper].
    pub slope_ci: [f64; 2],
    /// Whether trend is significant (p < 0.05).
    pub significant: bool,
    /// Trend direction.
    pub direction: TrendDirection,
}

/// Direction of a detected trend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    NoTrend,
}

/// Change point result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Index of the change point.
    pub index: usize,
    /// Confidence / magnitude of the change.
    pub magnitude: f64,
    /// Seasonal component of the change.
    pub seasonal_component: f64,
}

// ═════════════════════════════════════════════════════════════════════════════
// Module implementations
// ═════════════════════════════════════════════════════════════════════════════

pub mod stl {
    //! Seasonal-Trend decomposition using Loess (STL).
    //!
    //! Decomposes a time series `y` into `y = trend + seasonal + remainder`.
    //! Uses centered moving averages for trend extraction and iterative
    //! seasonal smoothing.

    use super::*;

    /// Configuration for STL decomposition.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct STLConfig {
        /// Seasonal period (e.g. 12 for monthly data).
        pub period: usize,
        /// Number of inner iterations (default: 2).
        pub inner_iterations: usize,
        /// Number of outer iterations for robustness (default: 0 = non-robust).
        pub outer_iterations: usize,
        /// Trend smoothing window (must be odd, 0 = auto).
        pub trend_window: usize,
        /// Seasonal smoothing window (must be odd, 0 = auto).
        pub seasonal_window: usize,
    }

    impl Default for STLConfig {
        fn default() -> Self {
            Self {
                period: 12,
                inner_iterations: 2,
                outer_iterations: 0,
                trend_window: 0,
                seasonal_window: 0,
            }
        }
    }

    /// Run STL decomposition on the given time series.
    pub fn stl_decompose(data: &[f64], config: &STLConfig) -> STLResult {
        let n = data.len();
        let period = config.period;
        assert!(period >= 2, "period must be >= 2");
        assert!(n >= 2 * period, "data length must be >= 2 * period");

        // Auto-select windows
        let trend_window = if config.trend_window == 0
        {
            let mut tw = (1.5 * period as f64 / (1.0 - 1.5 / period as f64)) as usize;
            if tw.is_multiple_of(2)
            {
                tw += 1;
            }
            tw.max(period + 1)
        }
        else
        {
            config.trend_window
        };

        let seasonal_window = if config.seasonal_window == 0
        {
            let mut sw = period.saturating_add(if period.is_multiple_of(2) { 1 } else { 0 });
            if sw % 2 == 0
            {
                sw += 1;
            }
            sw
        }
        else
        {
            config.seasonal_window
        };

        let mut trend = vec![0.0; n];
        let mut seasonal = vec![0.0; n];
        let mut remainder: Vec<f64> = data.to_vec();

        // Robustness weights, all 1.0 on the first (non-robust) outer pass.
        let mut robust_weights = vec![1.0; n];

        for outer in 0..=config.outer_iterations
        {
            // Inner loop
            for _inner in 0..config.inner_iterations
            {
                // Step 1: Detrend — subtract current trend estimate
                let detrended: Vec<f64> = data
                    .iter()
                    .zip(trend.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();

                // Step 2: Seasonal smoothing — smooth each cycle-subseries
                // position-by-position and scatter the per-cycle smoothed values
                // back to their original indices, yielding a full-length seasonal
                // that may evolve cycle-to-cycle. Robustness weights down-weight
                // outliers within each subseries.
                let cycle_subseries = extract_cycle_subseries(&detrended, period);
                let weight_subseries = extract_cycle_subseries(&robust_weights, period);
                seasonal = smooth_cycle_subseries(
                    &cycle_subseries,
                    &weight_subseries,
                    n,
                    period,
                    seasonal_window,
                );

                // Step 2b: Low-pass filter + detrend the seasonal. Smoothing each
                // cycle-subseries lets trend leak into the seasonal (a ramped
                // subseries is fit by a ramp, not a constant). Standard STL removes
                // that leakage by subtracting a low-pass filter (a moving average of
                // length `period`) of the seasonal, leaving a trend-free seasonal
                // that may still evolve cycle-to-cycle.
                let mut low_pass = centered_moving_average(&seasonal, period);
                pad_centered(&mut low_pass, n);
                for (s, &lp) in seasonal.iter_mut().zip(low_pass.iter())
                {
                    *s -= lp;
                }

                // Step 3: Deseasonalize
                let deseasonalized: Vec<f64> = data
                    .iter()
                    .zip(seasonal.iter())
                    .map(|(&y, &s)| y - s)
                    .collect();

                // Step 4: Trend smoothing (robustness-weighted)
                trend = weighted_centered_moving_average(
                    &deseasonalized,
                    &robust_weights,
                    trend_window,
                );
                // Pad trend to full length
                pad_centered(&mut trend, n);
            }

            // Compute remainder
            remainder = data
                .iter()
                .zip(trend.iter().zip(seasonal.iter()))
                .map(|(&y, (&t, &s))| y - t - s)
                .collect();

            // Robustness weights for the NEXT outer iteration: bisquare weights of
            // the residuals, w_i = (1 - (r_i / (6·MAD))^2)^2 for |r_i| < 6·MAD,
            // else 0. These are fed back into the trend and seasonal smoothers.
            if outer < config.outer_iterations
            {
                robust_weights = bisquare_weights(&remainder);
            }
        }

        STLResult {
            length: n,
            period,
            trend,
            seasonal,
            remainder,
        }
    }

    /// Extract the cycle subseries: for each position within a period,
    /// collect all values at that phase.
    fn extract_cycle_subseries(data: &[f64], period: usize) -> Vec<Vec<f64>> {
        let mut subseries = vec![Vec::new(); period];
        for (i, &val) in data.iter().enumerate()
        {
            subseries[i % period].push(val);
        }
        subseries
    }

    /// Smooth each cycle-subseries position-by-position and scatter the smoothed
    /// values back to their original indices, producing a full-length seasonal of
    /// length `n`. Each subseries (all values sharing a phase `i % period`) is
    /// smoothed with a centered moving average of size `window`; the smoothed value
    /// at cycle position `c` is written back to original index `c * period + phase`.
    /// The result is then centered so each phase's subseries has mean zero, which
    /// removes the trend/level contribution from the seasonal component.
    fn smooth_cycle_subseries(
        subseries: &[Vec<f64>],
        weight_subseries: &[Vec<f64>],
        n: usize,
        period: usize,
        window: usize,
    ) -> Vec<f64> {
        let mut seasonal = vec![0.0; n];
        for (phase, s) in subseries.iter().enumerate()
        {
            if s.is_empty()
            {
                continue;
            }
            let weights = &weight_subseries[phase];
            let smoothed = smooth_subseries_full(s, weights, window.min(s.len().max(1)));
            // Scatter each smoothed value back to its original index. Do NOT center
            // per-subseries here (that would erase the per-phase seasonal level);
            // the seasonal is made trend-free by the caller's low-pass/detrend step.
            for (cycle, &val) in smoothed.iter().enumerate()
            {
                let idx = cycle * period + phase;
                if idx < n
                {
                    seasonal[idx] = val;
                }
            }
        }
        seasonal
    }

    /// Smooth a subseries to the SAME length as the input via a centered,
    /// robustness-weighted moving average. Each output position is the weighted
    /// mean of the `window` neighbors centered on it; near the ends the window
    /// shrinks symmetrically so every position stays centered on available data.
    fn smooth_subseries_full(s: &[f64], weights: &[f64], window: usize) -> Vec<f64> {
        let len = s.len();
        if len == 0
        {
            return Vec::new();
        }
        if window <= 1
        {
            return s.to_vec();
        }
        let half = window / 2;
        (0..len)
            .map(|i| {
                // Symmetric half-width limited by both ends keeps the window
                // centered on `i` even at the boundaries.
                let reach = half.min(i).min(len - 1 - i);
                let lo = i - reach;
                let hi = i + reach;
                let mut wsum = 0.0;
                let mut vsum = 0.0;
                for (offset, &x) in s[lo..=hi].iter().enumerate()
                {
                    let w = weights.get(lo + offset).copied().unwrap_or(1.0);
                    wsum += w;
                    vsum += w * x;
                }
                if wsum > f64::EPSILON
                {
                    vsum / wsum
                }
                else
                {
                    // All neighbors had zero robustness weight: fall back to the
                    // unweighted center value.
                    s[i]
                }
            })
            .collect()
    }

    /// Extract trend component from STL result.
    pub fn extract_trend(result: &STLResult) -> &[f64] {
        &result.trend
    }

    /// Extract seasonal component from STL result.
    pub fn extract_seasonal(result: &STLResult) -> &[f64] {
        &result.seasonal
    }

    /// Compute remainder: original - trend - seasonal.
    pub fn compute_remainder(data: &[f64], result: &STLResult) -> Vec<f64> {
        data.iter()
            .zip(result.trend.iter().zip(result.seasonal.iter()))
            .map(|(&y, (&t, &s))| y - t - s)
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn synthetic_seasonal(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    let trend = 0.01 * t;
                    let seasonal = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                    trend + seasonal
                })
                .collect()
        }

        #[test]
        fn stl_recovers_seasonal_and_trend() {
            let period = 12;
            let n = 240;
            let data = synthetic_seasonal(n, period);
            let config = STLConfig {
                period,
                ..Default::default()
            };
            let result = stl_decompose(&data, &config);

            assert_eq!(result.length, n);
            assert_eq!(result.period, period);

            // The remainder should be much smaller than the original signal
            let orig_var = variance(&data);
            let rem_var = variance(&result.remainder);
            assert!(
                rem_var < orig_var * 0.1,
                "remainder variance {} should be < 10% of original {}",
                rem_var,
                orig_var
            );
        }

        #[test]
        fn stl_trend_length_matches() {
            let period = 7;
            let n = 140;
            let data = synthetic_seasonal(n, period);
            let config = STLConfig {
                period,
                ..Default::default()
            };
            let result = stl_decompose(&data, &config);
            assert_eq!(result.trend.len(), n);
            assert_eq!(result.seasonal.len(), n);
            assert_eq!(result.remainder.len(), n);
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn stl_decomposition_sums_correctly() {
            let period = 12;
            let n = 120;
            let data = synthetic_seasonal(n, period);
            let config = STLConfig {
                period,
                inner_iterations: 3,
                ..Default::default()
            };
            let result = stl_decompose(&data, &config);

            for i in 0..n
            {
                let reconstructed = result.trend[i] + result.seasonal[i] + result.remainder[i];
                assert!(
                    (data[i] - reconstructed).abs() < 1e-10,
                    "decomposition doesn't sum at {}: {} vs {}",
                    i,
                    data[i],
                    reconstructed
                );
            }
        }
    }
}

pub mod detection {
    //! Seasonal period detection using autocorrelation, PACF, periodogram,
    //! and dominant frequency analysis.

    use scirust_signal::fft_real;

    use super::*;

    /// Compute the Autocorrelation Function (ACF) for lags 0..max_lag.
    ///
    /// Normalizes by the variance so that lag 0 equals 1.0. The result is
    /// invariant to the scale of `data`. A constant series (all values equal)
    /// has no defined ACF; it returns `1.0` at every lag.
    pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        if n < 2
        {
            return vec![1.0];
        }
        let max_lag = max_lag.min(n - 1);
        if data.iter().all(|&x| x == data[0])
        {
            return vec![1.0; max_lag + 1];
        }
        let m = mean(data);
        // Divide by the largest deviation before squaring: the ACF is
        // scale-invariant, and this keeps Σc² from underflowing for tiny
        // amplitudes or overflowing for huge ones.
        let scale = data.iter().fold(0.0_f64, |acc, &x| acc.max((x - m).abs()));
        let scale = if scale > 0.0 && scale.is_finite()
        {
            scale
        }
        else
        {
            1.0
        };
        let centered: Vec<f64> = data.iter().map(|&x| (x - m) / scale).collect();
        let var = centered.iter().map(|&c| c * c).sum::<f64>();
        let mut result = Vec::with_capacity(max_lag + 1);
        for lag in 0..=max_lag
        {
            let mut sum = 0.0;
            for i in 0..n - lag
            {
                sum += centered[i] * centered[i + lag];
            }
            result.push(sum / var);
        }
        result
    }

    /// Compute the Partial Autocorrelation Function (PACF) using the
    /// Durbin-Levinson recursion.
    ///
    /// Returns PACF values for lags 1..=max_lag.
    pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
        let acf_vals = acf(data, max_lag);
        let m = max_lag.min(acf_vals.len() - 1);
        if m == 0
        {
            return Vec::new();
        }

        let mut phi = vec![0.0; m + 1];
        let mut phi_prev = vec![0.0; m + 1];
        let mut pacf_vals = Vec::with_capacity(m);

        // k = 1
        phi[1] = acf_vals[1];
        pacf_vals.push(phi[1]);

        for k in 2..=m
        {
            // Compute numerator
            let mut num = acf_vals[k];
            for j in 1..k
            {
                num -= phi_prev[j] * acf_vals[k - j];
            }
            // Compute denominator
            let mut denom = 1.0;
            for j in 1..k
            {
                denom -= phi_prev[j] * acf_vals[j];
            }
            if denom.abs() < f64::EPSILON
            {
                pacf_vals.push(0.0);
                continue;
            }
            phi[k] = num / denom;
            pacf_vals.push(phi[k]);

            // Update phi_prev for next iteration
            for j in 1..k
            {
                phi[j] = phi_prev[j] - phi[k] * phi_prev[k - j];
            }
            phi_prev = phi.clone();
        }

        pacf_vals
    }

    /// Compute the periodogram (power spectrum) of the time series.
    ///
    /// Returns (frequencies, power) pairs for positive frequencies.
    /// The sample mean is removed first, then the series is zero-padded to
    /// the next power of 2 for the FFT, so a constant level does not change
    /// the spectrum.
    pub fn periodogram(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n == 0
        {
            return (Vec::new(), Vec::new());
        }

        let fft_size = next_pow2(n);
        let padded = demean_and_zero_pad(data, fft_size);

        let spectrum = fft_real(&padded);
        let n_f64 = fft_size as f64;
        let frequencies: Vec<f64> = (0..spectrum.len()).map(|k| k as f64 / n_f64).collect();
        let power: Vec<f64> = spectrum.iter().map(|c| c.mag_sq() / n_f64).collect();

        (frequencies, power)
    }

    /// Detect the dominant period from the periodogram.
    ///
    /// Finds the frequency bin (excluding DC) with the highest power,
    /// then returns `1/frequency` as the period. The strength is the share
    /// of that bin in the total non-DC power, so it does not depend on the
    /// amplitude of `data`. A constant series has no spectral peak and
    /// returns period `0`, strength `0.0` and no candidates.
    pub fn dominant_frequency(data: &[f64]) -> PeriodDetection {
        let no_period = PeriodDetection {
            period: 0,
            strength: 0.0,
            candidates: Vec::new(),
        };
        if data.is_empty() || data.iter().all(|&x| x == data[0])
        {
            return no_period;
        }
        // Period and strength are scale-invariant. Dividing by the largest
        // magnitude first keeps the power from underflowing for tiny
        // amplitudes or overflowing to `inf` for huge ones.
        let scale = data.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let (frequencies, power) = if scale > 0.0 && scale.is_finite()
        {
            let scaled: Vec<f64> = data.iter().map(|&x| x / scale).collect();
            periodogram(&scaled)
        }
        else
        {
            periodogram(data)
        };
        if power.len() <= 1
        {
            return no_period;
        }

        // Skip DC (bin 0), only look at positive frequencies up to Nyquist.
        // fft_real returns fft_size/2 + 1 bins, so the true Nyquist bin is the last one.
        let nyquist = power.len() - 1;
        let mut candidates: Vec<(usize, f64)> = (1..=nyquist.min(power.len() - 1))
            .map(|k| (k, power[k]))
            .collect();

        // Sort by power descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let total_power: f64 = power[1..=nyquist.min(power.len() - 1)].iter().sum();

        let mut result_candidates = Vec::new();
        for &(k, pwr) in candidates.iter().take(5)
        {
            let freq = frequencies[k];
            let period = if freq > f64::EPSILON
            {
                (1.0 / freq).round() as usize
            }
            else
            {
                0
            };
            let strength = if total_power > 0.0
            {
                pwr / total_power
            }
            else
            {
                0.0
            };
            result_candidates.push(PeriodCandidate { period, strength });
        }

        let (dominant_k, dominant_power) = candidates[0];
        let dominant_freq = frequencies[dominant_k];
        let period = if dominant_freq > f64::EPSILON
        {
            (1.0 / dominant_freq).round() as usize
        }
        else
        {
            0
        };
        let strength = if total_power > 0.0
        {
            dominant_power / total_power
        }
        else
        {
            0.0
        };

        PeriodDetection {
            period,
            strength,
            candidates: result_candidates,
        }
    }

    /// Detect period using ACF peaks.
    ///
    /// Finds the first significant peak in the ACF, which corresponds
    /// to the dominant seasonal period.
    pub fn detect_period_acf(data: &[f64], max_lag: Option<usize>) -> PeriodDetection {
        let n = data.len();
        let max_lag = max_lag.unwrap_or(n / 2);
        let acf_vals = acf(data, max_lag);

        if acf_vals.len() < 3
        {
            return PeriodDetection {
                period: 0,
                strength: 0.0,
                candidates: Vec::new(),
            };
        }

        // Find peaks in ACF (local maxima above significance threshold)
        let threshold = 1.96 / (n as f64).sqrt(); // 95% confidence band
        let mut peaks: Vec<(usize, f64)> = Vec::new();

        for i in 1..acf_vals.len() - 1
        {
            if acf_vals[i] > acf_vals[i - 1]
                && acf_vals[i] > acf_vals[i + 1]
                && acf_vals[i] > threshold
            {
                peaks.push((i, acf_vals[i]));
            }
        }

        // Sort by ACF value
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let candidates: Vec<PeriodCandidate> = peaks
            .iter()
            .take(5)
            .map(|&(lag, val)| PeriodCandidate {
                period: lag,
                strength: val.abs(),
            })
            .collect();

        let (period, strength) = if let Some(&(lag, val)) = peaks.first()
        {
            (lag, val.abs())
        }
        else
        {
            (0, 0.0)
        };

        PeriodDetection {
            period,
            strength,
            candidates,
        }
    }

    /// Combined period detection: uses both ACF and periodogram,
    /// returns the consensus result.
    pub fn detect_period(data: &[f64]) -> PeriodDetection {
        let acf_result = detect_period_acf(data, None);
        let freq_result = dominant_frequency(data);

        // If both agree (within tolerance), high confidence
        if acf_result.period > 0 && freq_result.period > 0
        {
            let diff = (acf_result.period as f64 - freq_result.period as f64).abs();
            let tol = (acf_result.period as f64 * 0.1).max(1.0);
            if diff <= tol
            {
                return PeriodDetection {
                    period: ((acf_result.period + freq_result.period) / 2).max(1),
                    strength: ((acf_result.strength + freq_result.strength) / 2.0).min(1.0),
                    candidates: acf_result.candidates,
                };
            }
        }

        // Prefer ACF result if it has higher confidence
        if acf_result.strength >= freq_result.strength
        {
            acf_result
        }
        else
        {
            freq_result
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sine_wave(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
                .collect()
        }

        #[test]
        fn acf_exact_lag1() {
            // data = [1,2,3,4]; mean=2.5, centered=[-1.5,-0.5,0.5,1.5].
            // var = Σc² = 5.0. The crate normalizes the autocovariance by Σc².
            let data = [1.0, 2.0, 3.0, 4.0];
            let a = acf(&data, 3);
            assert!((a[0] - 1.0).abs() < 1e-12, "acf[0]={}", a[0]);
            assert!((a[1] - 0.25).abs() < 1e-12, "acf[1]={}", a[1]);
            assert!((a[2] - (-0.30)).abs() < 1e-12, "acf[2]={}", a[2]);
            assert!((a[3] - (-0.45)).abs() < 1e-12, "acf[3]={}", a[3]);
        }

        #[test]
        fn dominant_frequency_high_freq_not_missed() {
            // A pure period-3 sine has its spectral peak at FFT bin round(256/3)=85.
            // fft_real yields 129 bins; the true Nyquist is bin 128, so bin 85 is a
            // valid positive-frequency component. The previous nyquist=len/2≈64 bound
            // excluded it. 1/(85/256)=3.01 → period 3.
            let data: Vec<f64> = (0..256)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 3.0).sin())
                .collect();
            let result = dominant_frequency(&data);
            assert_eq!(result.period, 3, "expected period 3, got {}", result.period);
        }

        #[test]
        fn acf_detects_period() {
            let n = 200;
            let period = 12;
            let data = sine_wave(n, period);
            let acf_vals = acf(&data, n / 2);

            // ACF at lag = period should be near 1.0
            assert!(
                acf_vals[period] > 0.9,
                "ACF at lag {} should be ~1, got {}",
                period,
                acf_vals[period]
            );
        }

        #[test]
        fn pacf_detects_autoregressive() {
            // AR(1) process: x[t] = 0.8*x[t-1] + noise
            let n = 500;
            let mut data = vec![0.0; n];
            let mut rng_state: u64 = 42;
            for i in 1..n
            {
                // Simple LCG pseudo-random
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
                data[i] = 0.8 * data[i - 1] + noise;
            }
            let pacf_vals = pacf(&data, 10);
            // PACF at lag 1 should be near 0.8
            assert!(
                (pacf_vals[0] - 0.8).abs() < 0.15,
                "PACF at lag 1 should be ~0.8, got {}",
                pacf_vals[0]
            );
            // PACF at lags > 1 should be small
            for &v in &pacf_vals[2..]
            {
                assert!(
                    v.abs() < 0.2,
                    "PACF at higher lag should be small, got {}",
                    v
                );
            }
        }

        #[test]
        fn periodogram_finds_sine_period() {
            let n = 256;
            let period = 16;
            let data = sine_wave(n, period);
            let result = dominant_frequency(&data);
            assert!(
                (result.period as isize - period as isize).abs() <= 1,
                "periodogram should detect period ~{}, got {}",
                period,
                result.period
            );
        }

        /// The ACF is scale-invariant: acf(c·x) must equal acf(x) for any c > 0.
        /// A fixed absolute variance cut-off returned all ones for a 1e-9
        /// amplitude series, and the unscaled sum of squares overflowed to
        /// `inf` (giving NaN) at 1e160.
        #[test]
        fn acf_is_scale_invariant_for_small_and_large_amplitudes() {
            let n = 48;
            let period = 12;
            let base = sine_wave(n, period);
            let reference = acf(&base, 24);
            // Lag `period` of a 4-cycle sine, normalised by Σc²: (n - period) / n = 0.75.
            assert!(
                (reference[period] - 0.75).abs() < 1e-9,
                "acf[12]={}",
                reference[period]
            );
            for &scale in &[1e-9, 1e-12, 1e-160, 1e160]
            {
                let scaled: Vec<f64> = base.iter().map(|&x| x * scale).collect();
                let got = acf(&scaled, 24);
                assert_eq!(got.len(), reference.len());
                for (lag, (&g, &r)) in got.iter().zip(reference.iter()).enumerate()
                {
                    assert!(
                        (g - r).abs() < 1e-9,
                        "scale {scale:e}, lag {lag}: acf={g}, expected {r}"
                    );
                }
            }
        }

        #[test]
        fn small_amplitude_acf_period_detection_matches_unit_amplitude() {
            let n = 48;
            let period = 12;
            let base = sine_wave(n, period);
            let small: Vec<f64> = base.iter().map(|&x| x * 1e-9).collect();

            let acf_base = detect_period_acf(&base, None);
            let acf_small = detect_period_acf(&small, None);
            assert_eq!(acf_base.period, period);
            assert_eq!(acf_small.period, period, "ACF period at 1e-9 amplitude");
            assert!(
                (acf_small.strength - acf_base.strength).abs() < 1e-9,
                "strength {} vs {}",
                acf_small.strength,
                acf_base.strength
            );
            assert_eq!(detect_period(&small).period, period);
        }

        #[test]
        fn acf_constant_series_returns_ones() {
            // Constant input has no defined ACF; keep the convention of ones,
            // including when the computed mean is not exactly `c` and the
            // centred values are rounding noise rather than signal.
            for &c in &[0.0, 0.1, 5.0, 1e-300, 123_456.789, 1e8 + 0.1, 1e300]
            {
                let data = vec![c; 30];
                let a = acf(&data, 5);
                assert_eq!(a.len(), 6);
                assert!(a.iter().all(|&v| v == 1.0), "c={c}: {a:?}");
            }
        }

        /// The mean must be removed from the samples only, before zero-padding.
        /// Subtracting it after padding turned the padded tail into a block of
        /// `-mean`, whose low-frequency leakage swamped the real spectrum for
        /// any series with a non-zero level (n = 200 is padded to 256).
        #[test]
        fn periodogram_is_invariant_to_a_constant_offset() {
            let n = 200;
            let period = 12;
            let base = sine_wave(n, period);
            let (f0, p0) = periodogram(&base);
            for &offset in &[1.0, 100.0, -1e4]
            {
                let shifted: Vec<f64> = base.iter().map(|&x| x + offset).collect();
                let (f1, p1) = periodogram(&shifted);
                assert_eq!(f0, f1);
                let peak = p0.iter().cloned().fold(0.0, f64::max);
                for (k, (&a, &b)) in p0.iter().zip(p1.iter()).enumerate()
                {
                    assert!(
                        (a - b).abs() <= 1e-9 * peak.max(offset.abs()),
                        "offset {offset}, bin {k}: power {b} vs {a}"
                    );
                }
                let detected = dominant_frequency(&shifted);
                assert_eq!(detected.period, period, "offset {offset}");
                assert!(
                    detected.strength > 0.5,
                    "offset {offset}: strength {}",
                    detected.strength
                );
            }
            // The zero-padded tail of a demeaned constant series carries no power.
            let (_, flat) = periodogram(&vec![42.0; 200]);
            assert!(flat.iter().all(|&v| v < 1e-18), "{flat:?}");
        }

        /// The spectral strength `P(peak) / sum(P)` is a ratio, so it must not
        /// depend on the amplitude of the series. A fixed absolute cut-off on the
        /// total power returned strength 0 for a 1e-9 amplitude sine, and the
        /// unscaled power overflowed to `inf` at 1e160 (strength NaN, and
        /// period 256 instead of 12 for n = 200).
        #[test]
        fn dominant_frequency_is_scale_invariant() {
            for &(n, period, min_strength) in &[(256, 16, 0.9), (200, 12, 0.5)]
            {
                let base = sine_wave(n, period);
                let reference = dominant_frequency(&base);
                assert_eq!(reference.period, period, "n={n}");
                assert!(
                    reference.strength > min_strength,
                    "n={n}: strength {}",
                    reference.strength
                );
                for &scale in &[1e-9, 1e-12, 1e-160, 1e160]
                {
                    let scaled: Vec<f64> = base.iter().map(|&x| x * scale).collect();
                    let got = dominant_frequency(&scaled);
                    assert_eq!(got.period, period, "n={n}, scale {scale:e}");
                    assert!(
                        (got.strength - reference.strength).abs() < 1e-9,
                        "n={n}, scale {scale:e}: strength {} vs {}",
                        got.strength,
                        reference.strength
                    );
                    assert_eq!(got.candidates.len(), reference.candidates.len());
                    for (g, r) in got.candidates.iter().zip(reference.candidates.iter())
                    {
                        // Bins holding only rounding noise have no stable order.
                        if r.strength > 1e-6
                        {
                            assert_eq!(g.period, r.period, "n={n}, scale {scale:e}");
                        }
                        assert!(
                            (g.strength - r.strength).abs() < 1e-9,
                            "n={n}, scale {scale:e}: candidate strength {} vs {}",
                            g.strength,
                            r.strength
                        );
                    }
                }
            }
        }

        #[test]
        fn dominant_frequency_constant_series_has_no_period() {
            // Includes levels whose computed mean is not exactly `c`, where the
            // centred values are rounding noise rather than signal.
            for &c in &[0.0, 0.1, 5.0, 123_456.789, 1e8 + 0.1, 1e300]
            {
                let got = dominant_frequency(&vec![c; 30]);
                assert_eq!(got.strength, 0.0, "c={c}");
                assert!(
                    got.candidates.iter().all(|cand| cand.strength == 0.0),
                    "c={c}"
                );
            }
        }

        #[test]
        fn combined_detect_period() {
            let n = 240;
            let period = 24;
            let data: Vec<f64> = (0..n)
                .map(|i| {
                    (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                        + 0.5 * (4.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                })
                .collect();
            let result = detect_period(&data);
            assert!(
                (result.period as isize - period as isize).abs() <= 2,
                "combined detection should find period ~{}, got {}",
                period,
                result.period
            );
        }
    }
}

pub mod cyclic {
    //! Cyclic pattern detection: Fourier analysis, windowed FFT (STFT),
    //! cycle length estimation, and phase detection.

    use scirust_signal::{Complex, fft_real, hanning};

    use super::*;

    /// Perform Fourier analysis to find dominant cyclic components.
    ///
    /// Returns the `CyclicPattern` with detected harmonics, amplitude, and phase.
    pub fn fourier_analysis(data: &[f64]) -> CyclicPattern {
        let n = data.len();
        if n < 4
        {
            return CyclicPattern {
                cycle_length: n as f64,
                phase: 0.0,
                amplitude: 0.0,
                harmonics: Vec::new(),
            };
        }

        let fft_size = next_pow2(n);
        let padded = demean_and_zero_pad(data, fft_size);

        let spectrum = fft_real(&padded);
        let n_f64 = fft_size as f64;

        // Find dominant frequency (excluding DC). `fft_real` returns the
        // positive-frequency bins 0..=fft_size/2, so the highest usable bin
        // (the Nyquist frequency) is the last element of `spectrum`.
        let nyquist = spectrum.len() - 1;
        let mut max_power = 0.0;
        let mut max_bin = 0usize;
        #[allow(clippy::needless_range_loop)]
        for k in 1..=nyquist
        {
            let power = spectrum[k].mag_sq();
            if power > max_power
            {
                max_power = power;
                max_bin = k;
            }
        }

        let amplitude = if max_bin < spectrum.len()
        {
            2.0 * spectrum[max_bin].mag() / n_f64
        }
        else
        {
            0.0
        };
        let phase = if max_bin < spectrum.len()
        {
            spectrum[max_bin].phase()
        }
        else
        {
            0.0
        };
        let cycle_length = if max_bin > 0
        {
            n_f64 / max_bin as f64
        }
        else
        {
            n as f64
        };

        // Extract harmonics
        let mut harmonics = Vec::new();
        let power_threshold = max_power * 0.05; // 5% of dominant
        #[allow(clippy::needless_range_loop)]
        for k in 1..=nyquist
        {
            let power = spectrum[k].mag_sq();
            if power > power_threshold
            {
                let freq = k as f64 / n_f64;
                harmonics.push(Harmonic {
                    frequency_index: k,
                    frequency: freq,
                    amplitude: 2.0 * spectrum[k].mag() / n_f64,
                    phase: spectrum[k].phase(),
                });
            }
        }
        harmonics.sort_by(|a, b| {
            b.amplitude
                .partial_cmp(&a.amplitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        CyclicPattern {
            cycle_length,
            phase,
            amplitude,
            harmonics,
        }
    }

    /// Windowed FFT (Short-Time Fourier Transform) for time-varying frequency analysis.
    ///
    /// Returns a matrix of complex spectra: `result[window_idx][freq_bin]`.
    /// Each window is multiplied by a Hanning window before FFT.
    pub fn windowed_fft(data: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<Complex>> {
        if data.len() < window_size || window_size == 0 || hop_size == 0
        {
            return Vec::new();
        }

        let fft_size = next_pow2(window_size);
        let window = hanning(window_size);
        let num_windows = (data.len() - window_size) / hop_size + 1;
        let mut result = Vec::with_capacity(num_windows);

        for w in 0..num_windows
        {
            let start = w * hop_size;
            let mut segment: Vec<f64> = data[start..start + window_size]
                .iter()
                .zip(window.iter())
                .map(|(&x, &win)| x * win)
                .collect();
            segment.resize(fft_size, 0.0);

            // Remove mean
            let m = mean(&segment);
            for v in segment.iter_mut()
            {
                *v -= m;
            }

            let spectrum = fft_real(&segment);
            result.push(spectrum);
        }

        result
    }

    /// Estimate cycle length from the time series using zero-crossings and
    /// peak-to-peak analysis.
    pub fn estimate_cycle_length(data: &[f64]) -> f64 {
        if data.len() < 3
        {
            return data.len() as f64;
        }

        let m = mean(data);
        let centered: Vec<f64> = data.iter().map(|&x| x - m).collect();

        // Find zero crossings
        let mut crossings = Vec::new();
        for i in 1..centered.len()
        {
            if (centered[i - 1] >= 0.0 && centered[i] < 0.0)
                || (centered[i - 1] < 0.0 && centered[i] >= 0.0)
            {
                // Linear interpolation for precise crossing point
                let t = centered[i - 1] / (centered[i - 1] - centered[i]);
                crossings.push((i - 1) as f64 + t);
            }
        }

        if crossings.len() < 2
        {
            // Fallback: use peak-to-peak
            let mut peaks = Vec::new();
            for i in 1..centered.len() - 1
            {
                if centered[i] > centered[i - 1] && centered[i] > centered[i + 1]
                {
                    peaks.push(i);
                }
            }
            if peaks.len() >= 2
            {
                let diffs: Vec<f64> = peaks.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
                return mean(&diffs);
            }
            return data.len() as f64;
        }

        // Average half-period from consecutive crossings
        let half_periods: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
        let avg_half = mean(&half_periods);
        2.0 * avg_half
    }

    /// Detect the phase of a known period in the time series.
    ///
    /// Returns phase in radians [0, 2π).
    pub fn detect_phase(data: &[f64], period: usize) -> f64 {
        if data.len() < period || period == 0
        {
            return 0.0;
        }

        // Compute phase via correlation with sine and cosine
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;
        for (i, &val) in data.iter().enumerate()
        {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
            sin_sum += val * angle.sin();
            cos_sum += val * angle.cos();
        }

        // The DFT convention is X_k = Σ x·(cos − i·sin), so the spectral phase is
        // atan2(Im, Re) = atan2(−sin_sum, cos_sum). This matches detect_phase_fft.
        let phase = (-sin_sum).atan2(cos_sum);
        // Normalize to [0, 2π)
        if phase < 0.0
        {
            phase + 2.0 * std::f64::consts::PI
        }
        else
        {
            phase
        }
    }

    /// Detect phase using FFT for higher accuracy.
    pub fn detect_phase_fft(data: &[f64], period: usize) -> f64 {
        if data.len() < period || period == 0
        {
            return 0.0;
        }

        let fft_size = next_pow2(data.len());
        let padded = demean_and_zero_pad(data, fft_size);

        let spectrum = fft_real(&padded);
        let bin = (fft_size as f64 / period as f64).round() as usize;

        if bin > 0 && bin < spectrum.len()
        {
            let phase = spectrum[bin].phase();
            if phase < 0.0
            {
                phase + 2.0 * std::f64::consts::PI
            }
            else
            {
                phase
            }
        }
        else
        {
            0.0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sine_wave(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
                .collect()
        }

        #[test]
        fn fourier_finds_cycle_length() {
            let n = 256;
            let period = 20;
            let data = sine_wave(n, period);
            let pattern = fourier_analysis(&data);
            assert!(
                (pattern.cycle_length - period as f64).abs() < 2.0,
                "expected cycle_length ~{}, got {}",
                period,
                pattern.cycle_length
            );
            assert!(
                pattern.amplitude > 0.5,
                "amplitude should be ~1.0, got {}",
                pattern.amplitude
            );
        }

        #[test]
        fn windowed_fft_output_shape() {
            let n = 256;
            let data = sine_wave(n, 16);
            let result = windowed_fft(&data, 64, 32);
            assert!(!result.is_empty());
            // fft_real returns N/2+1 positive-frequency bins
            assert_eq!(result[0].len(), 33);
        }

        #[test]
        fn cycle_length_estimation() {
            let n = 200;
            let period = 24;
            let data = sine_wave(n, period);
            let est = estimate_cycle_length(&data);
            assert!(
                (est - period as f64).abs() < 3.0,
                "expected ~{}, got {}",
                period,
                est
            );
        }

        #[test]
        fn detect_phase_matches_fft_sign() {
            // For real x, X_k = Σ x·(cos − i·sin); for a pure sine Re≈0,
            // Im = −Σsin² < 0, so atan2(Im, Re) = −π/2, normalized to 3π/2.
            // detect_phase must agree with the DFT-based detect_phase_fft.
            let n = 128;
            let period = 32;
            let data: Vec<f64> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
                .collect();
            let phase = detect_phase(&data, period);
            let expected = 3.0 * std::f64::consts::PI / 2.0; // ≈ 4.7124
            assert!(
                (phase - expected).abs() < 0.2,
                "expected phase ~3π/2≈4.7124, got {}",
                phase
            );
            // Sibling FFT-based implementation must give the same sign/value.
            let phase_fft = detect_phase_fft(&data, period);
            assert!(
                (phase - phase_fft).abs() < 0.2,
                "detect_phase ({}) and detect_phase_fft ({}) disagree",
                phase,
                phase_fft
            );
        }

        #[test]
        fn fourier_finds_short_period_cycle() {
            // Regression: the bin scan must cover the full positive-frequency
            // spectrum up to Nyquist, not just the lower quarter. A short-period
            // (high-frequency) cycle lands in the upper half of the bins; with
            // the old `nyquist = spectrum.len() / 2` bound it was never scanned,
            // so cycle_length/amplitude collapsed to the degenerate fallback.
            let n = 128;
            let period = 3; // bin ≈ 43, above the old (≈32) cutoff
            let data = sine_wave(n, period);
            let pattern = fourier_analysis(&data);
            assert!(
                (pattern.cycle_length - period as f64).abs() < 1.0,
                "expected short cycle_length ~{}, got {}",
                period,
                pattern.cycle_length
            );
            assert!(
                pattern.amplitude > 0.5,
                "amplitude should be ~1.0 for the short-period sine, got {}",
                pattern.amplitude
            );
        }

        /// fourier_analysis and detect_phase_fft must remove the sample mean
        /// before zero-padding, so a constant level does not change the
        /// detected cycle, amplitude or phase (n = 200 is padded to 256).
        #[test]
        fn fourier_and_phase_fft_are_invariant_to_a_constant_offset() {
            let n = 200;
            let period = 20;
            let base: Vec<f64> = (0..n)
                .map(|i| 3.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64 + 0.7).sin())
                .collect();
            let ref_pattern = fourier_analysis(&base);
            let ref_phase = detect_phase_fft(&base, period);
            assert!((ref_pattern.cycle_length - period as f64).abs() < 1.0);
            for &offset in &[10.0, 500.0, -2e3]
            {
                let shifted: Vec<f64> = base.iter().map(|&x| x + offset).collect();
                let pattern = fourier_analysis(&shifted);
                assert!(
                    (pattern.cycle_length - ref_pattern.cycle_length).abs() < 1e-9,
                    "offset {offset}: cycle_length {} vs {}",
                    pattern.cycle_length,
                    ref_pattern.cycle_length
                );
                assert!(
                    (pattern.amplitude - ref_pattern.amplitude).abs()
                        < 1e-9 * offset.abs().max(1.0),
                    "offset {offset}: amplitude {} vs {}",
                    pattern.amplitude,
                    ref_pattern.amplitude
                );
                let phase = detect_phase_fft(&shifted, period);
                assert!(
                    (phase - ref_phase).abs() < 1e-9,
                    "offset {offset}: phase {phase} vs {ref_phase}"
                );
            }
        }

        #[test]
        fn windowed_fft_zero_hop_is_empty() {
            // Regression: hop_size == 0 must not divide-by-zero panic.
            let data = sine_wave(256, 16);
            let result = windowed_fft(&data, 64, 0);
            assert!(result.is_empty(), "zero hop_size should yield no windows");
        }

        #[test]
        fn phase_detection_sine() {
            // cos wave: cos(x) = sin(x + π/2). The DFT phase at the fundamental
            // frequency for a pure cosine is 0 (since it aligns with the cos basis).
            let n = 128;
            let period = 32;
            let data: Vec<f64> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).cos())
                .collect();
            let phase = detect_phase(&data, period);
            // For cos(x), the DFT coefficient is real and positive → phase ≈ 0
            assert!(
                phase.abs() < 0.3 || (phase - 2.0 * std::f64::consts::PI).abs() < 0.3,
                "expected phase ~0 or ~2π for cosine, got {}",
                phase
            );
        }
    }
}

pub mod adjustment {
    //! Seasonal adjustment methods: moving-average decomposition, X-11 style
    //! adjustment, and deseasonalization.

    use super::*;

    /// Seasonal adjustment using moving-average decomposition.
    ///
    /// Decomposes `y = trend * seasonal * irregular` (multiplicative) or
    /// `y = trend + seasonal + irregular` (additive).
    pub fn moving_average_adjustment(data: &[f64], period: usize, additive: bool) -> Vec<f64> {
        let n = data.len();
        if n < 2 * period
        {
            return data.to_vec();
        }

        // Estimate trend via centered MA
        let trend = centered_moving_average(data, period);
        let mut trend_full = trend.clone();
        pad_centered(&mut trend_full, n);

        if additive
        {
            // Remove trend
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| y - t)
                .collect();
            // Compute seasonal factors
            let factors = compute_seasonal_factors_additive(&detrended, period);
            // Remove seasonal
            data.iter()
                .zip(factors.iter().cycle())
                .map(|(&y, &s)| y - s)
                .collect()
        }
        else
        {
            // Multiplicative
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| if t.abs() > f64::EPSILON { y / t } else { 1.0 })
                .collect();
            let factors = compute_seasonal_factors_multiplicative(&detrended, period);
            data.iter()
                .zip(factors.iter().cycle())
                .map(|(&y, &s)| if s.abs() > f64::EPSILON { y / s } else { y })
                .collect()
        }
    }

    /// X-11 style seasonal adjustment.
    ///
    /// Iterative procedure: estimate trend, compute seasonal factors,
    /// smooth factors, and remove. Typically converges in 2-3 iterations.
    pub fn x11_adjustment(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < 2 * period
        {
            return data.to_vec();
        }

        let mut current = data.to_vec();
        let iterations = 3;

        for _ in 0..iterations
        {
            // Step 1: Estimate trend (centered 2×period MA for even period)
            let trend_ma = if period.is_multiple_of(2)
            {
                let ma1 = moving_average(&current, period);
                let ma2 = moving_average(&ma1, 2);
                // Re-center
                moving_average(&ma2, 2)
            }
            else
            {
                moving_average(&current, period)
            };
            let mut trend = trend_ma.clone();
            pad_centered(&mut trend, n);

            // Step 2: Detrend (multiplicative)
            let detrended: Vec<f64> = current
                .iter()
                .zip(trend.iter())
                .map(|(&y, &t)| if t.abs() > f64::EPSILON { y / t } else { 1.0 })
                .collect();

            // Step 3: Compute unsmoothed seasonal factors
            let mut seasonal = vec![0.0; n];
            #[allow(clippy::needless_range_loop)]
            for i in 0..n
            {
                let phase = i % period;
                // Collect all values at this phase
                let mut phase_vals: Vec<f64> = detrended
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| j % period == phase)
                    .map(|(_, &v)| v)
                    .collect();
                phase_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                seasonal[i] = trimmed_mean(&phase_vals, phase_vals.len() / 4);
            }

            // Step 4: Normalize seasonal factors (each cycle sums to period)
            for cycle_start in (0..n).step_by(period)
            {
                let cycle_end = (cycle_start + period).min(n);
                let cycle_sum: f64 = seasonal[cycle_start..cycle_end].iter().sum();
                let adjustment = cycle_sum / period as f64;
                for s in seasonal[cycle_start..cycle_end].iter_mut()
                {
                    *s -= adjustment;
                }
            }

            // Step 5: Smooth the seasonal factors ACROSS cycles, per phase. Each
            // phase's run of factors (positions phase, phase+period, …) is smoothed
            // with a short moving average so the seasonal can evolve slowly while
            // keeping every value aligned to its own phase. (The previous
            // implementation smoothed the interleaved full-length series with a
            // 3×period window and carried values forward across gaps, which mixed
            // and misaligned phases and left the seasonal in the adjusted series.)
            let smooth_cycles = 3;
            for phase in 0..period
            {
                let idxs: Vec<usize> = (phase..n).step_by(period).collect();
                if idxs.len() < 2
                {
                    continue;
                }
                let phase_series: Vec<f64> = idxs.iter().map(|&i| seasonal[i]).collect();
                let win = smooth_cycles.min(phase_series.len());
                let ma = moving_average(&phase_series, win);
                let half = win / 2;
                for (k, &i) in idxs.iter().enumerate()
                {
                    let j = k.saturating_sub(half).min(ma.len() - 1);
                    seasonal[i] = ma[j];
                }
            }

            // Step 6: Deseasonalize
            current = current
                .iter()
                .zip(seasonal.iter())
                .map(|(&y, &s)| {
                    if (1.0 + s).abs() > f64::EPSILON
                    {
                        y / (1.0 + s)
                    }
                    else
                    {
                        y
                    }
                })
                .collect();
        }

        current
    }

    /// Deseasonalize a time series by subtracting (additive) or dividing
    /// (multiplicative) the estimated seasonal component.
    pub fn deseasonalize(data: &[f64], period: usize, additive: bool) -> Vec<f64> {
        let n = data.len();
        if n < 2 * period
        {
            return data.to_vec();
        }

        // Estimate seasonal component
        let trend = centered_moving_average(data, period);
        let mut trend_full = trend.clone();
        pad_centered(&mut trend_full, n);

        if additive
        {
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| y - t)
                .collect();
            let seasonal = compute_seasonal_factors_additive(&detrended, period);
            data.iter()
                .zip(seasonal.iter().cycle())
                .map(|(&y, &s)| y - s)
                .collect()
        }
        else
        {
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| if t.abs() > f64::EPSILON { y / t } else { 1.0 })
                .collect();
            let seasonal = compute_seasonal_factors_multiplicative(&detrended, period);
            data.iter()
                .zip(seasonal.iter().cycle())
                .map(|(&y, &s)| if s.abs() > f64::EPSILON { y / s } else { y })
                .collect()
        }
    }

    fn compute_seasonal_factors_additive(detrended: &[f64], period: usize) -> Vec<f64> {
        let mut factors = vec![0.0; period];
        #[allow(clippy::needless_range_loop)]
        for p in 0..period
        {
            let vals: Vec<f64> = detrended
                .iter()
                .enumerate()
                .filter(|(i, _)| i % period == p)
                .map(|(_, &v)| v)
                .collect();
            factors[p] = mean(&vals);
        }
        // Normalize: subtract mean of factors
        let m = mean(&factors);
        for f in factors.iter_mut()
        {
            *f -= m;
        }
        factors
    }

    fn compute_seasonal_factors_multiplicative(detrended: &[f64], period: usize) -> Vec<f64> {
        let mut factors = vec![1.0; period];
        #[allow(clippy::needless_range_loop)]
        for p in 0..period
        {
            let vals: Vec<f64> = detrended
                .iter()
                .enumerate()
                .filter(|(i, _)| i % period == p)
                .map(|(_, &v)| v)
                .collect();
            factors[p] = mean(&vals);
        }
        // Normalize: divide by mean of factors
        let m = mean(&factors);
        if m.abs() > f64::EPSILON
        {
            for f in factors.iter_mut()
            {
                *f /= m;
            }
        }
        factors
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn seasonal_data(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let trend = 100.0 + 0.1 * i as f64;
                    let seasonal =
                        5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                    trend + seasonal
                })
                .collect()
        }

        #[test]
        fn ma_adjustment_reduces_seasonality() {
            let period = 12;
            let n = 240;
            let data = seasonal_data(n, period);
            let adjusted = moving_average_adjustment(&data, period, true);

            // Compute variance of seasonal component before/after
            let seasonal_before: Vec<f64> = data
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let t = 100.0 + 0.1 * i as f64;
                    y - t
                })
                .collect();
            let seasonal_after: Vec<f64> = adjusted
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let t = 100.0 + 0.1 * i as f64;
                    y - t
                })
                .collect();

            let var_before = variance(&seasonal_before);
            let var_after = variance(&seasonal_after);
            assert!(
                var_after < var_before * 0.5,
                "adjustment should reduce seasonal variance: {} vs {}",
                var_before,
                var_after
            );
        }

        #[test]
        fn x11_produces_valid_output() {
            let period = 12;
            let n = 240;
            let data = seasonal_data(n, period);
            let adjusted = x11_adjustment(&data, period);
            assert_eq!(adjusted.len(), n);
            // Adjusted should have lower variance in seasonal component
            for &v in &adjusted
            {
                assert!(v.is_finite(), "adjusted values must be finite");
            }
        }

        #[test]
        fn x11_removes_seasonal_to_constant() {
            // data[i] = 100*(1 + 0.2 sin(2πi/12)) = const-trend(100) * seasonal.
            // A correct multiplicative X-11 removes the seasonal factor, returning
            // ~100 everywhere; residual variance must be small. The original series
            // has seasonal variance ≈ (100*0.2)²/2 = 200.
            let period = 12;
            let n = 240;
            let data: Vec<f64> = (0..n)
                .map(|i| 100.0 * (1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()))
                .collect();
            let original_var = variance(&data);
            assert!(
                original_var > 100.0,
                "sanity: original seasonal variance should be large, got {}",
                original_var
            );
            let adjusted = x11_adjustment(&data, period);
            // Interior points only (edges of the MA-based trend are unreliable).
            let interior = &adjusted[period..n - period];
            let adj_var = variance(interior);
            assert!(
                adj_var < 1.0,
                "X-11 should drive seasonal variance to ~0, got {}",
                adj_var
            );
        }

        #[test]
        fn deseasonalize_removes_seasonal() {
            let period = 12;
            let n = 240;
            let data = seasonal_data(n, period);
            let adjusted = deseasonalize(&data, period, true);
            // The deseasonalized series should be smoother
            let var_original = variance(&data);
            let var_adjusted = variance(&adjusted);
            assert!(
                var_adjusted < var_original,
                "deseasonalized should be smoother: {} vs {}",
                var_adjusted,
                var_original
            );
        }
    }
}

pub mod trend {
    //! Trend detection: Mann-Kendall test, Sen's slope estimator,
    //! and linear trend with confidence intervals.

    use super::*;

    /// Variance of the Mann-Kendall S statistic (without ties correction):
    /// `n·(n-1)·(2n+5) / 18`.
    ///
    /// Computed in `f64` so the intermediate product cannot overflow `usize`
    /// for very large series (`n·(n-1)·(2n+5)` exceeds `usize::MAX` well within
    /// realistic sample sizes).
    #[inline]
    fn mann_kendall_var_s(n: usize) -> f64 {
        let nf = n as f64;
        nf * (nf - 1.0) * (2.0 * nf + 5.0) / 18.0
    }

    /// Tie term of the Mann-Kendall variance: `Σ t_p·(t_p-1)·(2·t_p+5)` over
    /// every group of `t_p` equal values in `data`.
    ///
    /// The tie-corrected variance (Kendall 1975; Hipel & McLeod 1994) is
    /// `Var(S) = [n·(n-1)·(2n+5) - Σ t_p·(t_p-1)·(2·t_p+5)] / 18`, which is what
    /// pyMannKendall `original_test` and R `Kendall::MannKendall` use. Values
    /// are grouped with `==`, so `-0.0` and `0.0` form one group and each NaN
    /// is a group of its own (contributing nothing).
    fn mann_kendall_tie_term(data: &[f64]) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut term = 0.0;
        let mut i = 0;
        while i < sorted.len()
        {
            let mut j = i + 1;
            while j < sorted.len() && sorted[j] == sorted[i]
            {
                j += 1;
            }
            let t = (j - i) as f64;
            term += t * (t - 1.0) * (2.0 * t + 5.0);
            i = j;
        }
        term
    }

    /// Perform the Mann-Kendall trend test.
    ///
    /// Returns the S statistic, Z statistic (with continuity correction), and
    /// the two-sided normal-approximation p-value. The variance of S includes
    /// the standard correction for tied values, matching pyMannKendall
    /// `original_test`.
    /// S > 0 indicates increasing trend, S < 0 indicates decreasing.
    pub fn mann_kendall(data: &[f64]) -> (f64, f64, f64) {
        let n = data.len();
        if n < 3
        {
            return (0.0, 0.0, 1.0);
        }

        let mut s = 0.0;
        for i in 0..n - 1
        {
            for j in i + 1..n
            {
                let diff = data[j] - data[i];
                if diff > 0.0
                {
                    s += 1.0;
                }
                else if diff < 0.0
                {
                    s -= 1.0;
                }
            }
        }

        // Variance of S with the tie correction; ignoring ties overstates the
        // variance and therefore the p-value for data with repeated values.
        let var_s = mann_kendall_var_s(n) - mann_kendall_tie_term(data) / 18.0;
        if var_s <= 0.0
        {
            // Every value is tied (constant series): S = 0, no trend evidence.
            return (s, 0.0, 1.0);
        }

        // Z statistic
        let z = if s > 0.0
        {
            (s - 1.0) / var_s.sqrt()
        }
        else if s < 0.0
        {
            (s + 1.0) / var_s.sqrt()
        }
        else
        {
            0.0
        };

        // Two-tailed p-value from normal distribution (approximation)
        let p = 2.0 * (1.0 - normal_cdf(z.abs()));

        (s, z, p)
    }

    /// Sen's slope estimator: median of all pairwise slopes.
    pub fn sens_slope(data: &[f64]) -> f64 {
        let n = data.len();
        if n < 2
        {
            return 0.0;
        }

        let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n - 1
        {
            for j in i + 1..n
            {
                let dx = (j - i) as f64;
                if dx > f64::EPSILON
                {
                    slopes.push((data[j] - data[i]) / dx);
                }
            }
        }

        if slopes.is_empty()
        {
            return 0.0;
        }

        slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        quantile(&slopes, 0.5)
    }

    /// Complete trend analysis with Mann-Kendall test, Sen's slope,
    /// linear regression, and 95% confidence intervals.
    pub fn trend_analysis(data: &[f64]) -> TrendResult {
        let n = data.len();
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let (slope, intercept) = linear_regression(&x, data);
        let (s_stat, z_stat, p_val) = mann_kendall(data);
        let s_slope = sens_slope(data);

        // Standard error of the OLS slope: `sqrt(s² / Sxx)` with the unbiased
        // residual variance `s² = SSR / (n − 2)` (two fitted parameters), the
        // same estimate whose ratio to the slope follows Student-t with
        // `n − 2` degrees of freedom. Dividing SSR by `n` instead understates
        // the standard error by `sqrt((n − 2)/n)` and makes the interval too
        // narrow for its stated 95 % level (≈18 % too narrow at `n = 6`).
        let ssr: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| (y - (slope * i as f64 + intercept)).powi(2))
            .sum();
        let residual_var = if n > 2 { ssr / (n as f64 - 2.0) } else { 0.0 };
        let x_var: f64 = x.iter().map(|&v| (v - mean(&x)).powi(2)).sum();
        let se_slope = if x_var > f64::EPSILON
        {
            (residual_var / x_var).sqrt()
        }
        else
        {
            0.0
        };

        // Student-t 0.975 critical value for a 95% CI with df = n - 2.
        let t_crit = if n > 2
        {
            t_critical(0.95, n as f64 - 2.0)
        }
        else
        {
            t_critical(0.95, 1.0)
        };

        let ci_lower = slope - t_crit * se_slope;
        let ci_upper = slope + t_crit * se_slope;

        let direction = if p_val < 0.05
        {
            if slope > 0.0
            {
                TrendDirection::Increasing
            }
            else
            {
                TrendDirection::Decreasing
            }
        }
        else
        {
            TrendDirection::NoTrend
        };

        TrendResult {
            slope,
            intercept,
            p_value: p_val,
            s_statistic: s_stat,
            z_statistic: z_stat,
            sens_slope: s_slope,
            slope_ci: [ci_lower, ci_upper],
            significant: p_val < 0.05,
            direction,
        }
    }

    /// Cumulative sum (CUSUM) for trend detection.
    ///
    /// Returns the cumulative sum of deviations from the mean.
    pub fn cusum(data: &[f64]) -> Vec<f64> {
        let m = mean(data);
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0.0;
        for &val in data
        {
            sum += val - m;
            result.push(sum);
        }
        result
    }

    /// Approximate CDF of standard normal distribution.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz & Stegun).
    fn erf(x: f64) -> f64 {
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        let poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3 - 1.453152027 * t4
            + 1.061405429 * t5;
        sign * (1.0 - poly * (-x * x).exp())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn increasing_trend() {
            let data: Vec<f64> = (0..100)
                .map(|i| i as f64 + 0.5 * (i as f64).sin())
                .collect();
            let result = trend_analysis(&data);
            assert!(
                result.slope > 0.0,
                "slope should be positive, got {}",
                result.slope
            );
            assert!(
                result.slope > 0.8,
                "slope should be ~1.0, got {}",
                result.slope
            );
            assert!(result.significant, "should be significant");
            assert_eq!(result.direction, TrendDirection::Increasing);
        }

        #[test]
        fn decreasing_trend() {
            let data: Vec<f64> = (0..100)
                .map(|i| 100.0 - i as f64 + 0.5 * (i as f64).sin())
                .collect();
            let result = trend_analysis(&data);
            assert!(result.slope < 0.0, "slope should be negative");
            assert!(result.significant, "should be significant");
            assert_eq!(result.direction, TrendDirection::Decreasing);
        }

        #[test]
        fn no_trend() {
            let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
            let result = trend_analysis(&data);
            assert!(
                result.slope.abs() < 0.1,
                "slope should be near zero, got {}",
                result.slope
            );
        }

        #[test]
        fn sens_slope_matches_linear() {
            let data: Vec<f64> = (0..50).map(|i| 2.0 * i as f64).collect();
            let result = trend_analysis(&data);
            assert!(
                (result.sens_slope - 2.0).abs() < 0.01,
                "Sen's slope should be ~2.0, got {}",
                result.sens_slope
            );
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn cusum_detects_shift() {
            let mut data = vec![0.0; 100];
            for i in 50..100
            {
                data[i] = 1.0;
            }
            let cs = cusum(&data);
            // CUSUM should increase after index 50
            assert!(cs[99] > cs[49], "CUSUM should detect the shift");
        }

        #[test]
        fn mann_kendall_exact_small_sample() {
            // data = [1,2,3,4,5]: all C(5,2)=10 pairs are concordant increasing,
            // so S = 10. var_s = 5·4·15/18 = 16.6667, sqrt ≈ 4.0825.
            // Z = (S−1)/sqrt(var_s) = 9/4.0825 ≈ 2.2045.
            // p = 2·(1−Φ(2.2045)) ≈ 0.0275.
            let data = [1.0, 2.0, 3.0, 4.0, 5.0];
            let (s, z, p) = mann_kendall(&data);
            assert!((s - 10.0).abs() < 1e-12, "S={}", s);
            assert!((z - 2.2045).abs() < 0.01, "Z={}", z);
            assert!((p - 0.0275).abs() < 0.005, "p={}", p);
        }

        #[test]
        fn mann_kendall_applies_tie_correction() {
            // Regression: Var(S) ignored ties, so p-values were too large for
            // data with repeated values. Reference values from pyMannKendall
            // 1.4.3 `original_test` (scipy 1.18.1), checked by hand:
            // n = 10, two groups of 5 ties → Σ t(t-1)(2t+5) = 2·300 = 600,
            // Var(S) = (10·9·25 − 600)/18 = 91.6667, S = 21,
            // Z = 20/√91.6667 = 2.088932, p = 0.036714.
            // Without the correction: Var(S) = 125, Z = 1.788854, p = 0.073638.
            let data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
            let (s, z, p) = mann_kendall(&data);
            assert!((s - 21.0).abs() < 1e-12, "S={}", s);
            assert!((z - 2.088_931_871_47).abs() < 1e-9, "Z={}", z);
            assert!((p - 0.036_713_856_36).abs() < 1e-5, "p={}", p);

            // Mixed tie groups (sizes 2, 3, 2, 3): pyMannKendall gives
            // S = 37, Var(S) = 115.6667, Z = 3.347329, p = 0.000816.
            let data = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
            let (s, z, p) = mann_kendall(&data);
            assert!((s - 37.0).abs() < 1e-12, "S={}", s);
            assert!((z - 3.347_328_927_82).abs() < 1e-9, "Z={}", z);
            assert!((p - 0.000_815_943_31).abs() < 1e-5, "p={}", p);
        }

        #[test]
        fn mann_kendall_constant_series_is_no_trend() {
            // All values tied: the tie-corrected Var(S) is 0, so the result
            // must stay finite (S = 0, Z = 0, p = 1) rather than 0/0.
            let (s, z, p) = mann_kendall(&[5.0; 6]);
            assert_eq!(s, 0.0);
            assert_eq!(z, 0.0);
            assert!((p - 1.0).abs() < 1e-6, "p={}", p);
        }

        #[test]
        fn mann_kendall_tie_term_groups_equal_values() {
            // Groups {1,1}, {2,2,2}, {3}: 2·1·9 + 3·2·11 + 0 = 18 + 66 = 84.
            assert!((mann_kendall_tie_term(&[2.0, 1.0, 3.0, 2.0, 1.0, 2.0]) - 84.0).abs() < 1e-12);
            // No ties → zero.
            assert_eq!(mann_kendall_tie_term(&[1.0, 2.0, 3.0]), 0.0);
        }

        #[test]
        fn slope_ci_uses_unbiased_residual_variance() {
            // Reference: scipy.stats.linregress(x, y) with x = 0..6 gives
            // slope = 0.885714285714, stderr = 0.232115382990 (residual
            // variance SSR/(n-2)), and t.ppf(0.975, 4) = 2.776445105198, so the
            // 95 % CI is [0.241258666772, 1.530169904657]. Dividing SSR by n
            // instead gave a half-width ≈18 % too small (0.5262 vs 0.6445).
            let data = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
            let result = trend_analysis(&data);
            assert!((result.slope - 0.885_714_285_714_285_8).abs() < 1e-12);
            let half = 0.5 * (result.slope_ci[1] - result.slope_ci[0]);
            let want_half = 2.776_445_105_197_793 * 0.232_115_382_989_598_83;
            // Tolerance covers the Cornish-Fisher t quantile (≈5e-4 relative
            // at df = 4), not the variance estimator (≈18 % off before).
            assert!(
                ((half - want_half) / want_half).abs() < 1e-3,
                "CI half-width {half}, want {want_half}"
            );
            assert!((result.slope_ci[0] - 0.241_258_666_771_702_93).abs() < 1e-3);
            assert!((result.slope_ci[1] - 1.530_169_904_656_868_6).abs() < 1e-3);
        }

        #[test]
        fn sens_slope_median_of_pairwise() {
            // data = [1,5,2,8] at x=0,1,2,3. Pairwise slopes (data[j]-data[i])/(j-i):
            // 4, 0.5, 2.3333, -3, 1.5, 6 → sorted [-3, 0.5, 1.5, 2.3333, 4, 6].
            // Linear-interpolated median: (1.5 + 2.3333)/2 = 1.91667.
            let data = [1.0, 5.0, 2.0, 8.0];
            let slope = sens_slope(&data);
            assert!(
                (slope - 1.916_666_666_666_666_5).abs() < 1e-9,
                "Sen's slope={}",
                slope
            );
        }

        #[test]
        fn mann_kendall_var_s_no_overflow_large_n() {
            // Regression: computing n·(n-1)·(2n+5) in usize overflows for large
            // n (this product exceeds usize::MAX around n ≈ 2.4e6 on 64-bit and
            // panics in debug builds). The f64 formula must stay finite and match
            // the exact value for a large sample size.
            let n = 3_000_000usize;
            let var_s = mann_kendall_var_s(n);
            assert!(var_s.is_finite(), "var_s must be finite, got {}", var_s);
            let nf = n as f64;
            let expected = nf * (nf - 1.0) * (2.0 * nf + 5.0) / 18.0;
            assert!(
                (var_s - expected).abs() <= expected * 1e-12,
                "var_s={} expected~{}",
                var_s,
                expected
            );
            // The exact small-sample value must be unchanged by the refactor.
            assert!(
                (mann_kendall_var_s(5) - (5.0 * 4.0 * 15.0 / 18.0)).abs() < 1e-12,
                "small-n var_s regressed"
            );
        }

        #[test]
        fn cusum_exact_values() {
            // data = [1,2,3,4], mean=2.5. Deviations: -1.5,-0.5,0.5,1.5.
            // Cumulative: [-1.5, -2.0, -1.5, 0.0]; the closing 0.0 is invariant.
            let data = [1.0, 2.0, 3.0, 4.0];
            let cs = cusum(&data);
            let expected = [-1.5, -2.0, -1.5, 0.0];
            for (i, (&got, &want)) in cs.iter().zip(expected.iter()).enumerate()
            {
                assert!(
                    (got - want).abs() < 1e-12,
                    "cusum[{}]={} want {}",
                    i,
                    got,
                    want
                );
            }
        }
    }
}

pub mod changepoint {
    //! Change-point detection with seasonality: seasonal CUSUM and
    //! seasonal break detection.

    use super::*;

    /// Mean of each phase `i % period` over a series; returns a `period`-length
    /// vector. Empty phases are 0.0.
    fn phase_means(series: &[f64], period: usize) -> Vec<f64> {
        let mut sums = vec![0.0; period];
        let mut counts = vec![0usize; period];
        for (i, &v) in series.iter().enumerate()
        {
            let p = i % period;
            sums[p] += v;
            counts[p] += 1;
        }
        for p in 0..period
        {
            if counts[p] > 0
            {
                sums[p] /= counts[p] as f64;
            }
        }
        sums
    }

    /// Variances and variance gains below this value, measured on a series
    /// divided by its largest magnitude, are treated as rounding noise. It
    /// corresponds to a relative standard deviation of `64 * f64::EPSILON`
    /// (about `1.4e-14`) and does not depend on the units of the data.
    const NEGLIGIBLE_SCALED_VARIANCE: f64 = (64.0 * f64::EPSILON) * (64.0 * f64::EPSILON);

    /// Divide `data` by its largest finite magnitude and return the scaled
    /// series with that magnitude. Variances of the scaled series neither
    /// underflow below fixed cut-offs for tiny amplitudes nor overflow to `inf`
    /// for huge ones. A series that is all zeros, or holds a non-finite value,
    /// is returned unchanged with scale `1.0`.
    fn normalize_by_max_abs(data: &[f64]) -> (Vec<f64>, f64) {
        let scale = data.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        if scale > 0.0 && scale.is_finite() && data.iter().all(|x| x.is_finite())
        {
            (data.iter().map(|&x| x / scale).collect(), scale)
        }
        else
        {
            (data.to_vec(), 1.0)
        }
    }

    /// Seasonal CUSUM (Cumulative Sum) control chart.
    ///
    /// Detects changes in the seasonal pattern by computing separate
    /// CUSUM statistics for each phase within the period.
    pub fn seasonal_cusum(data: &[f64], period: usize, threshold: f64) -> Vec<ChangePoint> {
        let n = data.len();
        if n < 2 * period
        {
            return Vec::new();
        }

        // Estimate seasonal baseline
        let trend = centered_moving_average(data, period);
        let mut trend_full = trend.clone();
        pad_centered(&mut trend_full, n);

        let deseasonalized: Vec<f64> = data
            .iter()
            .zip(trend_full.iter())
            .map(|(&y, &t)| y - t)
            .collect();

        // Compute baseline mean for each phase
        let mut phase_means = vec![0.0; period];
        let mut phase_counts = vec![0usize; period];
        for (i, &val) in deseasonalized.iter().enumerate()
        {
            let phase = i % period;
            phase_means[phase] += val;
            phase_counts[phase] += 1;
        }
        for p in 0..period
        {
            if phase_counts[p] > 0
            {
                phase_means[p] /= phase_counts[p] as f64;
            }
        }

        // Compute CUSUM for each phase
        let mut cusum_vals = vec![0.0f64; n];
        for i in 0..n
        {
            let phase = i % period;
            let deviation = deseasonalized[i] - phase_means[phase];
            cusum_vals[i] = if i == 0
            {
                deviation
            }
            else
            {
                cusum_vals[i - 1] + deviation
            };
        }

        // Detect change points: where CUSUM exceeds threshold
        let mut change_points = Vec::new();
        let mut last_cp = 0usize;
        let mut max_abs = 0.0f64;
        let mut max_idx = 0usize;

        for i in 1..n
        {
            if (cusum_vals[i] - cusum_vals[last_cp]).abs() > max_abs
            {
                max_abs = (cusum_vals[i] - cusum_vals[last_cp]).abs();
                max_idx = i;
            }
            if (cusum_vals[i] - cusum_vals[last_cp]).abs() > threshold
            {
                let seasonal_comp = phase_means[max_idx % period];
                change_points.push(ChangePoint {
                    index: max_idx,
                    magnitude: max_abs,
                    seasonal_component: seasonal_comp,
                });
                last_cp = max_idx;
                max_abs = 0.0;
                max_idx = i;
            }
        }

        change_points
    }

    /// Detect seasonal breaks using moving-window variance comparison.
    ///
    /// Compares variance in a left window vs right window at each position.
    /// A significant ratio indicates a change point.
    ///
    /// The test statistics are ratios, so the detected indices and magnitudes
    /// do not depend on the units of `data` (multiplying the series by a
    /// constant leaves them unchanged). Two windows that are both constant (up
    /// to rounding) are not a variance change, so a constant series has no
    /// breaks.
    pub fn seasonal_break_detection(
        data: &[f64],
        period: usize,
        window_size: usize,
        significance: f64,
    ) -> Vec<ChangePoint> {
        let n = data.len();
        let half_win = window_size / 2;
        if n < 2 * window_size + period
        {
            return Vec::new();
        }

        // Work on the series divided by its largest magnitude so the variance
        // cut-offs below are relative to the data scale: a fixed absolute
        // cut-off treated every window of a series with amplitude <= 1e-8 as
        // constant, and the raw variances overflowed to `inf` at 1e160.
        let (scaled, scale) = normalize_by_max_abs(data);

        // Deseasonalize first
        let adjusted = super::adjustment::deseasonalize(&scaled, period, true);

        // Seasonal component (additive): the part removed by deseasonalizing,
        // in the units of `data`. Averaged per phase so each change point can
        // report the seasonal shift at its phase, mirroring `seasonal_cusum`.
        let seasonal: Vec<f64> = scaled
            .iter()
            .zip(adjusted.iter())
            .map(|(&y, &a)| (y - a) * scale)
            .collect();
        let phase_seasonal = phase_means(&seasonal, period);

        // `significance` is the two-sided alpha level of the mean-shift test;
        // convert it to a Student-t critical value with
        // df = (n_left - 1) + (n_right - 1), both windows holding `half_win`
        // points. It does not depend on the position, so compute it once.
        // A significance outside (0, 1) falls back to the conventional 0.05.
        let alpha = if significance > 0.0 && significance < 1.0
        {
            significance
        }
        else
        {
            0.05
        };
        let df = 2.0 * half_win as f64 - 2.0;
        let critical_t = t_critical(1.0 - alpha, df);

        let mut change_points = Vec::new();
        let mut last_cp = 0usize;

        for i in half_win + period..n - half_win - period
        {
            // Skip if too close to last change point
            if i - last_cp < period
            {
                continue;
            }

            let left = &adjusted[i - half_win..i];
            let right = &adjusted[i..i + half_win];

            let var_left = variance(left);
            let var_right = variance(right);

            // F-test for variance ratio. Two windows without variance are not
            // a variance change.
            let f_stat = if var_right > NEGLIGIBLE_SCALED_VARIANCE
            {
                var_left / var_right
            }
            else if var_left > NEGLIGIBLE_SCALED_VARIANCE
            {
                f64::INFINITY
            }
            else
            {
                1.0
            };

            // Also check mean shift
            let mean_left = mean(left);
            let mean_right = mean(right);
            let pooled_var = (var_left + var_right) / 2.0;
            let pooled_std = pooled_var.sqrt();
            let t_stat = if pooled_var > NEGLIGIBLE_SCALED_VARIANCE
            {
                (mean_left - mean_right).abs() / (pooled_std * (2.0 / half_win as f64).sqrt())
            }
            else
            {
                0.0
            };

            // Combined criterion: significant mean shift or variance change.
            if t_stat > critical_t || (f_stat > 2.0 || f_stat < 0.5)
            {
                let magnitude = t_stat.max((f_stat - 1.0).abs());
                change_points.push(ChangePoint {
                    index: i,
                    magnitude,
                    seasonal_component: phase_seasonal[i % period],
                });
                last_cp = i;
            }
        }

        // Merge nearby change points (within one period)
        merge_nearby(&mut change_points, period);
        change_points
    }

    /// Binary segmentation for seasonal change points.
    ///
    /// Recursively splits the series at the point of maximum likelihood ratio.
    ///
    /// The split indices do not depend on the units of `data`; each
    /// change point's `magnitude` is the variance reduction of its split, in
    /// squared units of `data`. Splits whose variance reduction is only
    /// rounding noise relative to the data scale are not reported.
    pub fn seasonal_binary_segmentation(
        data: &[f64],
        period: usize,
        min_segment: usize,
        max_segments: usize,
    ) -> Vec<ChangePoint> {
        let n = data.len();
        if n < 2 * min_segment
        {
            return Vec::new();
        }

        // Per-phase seasonal contribution (phase mean minus the global mean), so
        // each change point can report the seasonal value at its phase rather than
        // a hard-coded 0.0.
        let global_mean = mean(data);
        let phase_seasonal: Vec<f64> = phase_means(data, period)
            .into_iter()
            .map(|m| m - global_mean)
            .collect();

        // Variances are computed on the series divided by its largest
        // magnitude: a fixed absolute cut-off on the raw variance gain dropped
        // splits of small-amplitude series (all of them at 1e-9), and the raw
        // variances overflowed to `inf` (gain NaN) at 1e160.
        let (scaled, scale) = normalize_by_max_abs(data);

        let mut change_points = Vec::new();
        let mut segments: Vec<(usize, usize)> = vec![(0, n)];

        while change_points.len() < max_segments && !segments.is_empty()
        {
            let mut best_gain = 0.0f64;
            let mut best_segment_idx = 0usize;
            let mut best_split = 0usize;

            for (seg_idx, &(start, end)) in segments.iter().enumerate()
            {
                let seg_len = end - start;
                if seg_len < 2 * min_segment
                {
                    continue;
                }

                let segment = &scaled[start..end];
                let total_var = variance(segment);

                // Try each split point
                for split in (start + min_segment..end - min_segment).step_by(period.max(1))
                {
                    let left = &scaled[start..split];
                    let right = &scaled[split..end];

                    let left_var = variance(left);
                    let right_var = variance(right);
                    let left_w = left.len() as f64 / seg_len as f64;
                    let right_w = right.len() as f64 / seg_len as f64;

                    let pooled_var = left_w * left_var + right_w * right_var;
                    let gain = total_var - pooled_var;

                    if gain > best_gain
                    {
                        best_gain = gain;
                        best_segment_idx = seg_idx;
                        best_split = split;
                    }
                }
            }

            if best_gain > NEGLIGIBLE_SCALED_VARIANCE
            {
                let (start, end) = segments[best_segment_idx];
                segments.remove(best_segment_idx);
                segments.push((start, best_split));
                segments.push((best_split, end));

                change_points.push(ChangePoint {
                    index: best_split,
                    magnitude: best_gain * scale * scale,
                    seasonal_component: phase_seasonal[best_split % period],
                });
            }
            else
            {
                break;
            }
        }

        change_points.sort_by_key(|cp| cp.index);
        change_points
    }

    /// Merge change points that are closer than `min_distance` apart.
    fn merge_nearby(change_points: &mut Vec<ChangePoint>, min_distance: usize) {
        if change_points.is_empty()
        {
            return;
        }
        change_points.sort_by_key(|cp| cp.index);
        let mut merged = vec![change_points[0].clone()];
        for cp in change_points.iter().skip(1)
        {
            let last = merged.last().unwrap();
            if cp.index - last.index >= min_distance
            {
                merged.push(cp.clone());
            }
            else if cp.magnitude > last.magnitude
            {
                *merged.last_mut().unwrap() = cp.clone();
            }
        }
        *change_points = merged;
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn cusum_detects_level_shift() {
            let mut data = vec![0.0; 200];
            // Add seasonal pattern
            for i in 0..200
            {
                data[i] = (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            }
            // Shift mean at index 100
            for i in 100..200
            {
                data[i] += 3.0;
            }

            let cps = seasonal_cusum(&data, 12, 3.0);
            assert!(!cps.is_empty(), "should detect at least one change point");
            // The change point should be near index 100
            let closest = cps
                .iter()
                .min_by_key(|cp| (cp.index as isize - 100).abs())
                .unwrap();
            assert!(
                (closest.index as isize - 100).abs() < 20,
                "change point should be near 100, got {}",
                closest.index
            );
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn break_detection_finds_shift() {
            let mut data = vec![0.0; 240];
            let period = 12;
            for i in 0..240
            {
                data[i] = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            }
            // Large shift at index 120
            for i in 120..240
            {
                data[i] += 5.0;
            }

            let cps = seasonal_break_detection(&data, period, 48, 0.0);
            assert!(!cps.is_empty(), "should detect at least one break");
            let closest = cps
                .iter()
                .min_by_key(|cp| (cp.index as isize - 120).abs())
                .unwrap();
            assert!(
                (closest.index as isize - 120).abs() < 20,
                "break should be near 120, got {}",
                closest.index
            );
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn binary_segmentation_finds_multiple() {
            let mut data = vec![0.0; 360];
            let period = 12;
            for i in 0..360
            {
                data[i] = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            }
            // Two shifts
            for i in 120..240
            {
                data[i] += 3.0;
            }
            for i in 240..360
            {
                data[i] += 6.0;
            }

            let cps = seasonal_binary_segmentation(&data, period, 30, 5);
            assert!(
                cps.len() >= 2,
                "should find at least 2 change points, found {}",
                cps.len()
            );
        }

        /// Monthly sine plus a level shift of 5 at index 120 plus deterministic
        /// uniform noise in [-0.25, 0.25).
        fn noisy_shifted_sine(n: usize) -> Vec<f64> {
            let mut state: u64 = 12345;
            (0..n)
                .map(|i| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.5;
                    let shift = if i >= 120 { 5.0 } else { 0.0 };
                    (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() + shift + noise
                })
                .collect()
        }

        fn assert_rel_close(got: f64, want: f64, what: &str) {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(f64::MIN_POSITIVE),
                "{what}: {got:e} vs {want:e}"
            );
        }

        /// The break statistics are ratios, so scaling the series must not
        /// change the detected breaks. A fixed absolute variance cut-off made
        /// every window of a 1e-9 amplitude series look constant (14 breaks,
        /// one per period, instead of 5), and the raw variances overflowed to
        /// `inf` at 1e160 (0 breaks).
        #[test]
        fn break_detection_is_scale_invariant() {
            let base = noisy_shifted_sine(240);
            let reference = seasonal_break_detection(&base, 12, 48, 0.05);
            assert!(!reference.is_empty());
            for &scale in &[1e-9, 1e-12, 1e-100, 1e8, 1e160]
            {
                let scaled: Vec<f64> = base.iter().map(|&x| x * scale).collect();
                let got = seasonal_break_detection(&scaled, 12, 48, 0.05);
                let got_idx: Vec<usize> = got.iter().map(|cp| cp.index).collect();
                let ref_idx: Vec<usize> = reference.iter().map(|cp| cp.index).collect();
                assert_eq!(got_idx, ref_idx, "scale {scale:e}");
                for (g, r) in got.iter().zip(reference.iter())
                {
                    assert_rel_close(g.magnitude, r.magnitude, "magnitude");
                    assert_rel_close(
                        g.seasonal_component / scale,
                        r.seasonal_component,
                        "seasonal_component",
                    );
                }
            }
        }

        /// Two constant windows are not a variance change: a constant series
        /// used to report a break every period (14 for n = 240).
        #[test]
        fn break_detection_constant_series_has_no_breaks() {
            for &level in &[0.0, 1.0, 5.0, 1e-9, 1e8 + 0.1]
            {
                let data = vec![level; 240];
                let cps = seasonal_break_detection(&data, 12, 48, 0.05);
                assert!(cps.is_empty(), "level {level:e}: {cps:?}");
            }
        }

        /// Split indices must not depend on the units of the data, and the
        /// reported variance reduction scales with the square of the data. A
        /// fixed absolute cut-off on the gain dropped every split at 1e-9
        /// (0 change points instead of 5), and the raw variances overflowed to
        /// `inf` at 1e160 (0 change points).
        #[test]
        fn binary_segmentation_is_scale_invariant() {
            let base = noisy_shifted_sine(240);
            let reference = seasonal_binary_segmentation(&base, 12, 30, 5);
            assert_eq!(reference.len(), 5);
            for &scale in &[1e-9, 1e-12, 1e-100, 1e8, 1e100, 1e160]
            {
                let scaled: Vec<f64> = base.iter().map(|&x| x * scale).collect();
                let got = seasonal_binary_segmentation(&scaled, 12, 30, 5);
                let got_idx: Vec<usize> = got.iter().map(|cp| cp.index).collect();
                let ref_idx: Vec<usize> = reference.iter().map(|cp| cp.index).collect();
                assert_eq!(got_idx, ref_idx, "scale {scale:e}");
                for (g, r) in got.iter().zip(reference.iter())
                {
                    assert_rel_close(
                        g.seasonal_component / scale,
                        r.seasonal_component,
                        "seasonal_component",
                    );
                    // The squared-unit gain is not representable at 1e160.
                    if (r.magnitude * scale * scale).is_finite()
                    {
                        assert_rel_close(g.magnitude / (scale * scale), r.magnitude, "magnitude");
                    }
                }
            }
        }

        /// A constant series has no variance to reduce; rounding noise of the
        /// level 1e8 + 0.1 used to pass the absolute cut-off and yield a split.
        #[test]
        fn binary_segmentation_constant_series_has_no_splits() {
            for &level in &[0.0, 1.0, 1e-9, 1e8 + 0.1]
            {
                let data = vec![level; 240];
                let cps = seasonal_binary_segmentation(&data, 12, 30, 5);
                assert!(cps.is_empty(), "level {level:e}: {cps:?}");
            }
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::adjustment::deseasonalize;
    use super::changepoint::seasonal_cusum;
    use super::cyclic::fourier_analysis;
    use super::detection::detect_period;
    use super::stl::STLConfig;
    use super::stl::stl_decompose;
    use super::trend::trend_analysis;

    /// Generate a synthetic time series with trend, seasonality, and noise.
    fn synthetic_series(n: usize, period: usize) -> Vec<f64> {
        let mut rng_state: u64 = 12345;
        (0..n)
            .map(|i| {
                let t = i as f64;
                let trend = 50.0 + 0.05 * t;
                let seasonal = 3.0 * (2.0 * std::f64::consts::PI * t / period as f64).sin()
                    + 1.5 * (4.0 * std::f64::consts::PI * t / period as f64).cos();
                // Simple noise
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.5;
                trend + seasonal + noise
            })
            .collect()
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn full_pipeline() {
        let period = 12;
        let n = 240;
        let data = synthetic_series(n, period);

        // 1. Detect period
        let detection = detect_period(&data);
        assert!(
            (detection.period as isize - period as isize).abs() <= 3,
            "detected period: {}",
            detection.period
        );

        // 2. STL decomposition
        let config = STLConfig {
            period,
            inner_iterations: 2,
            ..Default::default()
        };
        let stl_result = stl_decompose(&data, &config);
        assert_eq!(stl_result.length, n);
        // Decomposition should sum correctly
        for i in 0..n
        {
            let reconstructed =
                stl_result.trend[i] + stl_result.seasonal[i] + stl_result.remainder[i];
            assert!(
                (data[i] - reconstructed).abs() < 1e-10,
                "STL decomposition error at {}",
                i
            );
        }

        // 3. Fourier analysis
        let cyclic = fourier_analysis(&stl_result.remainder);
        // Remainder should have low amplitude
        assert!(
            cyclic.amplitude < 2.0,
            "remainder amplitude: {}",
            cyclic.amplitude
        );

        // 4. Deseasonalize
        let adjusted = deseasonalize(&data, period, true);
        assert_eq!(adjusted.len(), n);

        // 5. Trend analysis
        let trend_result = trend_analysis(&data);
        assert!(trend_result.slope > 0.0, "should detect positive trend");
        assert!(
            trend_result.slope > 0.03,
            "slope should be ~0.05, got {}",
            trend_result.slope
        );

        // 6. Change point detection
        let mut data_with_shift = data.clone();
        for i in 180..n
        {
            data_with_shift[i] += 5.0;
        }
        let cps = seasonal_cusum(&data_with_shift, period, 5.0);
        assert!(!cps.is_empty(), "should detect change point");
    }

    #[test]
    fn edge_cases() {
        // Very short series
        let short_data = vec![1.0, 2.0, 3.0, 4.0];
        let result = trend_analysis(&short_data);
        assert!(result.slope > 0.0);

        // Constant series
        let constant = vec![5.0; 100];
        let result = trend_analysis(&constant);
        assert!(result.slope.abs() < f64::EPSILON);
        assert!(!result.significant);

        // Single value
        let single = vec![42.0];
        let result = trend_analysis(&single);
        assert_eq!(result.slope, 0.0);
    }
}

#[cfg(test)]
mod t_critical_tests {
    use super::{inv_normal_cdf, t_critical};

    /// Relative-error check (never bit-for-bit).
    fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
        let rel = ((got - want) / want).abs();
        assert!(
            rel < tol,
            "{what}: got {got}, want {want} (rel err {rel:e})"
        );
    }

    // Reference values: scipy.stats.t.ppf((1 + conf) / 2, df), SciPy 1.18.1.
    const DFS: [f64; 13] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0,
    ];
    const REF_090: [f64; 13] = [
        6.313751514675037,
        2.919985580353724,
        2.353363434801823,
        2.13184678632665,
        2.015048373333023,
        1.943180280515304,
        1.859548037530897,
        1.812461122811676,
        1.753050355692572,
        1.724718242920787,
        1.697260886593957,
        1.670648864904636,
        1.657650899355235,
    ];
    const REF_095: [f64; 13] = [
        12.706204736174694,
        4.302652729749462,
        3.182446305283708,
        2.776445105197793,
        2.570581835636315,
        2.446911851144979,
        2.306004135204166,
        2.228138851986274,
        2.131449545559776,
        2.085963447265864,
        2.042272456301238,
        2.00029782201426,
        1.97993040508244,
    ];
    const REF_099: [f64; 13] = [
        63.656741162871526,
        9.924843200918287,
        5.840909309733355,
        4.604094871349992,
        4.032142983555228,
        3.707428021324807,
        3.355387331333395,
        3.16927267261695,
        2.946712883475238,
        2.845339709786108,
        2.749995653567225,
        2.660283028855037,
        2.617421145106865,
    ];
    const REF_0999: [f64; 13] = [
        636.6192487687897,
        31.599054576445365,
        12.923978636687961,
        8.610301581379522,
        6.868826625881276,
        5.95881617881889,
        5.041305433373456,
        4.586893858702708,
        4.072765195903846,
        3.849516274930874,
        3.645958635042063,
        3.460200469196392,
        3.373453768562533,
    ];

    #[test]
    fn small_df_95_matches_student_t_table() {
        // The truncated Cornish-Fisher series returned 11.300 at df = 1 and
        // 4.2706 at df = 2 (11 % and 0.75 % too small).
        assert_rel(t_critical(0.95, 1.0), 12.706_204_736_174_694, 1e-9, "df=1");
        assert_rel(t_critical(0.95, 2.0), 4.302_652_729_749_462, 1e-9, "df=2");
        assert_rel(t_critical(0.95, 3.0), 3.182_446_305_283_708, 1e-9, "df=3");
    }

    #[test]
    fn matches_scipy_across_df_and_confidence_levels() {
        for (conf, refs) in [
            (0.90, &REF_090),
            (0.95, &REF_095),
            (0.99, &REF_099),
            (0.999, &REF_0999),
        ]
        {
            for (&df, &want) in DFS.iter().zip(refs.iter())
            {
                assert_rel(
                    t_critical(conf, df),
                    want,
                    1e-9,
                    &format!("conf={conf}, df={df}"),
                );
            }
        }
    }

    #[test]
    fn closed_forms_for_one_and_two_df() {
        // df = 1: tan(pi (p - 1/2)); df = 2: (2p - 1) / sqrt(2 p (1 - p)).
        for conf in [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]
        {
            let p: f64 = 0.5 + conf / 2.0;
            let cauchy = (std::f64::consts::PI * (p - 0.5)).tan();
            let two = (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
            assert_rel(t_critical(conf, 1.0), cauchy, 1e-12, "df=1 closed form");
            assert_rel(t_critical(conf, 2.0), two, 1e-12, "df=2 closed form");
        }
    }

    #[test]
    fn non_integer_df() {
        assert_rel(t_critical(0.95, 1.5), 6.016_663_104_427_929, 1e-9, "df=1.5");
        assert_rel(t_critical(0.95, 2.5), 3.574_654_842_003_682, 1e-9, "df=2.5");
        assert_rel(t_critical(0.95, 7.3), 2.345_066_736_547_703, 1e-9, "df=7.3");
    }

    #[test]
    fn large_and_degenerate_df_approach_normal_quantile() {
        let z = inv_normal_cdf(0.975);
        assert_rel(z, 1.959_963_984_540_054, 1e-8, "normal quantile");
        // scipy.stats.t.ppf(0.975, 1e5) = 1.9599877075346095.
        assert_rel(
            t_critical(0.95, 1e5),
            1.959_987_707_534_609_5,
            1e-9,
            "df=1e5",
        );
        assert_eq!(t_critical(0.95, f64::INFINITY), z);
        assert_eq!(t_critical(0.95, 0.0), z);
        // Monotone decreasing in df towards z.
        let mut prev = f64::INFINITY;
        for df in 1..=200
        {
            let t = t_critical(0.95, df as f64);
            assert!(t < prev && t > z, "df={df}: {t} (prev {prev}, z {z})");
            prev = t;
        }
    }
}
