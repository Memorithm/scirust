//! Acoustic leak correlation.
//!
//! A pressurised leak radiates broadband noise that travels along the pipe wall
//! and water column to sensors at each end of a segment. If the leak sits closer
//! to sensor A, its noise reaches A first; the *delay* between the two arrivals
//! fixes the leak's position. We recover that delay as the lag of the peak of
//! the cross-correlation of the two sensor signals, then convert it to a
//! distance with the segment length and wave speed.
//!
//! For a leak distance `d_a` from sensor A on a segment of length `L`
//! (`d_b = L − d_a`), the noise reaches B later than A by `(d_b − d_a)/c`. So the
//! correlation peak lag `τ` (seconds, B relative to A) gives
//! `d_a = (L − c·τ)/2`.

use serde::{Deserialize, Serialize};

/// A localized leak.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LeakLocation {
    /// Distance of the leak from sensor A (metres).
    pub dist_from_a: f64,
    /// Cross-correlation peak lag (samples, B relative to A; positive = B later).
    pub lag_samples: i64,
    /// Normalised peak correlation in `[-1, 1]` (a confidence proxy).
    pub peak_corr: f64,
}

/// Lag (in samples, in `[-max_lag, max_lag]`) at which `y` best matches `x`, by
/// maximising the normalised cross-correlation `Σ x[t]·y[t+lag]`. A positive lag
/// means `y` is delayed relative to `x`.
///
/// Lags whose magnitude reaches the longer signal's length have no overlapping
/// samples, so the search is capped there: an oversized `max_lag` (up to
/// `usize::MAX`) costs no more than `max(x.len(), y.len())` and never overflows.
#[allow(clippy::needless_range_loop)]
pub fn best_lag(x: &[f64], y: &[f64], max_lag: usize) -> (i64, f64) {
    let nx = x.len();
    let ny = y.len();
    let max_lag = max_lag.min(nx.max(ny));
    let energy = |s: &[f64]| s.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm = energy(x) * energy(y);
    let mut best = f64::MIN;
    let mut best_lag = 0i64;
    for lag in -(max_lag as i64)..=(max_lag as i64)
    {
        let mut acc = 0.0;
        for t in 0..nx
        {
            let j = t as i64 + lag;
            if j >= 0 && (j as usize) < ny
            {
                acc += x[t] * y[j as usize];
            }
        }
        if acc > best
        {
            best = acc;
            best_lag = lag;
        }
    }
    let corr = if norm > 0.0 { best / norm } else { 0.0 };
    (best_lag, corr)
}

/// Locate a leak on a pipe segment of length `pipe_length` (m) from two sensor
/// recordings (`sensor_a`, `sensor_b`), the acoustic `wave_speed` (m/s) and the
/// `sample_rate` (Hz). Searches lags up to the segment's acoustic length.
///
/// The geometry must be physical: `pipe_length` finite and non-negative,
/// `wave_speed` and `sample_rate` finite and strictly positive. Otherwise no
/// position can be derived and the result has `dist_from_a = NaN`,
/// `lag_samples = 0` and `peak_corr = 0.0` instead of panicking.
pub fn locate_leak(
    sensor_a: &[f64],
    sensor_b: &[f64],
    pipe_length: f64,
    wave_speed: f64,
    sample_rate: f64,
) -> LeakLocation {
    let valid = pipe_length.is_finite()
        && pipe_length >= 0.0
        && wave_speed.is_finite()
        && wave_speed > 0.0
        && sample_rate.is_finite()
        && sample_rate > 0.0;
    if !valid
    {
        return LeakLocation {
            dist_from_a: f64::NAN,
            lag_samples: 0,
            peak_corr: 0.0,
        };
    }
    // The physical delay can never exceed the whole segment's transit time.
    // The float-to-int cast saturates; `best_lag` caps the search at the
    // signal length, so an enormous transit time cannot overflow.
    let max_lag = (((pipe_length / wave_speed) * sample_rate).ceil() as usize).saturating_add(1);
    let (lag, corr) = best_lag(sensor_a, sensor_b, max_lag);
    let tau = lag as f64 / sample_rate;
    let mut d_a = (pipe_length - wave_speed * tau) / 2.0;
    d_a = d_a.clamp(0.0, pipe_length); // a leak is physically on the segment
    LeakLocation {
        dist_from_a: d_a,
        lag_samples: lag,
        peak_corr: corr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic white noise in [-1, 1).
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn best_lag_recovers_a_planted_delay() {
        // y is x delayed by exactly 5 samples → best lag is +5, and the
        // normalised correlation at that lag is 1 (identical shifted signals).
        let x = noise(200, 0x5A1);
        let mut y = vec![0.0; x.len() + 5];
        for (i, &v) in x.iter().enumerate()
        {
            y[i + 5] = v;
        }
        let (lag, corr) = best_lag(&x, &y, 20);
        assert_eq!(lag, 5);
        assert!((corr - 1.0).abs() < 1e-9, "corr {corr}");
    }

    #[test]
    fn recovers_a_known_leak_position() {
        // Geometry chosen so arrival delays are whole samples:
        // c = 1000 m/s, fs = 10 kHz → 10 samples per metre.
        // Leak 30 m from A on a 100 m segment → A delay 300, B delay 700 samples.
        let (l, c, fs) = (100.0, 1000.0, 10_000.0);
        let (da, db) = (30.0, 70.0);
        let src = noise(4000, 0x1EA4);
        let delay_a = (da / c * fs) as usize; // 300
        let delay_b = (db / c * fs) as usize; // 700
        let total = src.len() + delay_b + 10;
        let mut a = vec![0.0; total];
        let mut b = vec![0.0; total];
        for (i, &v) in src.iter().enumerate()
        {
            a[i + delay_a] = v;
            b[i + delay_b] = v;
        }
        let loc = locate_leak(&a, &b, l, c, fs);
        assert!(
            (loc.dist_from_a - 30.0).abs() < 0.5,
            "got {} (lag {})",
            loc.dist_from_a,
            loc.lag_samples
        );
        assert!(loc.peak_corr > 0.9, "weak correlation {}", loc.peak_corr);
    }

    #[test]
    fn a_centered_leak_reads_mid_span() {
        let (l, c, fs) = (80.0, 1200.0, 12_000.0);
        let src = noise(3000, 0x2B17);
        // 40 m each way → 40/1200*12000 = 400 samples both sides (zero lag).
        let d = (40.0 / c * fs) as usize;
        let total = src.len() + d + 10;
        let mut a = vec![0.0; total];
        let mut b = vec![0.0; total];
        for (i, &v) in src.iter().enumerate()
        {
            a[i + d] = v;
            b[i + d] = v;
        }
        let loc = locate_leak(&a, &b, l, c, fs);
        assert_eq!(loc.lag_samples, 0);
        assert!(
            (loc.dist_from_a - 40.0).abs() < 0.5,
            "got {}",
            loc.dist_from_a
        );
    }

    #[test]
    fn invalid_geometry_returns_nan_instead_of_panicking() {
        let a = noise(256, 7);
        let b = a.clone();
        for (l, c, fs) in [
            (f64::NAN, 1200.0, 8000.0),
            (-10.0, 1200.0, 8000.0),
            (f64::INFINITY, 1200.0, 8000.0),
            (100.0, 0.0, 8000.0),
            (100.0, -1200.0, 8000.0),
            (100.0, f64::NAN, 8000.0),
            (100.0, 1200.0, 0.0),
            (100.0, 1200.0, f64::NAN),
        ]
        {
            let loc = locate_leak(&a, &b, l, c, fs);
            assert!(loc.dist_from_a.is_nan(), "L={l}, c={c}, fs={fs}: {loc:?}");
            assert_eq!(loc.lag_samples, 0);
            assert_eq!(loc.peak_corr, 0.0);
        }
    }

    #[test]
    fn huge_transit_time_does_not_overflow_the_lag_search() {
        // wave speed so small that the transit time saturates the usize cast.
        let a = noise(128, 3);
        let b = a.clone();
        let loc = locate_leak(&a, &b, 100.0, 1e-300, 8000.0);
        assert_eq!(loc.lag_samples, 0);
        assert!((loc.dist_from_a - 50.0).abs() < 1e-9, "{loc:?}");
        let (lag, corr) = best_lag(&a, &b, usize::MAX);
        assert_eq!(lag, 0);
        assert!((corr - 1.0).abs() < 1e-12);
    }
}
