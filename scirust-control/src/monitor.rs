//! Control-loop performance monitoring: oscillation detection.
//!
//! A persistently oscillating loop (bad tuning, valve stiction) shows a regular,
//! periodic error. This detector measures the regularity of the error's
//! zero-crossings: regularly spaced crossings ⇒ sustained oscillation.

use serde::{Deserialize, Serialize};

/// Oscillation diagnosis for an error trace.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OscillationReport {
    /// Whether a regular oscillation was detected.
    pub oscillating: bool,
    /// Estimated oscillation period in samples (0 if undetected).
    pub period_samples: f64,
    /// Crossing-interval regularity in `[0, 1]` (1 = perfectly periodic).
    pub regularity: f64,
}

/// Late-window RMS floor of [`detect_oscillation`], as a fraction of the RMS
/// of the whole error trace about its mean.
const SUSTAINED_RMS_FLOOR: f64 = 1e-9;

/// Detect a sustained oscillation in a control `error` trace. Requires at least
/// `min_crossings` mean-crossings, and a crossing-interval regularity at or
/// above `regularity_thresh` (e.g. `0.8`).
///
/// The oscillation also has to be sustained: the RMS of the last quarter of
/// the trace about that quarter's own mean must be at least half that of the
/// first quarter (about its own mean) and above `1e-9` times the RMS of the
/// whole trace about its mean. Measuring each quarter about its own mean
/// means a constant stretch has zero RMS whatever level it sits at, so a
/// burst that has ended, or a loop that has settled at a steady-state error,
/// is not taken as sustained. These thresholds are all relative, so scaling
/// `error` by a positive factor (a change of units) leaves the report
/// unchanged apart from round-off.
pub fn detect_oscillation(
    error: &[f64],
    min_crossings: usize,
    regularity_thresh: f64,
) -> OscillationReport {
    let n = error.len();
    let none = OscillationReport {
        oscillating: false,
        period_samples: 0.0,
        regularity: 0.0,
    };
    if n < 3
    {
        return none;
    }
    let mean = error.iter().sum::<f64>() / n as f64;
    let mut crossings = Vec::new();
    for i in 1..n
    {
        let prev = error[i - 1] - mean;
        let cur = error[i] - mean;
        if prev == 0.0
        {
            continue;
        }
        if (prev < 0.0) != (cur < 0.0) && cur != 0.0
        {
            crossings.push(i as f64);
        }
    }
    if crossings.len() < min_crossings
    {
        return none;
    }
    let intervals: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
    let imean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if imean <= 0.0
    {
        return none;
    }
    let var = intervals.iter().map(|&x| (x - imean).powi(2)).sum::<f64>() / intervals.len() as f64;
    let cv = var.sqrt() / imean;
    let regularity = (1.0 - cv).clamp(0.0, 1.0);

    // Sustained-amplitude check: a decaying transient also has regular
    // crossings, so require the late-window RMS to be a real fraction of the
    // early-window RMS (the oscillation does not die out), and to stand above
    // a round-off floor relative to the RMS of the whole trace. The floor is
    // relative so the verdict does not depend on the units of `error`.
    //
    // Each end window is measured about its own mean, not the mean of the
    // whole trace: a flat window that sits away from the global mean (a
    // burst that has ended, or a loop settled at a steady-state error) would
    // otherwise show an RMS equal to that offset and pass as sustained.
    let qn = (n / 4).max(1);
    // Deviations are divided by the largest one before squaring, so the
    // squares neither underflow for tiny nor overflow for huge amplitudes;
    // only ratios of these RMS values are used below.
    let dev_max = error.iter().fold(0.0_f64, |m, &x| m.max((x - mean).abs()));
    let unit = if dev_max > 0.0 && dev_max.is_finite()
    {
        dev_max
    }
    else
    {
        1.0
    };
    let rms_about = |slice: &[f64], centre: f64| -> f64 {
        (slice
            .iter()
            .map(|&x| ((x - centre) / unit).powi(2))
            .sum::<f64>()
            / slice.len() as f64)
            .sqrt()
    };
    // RMS of a window about its own mean. The deviations are taken from the
    // window's first sample before averaging, so an exactly constant window
    // gives exactly zero instead of the round-off of its mean.
    let window_rms = |slice: &[f64]| -> f64 {
        let x0 = slice[0];
        let shifted_mean = slice.iter().map(|&x| x - x0).sum::<f64>() / slice.len() as f64;
        rms_about(slice, x0 + shifted_mean)
    };
    let total_rms = rms_about(error, mean);
    let first_rms = window_rms(&error[..qn]);
    let last_rms = window_rms(&error[n - qn..]);
    let sustained = last_rms > SUSTAINED_RMS_FLOOR * total_rms && last_rms >= 0.5 * first_rms;

    OscillationReport {
        oscillating: regularity >= regularity_thresh && sustained,
        period_samples: 2.0 * imean,
        regularity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn detects_a_regular_oscillation() {
        // Error oscillating with period 40 samples.
        let period = 40.0;
        let err: Vec<f64> = (0..800)
            .map(|i| (2.0 * PI * i as f64 / period).sin())
            .collect();
        let r = detect_oscillation(&err, 6, 0.8);
        assert!(r.oscillating, "regularity {}", r.regularity);
        assert!(
            (r.period_samples - period).abs() < 3.0,
            "period {}",
            r.period_samples
        );
    }

    #[test]
    fn ignores_a_settled_loop() {
        // Decaying transient then flat: not a sustained oscillation.
        let err: Vec<f64> = (0..800)
            .map(|i| (-0.05 * i as f64).exp() * (2.0 * PI * i as f64 / 40.0).sin())
            .collect();
        let r = detect_oscillation(&err, 8, 0.8);
        assert!(!r.oscillating, "should not flag a settled loop");
    }

    #[test]
    fn small_amplitude_oscillation_is_detected() {
        // Regression: the sustained-amplitude check used an absolute
        // `last_rms > 1e-9` floor, so a loop oscillating with a 1 nm
        // amplitude in SI units (RMS ~7.1e-10 m) was reported as not
        // oscillating while the same trace in units of 1 m was flagged.
        let period = 40.0;
        let unit: Vec<f64> = (0..800)
            .map(|i| (2.0 * PI * i as f64 / period).sin())
            .collect();
        let reference = detect_oscillation(&unit, 6, 0.8);
        assert!(reference.oscillating);
        for scale in [1e-9, 1e-12, 1e-200, 1e10, 1e150]
        {
            let err: Vec<f64> = unit.iter().map(|&e| e * scale).collect();
            let r = detect_oscillation(&err, 6, 0.8);
            assert!(r.oscillating, "scale {scale}: {r:?}");
            assert_eq!(r.period_samples, reference.period_samples, "scale {scale}");
            assert!(
                (r.regularity - reference.regularity).abs() < 1e-12,
                "scale {scale}: {} vs {}",
                r.regularity,
                reference.regularity
            );
        }
    }

    #[test]
    fn oscillation_confined_to_the_middle_is_not_sustained_at_any_scale() {
        // A burst that is over before the last quarter of the trace must not
        // count as sustained, whatever the units of the error signal.
        let unit: Vec<f64> = (0..800)
            .map(|i| {
                if (300..500).contains(&i)
                {
                    (2.0 * PI * i as f64 / 40.0).sin()
                }
                else
                {
                    0.0
                }
            })
            .collect();
        for scale in [1.0, 1e-9, 1e-12, 1e-200, 1e10, 1e150]
        {
            let err: Vec<f64> = unit.iter().map(|&e| e * scale).collect();
            let r = detect_oscillation(&err, 6, 0.8);
            assert!(!r.oscillating, "scale {scale}: {r:?}");
        }
    }

    #[test]
    fn burst_that_ends_away_from_the_trace_mean_is_not_sustained() {
        // Regression: the end-window RMS was taken about the mean of the
        // whole trace. A burst of 5.5 periods (samples 300..520) shifts that
        // mean to about -0.0159, so the flat, exactly-zero first and last
        // quarters both had an RMS of about 0.0159 (ratio 1, above the 1e-9
        // floor) and the trace was reported as a sustained oscillation
        // (regularity 0.95) although nothing moves after sample 520.
        let burst: Vec<f64> = (0..800)
            .map(|i| {
                if (300..520).contains(&i)
                {
                    (2.0 * PI * i as f64 / 40.0).sin()
                }
                else
                {
                    0.0
                }
            })
            .collect();
        for offset in [0.0, 3.0, -1e3]
        {
            let err: Vec<f64> = burst.iter().map(|&e| e + offset).collect();
            let r = detect_oscillation(&err, 6, 0.8);
            assert!(!r.oscillating, "offset {offset}: {r:?}");
        }

        // Same for a loop that oscillates and then settles at a steady-state
        // error of 1.0 (the last 320 samples are constant).
        let settled: Vec<f64> = (0..800)
            .map(|i| {
                if i < 480
                {
                    (2.0 * PI * i as f64 / 40.0).sin()
                }
                else
                {
                    1.0
                }
            })
            .collect();
        let r = detect_oscillation(&settled, 6, 0.6);
        assert!(!r.oscillating, "settled at an offset: {r:?}");

        // An oscillation that is still running at the end of the trace, on
        // top of a level shift, is still sustained.
        let shifted: Vec<f64> = (0..800)
            .map(|i| {
                let level = if i < 400 { 0.0 } else { 0.5 };
                level + (2.0 * PI * i as f64 / 40.0).sin()
            })
            .collect();
        let r = detect_oscillation(&shifted, 6, 0.6);
        assert!(r.oscillating, "level shift: {r:?}");
    }
}
