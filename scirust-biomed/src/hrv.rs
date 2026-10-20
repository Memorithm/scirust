//! Heart-Rate Variability (HRV) time-domain metrics.
//!
//! From the RR-interval series, the standard clinical descriptors of autonomic
//! tone: **SDNN** (overall variability), **RMSSD** and **pNN50** (short-term /
//! parasympathetic), plus the mean heart rate. Durations are reported in
//! milliseconds, the clinical convention.

use serde::{Deserialize, Serialize};

/// HRV time-domain metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HrvMetrics {
    /// Mean heart rate (bpm).
    pub mean_hr_bpm: f64,
    /// Standard deviation of NN intervals (ms).
    pub sdnn_ms: f64,
    /// Root mean square of successive differences (ms).
    pub rmssd_ms: f64,
    /// Percentage of successive NN differences greater than 50 ms.
    pub pnn50: f64,
}

/// pNN50 threshold on successive RR differences (seconds).
const PNN50_THRESHOLD_S: f64 = 0.05;

/// Differences within this distance of [`PNN50_THRESHOLD_S`] are treated as
/// equal to it (seconds). 1 ns is far above the rounding error of a difference
/// of two RR intervals in seconds (~1e-16 s) and far below any RR resolution
/// (0.1 ms at 10 kHz).
const PNN50_TIE_TOLERANCE_S: f64 = 1e-9;

/// Compute HRV time-domain metrics from RR intervals (seconds).
///
/// `pnn50` is the percentage of successive differences strictly greater than
/// 50 ms; a difference equal to 50 ms (to within 1 ns) is not counted.
pub fn compute_hrv(rr_seconds: &[f64]) -> HrvMetrics {
    let n = rr_seconds.len();
    if n == 0
    {
        return HrvMetrics {
            mean_hr_bpm: 0.0,
            sdnn_ms: 0.0,
            rmssd_ms: 0.0,
            pnn50: 0.0,
        };
    }
    let mean = rr_seconds.iter().sum::<f64>() / n as f64;
    let mean_hr_bpm = if mean > 0.0 { 60.0 / mean } else { 0.0 };
    // Bessel's correction: SDNN is conventionally the *sample* standard
    // deviation (dividing by n-1, degrees of freedom), matching clinical HRV
    // toolkits (Kubios, NeuroKit2, pyHRV). Dividing by n underestimates
    // SDNN, most severely for short recordings.
    let var =
        rr_seconds.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let sdnn_ms = var.sqrt() * 1000.0;

    let diffs: Vec<f64> = rr_seconds.windows(2).map(|w| w[1] - w[0]).collect();
    let (rmssd_ms, pnn50) = if diffs.is_empty()
    {
        (0.0, 0.0)
    }
    else
    {
        let ms = (diffs.iter().map(|&d| d * d).sum::<f64>() / diffs.len() as f64).sqrt() * 1000.0;
        // Strictly greater than 50 ms (Task Force 1996). A difference within
        // `PNN50_TIE_TOLERANCE_S` of 50 ms is a tie: RR intervals built from
        // sample indices (e.g. at 200, 360, 500 or 1000 Hz) can differ by
        // exactly 50 ms, and in seconds that difference lands a few ulps either
        // side of 0.05, which must not decide whether it counts.
        let over = diffs
            .iter()
            .filter(|&&d| d.abs() > PNN50_THRESHOLD_S + PNN50_TIE_TOLERANCE_S)
            .count();
        (ms, 100.0 * over as f64 / diffs.len() as f64)
    };

    HrvMetrics {
        mean_hr_bpm,
        sdnn_ms,
        rmssd_ms,
        pnn50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfectly_regular_rhythm_has_zero_variability() {
        let m = compute_hrv(&[0.8; 20]);
        assert!((m.mean_hr_bpm - 75.0).abs() < 1e-9);
        assert!(m.sdnn_ms < 1e-9 && m.rmssd_ms < 1e-9 && m.pnn50 < 1e-9);
    }

    #[test]
    fn alternating_rr_matches_hand_computation() {
        // 0.8 / 0.9 s alternating, n=20: deviations are +/-0.05s each.
        // Sample variance (Bessel's correction, /(n-1)): (20*0.05^2)/19 =
        // 0.05/19 s^2 -> SDNN = 1000*0.05*sqrt(20/19) ms. Successive diffs
        // +/-100 ms.
        let rr: Vec<f64> = (0..20)
            .map(|k| if k % 2 == 0 { 0.8 } else { 0.9 })
            .collect();
        let m = compute_hrv(&rr);
        let expected_sdnn = 1000.0 * 0.05 * (20.0_f64 / 19.0).sqrt();
        assert!(
            (m.sdnn_ms - expected_sdnn).abs() < 1e-6,
            "SDNN {}",
            m.sdnn_ms
        );
        assert!((m.rmssd_ms - 100.0).abs() < 1e-6, "RMSSD {}", m.rmssd_ms);
        assert!((m.pnn50 - 100.0).abs() < 1e-6, "pNN50 {}", m.pnn50);
    }

    #[test]
    fn sdnn_uses_bessel_corrected_sample_variance() {
        // n=2 maximizes the relative gap between the population (/n) and
        // sample (/(n-1)) variance estimators. mean=0.9s, deviations =
        // [-0.1, 0.1]s, sum of squared deviations = 0.02 s^2. Sample
        // variance = 0.02/(2-1) = 0.02 s^2, SDNN = sqrt(0.02)*1000 =
        // 100*sqrt(2) ms -- not the population-variance value of 100 ms.
        let m = compute_hrv(&[0.8, 1.0]);
        assert!(
            (m.sdnn_ms - 100.0 * 2.0_f64.sqrt()).abs() < 1e-9,
            "SDNN {}",
            m.sdnn_ms
        );
    }

    #[test]
    fn small_variations_do_not_trigger_pnn50() {
        // ±20 ms successive diffs are below the 50 ms threshold.
        let rr: Vec<f64> = (0..20)
            .map(|k| if k % 2 == 0 { 0.80 } else { 0.82 })
            .collect();
        let m = compute_hrv(&rr);
        assert!(m.pnn50 < 1e-9, "pNN50 {}", m.pnn50);
        assert!(m.rmssd_ms > 0.0);
    }

    #[test]
    fn sdnn_and_rmssd_are_distinct_formulas() {
        // RR = [0.8, 0.8, 0.8, 1.2] s. Hand computation:
        //   mean = 0.9 s -> mean HR = 60/0.9 = 66.6667 bpm.
        //   deviations = [-0.1, -0.1, -0.1, 0.3]; sample var (Bessel's
        //   correction, /(n-1)) = 0.12/3 = 0.04; SDNN = sqrt(0.04)*1000 =
        //   200.0 ms.
        //   successive diffs = [0, 0, 0.4]; RMSSD = sqrt(0.16/3)*1000 = 230.9401 ms.
        //   pNN50: one of three diffs (0.4 s) exceeds 50 ms -> 33.3333 %.
        // RMSSD > SDNN here, so a test that conflated the two would fail.
        let m = compute_hrv(&[0.8, 0.8, 0.8, 1.2]);
        assert!(
            (m.mean_hr_bpm - 66.666_666_666_7).abs() < 1e-6,
            "HR {}",
            m.mean_hr_bpm
        );
        assert!((m.sdnn_ms - 200.0).abs() < 1e-6, "SDNN {}", m.sdnn_ms);
        assert!(
            (m.rmssd_ms - 230.940_107_676).abs() < 1e-6,
            "RMSSD {}",
            m.rmssd_ms
        );
        assert!((m.pnn50 - 100.0 / 3.0).abs() < 1e-9, "pNN50 {}", m.pnn50);
        assert!(
            m.rmssd_ms > m.sdnn_ms,
            "RMSSD should exceed SDNN for this series"
        );
    }

    #[test]
    fn successive_differences_of_exactly_50_ms_are_not_counted() {
        // pNN50 counts differences strictly greater than 50 ms. RR intervals
        // derived from sample indices at 1000 Hz (or 200, 360, 500 Hz) can
        // differ by exactly 50 ms; in seconds, `b - a` then lands a few ulps
        // either side of 0.05 (0.9 - 0.85 = 0.050000000000000044,
        // 0.85 - 0.8 = 0.04999999999999993), so such a tie must not be
        // classified by rounding noise.
        let rr = [0.80, 0.85, 0.90, 0.95, 1.00, 0.95, 0.90, 0.85, 0.80];
        let m = compute_hrv(&rr);
        assert!(m.pnn50.abs() < 1e-12, "pNN50 {}", m.pnn50);
        assert!((m.rmssd_ms - 50.0).abs() < 1e-9, "RMSSD {}", m.rmssd_ms);

        // Same through the R-peak pipeline: peaks at 1000 Hz with successive
        // RR of 800, 850, 900, 950, 1000 ms (differences of exactly 50 ms)
        // followed by a 1051 ms beat (a 51 ms difference, which counts).
        let peaks = [0usize, 800, 1650, 2550, 3500, 4500, 5551];
        let rr = crate::ecg::rr_intervals(&peaks, 1000.0);
        let m = compute_hrv(&rr);
        assert!((m.pnn50 - 20.0).abs() < 1e-9, "pNN50 {}", m.pnn50);
    }

    #[test]
    fn differences_just_above_50_ms_still_count() {
        // 50.001 ms (1 µs above the threshold) is a genuine exceedance.
        let m = compute_hrv(&[0.8, 0.850_001]);
        assert!((m.pnn50 - 100.0).abs() < 1e-12, "pNN50 {}", m.pnn50);
        let m = compute_hrv(&[0.850_001, 0.8]);
        assert!((m.pnn50 - 100.0).abs() < 1e-12, "pNN50 {}", m.pnn50);
    }

    #[test]
    fn empty_and_single_interval_edge_cases() {
        // No intervals -> all metrics zero (no panic / NaN).
        let z = compute_hrv(&[]);
        assert_eq!(
            (z.mean_hr_bpm, z.sdnn_ms, z.rmssd_ms, z.pnn50),
            (0.0, 0.0, 0.0, 0.0)
        );
        // A single RR has a defined mean HR and zero SDNN, but no successive
        // differences, so RMSSD and pNN50 are zero rather than NaN.
        let one = compute_hrv(&[0.75]);
        assert!(
            (one.mean_hr_bpm - 80.0).abs() < 1e-12,
            "HR {}",
            one.mean_hr_bpm
        );
        assert_eq!((one.sdnn_ms, one.rmssd_ms, one.pnn50), (0.0, 0.0, 0.0));
    }
}
