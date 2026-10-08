//! Voltage-event detection (IEC 61000-4-30): sags, swells, interruptions.
//!
//! A one-cycle sliding RMS, compared to the declared nominal, classifies each
//! window as normal, sag (dip), swell, or interruption — the core power-quality
//! events utilities must log. Deterministic.

use serde::{Deserialize, Serialize};

/// Voltage-event class for one measurement window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoltageEvent {
    /// RMS within `[0.9, 1.1]·nominal`.
    Normal,
    /// Dip: `[0.1, 0.9)·nominal`.
    Sag,
    /// `> 1.1·nominal`.
    Swell,
    /// `< 0.1·nominal`.
    Interruption,
}

/// Classify an RMS voltage against the nominal.
pub fn classify_voltage(rms: f64, nominal: f64) -> VoltageEvent {
    if nominal <= 0.0
    {
        return VoltageEvent::Normal;
    }
    let r = rms / nominal;
    if r < 0.1
    {
        VoltageEvent::Interruption
    }
    else if r < 0.9
    {
        VoltageEvent::Sag
    }
    else if r > 1.1
    {
        VoltageEvent::Swell
    }
    else
    {
        VoltageEvent::Normal
    }
}

/// One-cycle sliding RMS of `signal`, `samples_per_cycle` wide.
///
/// Returns one value per window start, `signal.len() - w + 1` values in total,
/// where `w = samples_per_cycle.max(1)`; an input shorter than one window gives
/// an empty vector. Runs in `O(n)` time with `O(n)` auxiliary storage.
///
/// Each window's sum of squares is assembled from at most two partial sums of
/// non-negative terms (the tail of one `w`-aligned block plus the head of the
/// next), never by adding the incoming sample and subtracting the outgoing one.
/// The result is therefore never negative, an all-zero window gives exactly
/// `0.0`, and a non-finite sample only affects the windows that contain it.
///
/// # Examples
///
/// A supply interruption after a normal cycle reads as zero volts, not `NaN`:
///
/// ```
/// use scirust_grid::cycle_rms;
/// let spc = 64;
/// let mut wave: Vec<f64> = (0..spc)
///     .map(|i| 325.27 * (2.0 * std::f64::consts::PI * i as f64 / spc as f64).sin())
///     .collect();
/// wave.extend(std::iter::repeat(0.0).take(2 * spc));
/// let rms = cycle_rms(&wave, spc);
/// assert_eq!(rms.len(), 2 * spc + 1);
/// assert!((rms[0] - 325.27 / 2.0_f64.sqrt()).abs() < 1e-9);
/// assert_eq!(rms[spc], 0.0);
/// assert_eq!(*rms.last().unwrap(), 0.0);
/// ```
pub fn cycle_rms(signal: &[f64], samples_per_cycle: usize) -> Vec<f64> {
    let w = samples_per_cycle.max(1);
    let n = signal.len();
    if n < w
    {
        return Vec::new();
    }
    // head[i]: sum of squares from the start of i's w-aligned block up to i.
    // tail[i]: sum of squares from i to the end of its block (or of the signal).
    let mut head = vec![0.0_f64; n];
    let mut tail = vec![0.0_f64; n];
    for block_start in (0..n).step_by(w)
    {
        let block_end = (block_start + w).min(n);
        let mut acc = 0.0;
        for i in block_start..block_end
        {
            acc += signal[i] * signal[i];
            head[i] = acc;
        }
        acc = 0.0;
        for i in (block_start..block_end).rev()
        {
            acc += signal[i] * signal[i];
            tail[i] = acc;
        }
    }
    (0..=n - w)
        .map(|start| {
            // A window starting mid-block is the tail of that block plus the
            // head of the next one; an aligned window is one whole block.
            let sq_sum = if start % w == 0
            {
                tail[start]
            }
            else
            {
                tail[start] + head[start + w - 1]
            };
            (sq_sum / w as f64).sqrt()
        })
        .collect()
}

/// A detected voltage event over a contiguous span of RMS windows.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EventSpan {
    pub event: VoltageEvent,
    /// First RMS-window index of the event.
    pub start: usize,
    /// One past the last RMS-window index.
    pub end: usize,
    /// Extreme RMS-to-nominal ratio reached during the event (depth/peak).
    pub extreme_ratio: f64,
}

/// Detect contiguous non-normal voltage events from a waveform.
pub fn detect_events(signal: &[f64], nominal: f64, samples_per_cycle: usize) -> Vec<EventSpan> {
    let rms = cycle_rms(signal, samples_per_cycle);
    let mut events = Vec::new();
    let mut i = 0;
    while i < rms.len()
    {
        let cls = classify_voltage(rms[i], nominal);
        if cls == VoltageEvent::Normal
        {
            i += 1;
            continue;
        }
        let start = i;
        let mut extreme = rms[i] / nominal;
        while i < rms.len() && classify_voltage(rms[i], nominal) == cls
        {
            let ratio = rms[i] / nominal;
            // Track the most extreme deviation from 1.0.
            if (ratio - 1.0).abs() > (extreme - 1.0).abs()
            {
                extreme = ratio;
            }
            i += 1;
        }
        events.push(EventSpan {
            event: cls,
            start,
            end: i,
            extreme_ratio: extreme,
        });
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn make_wave(n: usize, spc: usize, amp_at: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp_at(i) * (2.0 * PI * i as f64 / spc as f64).sin())
            .collect()
    }

    #[test]
    fn classifies_levels() {
        assert_eq!(classify_voltage(1.0, 1.0), VoltageEvent::Normal);
        assert_eq!(classify_voltage(0.5, 1.0), VoltageEvent::Sag);
        assert_eq!(classify_voltage(1.3, 1.0), VoltageEvent::Swell);
        assert_eq!(classify_voltage(0.02, 1.0), VoltageEvent::Interruption);
    }

    #[test]
    fn detects_a_sag_event() {
        let spc = 64; // samples per cycle
        let n = spc * 40;
        // Nominal peak 1.0 -> RMS ~0.707. Drop amplitude to 0.5 for cycles 10..20.
        let nominal_rms = 1.0 / 2.0_f64.sqrt();
        let wave = make_wave(n, spc, |i| {
            let cycle = i / spc;
            if (10..20).contains(&cycle) { 0.5 } else { 1.0 }
        });
        let events = detect_events(&wave, nominal_rms, spc);
        let sags: Vec<_> = events
            .iter()
            .filter(|e| e.event == VoltageEvent::Sag)
            .collect();
        assert_eq!(sags.len(), 1, "events {events:?}");
        assert!(
            (sags[0].extreme_ratio - 0.5).abs() < 0.05,
            "depth {}",
            sags[0].extreme_ratio
        );
    }

    #[test]
    fn clean_supply_has_no_events() {
        let spc = 64;
        let wave = make_wave(spc * 20, spc, |_| 1.0);
        let nominal_rms = 1.0 / 2.0_f64.sqrt();
        assert!(detect_events(&wave, nominal_rms, spc).is_empty());
    }

    /// Direct per-window reference: sum of squares of exactly that window.
    fn direct_rms(signal: &[f64], w: usize) -> Vec<f64> {
        signal
            .windows(w)
            .map(|win| (win.iter().map(|x| x * x).sum::<f64>() / w as f64).sqrt())
            .collect()
    }

    #[test]
    fn interruption_after_normal_supply_reads_zero_not_nan() {
        // Regression: the running sum (add incoming square, subtract outgoing
        // square) did not return to exactly zero once the window held only the
        // interruption's zero samples. It settled slightly below zero, so every
        // later window was sqrt(negative) = NaN, which classify_voltage maps to
        // Normal: a 20-cycle loss of supply produced no event at all.
        for spc in [64_usize, 128, 256]
        {
            let n = spc * 40;
            let wave = make_wave(n, spc, |i| if i < spc * 20 { 325.27 } else { 0.0 });
            let rms = cycle_rms(&wave, spc);
            assert!(rms.iter().all(|r| r.is_finite()), "spc {spc}: NaN in RMS");
            for (k, r) in rms.iter().enumerate().skip(spc * 20)
            {
                assert_eq!(*r, 0.0, "spc {spc}: window {k} of an all-zero span");
            }
            let nominal = 230.0;
            let events = detect_events(&wave, nominal, spc);
            let interruptions: Vec<_> = events
                .iter()
                .filter(|e| e.event == VoltageEvent::Interruption)
                .collect();
            assert_eq!(interruptions.len(), 1, "spc {spc}: events {events:?}");
            assert_eq!(interruptions[0].end, rms.len(), "spc {spc}");
            assert_eq!(interruptions[0].extreme_ratio, 0.0, "spc {spc}");
        }
    }

    #[test]
    fn sliding_rms_matches_direct_window_sums() {
        // Includes a 6 kV surge on a 230 V supply, a sag and an interruption,
        // and window widths that do and do not divide the signal length.
        let spc = 100;
        let mut wave = make_wave(spc * 30 + 37, spc, |i| match i / spc
        {
            5..=9 => 0.4 * 325.27,
            15..=19 => 0.0,
            _ => 325.27,
        });
        wave[spc * 12 + 3] = 6000.0;
        for w in [1_usize, 7, spc, spc + 13]
        {
            let fast = cycle_rms(&wave, w);
            let reference = direct_rms(&wave, w);
            assert_eq!(fast.len(), reference.len());
            for (k, (a, b)) in fast.iter().zip(&reference).enumerate()
            {
                assert!(*a >= 0.0, "w {w}: window {k} negative or NaN: {a}");
                assert!(
                    (a - b).abs() <= 1e-12 * b.max(1.0),
                    "w {w}: window {k}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn non_finite_sample_only_affects_windows_that_contain_it() {
        let spc = 32;
        let mut wave = make_wave(spc * 6, spc, |_| 1.0);
        let bad = spc * 2 + 5;
        wave[bad] = f64::NAN;
        let rms = cycle_rms(&wave, spc);
        for (k, r) in rms.iter().enumerate()
        {
            let contains_bad = k <= bad && bad < k + spc;
            assert_eq!(r.is_nan(), contains_bad, "window {k}: {r}");
        }
    }

    #[test]
    fn short_or_degenerate_inputs() {
        assert!(cycle_rms(&[1.0, 2.0], 3).is_empty());
        assert!(cycle_rms(&[], 4).is_empty());
        // samples_per_cycle = 0 is treated as 1: each sample's magnitude.
        assert_eq!(cycle_rms(&[-3.0, 4.0], 0), vec![3.0, 4.0]);
        assert_eq!(cycle_rms(&[3.0, 4.0], 2), vec![(12.5_f64).sqrt()]);
    }
}
