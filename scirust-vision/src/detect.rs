//! Small-target CFAR detection for EO/IR imagery.
//!
//! The radar chain thresholds a range-Doppler map with a constant-false-alarm-
//! rate (CFAR) detector; the same idea applies to a thermal/EO focal plane, where
//! the task is to pull small hot targets (an aircraft, a missile plume, a distant
//! vehicle) out of a spatially varying background. This module is the image-domain
//! analogue of the radar `ca_cfar_2d`: for each pixel it estimates the local
//! background from a **ring of training cells** around a **guard band** (so a
//! bright target does not corrupt its own background estimate) and flags the pixel
//! when it exceeds the local mean by `k` local standard deviations. Because the
//! threshold rides the *local* statistics, a target is found on a dim sky and on a
//! bright pedestal alike, at a false-alarm rate set by `k` rather than by an
//! absolute level.
//!
//! Thresholded pixels are grouped into targets by connected-component labelling
//! ([`crate::connected_components`]) and reduced to intensity-weighted centroids —
//! sub-pixel `(x, y)` locations ready to feed the tracking chain. Dependency-free.

use crate::{Image, connected_components};

/// A detected small target: an intensity-weighted centroid, the peak amplitude
/// over its pixels, and the number of thresholded pixels forming it.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetDetection {
    /// Sub-pixel centroid column (intensity-weighted).
    pub x: f64,
    /// Sub-pixel centroid row (intensity-weighted).
    pub y: f64,
    /// Peak pixel value over the target's pixels.
    pub amplitude: f64,
    /// Number of thresholded pixels in the target.
    pub pixels: usize,
}

/// The CFAR detection **mask**: `1.0` where a pixel exceeds its local background
/// mean by `k` local standard deviations, `0.0` elsewhere.
///
/// The background statistics for a pixel are gathered from the square ring of
/// side `2·(guard + train) + 1` centred on it, excluding the inner guard square
/// of side `2·guard + 1`. Cells outside the image are skipped; a pixel with
/// fewer than four training cells (a corner with a large window) is left
/// undetected.
pub fn cfar_mask(image: &Image, guard: usize, train: usize, k: f64) -> Image {
    let (w, h) = (image.width, image.height);
    let mut mask = Image::new(w, h);
    let outer = (guard + train) as isize;
    let guard = guard as isize;
    // Training-cell values of the pixel under test, reused across pixels.
    let mut ring: Vec<f64> = Vec::new();
    for y in 0..h as isize
    {
        for x in 0..w as isize
        {
            ring.clear();
            for dy in -outer..=outer
            {
                for dx in -outer..=outer
                {
                    // Skip the guard square (which contains the cell under test).
                    if dx.abs() <= guard && dy.abs() <= guard
                    {
                        continue;
                    }
                    let (ix, iy) = (x + dx, y + dy);
                    if ix >= 0 && iy >= 0 && (ix as usize) < w && (iy as usize) < h
                    {
                        ring.push(image.get(ix as usize, iy as usize));
                    }
                }
            }
            if ring.len() < 4
            {
                continue;
            }
            // Local statistics on data shifted by one training cell, with a
            // two-pass (corrected) variance. The one-pass `E[v²] − mean²` form
            // cancels catastrophically on a bright pedestal: at a level of 1e9
            // with unit-scale noise, `mean²` is ~1e18 and its rounding error
            // (~1e2) swamps the true variance, so the threshold collapsed to the
            // mean (false alarms everywhere) or blew up (targets missed).
            // Shifting makes the arithmetic depend only on differences between
            // cells, so adding a constant pedestal leaves the mask unchanged.
            let shift = ring[0];
            let n = ring.len() as f64;
            let mean_d = ring.iter().map(|&v| v - shift).sum::<f64>() / n;
            let (mut ss, mut s) = (0.0, 0.0);
            for &v in &ring
            {
                let e = (v - shift) - mean_d;
                ss += e * e;
                s += e;
            }
            let std = ((ss - s * s / n) / n).max(0.0).sqrt();
            if image.get(x as usize, y as usize) - shift > mean_d + k * std
            {
                mask.set(x as usize, y as usize, 1.0);
            }
        }
    }
    mask
}

/// Detect small targets in `image`: run [`cfar_mask`], group the thresholded
/// pixels into connected components, and reduce each to an intensity-weighted
/// centroid [`TargetDetection`]. `guard`/`train` size the CFAR window and `k` is
/// the threshold in local standard deviations.
pub fn detect_targets(image: &Image, guard: usize, train: usize, k: f64) -> Vec<TargetDetection> {
    let mask = cfar_mask(image, guard, train, k);
    connected_components(&mask)
        .iter()
        .map(|pixels| {
            let (mut sw, mut sx, mut sy, mut peak) = (0.0, 0.0, 0.0, f64::NEG_INFINITY);
            for &(px, py) in pixels
            {
                let v = image.get(px, py);
                let weight = v.max(0.0);
                sw += weight;
                sx += weight * px as f64;
                sy += weight * py as f64;
                if v > peak
                {
                    peak = v;
                }
            }
            let (x, y) = if sw > 0.0
            {
                (sx / sw, sy / sw)
            }
            else
            {
                // Degenerate (all-zero weights): fall back to the geometric mean.
                let n = pixels.len() as f64;
                (
                    pixels.iter().map(|p| p.0 as f64).sum::<f64>() / n,
                    pixels.iter().map(|p| p.1 as f64).sum::<f64>() / n,
                )
            };
            TargetDetection {
                x,
                y,
                amplitude: peak,
                pixels: pixels.len(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic LCG for reproducible background noise.
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn flat(w: usize, h: usize, level: f64) -> Image {
        Image::from_vec(w, h, vec![level; w * h])
    }

    #[test]
    fn detects_a_point_target_on_a_flat_background() {
        let mut img = flat(32, 32, 1.0);
        img.set(20, 12, 20.0);
        let dets = detect_targets(&img, 1, 3, 5.0);
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert!((dets[0].x - 20.0).abs() < 0.5 && (dets[0].y - 12.0).abs() < 0.5);
        assert_eq!(dets[0].amplitude, 20.0);
    }

    #[test]
    fn no_detections_on_a_uniform_background() {
        // A constant image has zero local variance, so nothing exceeds the mean.
        let img = flat(24, 24, 7.0);
        assert!(detect_targets(&img, 1, 3, 4.0).is_empty());
    }

    #[test]
    fn detects_a_target_on_a_bright_pedestal() {
        // CFAR rides the local level: a target on a bright uniform pedestal is
        // still found, and the pedestal itself raises no detections.
        let mut img = flat(32, 32, 500.0);
        img.set(16, 16, 560.0);
        let dets = detect_targets(&img, 1, 4, 5.0);
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert!((dets[0].x - 16.0).abs() < 0.5 && (dets[0].y - 16.0).abs() < 0.5);
    }

    #[test]
    fn weighted_centroid_is_subpixel() {
        // A two-pixel target with unequal intensities: the centroid is pulled
        // toward the brighter pixel, between the two integer coordinates.
        let mut img = flat(24, 24, 1.0);
        img.set(10, 10, 30.0);
        img.set(11, 10, 10.0);
        // guard = 1 so the two adjacent target pixels stay out of each other's
        // background ring.
        let dets = detect_targets(&img, 1, 3, 4.0);
        assert_eq!(dets.len(), 1, "{dets:?}");
        // Weighted x = (30·10 + 10·11)/40 = 10.25.
        assert!((dets[0].x - 10.25).abs() < 0.05, "x = {}", dets[0].x);
        assert!((dets[0].y - 10.0).abs() < 1e-9);
        assert_eq!(dets[0].pixels, 2);
    }

    #[test]
    fn resolves_two_separated_targets() {
        let mut img = flat(40, 40, 2.0);
        img.set(8, 8, 25.0);
        img.set(30, 28, 25.0);
        let dets = detect_targets(&img, 1, 3, 5.0);
        assert_eq!(dets.len(), 2, "{dets:?}");
    }

    /// Uniform noise in `[0, 1)` quantized to multiples of 1/1024, so adding a
    /// pedestal up to 2^40 is exact and the tests isolate the detector's own
    /// arithmetic from input rounding.
    fn quantized_noise(w: usize, h: usize, seed: u64, pedestal: f64) -> Image {
        let mut rng = Lcg(seed);
        let data: Vec<f64> = (0..(w * h))
            .map(|_| pedestal + (rng.unit() * 1024.0).floor() / 1024.0)
            .collect();
        Image::from_vec(w, h, data)
    }

    #[test]
    fn detects_a_faint_target_on_a_very_bright_noisy_pedestal() {
        // Regression: the local variance used to be computed as E[v²] − mean²,
        // which at a 1e9 pedestal loses the unit-scale noise variance to
        // rounding (mean² ≈ 1e18 has an ulp of 128), so the threshold collapsed
        // to the mean or exploded. A +10 target (~35 noise σ) on noise with
        // |v − mean| ≤ 1.8σ must give exactly one detection at k = 6.
        let mut img = quantized_noise(48, 48, 0x0B12_6E7D, 1.0e9);
        img.set(24, 24, img.get(24, 24) + 10.0);
        let dets = detect_targets(&img, 1, 4, 6.0);
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert!((dets[0].x - 24.0).abs() < 0.5 && (dets[0].y - 24.0).abs() < 0.5);
        assert_eq!(dets[0].pixels, 1);
    }

    #[test]
    fn mask_is_unchanged_by_a_constant_pedestal() {
        // CFAR is meant to ride the local level: adding an exactly representable
        // constant to every pixel must not change which pixels are flagged.
        let low = quantized_noise(40, 40, 0x5EED_CFA8, 0.0);
        let mut flagged = 0;
        for &pedestal in &[1.0e6, 2.0_f64.powi(30), 1.0e12]
        {
            let high = quantized_noise(40, 40, 0x5EED_CFA8, pedestal);
            let (a, b) = (cfar_mask(&low, 1, 3, 1.5), cfar_mask(&high, 1, 3, 1.5));
            let changed = a.data.iter().zip(&b.data).filter(|(p, q)| p != q).count();
            assert_eq!(
                changed, 0,
                "{changed} pixels changed at pedestal {pedestal:e}"
            );
            flagged = a.data.iter().filter(|&&m| m > 0.0).count();
        }
        // The comparison is not vacuous: k = 1.5 flags a few percent of pixels.
        assert!(flagged > 10, "only {flagged} pixels flagged");
    }

    #[test]
    fn higher_threshold_reduces_false_alarms() {
        // Pure noise background: raising k must not increase the false-alarm count.
        let mut rng = Lcg(0x00C1_FA12);
        let data: Vec<f64> = (0..(48 * 48)).map(|_| rng.unit()).collect();
        let img = Image::from_vec(48, 48, data);
        let low = detect_targets(&img, 1, 4, 3.0).len();
        let high = detect_targets(&img, 1, 4, 6.0).len();
        assert!(
            high <= low,
            "false alarms should not grow with k: {low} -> {high}"
        );
    }
}
