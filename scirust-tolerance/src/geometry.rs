//! ISO 1101 geometric characteristics (form, orientation, location) and their
//! inertial form.
//!
//! [`crate::position`] covers positional location; this module covers the rest
//! of the computable geometric tolerances — the ones defined as the width of a
//! zone containing an extracted feature:
//!
//! - **Form** — [`straightness`] (a line), [`flatness`] (a surface),
//!   [`roundness`] (a circle), [`cylindricity`] (an axis) : the peak-to-valley
//!   range of the deviations from the least-squares reference feature.
//! - **Orientation** — [`parallelism`], [`perpendicularity`], [`angularity`] :
//!   the zone width `L·sin(Δθ)` a feature of length `L` sweeps when its axis
//!   departs from the datum by `Δθ`.
//! - **Location / composite** — [`profile`] (deviation from a nominal profile),
//!   [`total_runout`] (the full-indicator range about a datum axis).
//!
//! Each form characteristic also has an **inertial** reading (`*_inertia`): the
//! RMS of the same deviations, `√((1/n) Σ dⱼ²)` — the [`crate::form`] surface
//! inertia of that specific geometric feature, so a form defect can be
//! toleranced by inertia instead of by peak-to-valley range, which is far less
//! sensitive to a single outlier point.
//!
//! Reference fits are **least-squares** (the Gaussian / L2 associated feature of
//! ISO 5459 / ISO 12781). The strict minimum-zone (Chebyshev) value is never
//! larger; least-squares is the estimator most CMM software reports and the one
//! that ties cleanly to inertia.

/// Root-mean-square of a deviation set, `√((1/n) Σ dⱼ²)`; 0 for empty input.
fn rms(d: &[f64]) -> f64 {
    if d.is_empty()
    {
        return 0.0;
    }
    (d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt()
}

/// Peak-to-valley range `max − min` of a deviation set; 0 for empty input.
fn range(d: &[f64]) -> f64 {
    if d.is_empty()
    {
        return 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in d
    {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    hi - lo
}

/// Arithmetic mean of one coordinate of a point set (non-empty input).
fn mean_of<const D: usize>(points: &[[f64; D]], axis: usize) -> f64 {
    points.iter().map(|p| p[axis]).sum::<f64>() / points.len() as f64
}

/// `true` when every point has the same coordinate on `axis` (exact test).
fn all_equal<const D: usize>(points: &[[f64; D]], axis: usize) -> bool {
    points.iter().all(|p| p[axis] == points[0][axis])
}

/// Solve the 2×2 symmetric system `[[a, b], [b, c]]·[s, t] = [r1, r2]` of
/// centred second moments. `None` when the moment matrix is singular *relative
/// to its own diagonal* (`a·c − b² ≤ 1e-12·a·c`, i.e. a squared correlation of
/// the two coordinates above `1 − 1e-12`), so the test does not depend on the
/// units of the data.
fn solve_centred_2x2(a: f64, b: f64, c: f64, r1: f64, r2: f64) -> Option<(f64, f64)> {
    if a.is_nan() || c.is_nan() || a <= 0.0 || c <= 0.0
    {
        return None;
    }
    let det = a * c - b * b;
    if det.is_nan() || det <= 1e-12 * a * c
    {
        return None;
    }
    Some(((c * r1 - b * r2) / det, (a * r2 - b * r1) / det))
}

/// Least-squares line in centred form: `(x̄, ȳ, slope)`, the line being
/// `y − ȳ = slope·(x − x̄)`.
fn centred_line(points: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
    if points.len() < 2 || all_equal(points, 0)
    {
        return None;
    }
    let (mx, my) = (mean_of(points, 0), mean_of(points, 1));
    let (mut suu, mut suv) = (0.0, 0.0);
    for p in points
    {
        let (u, v) = (p[0] - mx, p[1] - my);
        suu += u * u;
        suv += u * v;
    }
    if suu.is_nan() || suu <= 0.0
    {
        return None;
    }
    Some((mx, my, suv / suu))
}

/// Least-squares line `y = a + b·x` through 2D points, or `None` if fewer than
/// two distinct abscissae. Returns `(a, b)`.
///
/// The fit is computed on coordinates centred on the centroid, so it is
/// translation-invariant and does not depend on the units of the data (raw
/// normal-equation sums `n·Σx² − (Σx)²` cancel catastrophically when the points
/// sit far from the origin, e.g. in CMM machine coordinates). The returned
/// intercept `a` is the value at `x = 0`; for points far from the origin,
/// evaluating residuals as `y − (a + b·x)` would cancel again, so
/// [`straightness`] evaluates them in centred form instead of through `(a, b)`.
pub fn least_squares_line(points: &[[f64; 2]]) -> Option<(f64, f64)> {
    let (mx, my, b) = centred_line(points)?;
    Some((my - b * mx, b))
}

fn line_residuals(points: &[[f64; 2]]) -> Vec<f64> {
    match centred_line(points)
    {
        Some((mx, my, b)) => points
            .iter()
            .map(|p| (p[1] - my) - b * (p[0] - mx))
            .collect(),
        None => Vec::new(),
    }
}

/// Straightness: the peak-to-valley range of the deviations of a nominally
/// straight profile from its least-squares line. 0 if unfittable.
pub fn straightness(points: &[[f64; 2]]) -> f64 {
    range(&line_residuals(points))
}

/// Inertial straightness: the RMS deviation from the least-squares line.
pub fn straightness_inertia(points: &[[f64; 2]]) -> f64 {
    rms(&line_residuals(points))
}

/// Least-squares plane in centred form: `(x̄, ȳ, z̄, b, c)`, the plane being
/// `z − z̄ = b·(x − x̄) + c·(y − ȳ)`.
fn centred_plane(points: &[[f64; 3]]) -> Option<(f64, f64, f64, f64, f64)> {
    if points.len() < 3
    {
        return None;
    }
    let (mx, my, mz) = (mean_of(points, 0), mean_of(points, 1), mean_of(points, 2));
    let (mut suu, mut svv, mut suv, mut suw, mut svw) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points
    {
        let (u, v, w) = (p[0] - mx, p[1] - my, p[2] - mz);
        suu += u * u;
        svv += v * v;
        suv += u * v;
        suw += u * w;
        svw += v * w;
    }
    let (b, c) = solve_centred_2x2(suu, suv, svv, suw, svw)?;
    Some((mx, my, mz, b, c))
}

/// Least-squares plane `z = a + b·x + c·y` through 3D points, or `None` if the
/// points are collinear / too few. Returns `(a, b, c)`.
///
/// Computed on centred coordinates, with a collinearity test relative to the
/// spread of the points, so the fit is translation-invariant and independent
/// of the units of the data. As for [`least_squares_line`], `a` is the value at
/// the origin; [`flatness`] evaluates its residuals in centred form.
pub fn least_squares_plane(points: &[[f64; 3]]) -> Option<(f64, f64, f64)> {
    let (mx, my, mz, b, c) = centred_plane(points)?;
    Some((mz - b * mx - c * my, b, c))
}

fn plane_residuals(points: &[[f64; 3]]) -> Vec<f64> {
    match centred_plane(points)
    {
        Some((mx, my, mz, b, c)) => points
            .iter()
            .map(|p| (p[2] - mz) - (b * (p[0] - mx) + c * (p[1] - my)))
            .collect(),
        None => Vec::new(),
    }
}

/// Flatness: peak-to-valley range of deviations from the least-squares plane.
pub fn flatness(points: &[[f64; 3]]) -> f64 {
    range(&plane_residuals(points))
}

/// Inertial flatness: RMS deviation from the least-squares plane.
pub fn flatness_inertia(points: &[[f64; 3]]) -> f64 {
    rms(&plane_residuals(points))
}

/// Least-squares circle (Kåsa algebraic fit) through 2D points, or `None` if
/// unfittable (fewer than three points, or all points on one line). Returns
/// `(center_x, center_y, radius)`.
///
/// The algebraic fit `u² + v² + D·u + E·v + F = 0` is solved on coordinates
/// `(u, v)` centred on the centroid, where the normal equations decouple
/// (`F = −mean(u² + v²)`) and the remaining 2×2 system has its singularity
/// test relative to the spread of the points. Fitting raw coordinates instead
/// makes the third-order sums cancel catastrophically for a small feature far
/// from the origin (for a 2 mm bore at (500, 300) mm a raw-moment fit reports a
/// roundness about 19 times the true form error).
pub fn least_squares_circle(points: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
    let (mx, my, du, dv, r) = centred_circle(points)?;
    Some((mx + du, my + dv, r))
}

/// Kåsa circle in centred form: `(x̄, ȳ, cu, cv, r)`, the center being
/// `(x̄ + cu, ȳ + cv)`.
fn centred_circle(points: &[[f64; 2]]) -> Option<(f64, f64, f64, f64, f64)> {
    let n = points.len();
    if n < 3
    {
        return None;
    }
    let (mx, my) = (mean_of(points, 0), mean_of(points, 1));
    let (mut suu, mut svv, mut suv) = (0.0, 0.0, 0.0);
    let (mut suz, mut svz, mut sz) = (0.0, 0.0, 0.0);
    for p in points
    {
        let (u, v) = (p[0] - mx, p[1] - my);
        let z = u * u + v * v;
        suu += u * u;
        svv += v * v;
        suv += u * v;
        suz += u * z;
        svz += v * z;
        sz += z;
    }
    // With Σu = Σv = 0 the normal equations of z + D·u + E·v + F = 0 give
    // F = −Σz/n and [[Σuu, Σuv], [Σuv, Σvv]]·[D, E] = −[Σuz, Σvz].
    let (d, e) = solve_centred_2x2(suu, suv, svv, -suz, -svz)?;
    let f = -sz / n as f64;
    let (cu, cv) = (-0.5 * d, -0.5 * e);
    let r2 = cu * cu + cv * cv - f;
    if !r2.is_finite() || r2 < 0.0
    {
        return None;
    }
    Some((mx, my, cu, cv, r2.sqrt()))
}

/// Radial deviations `|P − C| − r` from the least-squares circle, evaluated in
/// centred coordinates.
fn circle_residuals(points: &[[f64; 2]]) -> Vec<f64> {
    match centred_circle(points)
    {
        Some((mx, my, cu, cv, r)) => points
            .iter()
            .map(|p| ((p[0] - mx - cu).powi(2) + (p[1] - my - cv).powi(2)).sqrt() - r)
            .collect(),
        None => Vec::new(),
    }
}

/// Roundness / circularity: peak-to-valley range of the radial deviations from
/// the least-squares circle.
pub fn roundness(points: &[[f64; 2]]) -> f64 {
    range(&circle_residuals(points))
}

/// Inertial roundness: RMS radial deviation from the least-squares circle.
pub fn roundness_inertia(points: &[[f64; 2]]) -> f64 {
    rms(&circle_residuals(points))
}

fn cylinder_residuals_axis_z(points: &[[f64; 3]]) -> Vec<f64> {
    let proj: Vec<[f64; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
    circle_residuals(&proj)
}

/// Cylindricity for a cylinder whose axis is nominally along `z`: fits the
/// least-squares circle to the `(x, y)` projection of every point and returns
/// the peak-to-valley range of the radial deviations over all points (so it
/// captures roundness *and* taper/waviness along the axis).
pub fn cylindricity(points: &[[f64; 3]]) -> f64 {
    range(&cylinder_residuals_axis_z(points))
}

/// Inertial cylindricity (axis along `z`): RMS radial deviation over all points.
pub fn cylindricity_inertia(points: &[[f64; 3]]) -> f64 {
    rms(&cylinder_residuals_axis_z(points))
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Angle in radians between two 3D vectors, in `[0, π]`. Returns 0 if either
/// vector is null.
pub fn angle_between(u: [f64; 3], v: [f64; 3]) -> f64 {
    let (nu, nv) = (norm3(u), norm3(v));
    if nu == 0.0 || nv == 0.0
    {
        return 0.0;
    }
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
    dot.clamp(-1.0, 1.0).acos()
}

/// Angularity zone: the width `L·sin(Δθ)` swept by a feature of length `length`
/// whose axis `measured_dir` departs by `Δθ` from the direction that makes
/// `nominal_angle` (radians) with the datum `datum_dir`
/// (`Δθ = |∠(measured, datum) − nominal_angle|`).
pub fn angularity(
    measured_dir: [f64; 3],
    datum_dir: [f64; 3],
    nominal_angle: f64,
    length: f64,
) -> f64 {
    let dtheta = (angle_between(measured_dir, datum_dir) - nominal_angle).abs();
    length.abs() * dtheta.sin()
}

/// Parallelism zone `L·sin(θ)` of a feature of length `length` whose axis makes
/// angle `θ` with the datum (0 when perfectly parallel). Special case of
/// [`angularity`] with a nominal angle of 0.
pub fn parallelism(measured_dir: [f64; 3], datum_dir: [f64; 3], length: f64) -> f64 {
    angularity(measured_dir, datum_dir, 0.0, length)
}

/// Perpendicularity zone `L·|cos θ|` of a feature of length `length` whose axis
/// makes angle `θ` with the datum (0 when perfectly perpendicular). Special case
/// of [`angularity`] with a nominal angle of `π/2`.
pub fn perpendicularity(measured_dir: [f64; 3], datum_dir: [f64; 3], length: f64) -> f64 {
    angularity(measured_dir, datum_dir, std::f64::consts::FRAC_PI_2, length)
}

/// Profile tolerance (equal-bilateral zone) from signed normal deviations `dⱼ`
/// of the extracted profile/surface from its nominal: `2·max|dⱼ|`, the width of
/// the symmetric zone that contains every point. 0 for empty input.
pub fn profile(deviations: &[f64]) -> f64 {
    2.0 * deviations.iter().fold(0.0, |m, d| f64::max(m, d.abs()))
}

/// Inertial profile: RMS of the signed normal deviations from the nominal
/// profile — the form inertia of the profile defect.
pub fn profile_inertia(deviations: &[f64]) -> f64 {
    rms(deviations)
}

/// Total runout: the full-indicator range `max − min` of a set of radial (or
/// axial) indicator readings taken about the datum axis. Unlike [`roundness`]
/// it is measured against the **datum**, so it also captures eccentricity.
pub fn total_runout(readings: &[f64]) -> f64 {
    range(readings)
}

/// Runout from measured points about a datum axis passing through `datum_center`
/// (in the plane of the points): the range of the point radii `|Pⱼ − datum|`.
pub fn runout_from_points(points: &[[f64; 2]], datum_center: [f64; 2]) -> f64 {
    let radii: Vec<f64> = points
        .iter()
        .map(|p| ((p[0] - datum_center[0]).powi(2) + (p[1] - datum_center[1]).powi(2)).sqrt())
        .collect();
    range(&radii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn perfect_features_have_zero_form_error() {
        // Collinear points ⇒ zero straightness.
        let line = [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]];
        assert_relative_eq!(straightness(&line), 0.0, epsilon = 1e-9);
        assert_relative_eq!(straightness_inertia(&line), 0.0, epsilon = 1e-9);
        // Coplanar points (z = 2 + x − y) ⇒ zero flatness.
        let plane = [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 3.0],
        ];
        assert_relative_eq!(flatness(&plane), 0.0, epsilon = 1e-9);
        // Points exactly on a circle ⇒ zero roundness.
        let circ: Vec<[f64; 2]> = (0..12)
            .map(|k| {
                let t = k as f64 / 12.0 * std::f64::consts::TAU;
                [2.0 + 0.5 * t.cos(), -1.0 + 0.5 * t.sin()]
            })
            .collect();
        assert_relative_eq!(roundness(&circ), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn straightness_captures_a_single_bump() {
        // Flat line y=0 with one point lifted by 0.1 ⇒ range of residuals ≈ the
        // bump (the LS line barely tilts for a symmetric layout).
        let pts = [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.1], [1.0, 0.0], [2.0, 0.0]];
        assert!(straightness(&pts) > 0.05);
        assert!(straightness_inertia(&pts) <= straightness(&pts));
    }

    #[test]
    fn circle_fit_recovers_center_and_radius() {
        let circ: Vec<[f64; 2]> = (0..20)
            .map(|k| {
                let t = k as f64 / 20.0 * std::f64::consts::TAU;
                [3.0 + 1.5 * t.cos(), 4.0 + 1.5 * t.sin()]
            })
            .collect();
        let (cx, cy, r) = least_squares_circle(&circ).unwrap();
        assert_relative_eq!(cx, 3.0, epsilon = 1e-9);
        assert_relative_eq!(cy, 4.0, epsilon = 1e-9);
        assert_relative_eq!(r, 1.5, epsilon = 1e-9);
    }

    #[test]
    fn orientation_zones_are_geometric() {
        let z = [0.0, 0.0, 1.0];
        let x = [1.0, 0.0, 0.0];
        // Parallel to itself ⇒ 0; perpendicular pair ⇒ 0 perpendicularity.
        assert_relative_eq!(parallelism(z, z, 10.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(perpendicularity(x, z, 10.0), 0.0, epsilon = 1e-12);
        // 30° tilt from datum z, length 10 ⇒ parallelism zone 10·sin30° = 5.
        let tilt = [0.5, 0.0, 3.0f64.sqrt() / 2.0]; // 30° from z
        assert_relative_eq!(parallelism(tilt, z, 10.0), 5.0, epsilon = 1e-9);
        // Same tilt ⇒ perpendicularity zone 10·|cos30°| = 8.660…
        assert_relative_eq!(
            perpendicularity(tilt, z, 10.0),
            10.0 * (3.0f64.sqrt() / 2.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn profile_and_runout_are_ranges() {
        let dev = [0.02, -0.03, 0.01, -0.015];
        assert_relative_eq!(profile(&dev), 0.06, epsilon = 1e-12); // 2·max|d| = 2·0.03
        assert_relative_eq!(profile_inertia(&dev), rms(&dev), epsilon = 1e-12);
        let readings = [10.02, 9.98, 10.05, 9.99];
        assert_relative_eq!(total_runout(&readings), 0.07, epsilon = 1e-12);
    }

    /// Three-lobed bore of nominal radius `r` and lobe amplitude `lobe`
    /// (roundness `2·lobe`), centred on `center`.
    fn lobed_bore(center: [f64; 2], r: f64, lobe: f64) -> Vec<[f64; 2]> {
        (0..36)
            .map(|k| {
                let t = k as f64 / 36.0 * std::f64::consts::TAU;
                let rho = r + lobe * (3.0 * t).cos();
                [center[0] + rho * t.cos(), center[1] + rho * t.sin()]
            })
            .collect()
    }

    #[test]
    fn roundness_is_translation_invariant_far_from_the_origin() {
        // 2 mm bore with a 0.1 µm three-lobe form error, in mm. At the origin
        // and at CMM machine coordinates (500, 300) mm the form error is the
        // same; the raw-moment Kåsa fit reported 3.85e-3 mm (19×) at the offset.
        let at_origin = roundness(&lobed_bore([0.0, 0.0], 1.0, 1e-4));
        let offset = roundness(&lobed_bore([500.0, 300.0], 1.0, 1e-4));
        assert_relative_eq!(at_origin, 2e-4, max_relative = 1e-6);
        assert_relative_eq!(offset, at_origin, max_relative = 1e-6);
        let (cx, cy, r) = least_squares_circle(&lobed_bore([500.0, 300.0], 1.0, 1e-4)).unwrap();
        assert!((cx - 500.0).abs() < 1e-9 && (cy - 300.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-8, "r = {r}");
        // Cylindricity uses the same circle fit.
        let mut cyl = Vec::new();
        for level in 0..3
        {
            for p in lobed_bore([500.0, 300.0], 1.0, 1e-4)
            {
                cyl.push([p[0], p[1], 40.0 + level as f64]);
            }
        }
        assert_relative_eq!(cylindricity(&cyl), 2e-4, max_relative = 1e-6);
    }

    /// 5×5 grid over a square of side `side` centred on `(x0, y0)`, with a
    /// paraboloid dome of height `dome` at the corners above `z0`: flatness of
    /// the dome is `dome` (corner) minus `0` (center).
    fn domed_square(x0: f64, y0: f64, z0: f64, side: f64, dome: f64) -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for i in 0..5
        {
            for j in 0..5
            {
                let (u, v) = ((i as f64 - 2.0) / 2.0, (j as f64 - 2.0) / 2.0);
                pts.push([
                    x0 + 0.5 * side * u,
                    y0 + 0.5 * side * v,
                    z0 + 0.5 * dome * (u * u + v * v),
                ]);
            }
        }
        pts
    }

    #[test]
    fn flatness_is_translation_and_scale_invariant() {
        // 0.1 mm square with a 1 nm dome, in mm, at the origin and at machine
        // coordinates (300, 200, 80) mm. The raw normal equations gave 1.41e-2
        // mm at the offset (about 14 000× the form error).
        let at_origin = flatness(&domed_square(0.0, 0.0, 0.0, 0.1, 1e-6));
        let offset = flatness(&domed_square(300.0, 200.0, 80.0, 0.1, 1e-6));
        assert_relative_eq!(at_origin, 1e-6, max_relative = 1e-6);
        assert_relative_eq!(offset, at_origin, max_relative = 1e-6);
        // A 50 µm membrane with a 1 nm dome expressed in metres: the absolute
        // 1e-14 determinant guard declared it singular and flatness returned 0.
        let metres = flatness(&domed_square(0.0, 0.0, 0.0, 5e-5, 1e-9));
        assert_relative_eq!(metres, 1e-9, max_relative = 1e-6);
        let (a, b, c) = least_squares_plane(&domed_square(0.0, 0.0, 0.0, 5e-5, 1e-9)).unwrap();
        assert!(a.is_finite() && b.abs() < 1e-9 && c.abs() < 1e-9);
    }

    #[test]
    fn straightness_is_translation_and_scale_invariant() {
        // 11 points with a 0.1 nm parabolic bow at 1 nm spacing, in metres:
        // the absolute 1e-14 guard on n·Σx² − (Σx)² rejected the fit and
        // straightness returned 0.
        let bow = |x0: f64, y0: f64, step: f64, depth: f64| -> Vec<[f64; 2]> {
            (0..11)
                .map(|i| {
                    let u = (i as f64 - 5.0) / 5.0;
                    [x0 + 5.0 * step * u, y0 + depth * u * u]
                })
                .collect()
        };
        assert_relative_eq!(
            straightness(&bow(0.0, 0.0, 1e-9, 1e-10)),
            1e-10,
            max_relative = 1e-6
        );
        // Far from the origin (positions in µm along a 1 m stage): the raw
        // sums gave 1.00025e-3 instead of 1e-3.
        let at_origin = straightness(&bow(0.0, 0.0, 1.0, 1e-3));
        let offset = straightness(&bow(1e6, 2e3, 1.0, 1e-3));
        assert_relative_eq!(at_origin, 1e-3, max_relative = 1e-9);
        assert_relative_eq!(offset, at_origin, max_relative = 1e-6);
        // The public (a, b) form still describes the same line.
        let (a, b) = least_squares_line(&[[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]).unwrap();
        assert_relative_eq!(a, 1.0, epsilon = 1e-12);
        assert_relative_eq!(b, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn degenerate_fits_are_still_rejected() {
        // One distinct abscissa: no line.
        assert!(least_squares_line(&[[2.0, 0.0], [2.0, 1.0], [2.0, 3.0]]).is_none());
        assert_eq!(straightness(&[[2.0, 0.0], [2.0, 1.0]]), 0.0);
        // Points collinear in (x, y): no plane, at any scale.
        let collinear = |s: f64| [[0.0, 0.0, 1.0], [s, 2.0 * s, 0.0], [2.0 * s, 4.0 * s, 3.0]];
        assert!(least_squares_plane(&collinear(1.0)).is_none());
        assert!(least_squares_plane(&collinear(1e-9)).is_none());
        assert!(least_squares_plane(&collinear(1e9)).is_none());
        // Collinear points: no circle.
        assert!(least_squares_circle(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_none());
        assert!(least_squares_circle(&[[0.0, 0.0], [1.0, 1.0]]).is_none());
    }

    #[test]
    fn cylindricity_of_a_perfect_cylinder_is_zero() {
        let mut pts = Vec::new();
        for level in 0..4
        {
            for k in 0..12
            {
                let t = k as f64 / 12.0 * std::f64::consts::TAU;
                pts.push([1.0 + 0.8 * t.cos(), -2.0 + 0.8 * t.sin(), level as f64]);
            }
        }
        assert_relative_eq!(cylindricity(&pts), 0.0, epsilon = 1e-9);
    }
}
