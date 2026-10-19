//! Planar 2-link arm kinematics.

/// Maximum reach of a 2-link arm.
pub fn max_reach(l1: f64, l2: f64) -> f64 {
    l1 + l2
}

/// Forward kinematics: end-effector `(x, y)` from joint angles (rad).
pub fn fk_2link(l1: f64, l2: f64, th1: f64, th2: f64) -> (f64, f64) {
    (
        l1 * th1.cos() + l2 * (th1 + th2).cos(),
        l1 * th1.sin() + l2 * (th1 + th2).sin(),
    )
}

/// Inverse kinematics (elbow-down branch): joint angles reaching `(x, y)`, or
/// `None` if the target is outside the annular workspace
/// `|l1 − l2| ≤ √(x² + y²) ≤ l1 + l2`.
///
/// Targets on the workspace boundary (arm fully stretched or fully folded,
/// e.g. `(max_reach(l1, l2), 0.0)`) are reachable: the elbow cosine computed
/// there can land a few ulps outside `[-1, 1]` through rounding alone, so it is
/// accepted within a bound on that rounding error (relative to the squared
/// lengths it is formed from, hence independent of the length unit) and
/// clamped to `[-1, 1]`.
pub fn ik_2link(l1: f64, l2: f64, x: f64, y: f64) -> Option<(f64, f64)> {
    let r2 = x * x + y * y;
    let den = 2.0 * l1 * l2;
    let c2 = (r2 - l1 * l1 - l2 * l2) / den;
    // Rounding-error bound of `c2`: a few ulps of the magnitudes summed in the
    // numerator, scaled by the denominator.
    let tol = 4.0 * f64::EPSILON * (r2 + l1 * l1 + l2 * l2) / den.abs();
    if c2.is_nan() || c2.abs() > 1.0 + tol
    {
        return None; // unreachable (or NaN input)
    }
    let c2 = c2.clamp(-1.0, 1.0);
    let s2 = -(1.0 - c2 * c2).sqrt(); // elbow-down
    let th2 = s2.atan2(c2);
    let th1 = y.atan2(x) - (l2 * s2).atan2(l1 + l2 * c2);
    Some((th1, th2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn fk_of_known_angles() {
        // θ1=0, θ2=π/2: tip at (l1, l2).
        let (x, y) = fk_2link(1.0, 0.5, 0.0, PI / 2.0);
        assert!(
            (x - 1.0).abs() < 1e-9 && (y - 0.5).abs() < 1e-9,
            "({x},{y})"
        );
    }

    #[test]
    fn ik_then_fk_round_trips() {
        let (l1, l2) = (1.0, 0.7);
        for &(x, y) in &[(1.2, 0.4), (0.5, 1.0), (-0.3, 0.9), (1.5, -0.2)]
        {
            let (th1, th2) = ik_2link(l1, l2, x, y).expect("reachable");
            let (xr, yr) = fk_2link(l1, l2, th1, th2);
            assert!(
                (xr - x).abs() < 1e-9 && (yr - y).abs() < 1e-9,
                "rt ({xr},{yr})"
            );
        }
    }

    #[test]
    fn workspace_boundary_targets_are_reachable() {
        // Fully stretched (r = l1 + l2) and fully folded (r = |l1 − l2|)
        // targets. Before the fix the elbow cosine rounded to e.g.
        // 1.0000000000000009 for (0.1, 1.0) at (1.1, 0) and the reachable
        // target was reported as unreachable.
        for &(l1, l2) in &[
            (0.1, 1.0),
            (0.1, 0.3),
            (1.0, 0.7),
            (0.1, 1.1),
            (2.5e-3, 4.0e3),
        ]
        {
            let outer = max_reach(l1, l2);
            let inner = (l1 - l2).abs();
            for &(x, y) in &[
                (outer, 0.0),
                (0.0, outer),
                (-outer, 0.0),
                (inner, 0.0),
                (0.0, -inner),
            ]
            {
                let (th1, th2) = ik_2link(l1, l2, x, y)
                    .unwrap_or_else(|| panic!("boundary target ({x}, {y}) of ({l1}, {l2})"));
                let (xr, yr) = fk_2link(l1, l2, th1, th2);
                let scale = l1 + l2;
                assert!(
                    (xr - x).abs() <= 1e-12 * scale && (yr - y).abs() <= 1e-12 * scale,
                    "rt ({xr},{yr}) vs ({x},{y})"
                );
            }
        }
        // Points produced by forward kinematics with a straight or folded elbow
        // must be accepted back by the inverse kinematics.
        for k in 0..64
        {
            let th1 = -PI + 2.0 * PI * (k as f64 + 0.37) / 64.0;
            for &th2 in &[0.0, PI]
            {
                let (x, y) = fk_2link(0.3, 1.7, th1, th2);
                assert!(ik_2link(0.3, 1.7, x, y).is_some(), "th1 {th1}, th2 {th2}");
            }
        }
    }

    #[test]
    fn targets_just_outside_the_workspace_stay_unreachable() {
        for &(l1, l2) in &[(0.1, 1.0), (1.0, 0.7), (2.5e-3, 4.0e3)]
        {
            let outer = max_reach(l1, l2) * (1.0 + 1e-12);
            let inner = (l1 - l2).abs() * (1.0 - 1e-12);
            assert!(ik_2link(l1, l2, outer, 0.0).is_none());
            assert!(ik_2link(l1, l2, 0.0, inner).is_none());
        }
        assert!(ik_2link(1.0, 0.7, f64::NAN, 0.0).is_none());
    }

    #[test]
    fn unreachable_target_returns_none() {
        // Beyond the max reach 1.7.
        assert!(ik_2link(1.0, 0.7, 2.0, 0.0).is_none());
        assert!(max_reach(1.0, 0.7) < 2.0);
    }
}
