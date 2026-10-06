//! Internal numerical helpers shared across the interpolation methods.
//!
//! Nothing in this module is part of the public API.

use crate::error::InterpError;

/// Validate node arrays for a multi-point method.
///
/// Checks, in order: equal lengths, at least `min_points` points, all values
/// finite (rejecting NaN/±∞), and strictly increasing abscissae.
pub(crate) fn validate_nodes(xs: &[f64], ys: &[f64], min_points: usize) -> Result<(), InterpError> {
    if xs.len() != ys.len()
    {
        return Err(InterpError::LengthMismatch {
            xs: xs.len(),
            ys: ys.len(),
        });
    }
    if xs.len() < min_points
    {
        return Err(InterpError::TooFewPoints {
            got: xs.len(),
            need: min_points,
        });
    }
    for (i, &v) in xs.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(InterpError::NonFinite { index: i });
        }
    }
    for (i, &v) in ys.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(InterpError::NonFinite { index: i });
        }
    }
    for (i, w) in xs.windows(2).enumerate()
    {
        if w[1] <= w[0]
        {
            return Err(InterpError::NotStrictlyIncreasing { index: i + 1 });
        }
    }
    Ok(())
}

/// Locate the segment containing `x`.
///
/// Returns the left index `i` of the bracketing interval `[xs[i], xs[i + 1]]`,
/// clamped to `0..=n - 2` so that queries below `xs[0]` map to the first
/// segment and queries above `xs[n - 1]` map to the last one (the basis for
/// each method's extrapolation). Callers guarantee `xs.len() >= 2`.
pub(crate) fn find_segment(xs: &[f64], x: f64) -> usize {
    let n = xs.len();
    if x <= xs[0]
    {
        return 0;
    }
    if x >= xs[n - 1]
    {
        return n - 2;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1
    {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    lo
}

/// Evaluate a cubic Hermite segment.
///
/// The segment spans a node with value `y0` and slope `d0` at its left end and
/// value `y1`, slope `d1` at its right end, with width `h`. `dx` is
/// `x - x_left`; values of `dx` outside `[0, h]` extrapolate the same cubic.
pub(crate) fn hermite(y0: f64, y1: f64, d0: f64, d1: f64, h: f64, dx: f64) -> f64 {
    let t = dx / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
}

/// Solve a tridiagonal linear system with the Thomas algorithm.
///
/// `a` is the sub-diagonal (`a[0]` unused), `b` the main diagonal, `c` the
/// super-diagonal (`c[n - 1]` unused) and `d` the right-hand side. The systems
/// assembled by this crate are diagonally dominant, so no pivoting is needed.
pub(crate) fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n
    {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev()
    {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

/// First derivative of a cubic Hermite segment (same arguments as [`hermite`]).
pub(crate) fn hermite_derivative(y0: f64, y1: f64, d0: f64, d1: f64, h: f64, dx: f64) -> f64 {
    let t = dx / h;
    let t2 = t * t;
    let dh00 = 6.0 * t2 - 6.0 * t;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 = 3.0 * t2 - 2.0 * t;
    (dh00 * y0 + dh01 * y1) / h + dh10 * d0 + dh11 * d1
}

/// Second derivative of a cubic Hermite segment (same arguments as [`hermite`]).
pub(crate) fn hermite_second_derivative(
    y0: f64,
    y1: f64,
    d0: f64,
    d1: f64,
    h: f64,
    dx: f64,
) -> f64 {
    let t = dx / h;
    let ddh00 = 12.0 * t - 6.0;
    let ddh10 = 6.0 * t - 4.0;
    let ddh01 = -12.0 * t + 6.0;
    let ddh11 = 6.0 * t - 2.0;
    (ddh00 * y0 + ddh01 * y1) / (h * h) + (ddh10 * d0 + ddh11 * d1) / h
}

/// Exact integral of a cubic Hermite segment between local offsets `p` and `q`.
///
/// `p` and `q` are offsets from the segment's left node (like `dx` in
/// [`hermite`]); values outside `[0, h]` integrate the extrapolated cubic.
pub(crate) fn hermite_integral(y0: f64, y1: f64, d0: f64, d1: f64, h: f64, p: f64, q: f64) -> f64 {
    // Antiderivative of the Hermite basis in the unit variable t = dx / h.
    let anti = |dx: f64| {
        let t = dx / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let i00 = 0.5 * t4 - t3 + t;
        let i10 = 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2;
        let i01 = -0.5 * t4 + t3;
        let i11 = 0.25 * t4 - t3 / 3.0;
        i00 * y0 + i10 * h * d0 + i01 * y1 + i11 * h * d1
    };
    h * (anti(q) - anti(p))
}

/// Integrate a piecewise function over `[a, b]` following the segment layout
/// of [`find_segment`].
///
/// `segment(i, lo, hi)` must return the exact integral of piece `i` over the
/// absolute interval `[lo, hi]`. The first piece is extended to `-∞` and the
/// last to `+∞`, matching the crate's boundary-piece extrapolation, so bounds
/// outside the node range integrate the extrapolated function. Reversed bounds
/// flip the sign; equal bounds give `0`; a non-finite bound gives `NaN`.
pub(crate) fn piecewise_integral<F>(xs: &[f64], a: f64, b: f64, segment: F) -> f64
where
    F: Fn(usize, f64, f64) -> f64,
{
    if !a.is_finite() || !b.is_finite()
    {
        return f64::NAN;
    }
    if a == b
    {
        return 0.0;
    }
    let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let n = xs.len();
    let first = find_segment(xs, lo);
    let last = find_segment(xs, hi);
    let mut total = 0.0;
    for i in first..=last
    {
        let seg_lo = if i == 0 { lo } else { lo.max(xs[i]) };
        let seg_hi = if i == n - 2 { hi } else { hi.min(xs[i + 1]) };
        if seg_hi > seg_lo
        {
            total += segment(i, seg_lo, seg_hi);
        }
    }
    sign * total
}
