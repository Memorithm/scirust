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

/// `floor(log2(v))` for a positive finite `v` (normal or subnormal).
pub(crate) fn floor_log2(v: f64) -> i32 {
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0
    {
        // Subnormal: value = mantissa * 2^-1074.
        let mantissa = bits & ((1u64 << 52) - 1);
        63 - mantissa.leading_zeros() as i32 - 1074
    }
    else
    {
        biased - 1023
    }
}

/// `v * 2^e`, applied in steps so that intermediate powers stay normal.
pub(crate) fn ldexp(mut v: f64, mut e: i32) -> f64 {
    while e != 0
    {
        let step = e.clamp(-1000, 1000);
        v *= f64::from_bits(((1023 + step) as u64) << 52);
        e -= step;
    }
    v
}

/// Binary exponent of the largest node spacing, so that every width scaled
/// by `2^-ex` lies in `(0, 2)`. Widths that overflow (node spans beyond
/// `f64::MAX`) are measured on halved abscissae.
pub(crate) fn abscissa_exponent(xs: &[f64]) -> i32 {
    xs.windows(2)
        .map(|w| {
            let h = w[1] - w[0];
            if h.is_finite()
            {
                floor_log2(h)
            }
            else
            {
                floor_log2(0.5 * w[1] - 0.5 * w[0]) + 1
            }
        })
        .max()
        .unwrap_or(0)
}

/// Segment widths scaled by `2^-ex` (see [`abscissa_exponent`]).
pub(crate) fn scaled_widths(xs: &[f64], ex: i32) -> Vec<f64> {
    xs.windows(2)
        .map(|w| {
            let h = w[1] - w[0];
            if h.is_finite()
            {
                ldexp(h, -ex)
            }
            else
            {
                ldexp(0.5 * w[1] - 0.5 * w[0], 1 - ex)
            }
        })
        .collect()
}

/// Binary exponent of the largest ordinate magnitude (`0` for all-zero data),
/// so that every ordinate scaled by `2^-ey` lies in `(-2, 2)`.
pub(crate) fn ordinate_exponent(ys: &[f64]) -> i32 {
    let ymax = ys.iter().fold(0.0_f64, |acc, &y| acc.max(y.abs()));
    if ymax > 0.0 { floor_log2(ymax) } else { 0 }
}

/// Normalised offsets `u = (x1 - x) / h` and `t = (x - x0) / h` of `x` in
/// the segment `[x0, x1]`.
///
/// At `x == x0` this gives exactly `(1, 0)` and at `x == x1` exactly
/// `(0, 1)`. When a difference overflows (node spans or queries beyond
/// `f64::MAX`), all three differences are taken on halved values instead.
pub(crate) fn unit_offsets(x0: f64, x1: f64, x: f64) -> (f64, f64) {
    let (mut a, mut b, mut h) = (x1 - x, x - x0, x1 - x0);
    if !(a.is_finite() && b.is_finite() && h.is_finite())
    {
        a = 0.5 * x1 - 0.5 * x;
        b = 0.5 * x - 0.5 * x0;
        h = 0.5 * x1 - 0.5 * x0;
    }
    (a / h, b / h)
}

/// Piecewise cubic Hermite interpolant stored in power-of-two scaled units.
///
/// Abscissae are scaled by `2^-ex` (largest width in `[1, 2)`) and ordinates
/// by `2^-ey` (largest magnitude in `[1, 2)`), so the node slopes, which the
/// caller computes from the scaled widths and ordinates, stay moderate even
/// when the slopes in the caller's units (`dy / dx`) would overflow or the
/// differences `y1 - y0`, `x1 - x0` exceed `f64::MAX`. Results are scaled
/// back with exact power-of-two factors, so they are finite whenever the true
/// value is representable.
#[derive(Debug, Clone)]
pub(crate) struct HermitePieces {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Ordinates scaled by `2^-ey`.
    yn: Vec<f64>,
    /// Node slopes in scaled units: `dy/dx = dn * 2^(ey - ex)`.
    dn: Vec<f64>,
    /// Segment widths scaled by `2^-ex`.
    hn: Vec<f64>,
    ex: i32,
    ey: i32,
}

impl HermitePieces {
    /// Scale validated nodes and compute the node slopes with `slopes`, which
    /// receives the scaled widths and ordinates and must be scale-covariant
    /// (scaling every slope by `sy / sx` when ordinates scale by `sy` and
    /// abscissae by `sx`).
    pub(crate) fn new(xs: &[f64], ys: &[f64], slopes: impl Fn(&[f64], &[f64]) -> Vec<f64>) -> Self {
        let ex = abscissa_exponent(xs);
        let ey = ordinate_exponent(ys);
        let hn = scaled_widths(xs, ex);
        let yn: Vec<f64> = ys.iter().map(|&y| ldexp(y, -ey)).collect();
        let dn = slopes(&hn, &yn);
        Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            yn,
            dn,
            hn,
            ex,
            ey,
        }
    }

    /// Segment index and normalised offsets `(u, t)` for `x`.
    fn local(&self, x: f64) -> (usize, f64, f64) {
        let i = find_segment(&self.xs, x);
        let (u, t) = unit_offsets(self.xs[i], self.xs[i + 1], x);
        (i, u, t)
    }

    /// Value of the interpolant at `x`; exact (bit for bit) at every node.
    pub(crate) fn eval(&self, x: f64) -> f64 {
        let (i, u, t) = self.local(x);
        let (y0, y1) = (self.yn[i], self.yn[i + 1]);
        let (d0, d1) = (self.dn[i], self.dn[i + 1]);
        // Hermite cubic rewritten as the linear interpolant plus a correction
        // that vanishes at both nodes, evaluated in scaled units so that
        // extrapolated values stay finite whenever they are representable:
        //   p = u y0 + t y1 + u t [(u - t)(y0 - y1) + h (u d0 - t d1)].
        // Node queries return the stored ordinate directly (t or u is exactly
        // 0 there), so scaling round-trips can never perturb them.
        if t == 0.0
        {
            return self.ys[i];
        }
        if u == 0.0
        {
            return self.ys[i + 1];
        }
        let correction = u * t * ((u - t) * (y0 - y1) + self.hn[i] * (u * d0 - t * d1));
        ldexp(u * y0 + t * y1 + correction, self.ey)
    }

    /// First derivative of the interpolant at `x`.
    pub(crate) fn derivative(&self, x: f64) -> f64 {
        let (i, u, t) = self.local(x);
        let (y0, y1) = (self.yn[i], self.yn[i + 1]);
        let (d0, d1) = (self.dn[i], self.dn[i + 1]);
        let dn = 6.0 * t * u * (y1 - y0) / self.hn[i]
            + (3.0 * t * t - 4.0 * t + 1.0) * d0
            + (3.0 * t * t - 2.0 * t) * d1;
        ldexp(dn, self.ey - self.ex)
    }

    /// Second derivative of the interpolant at `x` (right-hand piece at
    /// interior nodes, left-hand piece at the last node).
    pub(crate) fn second_derivative(&self, x: f64) -> f64 {
        let (i, _, t) = self.local(x);
        let (y0, y1) = (self.yn[i], self.yn[i + 1]);
        let (d0, d1) = (self.dn[i], self.dn[i + 1]);
        let h = self.hn[i];
        let ddn = (12.0 * t - 6.0) * (y0 - y1) / (h * h)
            + ((6.0 * t - 4.0) * d0 + (6.0 * t - 2.0) * d1) / h;
        ldexp(ddn, self.ey - 2 * self.ex)
    }

    /// Exact definite integral from `a` to `b` (see [`piecewise_integral`]).
    ///
    /// The pieces are summed in scaled units and rescaled once, so partial
    /// sums cannot overflow when the total is representable.
    pub(crate) fn integrate(&self, a: f64, b: f64) -> f64 {
        let scaled = piecewise_integral(&self.xs, a, b, |i, lo, hi| {
            let (y0, y1) = (self.yn[i], self.yn[i + 1]);
            let (d0, d1) = (self.dn[i], self.dn[i + 1]);
            let h = self.hn[i];
            // Antiderivative of the Hermite basis in the unit variable t.
            let anti = |x: f64| {
                let (_, t) = unit_offsets(self.xs[i], self.xs[i + 1], x);
                let t2 = t * t;
                let t3 = t2 * t;
                let t4 = t3 * t;
                let i00 = 0.5 * t4 - t3 + t;
                let i10 = 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2;
                let i01 = -0.5 * t4 + t3;
                let i11 = 0.25 * t4 - t3 / 3.0;
                i00 * y0 + i10 * h * d0 + i01 * y1 + i11 * h * d1
            };
            h * (anti(hi) - anti(lo))
        });
        ldexp(scaled, self.ex + self.ey)
    }
}
