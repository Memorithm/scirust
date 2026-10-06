//! Piecewise-linear interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, piecewise_integral, validate_nodes};

/// Piecewise-linear interpolant through the given nodes.
///
/// Between adjacent nodes the value is the straight-line segment joining them.
/// **Extrapolation is linear**: for queries outside `[xs[0], xs[n - 1]]` the
/// slope of the nearest boundary segment is continued, so an affine input is
/// reproduced exactly everywhere, including outside the node range.
#[derive(Debug, Clone)]
pub struct LinearInterp {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl LinearInterp {
    /// Build a linear interpolant.
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
        })
    }

    /// First derivative of the interpolant at `x`.
    ///
    /// Returns the slope of the segment containing `x`. At an interior node
    /// the right-hand segment's slope is used; outside the node range the
    /// boundary slope is continued (matching the linear extrapolation). A NaN
    /// query returns NaN.
    pub fn derivative(&self, x: f64) -> f64 {
        if x.is_nan()
        {
            return f64::NAN;
        }
        let i = find_segment(&self.xs, x);
        (self.ys[i + 1] - self.ys[i]) / (self.xs[i + 1] - self.xs[i])
    }

    /// Exact definite integral of the interpolant from `a` to `b`.
    ///
    /// Integrates the piecewise-linear function (trapezoidal rule on the
    /// nodes, exact by construction), including the linearly extrapolated
    /// parts when a bound lies outside the node range. Reversed bounds flip
    /// the sign, equal bounds give `0`, and a non-finite bound gives NaN.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        piecewise_integral(&self.xs, a, b, |i, lo, hi| {
            let x0 = self.xs[i];
            let y0 = self.ys[i];
            let s = (self.ys[i + 1] - y0) / (self.xs[i + 1] - x0);
            let (p, q) = (lo - x0, hi - x0);
            y0 * (q - p) + 0.5 * s * (q * q - p * p)
        })
    }
}

impl Interpolator for LinearInterp {
    fn eval(&self, x: f64) -> f64 {
        let i = find_segment(&self.xs, x);
        let (x0, x1) = (self.xs[i], self.xs[i + 1]);
        let (y0, y1) = (self.ys[i], self.ys[i + 1]);
        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}
