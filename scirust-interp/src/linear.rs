//! Piecewise-linear interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, piecewise_integral, validate_nodes};

/// Piecewise-linear interpolant through the given nodes.
///
/// Between adjacent nodes the value is the straight-line segment joining them.
/// **Extrapolation is linear**: for queries outside `[xs[0], xs[n - 1]]` the
/// slope of the nearest boundary segment is continued, so an affine input is
/// reproduced up to rounding everywhere, including outside the node range.
/// Every node is reproduced exactly.
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
        slope(self.xs[i], self.xs[i + 1], self.ys[i], self.ys[i + 1])
    }

    /// Exact definite integral of the interpolant from `a` to `b`.
    ///
    /// Integrates the piecewise-linear function (trapezoidal rule on the
    /// nodes, exact by construction), including the linearly extrapolated
    /// parts when a bound lies outside the node range. Reversed bounds flip
    /// the sign, equal bounds give `0`, and a non-finite bound gives NaN.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        piecewise_integral(&self.xs, a, b, |i, lo, hi| {
            let (x0, x1) = (self.xs[i], self.xs[i + 1]);
            let (y0, y1) = (self.ys[i], self.ys[i + 1]);
            // Exact for a linear piece: width times the mean of the end
            // values. Unlike expanding the antiderivative in squared offsets,
            // this cannot overflow or underflow before the area itself does.
            let (vl, vh) = (
                segment_value(x0, x1, y0, y1, lo),
                segment_value(x0, x1, y0, y1, hi),
            );
            let width = hi - lo;
            let sum = vl + vh;
            if width.is_finite() && sum.is_finite()
            {
                width * (0.5 * sum)
            }
            else
            {
                2.0 * ((0.5 * hi - 0.5 * lo) * (0.5 * vl + 0.5 * vh))
            }
        })
    }
}

impl Interpolator for LinearInterp {
    fn eval(&self, x: f64) -> f64 {
        let i = find_segment(&self.xs, x);
        segment_value(self.xs[i], self.xs[i + 1], self.ys[i], self.ys[i + 1], x)
    }
}

/// Value at `x` of the line through `(x0, y0)` and `(x1, y1)`, with `x0 < x1`.
///
/// Both nodes are reproduced exactly (`y0 + 1 * (y1 - y0)` can round away from
/// `y1`), and differences that overflow although every node is finite are
/// taken on halved operands instead.
fn segment_value(x0: f64, x1: f64, y0: f64, y1: f64, x: f64) -> f64 {
    if x == x1
    {
        return y1;
    }
    let (dx, offset) = (x1 - x0, x - x0);
    let t = if dx.is_finite() && offset.is_finite()
    {
        offset / dx
    }
    else
    {
        (0.5 * x - 0.5 * x0) / (0.5 * x1 - 0.5 * x0)
    };
    let dy = y1 - y0;
    if dy.is_finite()
    {
        y0 + t * dy
    }
    else
    {
        // Opposite-sign endpoints near f64::MAX: each weighted term is
        // bounded by its endpoint inside the segment.
        (1.0 - t) * y0 + t * y1
    }
}

/// Slope of the segment `(x0, y0)`–`(x1, y1)`, halving the operands when a
/// difference of finite nodes overflows.
fn slope(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let (dx, dy) = (x1 - x0, y1 - y0);
    if dx.is_finite() && dy.is_finite()
    {
        dy / dx
    }
    else
    {
        (0.5 * y1 - 0.5 * y0) / (0.5 * x1 - 0.5 * x0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_close(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs()
    }

    #[test]
    fn eval_reproduces_last_node_exactly() {
        // 0.9 + (0.1 - 0.9) rounds to 0.09999999999999998, not 0.1.
        let lin = LinearInterp::new(&[0.0, 1.0], &[0.9, 0.1]).unwrap();
        assert_eq!(lin.eval(0.0), 0.9);
        assert_eq!(lin.eval(1.0), 0.1);
        let lin = LinearInterp::new(&[0.0, 1.0, 2.0], &[0.3, 0.9, 0.1]).unwrap();
        assert_eq!(lin.eval(1.0), 0.9);
        assert_eq!(lin.eval(2.0), 0.1);
    }

    #[test]
    fn eval_and_derivative_survive_overflowing_differences() {
        // xs[1] - xs[0] overflows although every node is finite.
        let lin = LinearInterp::new(&[-f64::MAX, f64::MAX], &[0.0, 1.0]).unwrap();
        assert_eq!(lin.eval(0.0), 0.5);
        assert!(rel_close(lin.derivative(0.0), 0.5 / f64::MAX, 1e-12));
        // ys[1] - ys[0] overflows as well.
        let lin = LinearInterp::new(&[-f64::MAX, f64::MAX], &[-f64::MAX, f64::MAX]).unwrap();
        assert_eq!(lin.eval(0.0), 0.0);
        assert!(rel_close(lin.eval(f64::MAX / 2.0), f64::MAX / 2.0, 1e-12));
        assert_eq!(lin.derivative(0.0), 1.0);
        let lin = LinearInterp::new(&[0.0, 1.0], &[-f64::MAX, f64::MAX]).unwrap();
        assert_eq!(lin.eval(0.5), 0.0);
        assert!(rel_close(lin.eval(0.75), f64::MAX / 2.0, 1e-12));
        assert!(lin.derivative(0.5).is_infinite());
    }

    #[test]
    fn integrate_keeps_large_and_small_abscissa_scales() {
        // q * q overflowed for abscissae beyond ~1e154: the integral was inf.
        let lin = LinearInterp::new(&[0.0, 1e200], &[0.0, 1.0]).unwrap();
        assert!(rel_close(lin.integrate(0.0, 1e200), 5e199, 1e-12));
        let lin = LinearInterp::new(&[1e200, 3e200], &[2.0, 4.0]).unwrap();
        assert!(rel_close(lin.integrate(1e200, 3e200), 6e200, 1e-12));
        assert!(rel_close(lin.integrate(3e200, 1e200), -6e200, 1e-12));
        // q * q underflowed below ~1e-162: the integral was 0.
        let lin = LinearInterp::new(&[0.0, 1e-200], &[0.0, 1.0]).unwrap();
        assert!(rel_close(lin.integrate(0.0, 1e-200), 5e-201, 1e-12));
        // A span wider than f64::MAX with a finite area.
        let lin = LinearInterp::new(&[-f64::MAX, f64::MAX], &[0.0, 1e-300]).unwrap();
        assert!(rel_close(
            lin.integrate(-f64::MAX, f64::MAX),
            f64::MAX * 1e-300,
            1e-12
        ));
    }

    #[test]
    fn integrate_matches_trapezoid_on_ordinary_data() {
        let lin = LinearInterp::new(&[0.0, 1.0, 3.0], &[1.0, 3.0, -1.0]).unwrap();
        assert!(rel_close(lin.integrate(0.0, 3.0), 4.0, 1e-14));
        assert!(rel_close(lin.integrate(0.5, 2.0), 3.25, 1e-14));
        // Extrapolation continues the boundary slopes (2 on the left, -2 on the right).
        assert!(lin.integrate(-1.0, 0.0).abs() < 1e-14);
        assert!(rel_close(lin.integrate(3.0, 4.0), -2.0, 1e-14));
        assert_eq!(lin.integrate(2.0, 2.0), 0.0);
    }
}
