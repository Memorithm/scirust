//! Barycentric-form Lagrange interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::validate_nodes;

/// Global polynomial interpolant in the numerically stable barycentric form.
///
/// This represents the unique degree-`n-1` polynomial through the `n` nodes,
/// evaluated with the second (true) barycentric formula. The weights are
/// computed with a capacity scaling `(x_max - x_min) / 4` per factor to keep
/// their magnitudes near unity and avoid overflow.
///
/// **Extrapolation** is polynomial: outside the node range the value is that of
/// the interpolating polynomial, which — as with any high-degree polynomial —
/// can grow rapidly. Interior evaluation is well behaved for modest `n`.
///
/// A query exactly on a node returns that node's ordinate. Queries arbitrarily
/// close to a node and node spans wider than `f64::MAX` are handled without
/// intermediate overflow. A non-finite query returns NaN.
#[derive(Debug, Clone)]
pub struct BarycentricLagrange {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Barycentric weights.
    w: Vec<f64>,
    /// Whether abscissa differences are taken on halved values because the
    /// node span `x_max - x_min` overflows `f64`.
    halve: bool,
}

/// Difference `a - b`, halved first when `halve` is set so that it cannot
/// overflow for finite inputs. Barycentric weights and the evaluation formula
/// only use ratios of such differences, so the common factor cancels.
fn diff(a: f64, b: f64, halve: bool) -> f64 {
    if halve { 0.5 * a - 0.5 * b } else { a - b }
}

impl BarycentricLagrange {
    /// Build a barycentric-Lagrange interpolant.
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        let n = xs.len();
        let halve = !(xs[n - 1] - xs[0]).is_finite();
        let cap = diff(xs[n - 1], xs[0], halve) / 4.0;
        let w: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(j, &xj)| {
                let mut wj = 1.0;
                for (k, &xk) in xs.iter().enumerate()
                {
                    if k != j
                    {
                        wj *= cap / diff(xj, xk, halve);
                    }
                }
                wj
            })
            .collect();
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            w,
            halve,
        })
    }
}

impl Interpolator for BarycentricLagrange {
    fn eval(&self, x: f64) -> f64 {
        if !x.is_finite()
        {
            return f64::NAN;
        }
        // Distance to the closest node. Exactly on a node, return its ordinate
        // (avoids 0/0).
        let mut dmin = f64::INFINITY;
        for (&xj, &yj) in self.xs.iter().zip(&self.ys)
        {
            let d = diff(x, xj, self.halve);
            if d == 0.0
            {
                return yj;
            }
            dmin = dmin.min(d.abs());
        }
        // The second barycentric formula is invariant under a common factor
        // on the terms `w_j / (x - x_j)`. Multiplying them by the smallest
        // distance keeps every term at most `|w_j|` in magnitude, so a query
        // very close to a node cannot overflow (or produce inf * 0).
        let mut num = 0.0;
        let mut den = 0.0;
        for ((&xj, &yj), &wj) in self.xs.iter().zip(&self.ys).zip(&self.w)
        {
            let t = wj * (dmin / diff(x, xj, self.halve));
            num += t * yj;
            den += t;
        }
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_close(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn query_a_subnormal_distance_from_a_node_stays_finite() {
        // p(x) = x^2 + 1 through (0, 1), (1, 2), (2, 5).
        let b = BarycentricLagrange::new(&[0.0, 1.0, 2.0], &[1.0, 2.0, 5.0]).unwrap();
        for x in [5e-324, 1e-320, 1e-310, 1e-300]
        {
            let v = b.eval(x);
            assert!(rel_close(v, 1.0, 1e-12), "eval({x:e}) = {v}");
        }
        // A zero ordinate at the nearby node must not produce inf * 0.
        let sq = BarycentricLagrange::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap();
        let v = sq.eval(5e-324);
        assert!(v.is_finite() && v.abs() < 1e-300, "eval(5e-324) = {v}");
        // Just right of an interior node.
        let x = 1.0 + f64::EPSILON;
        let v = b.eval(x);
        assert!(rel_close(v, x * x + 1.0, 1e-12), "eval(1+eps) = {v}");
    }

    #[test]
    fn query_close_to_a_node_with_large_ordinates_does_not_overflow() {
        let b = BarycentricLagrange::new(&[0.0, 1.0, 2.0], &[1e308, 1e308, 1e308]).unwrap();
        for x in [1e-2, 1e-6, 1e-300]
        {
            let v = b.eval(x);
            assert!(rel_close(v, 1e308, 1e-12), "eval({x:e}) = {v}");
        }
    }

    #[test]
    fn node_span_beyond_f64_max_still_interpolates() {
        // p(x) = 2 + x / 1e308 through (-1e308, 1), (0, 2), (1e308, 3): the
        // span 2e308 overflows f64, but every node and value is finite.
        let b = BarycentricLagrange::new(&[-1e308, 0.0, 1e308], &[1.0, 2.0, 3.0]).unwrap();
        for (x, want) in [(5e307, 2.5), (-5e307, 1.5), (1.0, 2.0), (1.5e308, 3.5)]
        {
            let v = b.eval(x);
            assert!(rel_close(v, want, 1e-12), "eval({x:e}) = {v}, want {want}");
        }
        // Queries beyond the span on the far side of every node too.
        let v = b.eval(-f64::MAX);
        assert!(v.is_finite(), "eval(-MAX) = {v}");
    }

    #[test]
    fn non_finite_query_is_nan() {
        let b = BarycentricLagrange::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert!(b.eval(f64::NAN).is_nan());
        assert!(b.eval(f64::INFINITY).is_nan());
    }
}
