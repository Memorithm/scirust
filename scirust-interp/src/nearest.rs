//! Nearest-neighbour interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, validate_nodes};

/// Piecewise-constant nearest-neighbour interpolant.
///
/// Each query returns the ordinate of the closest node. Ties (a query exactly
/// halfway between two nodes) resolve to the **lower-indexed (left)** node.
/// The closest node is decided exactly: the query is compared with the true
/// midpoint of its bracketing nodes, so the choice is never flipped by rounding
/// in the two distances, at any scale up to node spans beyond `f64::MAX`.
/// A NaN query returns NaN, like every other method in this crate.
/// This is the only method that accepts a **single** node — in which case every
/// query returns that node's value.
///
/// **Extrapolation** returns the nearest endpoint's value, which is the natural
/// consequence of the nearest-node rule.
#[derive(Debug, Clone)]
pub struct NearestNeighbor {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl NearestNeighbor {
    /// Build a nearest-neighbour interpolant.
    ///
    /// Requires at least **one** node with finite `xs` (strictly increasing
    /// when more than one) and finite `ys` of matching length; otherwise
    /// returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 1)?;
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
        })
    }
}

impl Interpolator for NearestNeighbor {
    fn eval(&self, x: f64) -> f64 {
        if x.is_nan()
        {
            return f64::NAN;
        }
        let n = self.xs.len();
        if n == 1 || x <= self.xs[0]
        {
            return self.ys[0];
        }
        if x >= self.xs[n - 1]
        {
            return self.ys[n - 1];
        }
        let i = find_segment(&self.xs, x);
        if closer_to_left(self.xs[i], self.xs[i + 1], x)
        {
            self.ys[i]
        }
        else
        {
            self.ys[i + 1]
        }
    }
}

/// Whether `x` (with `a < x < b`) is at least as close to `a` as to `b`.
///
/// Comparing the rounded distances `x - a` and `b - x` misclassifies queries
/// within an ulp or so of the midpoint whenever those subtractions round (for
/// example `a = -3.3553304373906823`, `b = 6.762933338820778`,
/// `x = 1.7038014507150483` rounds both distances to the same value although
/// `x` is strictly closer to `b`), and overflows for spans beyond `f64::MAX`.
/// Instead `2x` is compared with `a + b`, using the exact rounding error of
/// the sum to break the remaining tie. When `a + b` overflows (both nodes
/// large with the same sign) all three values are halved first, which is
/// exact for such large magnitudes.
fn closer_to_left(a: f64, b: f64, x: f64) -> bool {
    let (a, b, x) = if (a + b).is_finite()
    {
        (a, b, x)
    }
    else
    {
        (0.5 * a, 0.5 * b, 0.5 * x)
    };
    // Two-sum: `s + e == a + b` exactly.
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    // `2x` is exact unless it overflows, which only happens when the true
    // `|2x|` exceeds `|a + b|`, so the ordering below is still correct.
    let t = 2.0 * x;
    if t != s
    {
        return t < s;
    }
    // `t == s`: the true midpoint is `(s + e) / 2`; `e >= 0` puts `x` on or
    // left of it (exact ties go to the left node).
    e >= 0.0
}
