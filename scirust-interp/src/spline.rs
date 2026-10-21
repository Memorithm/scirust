//! Natural and clamped cubic-spline interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{
    abscissa_exponent, find_segment, ldexp, ordinate_exponent, piecewise_integral, scaled_widths,
    thomas, unit_offsets, validate_nodes,
};

/// A C² cubic spline through the given nodes.
///
/// The spline is built by solving a tridiagonal system (via the Thomas
/// algorithm) for the second derivatives — the *moments* — at each node, which
/// makes the interpolant twice continuously differentiable across interior
/// nodes. Two boundary conditions are offered:
///
/// * [`CubicSpline::natural`] sets the second derivative to zero at both ends.
/// * [`CubicSpline::clamped`] fixes the first derivative at both ends.
///
/// **Extrapolation** continues the boundary cubic piece: queries outside the
/// node range are evaluated with the polynomial of the nearest end segment.
///
/// **Exactness and scale.** [`eval`](Interpolator::eval) returns every node's
/// ordinate exactly (bit for bit) when queried at that node. Internally the
/// abscissae and ordinates are rescaled by powers of two before the moment
/// system is solved, so the interpolant, its derivatives and its integral
/// stay finite whenever the true values are representable, including node
/// spacings far below `1e-154`, ordinates near `f64::MAX`, and node spans
/// wider than `f64::MAX`.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives (moments) at each node, in the caller's units.
    m: Vec<f64>,
    /// Moments in scaled units: `m[i] = mu[i] * 2^(ey - 2 ex)`.
    mu: Vec<f64>,
    /// Segment widths in scaled units: `h[i] = hn[i] * 2^ex`.
    hn: Vec<f64>,
    /// Binary exponent of the abscissa scale.
    ex: i32,
    /// Binary exponent of the ordinate scale.
    ey: i32,
}

impl CubicSpline {
    /// Build a natural cubic spline (`s''(x_0) = s''(x_{n-1}) = 0`).
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn natural(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        Ok(Self::build(xs, ys, BoundaryCondition::Natural))
    }

    /// Build a clamped cubic spline with prescribed end slopes.
    ///
    /// `dy0` is the required first derivative `s'(x_0)` and `dy_n` the required
    /// `s'(x_{n-1})`. Requires at least two nodes with strictly increasing,
    /// finite `xs` and finite `ys` of matching length; `dy0` and `dy_n` must be
    /// finite. Returns [`InterpError`] on any violation.
    pub fn clamped(xs: &[f64], ys: &[f64], dy0: f64, dy_n: f64) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        if !dy0.is_finite()
        {
            return Err(InterpError::NonFinite { index: 0 });
        }
        if !dy_n.is_finite()
        {
            return Err(InterpError::NonFinite {
                index: xs.len() - 1,
            });
        }
        Ok(Self::build(
            xs,
            ys,
            BoundaryCondition::Clamped { dy0, dy_n },
        ))
    }

    /// Rescale the validated nodes, solve for the moments and assemble.
    fn build(xs: &[f64], ys: &[f64], bc: BoundaryCondition) -> Self {
        let ex = abscissa_exponent(xs);
        let ey = ordinate_exponent(ys);
        let hn = scaled_widths(xs, ex);
        let yn: Vec<f64> = ys.iter().map(|&y| ldexp(y, -ey)).collect();
        // Clamped end slopes in scaled units (dy/dx scales by 2^(ex - ey)).
        let bc = match bc
        {
            BoundaryCondition::Natural => BoundaryCondition::Natural,
            BoundaryCondition::Clamped { dy0, dy_n } => BoundaryCondition::Clamped {
                dy0: ldexp(dy0, ex - ey),
                dy_n: ldexp(dy_n, ex - ey),
            },
        };
        let mu = solve_moments(&hn, &yn, bc);
        let m = mu.iter().map(|&v| ldexp(v, ey - 2 * ex)).collect();
        Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            m,
            mu,
            hn,
            ex,
            ey,
        }
    }

    /// The computed second derivatives (moments) at each node.
    ///
    /// Exposed mainly so callers can verify boundary behaviour (a natural
    /// spline has `moments()[0] == moments()[n - 1] == 0`). A moment whose
    /// magnitude exceeds `f64::MAX` (possible only for extreme ratios of
    /// ordinate scale to squared node spacing) is reported as `±inf`, while
    /// [`eval`](Interpolator::eval) and the calculus methods stay finite.
    pub fn moments(&self) -> &[f64] {
        &self.m
    }

    /// First derivative of the spline at `x`.
    ///
    /// Continuous everywhere (the spline is C²); outside the node range the
    /// boundary cubic is differentiated. A NaN query returns NaN.
    pub fn derivative(&self, x: f64) -> f64 {
        let (i, u, t) = self.local(x);
        let hn = self.hn[i];
        let (y0, y1) = (ldexp(self.ys[i], -self.ey), ldexp(self.ys[i + 1], -self.ey));
        let (m0, m1) = (self.mu[i], self.mu[i + 1]);
        let dn = (y1 - y0) / hn + hn / 6.0 * (m1 * (3.0 * t * t - 1.0) - m0 * (3.0 * u * u - 1.0));
        ldexp(dn, self.ey - self.ex)
    }

    /// Second derivative of the spline at `x`.
    ///
    /// Piecewise linear between the [`moments`](Self::moments) and equal to
    /// them at the nodes; outside the node range the boundary cubic is
    /// differentiated. A NaN query returns NaN.
    pub fn second_derivative(&self, x: f64) -> f64 {
        let (i, u, t) = self.local(x);
        ldexp(self.mu[i] * u + self.mu[i + 1] * t, self.ey - 2 * self.ex)
    }

    /// Exact definite integral of the spline from `a` to `b`.
    ///
    /// Integrates each cubic piece in closed form, including the extrapolated
    /// boundary cubics when a bound lies outside the node range. Reversed
    /// bounds flip the sign, equal bounds give `0`, and a non-finite bound
    /// gives NaN.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        piecewise_integral(&self.xs, a, b, |i, lo, hi| {
            let hn = self.hn[i];
            let (y0, y1) = (ldexp(self.ys[i], -self.ey), ldexp(self.ys[i + 1], -self.ey));
            let (m0, m1) = (self.mu[i], self.mu[i + 1]);
            // Antiderivative in the unit variable t (with u = 1 - t).
            let anti = |x: f64| {
                let (u, t) = self.weights(i, x);
                let (u2, t2) = (u * u, t * t);
                0.5 * (t2 * y1 - u2 * y0)
                    + hn * hn / 6.0
                        * (m1 * (0.25 * t2 * t2 - 0.5 * t2) - m0 * (0.25 * u2 * u2 - 0.5 * u2))
            };
            ldexp(hn * (anti(hi) - anti(lo)), self.ex + self.ey)
        })
    }

    /// Segment index and the normalised offsets `(u, t)` for `x`.
    fn local(&self, x: f64) -> (usize, f64, f64) {
        let i = find_segment(&self.xs, x);
        let (u, t) = self.weights(i, x);
        (i, u, t)
    }

    /// Normalised offsets `(u, t)` of `x` in segment `i` (see
    /// [`unit_offsets`]).
    fn weights(&self, i: usize, x: f64) -> (f64, f64) {
        unit_offsets(self.xs[i], self.xs[i + 1], x)
    }
}

/// Which boundary condition to impose when solving for the moments.
enum BoundaryCondition {
    /// Zero curvature at both ends.
    Natural,
    /// Prescribed first derivatives at both ends.
    Clamped { dy0: f64, dy_n: f64 },
}

/// Assemble and solve the tridiagonal moment system for the chosen boundary.
///
/// `h` holds the segment widths and `ys` the ordinates, both already scaled
/// to moderate magnitudes by the caller.
fn solve_moments(h: &[f64], ys: &[f64], bc: BoundaryCondition) -> Vec<f64> {
    let n = ys.len();
    if n == 2
    {
        // A single segment: handle boundaries directly to keep the solver
        // uniform. Both boundary conditions reduce to a 2x2 system.
        return solve_moments_two(h[0], ys, bc);
    }
    let slope: Vec<f64> = (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    for i in 1..n - 1
    {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * (slope[i] - slope[i - 1]);
    }

    match bc
    {
        BoundaryCondition::Natural =>
        {
            b[0] = 1.0;
            c[0] = 0.0;
            d[0] = 0.0;
            a[n - 1] = 0.0;
            b[n - 1] = 1.0;
            d[n - 1] = 0.0;
        },
        BoundaryCondition::Clamped { dy0, dy_n } =>
        {
            b[0] = 2.0 * h[0];
            c[0] = h[0];
            d[0] = 6.0 * (slope[0] - dy0);
            a[n - 1] = h[n - 2];
            b[n - 1] = 2.0 * h[n - 2];
            d[n - 1] = 6.0 * (dy_n - slope[n - 2]);
        },
    }

    thomas(&a, &b, &c, &d)
}

/// Closed-form moments for the two-node special case (segment width `h`).
fn solve_moments_two(h: f64, ys: &[f64], bc: BoundaryCondition) -> Vec<f64> {
    match bc
    {
        BoundaryCondition::Natural => vec![0.0, 0.0],
        BoundaryCondition::Clamped { dy0, dy_n } =>
        {
            let slope = (ys[1] - ys[0]) / h;
            // 2x2 system:
            //   2h*M0 +  h*M1 = 6(slope - dy0)
            //    h*M0 + 2h*M1 = 6(dy_n - slope)
            let r0 = 6.0 * (slope - dy0);
            let r1 = 6.0 * (dy_n - slope);
            let det = (2.0 * h) * (2.0 * h) - h * h; // 3h^2
            let m0 = (r0 * (2.0 * h) - h * r1) / det;
            let m1 = ((2.0 * h) * r1 - h * r0) / det;
            vec![m0, m1]
        },
    }
}

impl Interpolator for CubicSpline {
    fn eval(&self, x: f64) -> f64 {
        let (i, u, t) = self.local(x);
        let hn = self.hn[i];
        let (m0, m1) = (self.mu[i], self.mu[i + 1]);
        // Moment form of the cubic written in the normalised offsets: the
        // linear part reproduces the node ordinates exactly (u, t ∈ {0, 1}
        // there) and the curvature term vanishes exactly at both nodes.
        let curvature = hn * hn / 6.0 * (m0 * u * (u * u - 1.0) + m1 * t * (t * t - 1.0));
        u * self.ys[i] + t * self.ys[i + 1] + ldexp(curvature, self.ey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift generator in `[0, 1)`.
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn close(got: f64, want: f64, scale: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(scale)
    }

    const XS: [f64; 6] = [0.0, 0.3, 1.1, 1.7, 2.9, 3.4];
    const YS: [f64; 6] = [1.0, -0.5, 2.0, 0.25, 1.5, -1.0];

    fn both(xs: &[f64], ys: &[f64], d0: f64, dn: f64) -> [CubicSpline; 2] {
        [
            CubicSpline::natural(xs, ys).unwrap(),
            CubicSpline::clamped(xs, ys, d0, dn).unwrap(),
        ]
    }

    #[test]
    fn eval_reproduces_every_node_bit_for_bit() {
        // Before the fix the moment form `(y/h - m h/6) * h` rounded, so
        // about a third of node queries missed the ordinate by an ulp
        // (e.g. 0.3 -> 0.29999999999999993).
        let mut r = rng(12345);
        for _ in 0..500
        {
            let mut xs = [0.0; 6];
            let mut x = r();
            for slot in &mut xs
            {
                *slot = x;
                x += 0.01 + r();
            }
            let ys: Vec<f64> = (0..6).map(|_| 10.0 * r() - 5.0).collect();
            for s in both(&xs, &ys, r() - 0.5, r() - 0.5)
            {
                for (&x, &y) in xs.iter().zip(&ys)
                {
                    assert_eq!(s.eval(x), y, "node {x} not reproduced");
                }
            }
        }
        let s = CubicSpline::natural(&[0.0, 0.1, 0.3, 0.7], &[0.1, 0.7, 0.3, 0.9]).unwrap();
        assert_eq!(s.eval(0.3), 0.3);
        assert_eq!(s.eval(0.7), 0.9);
    }

    #[test]
    fn small_node_spacing_stays_finite_and_scale_invariant() {
        // Before the fix `y / h` and the moments (~ y / h²) overflowed for
        // node spacings below ~1e-154, so eval, derivative and integrate
        // returned NaN.
        let unit = both(&XS, &YS, 0.7, -1.2);
        for k in [1e-160, 1e-200, 1e-300]
        {
            let xs: Vec<f64> = XS.iter().map(|&x| x * k).collect();
            let scaled = both(&xs, &YS, 0.7 / k, -1.2 / k);
            for (u, s) in unit.iter().zip(&scaled)
            {
                for j in 0..=40
                {
                    let x = -0.3 + 0.1 * j as f64;
                    assert!(close(s.eval(x * k), u.eval(x), 1.0, 1e-12), "k={k:e} x={x}");
                    // Derivatives up to ~1e300 are representable here.
                    if k >= 1e-200
                    {
                        let d = s.derivative(x * k) * k;
                        assert!(close(d, u.derivative(x), 1.0, 1e-11), "k={k:e} x={x}: {d}");
                    }
                }
                let i = s.integrate(-0.2 * k, 3.6 * k) / k;
                assert!(close(i, u.integrate(-0.2, 3.6), 1.0, 1e-12), "k={k:e}: {i}");
            }
        }
    }

    #[test]
    fn large_node_spacing_keeps_its_curvature() {
        // Before the fix the moments (~ y / h²) underflowed to zero for node
        // spacings above ~1e154 and the spline silently degraded to linear
        // interpolation; spans beyond f64::MAX gave NaN.
        let unit = both(&XS, &YS, 0.7, -1.2);
        for k in [1e160, 1e200, 1e300]
        {
            let xs: Vec<f64> = XS.iter().map(|&x| x * k).collect();
            let scaled = both(&xs, &YS, 0.7 / k, -1.2 / k);
            for (u, s) in unit.iter().zip(&scaled)
            {
                for j in 0..=40
                {
                    let x = -0.3 + 0.1 * j as f64;
                    assert!(close(s.eval(x * k), u.eval(x), 1.0, 1e-12), "k={k:e} x={x}");
                    let d = s.derivative(x * k) * k;
                    assert!(close(d, u.derivative(x), 1.0, 1e-11), "k={k:e} x={x}: {d}");
                }
            }
        }
        // Node span wider than f64::MAX.
        let xs = [-1.5e308, -0.5e308, 0.5e308, 1.5e308];
        let ys = [1.0, 2.0, 0.0, 1.0];
        let s = CubicSpline::natural(&xs, &ys).unwrap();
        let u = CubicSpline::natural(&[-1.5, -0.5, 0.5, 1.5], &ys).unwrap();
        for j in 0..=30
        {
            let x = -1.5 + 0.1 * j as f64;
            assert!(close(s.eval(x * 1e308), u.eval(x), 1.0, 1e-12), "x={x}");
        }
    }

    #[test]
    fn ordinates_near_f64_max_stay_finite() {
        // Before the fix y / h and the secant differences overflowed, giving
        // NaN between nodes of alternating sign near f64::MAX.
        let xs = [0.0, 1.0, 2.0, 3.0];
        let unit = CubicSpline::natural(&xs, &[-1.0, 1.0, -1.0, 1.0]).unwrap();
        let big = CubicSpline::natural(&xs, &[-1e308, 1e308, -1e308, 1e308]).unwrap();
        for j in 0..=30
        {
            let x = 0.1 * j as f64;
            let got = big.eval(x);
            assert!(got.is_finite(), "x={x}: {got}");
            assert!(
                close(got, 1e308 * unit.eval(x), 1e308, 1e-12),
                "x={x}: {got:e}"
            );
            // Only compare slopes whose true value is representable.
            if unit.derivative(x).abs() < 1.7
            {
                let d = big.derivative(x);
                assert!(
                    close(d, 1e308 * unit.derivative(x), 1e308, 1e-12),
                    "x={x}: {d:e}"
                );
            }
        }
    }

    #[test]
    fn moments_and_second_derivative_match_unscaled_values() {
        let s = CubicSpline::natural(&XS, &YS).unwrap();
        assert_eq!(s.moments()[0], 0.0);
        assert_eq!(s.moments()[XS.len() - 1], 0.0);
        for (k, &x) in XS.iter().enumerate()
        {
            assert!(close(s.second_derivative(x), s.moments()[k], 1.0, 1e-12));
        }
    }
}
