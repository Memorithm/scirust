//! # `scirust-interp` — deterministic 1-D interpolation
//!
//! Pure-Rust, dependency-free, `#![forbid(unsafe_code)]` interpolation of
//! one-dimensional data. Every method is exposed behind the common
//! [`Interpolator`] trait and built from a validated set of nodes.
//!
//! ## Methods
//!
//! | Type | Kind | Min. points | Extrapolation |
//! |------|------|-------------|---------------|
//! | [`LinearInterp`] | piecewise linear | 2 | linear |
//! | [`CubicSpline`] | C² natural / clamped spline | 2 | boundary cubic |
//! | [`PchipInterp`] | monotone cubic Hermite | 2 | boundary cubic |
//! | [`AkimaSpline`] | Akima cubic | 5 | boundary cubic |
//! | [`BarycentricLagrange`] | global polynomial | 2 | polynomial |
//! | [`NearestNeighbor`] | piecewise constant | 1 | nearest endpoint |
//!
//! ## Calculus
//!
//! The piecewise methods ([`LinearInterp`], [`CubicSpline`], [`PchipInterp`],
//! [`AkimaSpline`]) also expose `derivative(x)` and an exact closed-form
//! `integrate(a, b)`; the cubic ones add `second_derivative(x)`. Both follow
//! the same boundary-piece extrapolation as `eval`.
//!
//! ```
//! use scirust_interp::CubicSpline;
//!
//! // A clamped spline with exact end slopes reproduces x³ on [0, 2].
//! let xs = [0.0, 0.5, 1.0, 1.5, 2.0];
//! let ys: Vec<f64> = xs.iter().map(|x| x * x * x).collect();
//! let s = CubicSpline::clamped(&xs, &ys, 0.0, 12.0).unwrap();
//! assert!((s.derivative(1.2) - 3.0 * 1.2 * 1.2).abs() < 1e-9);
//! assert!((s.integrate(0.0, 2.0) - 4.0).abs() < 1e-9);
//! ```
//!
//! ## Guarantees
//!
//! - **Deterministic**: no global state, no RNG — identical inputs give
//!   identical outputs everywhere.
//! - **Validated**: every multi-point constructor rejects length mismatches,
//!   non-finite values, and non-strictly-increasing abscissae, returning an
//!   [`InterpError`] instead of panicking.
//!
//! ## Example
//!
//! ```
//! use scirust_interp::{Interpolator, LinearInterp, CubicSpline};
//!
//! let xs = [0.0, 1.0, 2.0, 3.0];
//! let ys = [0.0, 1.0, 4.0, 9.0];
//!
//! // Piecewise-linear interpolation.
//! let lin = LinearInterp::new(&xs, &ys).unwrap();
//! assert!((lin.eval(1.5) - 2.5).abs() < 1e-12);
//!
//! // A natural cubic spline still passes through every node.
//! let spline = CubicSpline::natural(&xs, &ys).unwrap();
//! assert!((spline.eval(2.0) - 4.0).abs() < 1e-12);
//!
//! // Batch evaluation.
//! let out = lin.eval_all(&[0.0, 1.0, 2.0, 3.0]);
//! assert_eq!(out, vec![0.0, 1.0, 4.0, 9.0]);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod akima;
mod barycentric;
mod error;
mod linear;
mod nearest;
mod pchip;
mod spline;
mod traits;
mod util;

pub use akima::AkimaSpline;
pub use barycentric::BarycentricLagrange;
pub use error::InterpError;
pub use linear::LinearInterp;
pub use nearest::NearestNeighbor;
pub use pchip::PchipInterp;
pub use spline::CubicSpline;
pub use traits::Interpolator;

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference cubic and its derivative, used for exactness tests.
    fn cubic(x: f64) -> f64 {
        2.0 * x * x * x - 3.0 * x * x + x - 5.0
    }
    fn cubic_prime(x: f64) -> f64 {
        6.0 * x * x - 6.0 * x + 1.0
    }

    // ---------------------------------------------------------------- //
    //  Node exactness (all methods pass through their nodes).          //
    // ---------------------------------------------------------------- //

    #[test]
    fn node_exactness_all_methods() {
        let xs = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0];
        let ys = [1.0, -2.0, 3.5, 0.25, 6.0, -1.5];

        let interps: Vec<Box<dyn Interpolator>> = vec![
            Box::new(LinearInterp::new(&xs, &ys).unwrap()),
            Box::new(CubicSpline::natural(&xs, &ys).unwrap()),
            Box::new(CubicSpline::clamped(&xs, &ys, 0.0, 0.0).unwrap()),
            Box::new(PchipInterp::new(&xs, &ys).unwrap()),
            Box::new(AkimaSpline::new(&xs, &ys).unwrap()),
            Box::new(BarycentricLagrange::new(&xs, &ys).unwrap()),
            Box::new(NearestNeighbor::new(&xs, &ys).unwrap()),
        ];

        for interp in &interps
        {
            for (&x, &y) in xs.iter().zip(ys.iter())
            {
                assert!(
                    (interp.eval(x) - y).abs() < 1e-12,
                    "node ({x}, {y}) not reproduced: got {}",
                    interp.eval(x)
                );
            }
        }
    }

    // ---------------------------------------------------------------- //
    //  Linear reproduces an affine function exactly.                   //
    // ---------------------------------------------------------------- //

    #[test]
    fn linear_reproduces_affine() {
        let affine = |x: f64| 3.0 * x - 7.0;
        let xs = [-2.0, 0.0, 1.0, 4.0, 9.0];
        let ys: Vec<f64> = xs.iter().map(|&x| affine(x)).collect();
        let lin = LinearInterp::new(&xs, &ys).unwrap();

        // Interior, node, and extrapolated points all match exactly.
        for &x in &[-5.0, -2.0, -1.3, 0.5, 4.0, 6.7, 12.0]
        {
            assert!((lin.eval(x) - affine(x)).abs() < 1e-12);
        }
    }

    // ---------------------------------------------------------------- //
    //  Clamped spline & barycentric reproduce a cubic.                 //
    // ---------------------------------------------------------------- //

    #[test]
    fn clamped_spline_reproduces_cubic() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| cubic(x)).collect();
        let n = xs.len();
        let spline =
            CubicSpline::clamped(&xs, &ys, cubic_prime(xs[0]), cubic_prime(xs[n - 1])).unwrap();

        // Sample densely between the nodes: a clamped spline reproduces the
        // cubic exactly (up to rounding).
        for i in 0..=400
        {
            let x = 4.0 * f64::from(i) / 400.0;
            assert!(
                (spline.eval(x) - cubic(x)).abs() < 1e-9,
                "clamped spline off cubic at x={x}"
            );
        }
    }

    #[test]
    fn barycentric_reproduces_cubic() {
        // Five nodes on a cubic → the degree-4 interpolant IS the cubic.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| cubic(x)).collect();
        let bary = BarycentricLagrange::new(&xs, &ys).unwrap();
        for i in 0..=400
        {
            let x = 4.0 * f64::from(i) / 400.0;
            assert!(
                (bary.eval(x) - cubic(x)).abs() < 1e-9,
                "barycentric off cubic at x={x}"
            );
        }
    }

    // ---------------------------------------------------------------- //
    //  Barycentric matches a hand-computed Lagrange value.             //
    // ---------------------------------------------------------------- //

    #[test]
    fn barycentric_matches_known_lagrange() {
        // Quadratic through (0,1),(1,3),(2,7) is p(x) = x^2 + x + 1.
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let bary = BarycentricLagrange::new(&xs, &ys).unwrap();
        assert!((bary.eval(0.5) - 1.75).abs() < 1e-12); // p(0.5) = 1.75
        assert!((bary.eval(1.5) - 4.75).abs() < 1e-12); // p(1.5) = 4.75
        // And it reproduces the quadratic elsewhere.
        for &x in &[-1.0, 0.3, 2.5, 3.0]
        {
            let p = x * x + x + 1.0;
            assert!((bary.eval(x) - p).abs() < 1e-10);
        }
    }

    // ---------------------------------------------------------------- //
    //  PCHIP preserves monotonicity (no overshoot).                    //
    // ---------------------------------------------------------------- //

    #[test]
    fn pchip_monotone_no_overshoot() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.0, 0.3, 0.35, 0.9, 3.0, 10.0]; // strictly increasing
        let pchip = PchipInterp::new(&xs, &ys).unwrap();

        let mut prev = f64::NEG_INFINITY;
        let n_samples = 5000;
        for i in 0..=n_samples
        {
            let x = 5.0 * f64::from(i) / f64::from(n_samples);
            let v = pchip.eval(x);
            // Monotone non-decreasing along the sweep.
            assert!(v >= prev - 1e-12, "monotonicity broken at x={x}");
            // No overshoot outside the global data range.
            assert!(
                v >= ys[0] - 1e-12 && v <= ys[5] + 1e-12,
                "overshoot at x={x}"
            );
            prev = v;
        }
    }

    #[test]
    fn pchip_no_local_overshoot_between_nodes() {
        // Each segment must stay within its bracketing ordinate values.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.0, 2.0, 2.1, 2.15, 8.0, 8.1];
        let pchip = PchipInterp::new(&xs, &ys).unwrap();
        for seg in 0..xs.len() - 1
        {
            let (lo, hi) = (ys[seg].min(ys[seg + 1]), ys[seg].max(ys[seg + 1]));
            for j in 0..=50
            {
                let x = xs[seg] + (xs[seg + 1] - xs[seg]) * f64::from(j) / 50.0;
                let v = pchip.eval(x);
                assert!(
                    v >= lo - 1e-12 && v <= hi + 1e-12,
                    "segment overshoot at x={x}"
                );
            }
        }
    }

    // ---------------------------------------------------------------- //
    //  Natural spline: zero second derivative at the endpoints.        //
    // ---------------------------------------------------------------- //

    #[test]
    fn natural_spline_zero_curvature_ends() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.0, 0.0, 2.0, -1.0, 3.0];
        let spline = CubicSpline::natural(&xs, &ys).unwrap();
        let m = spline.moments();
        let n = xs.len();

        // The computed moments (second derivatives) vanish at both ends.
        assert!(m[0].abs() < 1e-12);
        assert!(m[n - 1].abs() < 1e-12);

        // Cross-check with a finite-difference second derivative just inside
        // the left endpoint. On segment 0, s''(x) ramps linearly from m[0]=0
        // at x0 to m[1] at x1, so s''(x0+e) = m[1]*e/h.
        let e = 1e-3;
        let h = xs[1] - xs[0];
        let spp = (spline.eval(xs[0]) - 2.0 * spline.eval(xs[0] + e)
            + spline.eval(xs[0] + 2.0 * e))
            / (e * e);
        assert!((spp - m[1] * e / h).abs() < 1e-6);
    }

    #[test]
    fn spline_c2_continuity_interior() {
        // A centered second difference straddling each interior node converges
        // to the shared moment M_k, confirming s'' is continuous there (C²).
        let xs = [0.0, 1.3, 2.1, 3.8, 5.0];
        let ys = [0.5, -1.0, 2.0, 0.0, 4.0];
        let spline = CubicSpline::natural(&xs, &ys).unwrap();
        let m = spline.moments();
        let e = 1e-4;
        for k in 1..xs.len() - 1
        {
            let xk = xs[k];
            let spp = (spline.eval(xk + e) - 2.0 * spline.eval(xk) + spline.eval(xk - e)) / (e * e);
            assert!(
                (spp - m[k]).abs() < 1e-3,
                "s'' mismatch across node {k}: fd={spp}, moment={}",
                m[k]
            );
        }
    }

    // ---------------------------------------------------------------- //
    //  Akima specifics.                                                //
    // ---------------------------------------------------------------- //

    #[test]
    fn akima_reproduces_affine() {
        let affine = |x: f64| -2.0 * x + 4.0;
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|&x| affine(x)).collect();
        let ak = AkimaSpline::new(&xs, &ys).unwrap();
        for &x in &[0.25, 1.7, 2.5, 3.9, 4.4]
        {
            assert!((ak.eval(x) - affine(x)).abs() < 1e-12);
        }
    }

    // ---------------------------------------------------------------- //
    //  Nearest-neighbour behaviour, incl. single-point support.        //
    // ---------------------------------------------------------------- //

    #[test]
    fn nearest_single_point() {
        let nn = NearestNeighbor::new(&[3.0], &[42.0]).unwrap();
        assert_eq!(nn.eval(-100.0), 42.0);
        assert_eq!(nn.eval(3.0), 42.0);
        assert_eq!(nn.eval(1000.0), 42.0);
    }

    #[test]
    fn nearest_rounding_and_ties() {
        let xs = [0.0, 2.0, 4.0];
        let ys = [10.0, 20.0, 30.0];
        let nn = NearestNeighbor::new(&xs, &ys).unwrap();
        assert_eq!(nn.eval(0.4), 10.0);
        assert_eq!(nn.eval(1.6), 20.0);
        assert_eq!(nn.eval(1.0), 10.0); // exact midpoint → left node
        assert_eq!(nn.eval(3.0), 20.0); // midpoint of [2,4] → left node
        assert_eq!(nn.eval(-5.0), 10.0); // extrapolation → nearest endpoint
        assert_eq!(nn.eval(9.0), 30.0);
    }

    // ---------------------------------------------------------------- //
    //  eval_all default matches per-point eval.                        //
    // ---------------------------------------------------------------- //

    #[test]
    fn eval_all_matches_eval() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 0.0, 5.0];
        let spline = CubicSpline::natural(&xs, &ys).unwrap();
        let query = [0.5, 1.5, 2.5, 3.5, -0.5];
        let batch = spline.eval_all(&query);
        for (i, &x) in query.iter().enumerate()
        {
            assert!((batch[i] - spline.eval(x)).abs() < 1e-15);
        }
    }

    // ---------------------------------------------------------------- //
    //  Error paths for every constructor.                              //
    // ---------------------------------------------------------------- //

    #[test]
    fn err_length_mismatch() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0];
        assert_eq!(
            LinearInterp::new(&xs, &ys).unwrap_err(),
            InterpError::LengthMismatch { xs: 3, ys: 2 }
        );
        assert!(CubicSpline::natural(&xs, &ys).is_err());
        assert!(CubicSpline::clamped(&xs, &ys, 0.0, 0.0).is_err());
        assert!(PchipInterp::new(&xs, &ys).is_err());
        assert!(BarycentricLagrange::new(&xs, &ys).is_err());
        assert!(NearestNeighbor::new(&xs, &ys).is_err());
        let xs5 = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys4 = [0.0, 1.0, 2.0, 3.0];
        assert!(AkimaSpline::new(&xs5, &ys4).is_err());
    }

    #[test]
    fn err_unsorted() {
        let xs = [0.0, 2.0, 1.0, 3.0];
        let ys = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(
            LinearInterp::new(&xs, &ys).unwrap_err(),
            InterpError::NotStrictlyIncreasing { index: 2 }
        );
        assert!(CubicSpline::natural(&xs, &ys).is_err());
        assert!(PchipInterp::new(&xs, &ys).is_err());
        assert!(BarycentricLagrange::new(&xs, &ys).is_err());
        assert!(NearestNeighbor::new(&xs, &ys).is_err());
    }

    #[test]
    fn err_duplicate_x() {
        let xs = [0.0, 1.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(
            LinearInterp::new(&xs, &ys).unwrap_err(),
            InterpError::NotStrictlyIncreasing { index: 2 }
        );
        assert!(CubicSpline::clamped(&xs, &ys, 0.0, 0.0).is_err());
        assert!(PchipInterp::new(&xs, &ys).is_err());
    }

    #[test]
    fn err_too_few_points() {
        // Linear / spline / pchip / barycentric need >= 2.
        assert_eq!(
            LinearInterp::new(&[1.0], &[1.0]).unwrap_err(),
            InterpError::TooFewPoints { got: 1, need: 2 }
        );
        assert!(CubicSpline::natural(&[1.0], &[1.0]).is_err());
        assert!(PchipInterp::new(&[1.0], &[1.0]).is_err());
        assert!(BarycentricLagrange::new(&[1.0], &[1.0]).is_err());
        // Akima needs >= 5.
        let xs4 = [0.0, 1.0, 2.0, 3.0];
        let ys4 = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(
            AkimaSpline::new(&xs4, &ys4).unwrap_err(),
            InterpError::TooFewPoints { got: 4, need: 5 }
        );
        // Nearest needs >= 1.
        assert_eq!(
            NearestNeighbor::new(&[], &[]).unwrap_err(),
            InterpError::TooFewPoints { got: 0, need: 1 }
        );
    }

    #[test]
    fn err_non_finite() {
        let xs = [0.0, 1.0, f64::NAN, 3.0];
        let ys = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(
            LinearInterp::new(&xs, &ys).unwrap_err(),
            InterpError::NonFinite { index: 2 }
        );
        // NaN / infinity in ys is rejected too.
        let xs2 = [0.0, 1.0, 2.0, 3.0];
        let ys2 = [0.0, f64::INFINITY, 2.0, 3.0];
        assert_eq!(
            PchipInterp::new(&xs2, &ys2).unwrap_err(),
            InterpError::NonFinite { index: 1 }
        );
        // Non-finite clamp derivative is rejected.
        assert!(CubicSpline::clamped(&xs2, &[0.0, 1.0, 2.0, 3.0], f64::NAN, 0.0).is_err());
    }

    #[test]
    fn error_display_is_nonempty() {
        let e = InterpError::TooFewPoints { got: 1, need: 2 };
        assert!(!format!("{e}").is_empty());
        // Exercise the std::error::Error impl.
        let _: &dyn std::error::Error = &e;
    }

    // ---------------------------------------------------------------- //
    //  Calculus: derivatives and exact definite integrals.             //
    // ---------------------------------------------------------------- //

    /// Antiderivative of the reference cubic.
    fn cubic_anti(x: f64) -> f64 {
        0.5 * x.powi(4) - x.powi(3) + 0.5 * x * x - 5.0 * x
    }

    /// A named interpolant paired with its exact integral, for oracle tests.
    type Case<'a> = (&'a str, &'a dyn Interpolator, &'a dyn Fn(f64, f64) -> f64);

    /// Composite Simpson's rule on `eval`, used as an independent oracle.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut s = f(a) + f(b);
        for k in 1..n
        {
            let w = if k % 2 == 1 { 4.0 } else { 2.0 };
            s += w * f(a + h * k as f64);
        }
        s * h / 3.0
    }

    #[test]
    fn clamped_spline_calculus_reproduces_cubic() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| cubic(x)).collect();
        let s = CubicSpline::clamped(&xs, &ys, cubic_prime(0.0), cubic_prime(4.0)).unwrap();
        for k in 0..=80
        {
            let x = -0.5 + 5.0 * f64::from(k) / 80.0;
            assert!((s.derivative(x) - cubic_prime(x)).abs() < 1e-9, "s'({x})");
            assert!(
                (s.second_derivative(x) - (12.0 * x - 6.0)).abs() < 1e-8,
                "s''({x})"
            );
        }
        for &(a, b) in &[(0.0, 4.0), (0.3, 2.7), (1.0, 1.5), (-1.0, 5.0), (3.2, 0.4)]
        {
            let want = cubic_anti(b) - cubic_anti(a);
            assert!(
                (s.integrate(a, b) - want).abs() < 1e-9,
                "∫[{a},{b}] got {} want {want}",
                s.integrate(a, b)
            );
        }
    }

    #[test]
    fn spline_second_derivative_matches_moments() {
        let xs = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0];
        let ys = [1.0, -2.0, 3.5, 0.25, 6.0, -1.5];
        let s = CubicSpline::natural(&xs, &ys).unwrap();
        for (k, &x) in xs.iter().enumerate()
        {
            assert!((s.second_derivative(x) - s.moments()[k]).abs() < 1e-10);
        }
        assert!(s.second_derivative(xs[0]).abs() < 1e-12);
        assert!(s.second_derivative(xs[5]).abs() < 1e-12);
    }

    #[test]
    fn linear_calculus_is_exact() {
        let affine = |x: f64| 3.0 * x - 7.0;
        let xs = [-2.0, 0.0, 1.0, 4.0, 9.0];
        let ys: Vec<f64> = xs.iter().map(|&x| affine(x)).collect();
        let lin = LinearInterp::new(&xs, &ys).unwrap();
        for &x in &[-5.0, -2.0, 0.5, 4.0, 12.0]
        {
            assert!((lin.derivative(x) - 3.0).abs() < 1e-12);
        }
        let anti = |x: f64| 1.5 * x * x - 7.0 * x;
        for &(a, b) in &[(-2.0, 9.0), (-6.0, 11.0), (0.5, 3.25), (9.0, -2.0)]
        {
            assert!((lin.integrate(a, b) - (anti(b) - anti(a))).abs() < 1e-10);
        }
        // Kinked data: trapezoid areas, slope switches at the node.
        let k = LinearInterp::new(&[0.0, 1.0, 3.0], &[0.0, 2.0, 0.0]).unwrap();
        assert!((k.integrate(0.0, 3.0) - 3.0).abs() < 1e-12);
        assert!((k.derivative(0.5) - 2.0).abs() < 1e-12);
        assert!((k.derivative(1.0) + 1.0).abs() < 1e-12);
        assert!(k.derivative(f64::NAN).is_nan());
    }

    #[test]
    fn hermite_methods_derivative_matches_node_slopes_and_fd() {
        let xs = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0];
        let ys = [1.0, -2.0, 3.5, 0.25, 6.0, -1.5];
        let p = PchipInterp::new(&xs, &ys).unwrap();
        let a = AkimaSpline::new(&xs, &ys).unwrap();
        let eps = 1e-6;
        for k in 0..=70
        {
            // Avoid the nodes themselves, where the second derivative jumps.
            let x = -0.45 + 7.9 * f64::from(k) / 70.0 + 1e-3;
            let fd_p = (p.eval(x + eps) - p.eval(x - eps)) / (2.0 * eps);
            let fd_a = (a.eval(x + eps) - a.eval(x - eps)) / (2.0 * eps);
            assert!((p.derivative(x) - fd_p).abs() < 1e-6, "pchip'({x})");
            assert!((a.derivative(x) - fd_a).abs() < 1e-6, "akima'({x})");
            let fd2 = (p.derivative(x + eps) - p.derivative(x - eps)) / (2.0 * eps);
            assert!((p.second_derivative(x) - fd2).abs() < 1e-4, "pchip''({x})");
        }
        // C¹: left and right limits of the derivative agree at interior nodes.
        for &x in &xs[1..5]
        {
            let l = p.derivative(x - 1e-12);
            let r = p.derivative(x);
            assert!((l - r).abs() < 1e-9);
        }
    }

    #[test]
    fn piecewise_integrals_match_simpson_oracle() {
        let xs = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0];
        let ys = [1.0, -2.0, 3.5, 0.25, 6.0, -1.5];
        let lin = LinearInterp::new(&xs, &ys).unwrap();
        let nat = CubicSpline::natural(&xs, &ys).unwrap();
        let p = PchipInterp::new(&xs, &ys).unwrap();
        let a = AkimaSpline::new(&xs, &ys).unwrap();
        let methods: [Case<'_>; 4] = [
            ("linear", &lin, &|u, v| lin.integrate(u, v)),
            ("natural", &nat, &|u, v| nat.integrate(u, v)),
            ("pchip", &p, &|u, v| p.integrate(u, v)),
            ("akima", &a, &|u, v| a.integrate(u, v)),
        ];
        let bounds = [(0.0, 7.0), (0.4, 6.1), (2.5, 5.5), (-1.0, 0.3), (6.2, 8.5)];
        for (name, f, int) in methods
        {
            for &(u, v) in &bounds
            {
                // Split at the nodes so Simpson is exact on every cubic piece.
                let mut cuts = vec![u];
                cuts.extend(xs.iter().copied().filter(|&x| x > u && x < v));
                cuts.push(v);
                let want: f64 = cuts
                    .windows(2)
                    .map(|w| simpson(|x| f.eval(x), w[0], w[1], 2))
                    .sum();
                let got = int(u, v);
                assert!(
                    (got - want).abs() < 1e-9,
                    "{name} ∫[{u},{v}] {got} vs {want}"
                );
                // Reversal antisymmetry.
                assert!((int(v, u) + got).abs() < 1e-12);
            }
            // Additivity across an arbitrary interior split point.
            let whole = int(-0.7, 7.9);
            let parts = int(-0.7, 3.3) + int(3.3, 7.9);
            assert!((whole - parts).abs() < 1e-10, "{name} additivity");
            assert_eq!(int(2.0, 2.0), 0.0);
            assert!(int(0.0, f64::INFINITY).is_nan());
            assert!(int(f64::NAN, 1.0).is_nan());
        }
    }

    #[test]
    fn two_node_calculus() {
        let xs = [1.0, 3.0];
        let ys = [2.0, 6.0];
        let lin = LinearInterp::new(&xs, &ys).unwrap();
        let p = PchipInterp::new(&xs, &ys).unwrap();
        let s = CubicSpline::natural(&xs, &ys).unwrap();
        for (d, i) in [
            (lin.derivative(2.0), lin.integrate(1.0, 3.0)),
            (p.derivative(2.0), p.integrate(1.0, 3.0)),
            (s.derivative(2.0), s.integrate(1.0, 3.0)),
        ]
        {
            assert!((d - 2.0).abs() < 1e-12);
            assert!((i - 8.0).abs() < 1e-12);
        }
    }

    // ---------------------------------------------------------------- //
    //  Scale invariance of the local-slope methods (regression).       //
    // ---------------------------------------------------------------- //

    /// Relative closeness with an absolute floor proportional to `scale`.
    fn close_rel(got: f64, want: f64, scale: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(scale)
    }

    #[test]
    fn pchip_and_akima_reproduce_linear_data_at_tiny_scale() {
        // Both methods reproduce straight-line data exactly (every node slope
        // equals the common secant). Scaling the ordinates by `s` must scale
        // the interpolant by `s`. Before the fix, the products of two secant
        // slopes of size ~1e-170 underflowed to zero, so PCHIP took every
        // interior node for a local extremum (slope 0) and Akima's weighted
        // average collapsed to 0, giving errors of tens of percent.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for s in [1.0, 1e-170, 1e-200, 1e-300]
        {
            let ys: Vec<f64> = xs.iter().map(|&x| s * x).collect();
            let p = PchipInterp::new(&xs, &ys).unwrap();
            let a = AkimaSpline::new(&xs, &ys).unwrap();
            for k in 0..=50
            {
                let x = 0.1 * k as f64;
                let want = s * x;
                for (name, got, slope) in [
                    ("pchip", p.eval(x), p.derivative(x)),
                    ("akima", a.eval(x), a.derivative(x)),
                ]
                {
                    assert!(
                        close_rel(got, want, s, 1e-12),
                        "{name} s={s:e} x={x}: {got:e} vs {want:e}"
                    );
                    assert!(
                        close_rel(slope, s, s, 1e-12),
                        "{name} s={s:e} x={x}: slope {slope:e} vs {s:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn pchip_and_akima_are_scale_invariant() {
        // Non-trivial monotone data (PCHIP) and generic data (Akima): the
        // interpolant of s*y must equal s times the interpolant of y.
        let xs = [0.0, 0.5, 1.7, 2.0, 3.2, 4.0, 5.5];
        let ys = [0.0, 0.3, 1.9, 2.0, 2.05, 4.0, 4.5];
        let p1 = PchipInterp::new(&xs, &ys).unwrap();
        let a1 = AkimaSpline::new(&xs, &ys).unwrap();
        for s in [1e-150, 1e-170, 1e-250, 1e-300, 1e150, 1e300]
        {
            let ys_s: Vec<f64> = ys.iter().map(|&y| s * y).collect();
            let ps = PchipInterp::new(&xs, &ys_s).unwrap();
            let as_ = AkimaSpline::new(&xs, &ys_s).unwrap();
            for k in 0..=60
            {
                let x = -0.25 + 0.1 * k as f64;
                for (name, got, want) in [
                    ("pchip", ps.eval(x), s * p1.eval(x)),
                    ("pchip'", ps.derivative(x), s * p1.derivative(x)),
                    ("akima", as_.eval(x), s * a1.eval(x)),
                    ("akima'", as_.derivative(x), s * a1.derivative(x)),
                ]
                {
                    assert!(
                        close_rel(got, want, s, 1e-12),
                        "{name} s={s:e} x={x}: {got:e} vs {want:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn pchip_interior_slope_matches_harmonic_mean_at_tiny_scale() {
        // Fritsch–Carlson interior slope with h = [1, 1]:
        // d = 2 / (1/dl + 1/dr). For dl = 1e-308 the naive 3/dl overflowed
        // to infinity and the slope collapsed to 0.
        let s = 1e-308;
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, s, 3.0 * s];
        let p = PchipInterp::new(&xs, &ys).unwrap();
        let want = 2.0 / (1.0 / 1.0 + 1.0 / 2.0) * s;
        let got = p.derivative(1.0);
        assert!(close_rel(got, want, s, 1e-12), "{got:e} vs {want:e}");
    }
}
