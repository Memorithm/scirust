//! Probability distributions with a unified [`Distribution`] trait.
//!
//! Continuous CDFs are expressed through the audited `scirust-special`
//! primitives (erf for the normal, the regularized incomplete gamma for χ²/gamma,
//! the regularized incomplete beta for Student-t / F / beta), so every tail
//! probability in the platform traces back to one validated numeric base.

use scirust_special::{
    binom_deviance, erfc, ln_beta, ln_gamma, regularized_gamma_p, regularized_gamma_q,
    regularized_incomplete_beta, stirling_error,
};

use crate::rng::SplitMix64;

use std::f64::consts::PI;

/// A univariate probability distribution.
///
/// The `sample` default draws by inverse-CDF transform of a uniform variate, so
/// any distribution with a `quantile` is sampleable deterministically from a
/// seeded [`SplitMix64`]. `sf` (survival / upper tail) defaults to `1 - cdf` but
/// is overridden where a direct form is more accurate in the far tail.
pub trait Distribution {
    /// Probability density (continuous) at `x`.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> f64;
    /// Inverse CDF (quantile / percent-point) for `p ∈ (0, 1)`.
    fn quantile(&self, p: f64) -> f64;
    /// Distribution mean (may be `NaN`/`∞` where undefined).
    fn mean(&self) -> f64;
    /// Distribution variance (may be `NaN`/`∞` where undefined).
    fn variance(&self) -> f64;

    /// Standard deviation, `sqrt(variance)`.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Survival function `P(X > x) = 1 − cdf(x)`. Override for tail accuracy.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    /// One deterministic draw via inverse-CDF from a seeded uniform source.
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        // Clamp away from the open-interval endpoints so ±∞ never appears.
        let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
        self.quantile(u)
    }
}

/// Robust inverse of a monotone-increasing CDF by bracket-and-bisect. Fully
/// deterministic (fixed iteration budget); narrows the bracket to
/// (near-)adjacent representable `f64`s rather than stopping at a loose
/// absolute width: for a very steep CDF (e.g. a Beta distribution with one
/// shape parameter close to 0, whose mass concentrates in a sliver near an
/// endpoint), a fixed `1e-13` bracket can still span most of `[0, 1]` in
/// `p`-space, so stopping there — as an earlier version of this function
/// did — silently returns an `x` whose `cdf(x)` is far from the requested
/// `p`. Using the tightest tolerance the mantissa allows costs nothing (the
/// loop is still capped at 128 iterations) and gives the best answer f64
/// can represent even in that regime.
fn invert_cdf(cdf: impl Fn(f64) -> f64, p: f64, mut lo: f64, mut hi: f64) -> f64 {
    // Expand the bracket outward until it straddles `p`. The span doubles on
    // every step, so the loops reach the end of the finite `f64` range in
    // about a thousand steps; they stop there (at `∓f64::MAX`) rather than at
    // an arbitrary step budget. An earlier budget of 200 doublings capped
    // the bracket near `±3.2e62`, so a heavy-tailed quantile beyond it (for
    // example `StudentT::new(1.0).quantile(1e-100) ≈ −3.18e99`) silently came
    // back as `≈ −3.2e62`.
    while cdf(lo) > p && lo > f64::MIN
    {
        let span = (hi - lo).max(1.0);
        let next = lo - span;
        lo = if next.is_finite() { next } else { f64::MIN };
    }
    while cdf(hi) < p && hi < f64::MAX
    {
        let span = (hi - lo).max(1.0);
        let next = hi + span;
        hi = if next.is_finite() { next } else { f64::MAX };
    }
    // The quantile lies beyond the largest finite `f64`: report it as the
    // infinity it rounds to instead of pinning it to `∓f64::MAX`.
    if lo == f64::MIN && cdf(lo) > p
    {
        return f64::NEG_INFINITY;
    }
    if hi == f64::MAX && cdf(hi) < p
    {
        return f64::INFINITY;
    }

    // Non-negative support (`lo == 0`): the quantile can sit many orders of
    // magnitude below `hi` — e.g. a Gamma with `shape ≪ 1` places its p-th
    // percentile astronomically close to 0 (for `shape = scale = 0.01`,
    // `quantile(0.01) ≈ 1e-202`, which is perfectly representable in f64 yet
    // unreachable by a linear bisection: that bottoms out near `hi·EPSILON`,
    // where `cdf` is still far from `p`). Searching in **log-space** resolves
    // the root to a *relative* precision regardless of its magnitude.
    if lo == 0.0
    {
        // Shrink a lower bound `a` geometrically until `cdf(a) < p`, keeping the
        // last bracket `b` with `cdf(b) >= p`.
        let mut a = hi;
        let mut b = hi;
        let mut guard = 0;
        while cdf(a) >= p && a > f64::MIN_POSITIVE && guard < 400
        {
            b = a;
            a *= 1e-8;
            guard += 1;
        }
        if cdf(a) >= p
        {
            // The quantile is below the smallest positive f64: return the best
            // representable value (a floating-point limit, not a convergence bug).
            return a;
        }
        let (mut la, mut lb) = (a.ln(), b.ln());
        for _ in 0..200
        {
            let m = 0.5 * (la + lb);
            if cdf(m.exp()) < p
            {
                la = m;
            }
            else
            {
                lb = m;
            }
            if (lb - la).abs() <= 4.0 * f64::EPSILON * (1.0 + la.abs())
            {
                break;
            }
        }
        return (0.5 * (la + lb)).exp();
    }

    // General support (may be negative): linear bisection.
    for _ in 0..128
    {
        // `0.5·lo + 0.5·hi` rather than `0.5·(lo + hi)`: the bracket may span
        // the whole finite range, where `lo + hi` would overflow.
        let mid = 0.5 * lo + 0.5 * hi;
        if cdf(mid) < p
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
        if (hi - lo).abs() <= 4.0 * f64::EPSILON * (1.0 + mid.abs())
        {
            break;
        }
    }
    0.5 * lo + 0.5 * hi
}

/// Quantile by bisection on whichever tail keeps full precision at `p`.
///
/// For `p ≤ ½` this is [`invert_cdf`] on the CDF. For `p > ½` it solves
/// `sf(x) = 1 − p` instead (`1 − p` is exact there): inverting the CDF near
/// `1` can only resolve the upper tail to the `≈1.1e-16` spacing of `f64`
/// values just below `1`, and some CDFs lose more than that when their own
/// argument rounds towards `1` (`FisherF` forms `d1·x/(d1·x + d2)`). That
/// limited upper-tail quantiles to a few significant digits, e.g.
/// `FisherF::new(1.0, 1.0).quantile(1.0 - 1e-8)` came back ≈11 % low and
/// `ChiSquared::new(1.0).quantile(1.0 - 1e-15)` ≈0.17 % low. The negated
/// survival function `−sf` is increasing, so [`invert_cdf`] applies as is.
fn invert_two_tailed(
    cdf: impl Fn(f64) -> f64,
    sf: impl Fn(f64) -> f64,
    p: f64,
    lo: f64,
    hi: f64,
) -> f64 {
    if p > 0.5
    {
        invert_cdf(|x| -sf(x), -(1.0 - p), lo, hi)
    }
    else
    {
        invert_cdf(cdf, p, lo, hi)
    }
}

/// Standard normal quantile `Φ⁻¹(p)` for `p ∈ (0, 1)`.
///
/// Wichura's algorithm AS 241 (`PPND16`, *Applied Statistics* 37 (1988)
/// 477–484), accurate to about 1e-16 relative over the whole open interval.
/// It works from `p` (or `1 − p`) directly, so the lower tail keeps full
/// relative precision; the previous `√2 · erfinv(2p − 1)` formulation lost
/// the low digits of `p` when forming `2p − 1` and returned `−∞` once
/// `p ≲ 1.1e-16`, even though `Φ⁻¹(p)` is finite down to the smallest
/// subnormal `p`.
fn standard_normal_quantile(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425
    {
        let r = 0.180_625 - q * q;
        let num = ((((((2.509_080_928_730_122_7e3 * r + 3.343_057_558_358_813e4) * r
            + 6.726_577_092_700_87e4)
            * r
            + 4.592_195_393_154_987e4)
            * r
            + 1.373_169_376_550_946e4)
            * r
            + 1.971_590_950_306_551_4e3)
            * r
            + 1.331_416_678_917_843_8e2)
            * r
            + 3.387_132_872_796_366_6;
        let den = ((((((5.226_495_278_852_854e3 * r + 2.872_908_573_572_194_3e4) * r
            + 3.930_789_580_009_271e4)
            * r
            + 2.121_379_430_158_659_6e4)
            * r
            + 5.394_196_021_424_751e3)
            * r
            + 6.871_870_074_920_579e2)
            * r
            + 4.231_333_070_160_091e1)
            * r
            + 1.0;
        return q * num / den;
    }
    // Tail: work with the smaller of p and 1 − p (both exact here).
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let mut r = (-tail.ln()).sqrt();
    let z = if r <= 5.0
    {
        r -= 1.6;
        let num = ((((((7.745_450_142_783_414e-4 * r + 2.272_384_498_926_918_5e-2) * r
            + 2.417_807_251_774_506e-1)
            * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_6)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_546)
            * r
            + 1.423_437_110_749_683_6;
        let den = ((((((1.050_750_071_644_416_8e-9 * r + 5.475_938_084_995_345e-4) * r
            + 1.519_866_656_361_645_7e-2)
            * r
            + 1.481_039_764_274_800_7e-1)
            * r
            + 6.897_673_349_851e-1)
            * r
            + 1.676_384_830_183_803_8)
            * r
            + 2.053_191_626_637_759)
            * r
            + 1.0;
        num / den
    }
    else
    {
        r -= 5.0;
        let num = ((((((2.010_334_399_292_288_1e-7 * r + 2.711_555_568_743_487_6e-5) * r
            + 1.242_660_947_388_078_4e-3)
            * r
            + 2.653_218_952_657_612_3e-2)
            * r
            + 2.965_605_718_285_048_9e-1)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_104;
        let den = ((((((2.044_263_103_389_939_8e-15 * r + 1.421_511_758_316_446e-7) * r
            + 1.846_318_317_510_054_7e-5)
            * r
            + 7.868_691_311_456_133e-4)
            * r
            + 1.487_536_129_085_061_5e-2)
            * r
            + 1.369_298_809_227_358e-1)
            * r
            + 5.998_322_065_558_88e-1)
            * r
            + 1.0;
        num / den
    };
    if q < 0.0 { -z } else { z }
}

// ============================================================ //
//  Normal                                                      //
// ============================================================ //

/// Normal (Gaussian) distribution `N(μ, σ²)`.
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    mean: f64,
    sd: f64,
}

impl Normal {
    /// `N(μ, σ)` with standard deviation `sd > 0`.
    pub fn new(mean: f64, sd: f64) -> Self {
        assert!(sd > 0.0, "Normal: standard deviation must be > 0");
        Self { mean, sd }
    }
    /// The standard normal `N(0, 1)`.
    pub fn standard() -> Self {
        Self { mean: 0.0, sd: 1.0 }
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        (-0.5 * z * z).exp() / (self.sd * (2.0 * PI).sqrt())
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        0.5 * erfc(-z / std::f64::consts::SQRT_2)
    }
    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        0.5 * erfc(z / std::f64::consts::SQRT_2)
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        self.mean + self.sd * standard_normal_quantile(p)
    }
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.sd * self.sd
    }
}

// ============================================================ //
//  Exponential & Uniform (closed-form)                         //
// ============================================================ //

/// Exponential distribution with rate `λ > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    rate: f64,
}
impl Exponential {
    /// Rate parameter `λ > 0` (mean `1/λ`).
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "Exponential: rate must be > 0");
        Self { rate }
    }
}
impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0
        {
            0.0
        }
        else
        {
            self.rate * (-self.rate * x).exp()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0
        {
            0.0
        }
        else
        {
            // `-expm1(-λx)` rather than `1 − exp(−λx)`: the subtraction
            // cancels catastrophically for small `λx` (it returns exactly 0
            // once `λx < EPSILON/2`, although the true CDF is `≈ λx`).
            -(-self.rate * x).exp_m1()
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 { 1.0 } else { (-self.rate * x).exp() }
    }
    fn quantile(&self, p: f64) -> f64 {
        // `−ln1p(−p)` rather than `−ln(1 − p)`: forming `1 − p` discards the
        // low digits of a small `p` (and rounds `p < EPSILON/2` to exactly 0).
        -(-p).ln_1p() / self.rate
    }
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }
    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }
}

/// Continuous uniform distribution on `[a, b]`.
#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    a: f64,
    b: f64,
}
impl Uniform {
    /// Uniform on `[a, b]` with `a < b`.
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a < b, "Uniform: require a < b");
        Self { a, b }
    }
}
impl Distribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b
        {
            0.0
        }
        else
        {
            1.0 / (self.b - self.a)
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a
        {
            0.0
        }
        else if x >= self.b
        {
            1.0
        }
        else
        {
            (x - self.a) / (self.b - self.a)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        self.a + p.clamp(0.0, 1.0) * (self.b - self.a)
    }
    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let d = self.b - self.a;
        d * d / 12.0
    }
}

// ============================================================ //
//  Gamma & Chi-squared                                         //
// ============================================================ //

// Shapes at or above this use the saddle-point density in `Gamma::pdf`
// (`k − 1 ≥ 1`, so the Stirling remainder `δ(k − 1)` stays finite).
const GAMMA_PDF_SADDLE_MIN: f64 = 2.0;

/// Gamma distribution with shape `k > 0` and scale `θ > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    shape: f64,
    scale: f64,
}
impl Gamma {
    /// Shape `k > 0`, scale `θ > 0` (mean `kθ`).
    pub fn new(shape: f64, scale: f64) -> Self {
        assert!(
            shape > 0.0 && scale > 0.0,
            "Gamma: shape, scale must be > 0"
        );
        Self { shape, scale }
    }
}
impl Distribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 0.0;
        }
        let k = self.shape;
        let t = self.scale;
        if k < GAMMA_PDF_SADDLE_MIN
        {
            return ((k - 1.0) * x.ln() - x / t - k * t.ln() - ln_gamma(k)).exp();
        }
        // `y^(k−1)·e^(−y)/Γ(k)` with `y = x/θ` is the Poisson "pmf" at
        // `k − 1` with rate `y`; Loader's saddle-point form of it (R's
        // `dgamma`) avoids the `ε·k·ln k` cancellation of the direct
        // exponent, which near the mode costs ~5e-5 relative at `k = 1e10`.
        let y = x / t;
        if y.is_infinite()
        {
            return 0.0;
        }
        let m = k - 1.0;
        (-stirling_error(m) - binom_deviance(m, y) - 0.5 * (2.0 * PI * m).ln()).exp() / t
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            0.0
        }
        else
        {
            regularized_gamma_p(self.shape, x / self.scale)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            1.0
        }
        else
        {
            regularized_gamma_q(self.shape, x / self.scale)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        let hi = self.mean() + 12.0 * self.std_dev();
        invert_two_tailed(|x| self.cdf(x), |x| self.sf(x), p, 0.0, hi.max(1.0))
    }
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
}

/// Chi-squared distribution with `k > 0` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    k: f64,
}
impl ChiSquared {
    /// `k` degrees of freedom (`k > 0`).
    pub fn new(k: f64) -> Self {
        assert!(k > 0.0, "ChiSquared: degrees of freedom must be > 0");
        Self { k }
    }
}
impl Distribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        // χ²(k) is Gamma(k/2, 2).
        Gamma::new(self.k / 2.0, 2.0).pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            0.0
        }
        else
        {
            regularized_gamma_p(self.k / 2.0, x / 2.0)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            1.0
        }
        else
        {
            regularized_gamma_q(self.k / 2.0, x / 2.0)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        let hi = self.mean() + 12.0 * self.std_dev();
        invert_two_tailed(|x| self.cdf(x), |x| self.sf(x), p, 0.0, hi.max(1.0))
    }
    fn mean(&self) -> f64 {
        self.k
    }
    fn variance(&self) -> f64 {
        2.0 * self.k
    }
}

// ============================================================ //
//  Student-t & Fisher-F & Beta                                 //
// ============================================================ //

/// Student's t distribution with `ν > 0` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    nu: f64,
}
impl StudentT {
    /// `ν` degrees of freedom (`ν > 0`).
    pub fn new(nu: f64) -> Self {
        assert!(nu > 0.0, "StudentT: degrees of freedom must be > 0");
        Self { nu }
    }
}
impl Distribution for StudentT {
    fn pdf(&self, t: f64) -> f64 {
        let nu = self.nu;
        // Γ((ν+1)/2) / (√(νπ)·Γ(ν/2)) = 1 / (√ν·B(ν/2, ½)). `ln_beta` keeps
        // this exact for huge ν, where the ln Γ difference used to cancel
        // (ν = 1e15 gave pdf(0) ≈ 0.159 instead of 1/√(2π) ≈ 0.399).
        let ln_norm = -ln_beta(0.5 * nu, 0.5) - 0.5 * nu.ln();
        (ln_norm - (nu + 1.0) / 2.0 * (t * t / nu).ln_1p()).exp()
    }
    fn cdf(&self, t: f64) -> f64 {
        // By symmetry, cdf(t) = sf(−t). Each side is evaluated as a tail
        // (or near-centre) quantity directly, never as `1 − small`.
        self.sf(-t)
    }
    /// Upper tail `P(T > t)`, evaluated without the `1 − cdf` cancellation.
    ///
    /// For `t ≥ 0` the tail `P(T > t) = ½·I_{ν/(ν+t²)}(ν/2, ½)` is computed
    /// directly, so it keeps full relative precision far below `f64::EPSILON`
    /// (it used to collapse to exactly `0` once the tail fell below ≈1e-16,
    /// e.g. for `ν = 20, t = 50`, true value ≈ 8.77e-23). Near the centre,
    /// where the incomplete beta would itself switch to its `1 − …` side, the
    /// complementary form `½ − ½·I_{t²/(ν+t²)}(½, ν/2)` is used instead, so
    /// that `x = ν/(ν+t²)` rounding to `1` for tiny `|t|` loses nothing.
    fn sf(&self, t: f64) -> f64 {
        if t.is_nan()
        {
            return f64::NAN;
        }
        let nu = self.nu;
        let t2 = t * t;
        let x = nu / (nu + t2);
        // Mass of one tail beyond |t|, P(T > |t|). The switch point is the
        // one `regularized_incomplete_beta(ν/2, ½, x)` uses internally, so
        // each branch evaluates its continued fraction on the direct side.
        let tail = if x < 1e-100
        {
            // Far tail: `I_x(a, ½) = x^a / (a·B(a, ½)) · (1 + O(x))`, so the
            // leading term is exact to f64 precision here. It is evaluated in
            // log space with `ln x = ln ν − 2·ln|t|`, which stays finite where
            // `t²` overflows (|t| ≳ 1.3e154) or `x` underflows; forming `x`
            // directly used to return a tail of exactly 0 there, although the
            // true tail of a small-ν t distribution is still far above the
            // smallest f64 (`ν = 1, t = 1e200`: ≈3.18e-201).
            let a = 0.5 * nu;
            let ln_x = nu.ln() - 2.0 * t.abs().ln();
            0.5 * (a * ln_x - a.ln() - ln_beta(a, 0.5)).exp()
        }
        else if x < (0.5 * nu + 1.0) / (0.5 * nu + 2.5)
        {
            0.5 * regularized_incomplete_beta(nu / 2.0, 0.5, x)
        }
        else
        {
            // P(|T| <= |t|) = I_{t²/(ν+t²)}(½, ν/2), computed without
            // forming 1 − x.
            0.5 - 0.5 * regularized_incomplete_beta(0.5, nu / 2.0, t2 / (nu + t2))
        };
        if t >= 0.0 { tail } else { 1.0 - tail }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        invert_two_tailed(|t| self.cdf(t), |t| self.sf(t), p, -100.0, 100.0)
    }
    fn mean(&self) -> f64 {
        if self.nu > 1.0 { 0.0 } else { f64::NAN }
    }
    fn variance(&self) -> f64 {
        if self.nu > 2.0
        {
            self.nu / (self.nu - 2.0)
        }
        else
        {
            f64::INFINITY
        }
    }
}

/// Fisher–Snedecor F distribution with `(d1, d2)` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct FisherF {
    d1: f64,
    d2: f64,
}
impl FisherF {
    /// Numerator `d1 > 0` and denominator `d2 > 0` degrees of freedom.
    pub fn new(d1: f64, d2: f64) -> Self {
        assert!(d1 > 0.0 && d2 > 0.0, "FisherF: both dof must be > 0");
        Self { d1, d2 }
    }
}
impl Distribution for FisherF {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 0.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        // ln pdf = (d1/2)ln(d1/d2) + (d1/2−1)ln x − ((d1+d2)/2)ln(1+d1 x/d2) − lnB(d1/2,d2/2)
        let ln = (d1 / 2.0) * (d1 / d2).ln() + (d1 / 2.0 - 1.0) * x.ln()
            - (d1 + d2) / 2.0 * (d1 * x / d2).ln_1p()
            - ln_beta(d1 / 2.0, d2 / 2.0);
        ln.exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 0.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        let y = d1 * x / (d1 * x + d2);
        regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, y)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 1.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        // Complement via the symmetry of the incomplete beta.
        let y = d2 / (d1 * x + d2);
        regularized_incomplete_beta(d2 / 2.0, d1 / 2.0, y)
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        invert_two_tailed(|x| self.cdf(x), |x| self.sf(x), p, 0.0, 100.0)
    }
    fn mean(&self) -> f64 {
        if self.d2 > 2.0
        {
            self.d2 / (self.d2 - 2.0)
        }
        else
        {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        let (d1, d2) = (self.d1, self.d2);
        if d2 > 4.0
        {
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0))
        }
        else
        {
            f64::NAN
        }
    }
}

/// Beta distribution on `[0, 1]` with shapes `a, b > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    a: f64,
    b: f64,
}
impl Beta {
    /// Shapes `a > 0`, `b > 0`.
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a > 0.0 && b > 0.0, "Beta: shapes must be > 0");
        Self { a, b }
    }
}
impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0
        {
            return 0.0;
        }
        // `ln_1p(−x)` keeps `ln(1 − x)` exact for tiny `x` (large-`b` shapes).
        ((self.a - 1.0) * x.ln() + (self.b - 1.0) * (-x).ln_1p() - ln_beta(self.a, self.b)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        regularized_incomplete_beta(self.a, self.b, x)
    }
    /// Upper tail `P(X > x) = I_{1−x}(b, a)`, evaluated without the
    /// `1 − cdf` cancellation.
    ///
    /// Above the point `(a+1)/(a+b+2)` where the incomplete-beta continued
    /// fraction switches sides, the tail is computed directly from the
    /// reflected form, so it keeps full relative precision far below
    /// `f64::EPSILON` (it used to collapse to exactly `0`, e.g. for
    /// `Beta(10, 10)` at `x = 0.999`, true value ≈ 9.16e-26). Below that point
    /// `1 − cdf` is already accurate (the cdf is not close to `1` there) and
    /// avoids rounding `1 − x` for tiny `x`.
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan()
        {
            return f64::NAN;
        }
        if x <= 0.0
        {
            return 1.0;
        }
        if x >= 1.0
        {
            return 0.0;
        }
        let (a, b) = (self.a, self.b);
        if x < (a + 1.0) / (a + b + 2.0)
        {
            1.0 - regularized_incomplete_beta(a, b, x)
        }
        else
        {
            regularized_incomplete_beta(b, a, 1.0 - x)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return 1.0;
        }
        invert_two_tailed(|x| self.cdf(x), |x| self.sf(x), p, 0.0, 1.0)
    }
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// Purely relative comparison, for tail values far below 1.
    fn rel_ok(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs()
    }

    #[test]
    fn normal_matches_reference() {
        let n = Normal::standard();
        assert!(close(n.cdf(0.0), 0.5, 1e-12));
        assert!(close(n.cdf(1.96), 0.975_002_104_851_780, 1e-10));
        assert!(close(n.quantile(0.975), 1.959_963_984_540_054, 1e-9));
        // pdf peak = 1/√(2π).
        assert!(close(n.pdf(0.0), 1.0 / (2.0 * PI).sqrt(), 1e-13));
        // round-trip cdf∘quantile.
        for &p in &[0.01, 0.3, 0.5, 0.84, 0.999]
        {
            assert!(close(n.cdf(n.quantile(p)), p, 1e-9), "p = {p}");
        }
        // shifted/scaled
        let m = Normal::new(10.0, 2.0);
        assert!(close(m.cdf(10.0), 0.5, 1e-12));
        assert!(close(m.std_dev(), 2.0, 1e-15));
    }

    #[test]
    fn normal_quantile_keeps_relative_precision_in_the_far_lower_tail() {
        // Regression test: the quantile used to be `√2 · erfinv(2p − 1)`.
        // Forming `2p − 1` throws away the low digits of a small `p`
        // (for p = 1e-10 only ~7 significant digits survive) and rounds to
        // exactly −1 once p ≲ 1.1e-16, so the old code returned `−∞` for
        // p = 1e-17, 1e-20, 1e-300, ... although Φ⁻¹(p) is finite down to the
        // smallest subnormal. References: mpmath (60 digits), solving
        // ncdf(z) = p for the exact f64 value of p.
        let n = Normal::standard();
        let cases: [(f64, f64); 6] = [
            (1e-10, -6.361_340_902_404_057),
            (1e-17, -8.493_793_224_109_599),
            (1e-20, -9.262_340_089_798_407),
            (1e-100, -21.273_453_560_965_326),
            (1e-300, -37.047_096_299_361_2),
            (5e-324, -38.467_405_617_144_344),
        ];
        for (p, want) in cases
        {
            let got = n.quantile(p);
            assert!(got.is_finite(), "p = {p:e}: quantile = {got}");
            assert!(
                rel_ok(got, want, 1e-13),
                "p = {p:e}: got {got}, want {want}"
            );
        }
        // Upper tail mirrors the lower tail; 1 − 2⁻⁴⁰ and 2⁻⁴⁰ are exact.
        let t = 2f64.powi(-40);
        assert!(rel_ok(n.quantile(1.0 - t), -n.quantile(t), 1e-14));
        // Shifted/scaled normals inherit the accuracy.
        let m = Normal::new(3.0, 2.0);
        assert!(rel_ok(
            m.quantile(1e-20),
            3.0 + 2.0 * -9.262_340_089_798_407,
            1e-13
        ));
    }

    #[test]
    fn exponential_small_p_quantile_and_small_x_cdf_are_not_cancelled() {
        // Regression test: `quantile` computed `−ln(1 − p)/λ` and `cdf`
        // computed `1 − exp(−λx)`. Both subtract from 1 and cancel
        // catastrophically: p = 1e-20 gave quantile 0 and x = 1e-20 gave
        // cdf 0 (relative error 100%), and p = 1e-10 kept only ~7 digits.
        // References: mpmath, −log1p(−p) and −expm1(−x).
        let e = Exponential::new(1.0);
        assert!(rel_ok(e.quantile(1e-20), 1e-20, 1e-14));
        assert!(rel_ok(e.quantile(1e-10), 1.000_000_000_05e-10, 1e-14));
        assert!(rel_ok(e.cdf(1e-20), 1e-20, 1e-14));
        assert!(rel_ok(e.cdf(1e-10), 9.999_999_999_500_001e-11, 1e-14));
        // Rate scaling: quantile(p) = −log1p(−p)/λ.
        let fast = Exponential::new(4.0);
        assert!(rel_ok(
            fast.quantile(2.5e-7),
            2.500_000_312_500_052e-7 / 4.0,
            1e-14
        ));
        // Unchanged in the bulk.
        assert!(close(e.quantile(0.5), 2.0_f64.ln(), 1e-15));
        assert!(close(e.cdf(e.quantile(0.9)), 0.9, 1e-15));
    }

    #[test]
    fn student_t_matches_reference() {
        let t = StudentT::new(10.0);
        assert!(close(t.cdf(0.0), 0.5, 1e-12));
        // t_{0.975, 10} = 2.228138852...
        assert!(close(t.quantile(0.975), 2.228_138_851_986_273, 1e-6));
        assert!(close(t.cdf(2.228_138_851_986_273), 0.975, 1e-9));
        // symmetry
        assert!(close(t.cdf(-1.5), 1.0 - t.cdf(1.5), 1e-12));
        // large ν → normal
        let big = StudentT::new(1e6);
        assert!(close(big.cdf(1.96), Normal::standard().cdf(1.96), 1e-4));
    }

    #[test]
    fn chi_squared_matches_reference() {
        let c = ChiSquared::new(5.0);
        assert!(close(c.mean(), 5.0, 1e-15));
        assert!(close(c.variance(), 10.0, 1e-15));
        // χ²_{0.95, 5} = 11.0704976935...
        assert!(close(c.quantile(0.95), 11.070_497_693_516_35, 1e-7));
        assert!(close(c.cdf(11.070_497_693_516_35), 0.95, 1e-9));
        // sf + cdf = 1
        assert!(close(c.cdf(7.0) + c.sf(7.0), 1.0, 1e-14));
    }

    #[test]
    fn fisher_f_matches_reference() {
        let f = FisherF::new(5.0, 10.0);
        // F_{0.95}(5,10) = 3.325834529...
        assert!(close(f.quantile(0.95), 3.325_834_529_923_105, 1e-5));
        assert!(close(f.cdf(3.325_834_529_923_105), 0.95, 1e-8));
        assert!(close(f.cdf(2.0) + f.sf(2.0), 1.0, 1e-12));
        assert!(close(f.mean(), 10.0 / 8.0, 1e-13));
    }

    #[test]
    fn gamma_beta_exponential_uniform() {
        // Gamma(k=1) is Exponential(1/scale).
        let g = Gamma::new(1.0, 2.0);
        let e = Exponential::new(0.5);
        assert!(close(g.cdf(3.0), e.cdf(3.0), 1e-12));
        assert!(close(e.quantile(0.5), 2.0 * 2.0_f64.ln(), 1e-12));
        // Beta(2,2) is symmetric about 0.5.
        let b = Beta::new(2.0, 2.0);
        assert!(close(b.cdf(0.5), 0.5, 1e-12));
        assert!(close(b.mean(), 0.5, 1e-15));
        // Uniform
        let u = Uniform::new(2.0, 6.0);
        assert!(close(u.cdf(4.0), 0.5, 1e-15));
        assert!(close(u.mean(), 4.0, 1e-15));
        assert!(close(u.variance(), 16.0 / 12.0, 1e-14));
    }

    #[test]
    // Ignored under Miri: this asserts a tight (1e-10) CDF∘quantile round-trip,
    // where the quantile is a bisection over a transcendental CDF. Miri perturbs
    // the float intrinsics non-deterministically, so the round-trip drifts past
    // the tolerance. The native Build & Test jobs enforce the real accuracy.
    #[cfg_attr(miri, ignore)]
    fn beta_quantile_round_trip_at_the_edge_of_f64_resolution() {
        // Regression test for a finding made by this crate's own property
        // tests: `invert_cdf` stopped bisecting as soon as its x-bracket
        // narrowed below a fixed 1e-13, regardless of how steep the CDF is
        // there. For a strongly skewed Beta (one shape parameter close to
        // 0), almost all of the probability mass sits within a sliver near
        // an endpoint far narrower than 1e-13 — so the old bracket-width
        // cutoff quit while `cdf(x)` was still far from the target `p`,
        // e.g. quantile(0.99) on Beta(390.12, 0.5) round-tripped to
        // cdf ≈ 0.7996 with the old fixed tolerance (error ~2e-10 — 100x
        // looser than what f64 can actually resolve here). Tightening the
        // stopping criterion to a few ULPs (`4·EPSILON`) costs nothing (the
        // loop is already capped at 128 iterations) and brings the
        // round-trip error down to ~1e-12 for this case.
        let beta = Beta::new(390.121, 0.5);
        for &p in &[0.1, 0.5, 0.873, 0.99]
        {
            let p_hat = beta.cdf(beta.quantile(p));
            assert!(close(p_hat, p, 1e-10), "p={p} p_hat={p_hat}");
        }
        // Below shape ~0.3-0.5, the mass compresses to less than one ULP
        // near the endpoint and even exact bisection cannot resolve it —
        // that's a floating-point representability limit, not a bug: the
        // best `invert_cdf` can do is return a value adjacent to the
        // endpoint, whose true `cdf` legitimately differs a lot from `p`.
        let extreme = Beta::new(390.121, 0.01);
        let x = extreme.quantile(0.99);
        assert!(x > 1.0 - 1e-9, "expected quantile pinned near 1.0, got {x}");
    }

    #[test]
    // Ignored under Miri (transcendental CDF bisection; see the Beta test above).
    #[cfg_attr(miri, ignore)]
    fn gamma_quantile_round_trip_at_tiny_lower_tail() {
        // Regression for a failure surfaced by this crate's property tests:
        // `invert_cdf`'s linear bisection could not reach the lower-tail
        // quantile of a Gamma with `shape ≪ 1`. For `Gamma(0.01, 0.01)` the
        // 1st percentile sits ~1e-202 from 0 — representable in f64, yet linear
        // bisection bottomed out near `hi·EPSILON`, returning an `x` whose
        // `cdf(x) ≈ 0.74` instead of `0.01`. Log-space bisection resolves it.
        let g = Gamma::new(0.01, 0.01);
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99]
        {
            let x = g.quantile(p);
            let p_hat = g.cdf(x);
            assert!(close(p_hat, p, 1e-6), "p={p} x={x} p_hat={p_hat}");
        }
        // A range of small shapes and scales.
        for &shape in &[0.02f64, 0.05, 0.1, 0.5]
        {
            for &scale in &[0.01f64, 1.0, 100.0]
            {
                let g = Gamma::new(shape, scale);
                let p_hat = g.cdf(g.quantile(0.01));
                assert!(
                    close(p_hat, 0.01, 1e-6),
                    "shape={shape} scale={scale} p_hat={p_hat}"
                );
            }
        }
    }

    #[test]
    // Same Miri policy as the other quantile round-trip tests.
    #[cfg_attr(miri, ignore)]
    fn deep_lower_tail_is_exponentially_small_and_chi_squared_shares_the_fix() {
        // Complements the test above (the CI failure that produced the seed in
        // proptest-regressions/dist.txt): the recovered quantile really is
        // exponentially small — not merely nonzero — and χ²(k) = Gamma(k/2, 2)
        // takes the same log-space path through `invert_cdf`.
        let x = Gamma::new(0.01, 0.01).quantile(0.01);
        assert!(
            x > 0.0 && x < 1e-150,
            "expected an exponentially small quantile, got {x}"
        );
        let c = ChiSquared::new(0.02);
        let p_hat = c.cdf(c.quantile(0.01));
        assert!(close(p_hat, 0.01, 1e-6), "chi2 p_hat={p_hat}");
    }

    #[test]
    // Same Miri policy as the other quantile round-trip tests.
    #[cfg_attr(miri, ignore)]
    fn upper_tail_quantiles_keep_full_precision() {
        // Regression: every bisection-based quantile inverted the CDF, which
        // near 1 can only resolve the upper tail to the ≈1.1e-16 spacing of
        // f64 values below 1 (and less where the CDF's own argument rounds
        // towards 1). References: mpmath at 60 digits, solving
        // sf(x) = 1 − p for the exact f64 `p`; they agree with SciPy 1.18.1.
        // Old results in the comments.
        let p15 = 1.0 - 1e-15;
        // 64.3255…, 0.17 % low.
        assert!(rel_ok(
            ChiSquared::new(1.0).quantile(p15),
            64.432_038_969_363_55,
            1e-11
        ));
        // 41.2816…, 0.14 % low.
        assert!(rel_ok(
            Gamma::new(3.0, 1.0).quantile(p15),
            41.338_374_259_893_29,
            1e-11
        ));
        // 14.8959…, 0.2 % low.
        assert!(rel_ok(
            StudentT::new(30.0).quantile(p15),
            14.926_307_996_864_4,
            1e-11
        ));
        // 3.6029e15, 11 % low: F's CDF forms d1·x/(d1·x + d2), which rounds to 1.
        assert!(rel_ok(
            FisherF::new(1.0, 1.0).quantile(1.0 - 1e-8),
            4.052_847_304_964_346e15,
            1e-10
        ));
        // Beta(2, 3): 1 − x was 6.41e-6 instead of 6.30e-6 (1.8 % off).
        let x = Beta::new(2.0, 3.0).quantile(p15);
        assert!(
            rel_ok(1.0 - x, 6.297_936_339_841_653e-6, 1e-9),
            "1 - x = {}",
            1.0 - x
        );
        // The defining identity, sf(quantile(p)) = 1 − p, across the upper half.
        for &q in &[0.4, 1e-3, 1e-6, 1e-10, 1e-14]
        {
            let p = 1.0 - q;
            let q = 1.0 - p; // the exact upper-tail mass for this f64 `p`
            let c = ChiSquared::new(4.0);
            assert!(rel_ok(c.sf(c.quantile(p)), q, 1e-9), "chi2 q={q}");
            let t = StudentT::new(3.0);
            assert!(rel_ok(t.sf(t.quantile(p)), q, 1e-9), "t q={q}");
            let f = FisherF::new(3.0, 7.0);
            assert!(rel_ok(f.sf(f.quantile(p)), q, 1e-9), "F q={q}");
        }
    }

    #[test]
    // Same Miri policy as the other quantile round-trip tests.
    #[cfg_attr(miri, ignore)]
    fn heavy_tailed_quantiles_are_not_capped_by_the_bracket_budget() {
        // Regression: `invert_cdf` stopped widening its bracket after 200
        // doublings (≈ ±3.2e62), so the far tail of a small-ν t distribution
        // came back as ≈ −3.2e62 whatever the true value. mpmath references.
        assert!(rel_ok(
            StudentT::new(1.0).quantile(1e-100),
            -3.183_098_861_837_907e99,
            1e-10
        ));
        assert!(rel_ok(
            StudentT::new(5.0).quantile(1e-300),
            -1.568_392_559_099_337_8e60,
            1e-10
        ));
        // ν = 0.3: the true 1e-100 quantile is beyond the f64 range
        // (sf(f64::MAX) ≈ 1.17e-93 > 1e-100), so it rounds to −∞.
        assert_eq!(StudentT::new(0.3).quantile(1e-100), f64::NEG_INFINITY);
        // Upper side: F(0.001, 0.001) keeps sf(f64::MAX) ≈ 0.35 (mpmath), so
        // its 0.7 quantile is beyond f64 too (it used to return ≈1.76e16).
        assert_eq!(FisherF::new(0.001, 0.001).quantile(0.7), f64::INFINITY);
    }

    #[test]
    fn student_t_far_tail_survives_t_squared_overflow() {
        // Regression: for |t| ≳ 1.3e154, t² overflowed, x = ν/(ν + t²) became
        // 0 and sf returned exactly 0. mpmath references.
        let cauchy = StudentT::new(1.0);
        assert!(rel_ok(cauchy.sf(1e200), 3.183_098_861_837_907e-201, 1e-12));
        assert!(rel_ok(
            cauchy.cdf(-1e200),
            3.183_098_861_837_907e-201,
            1e-12
        ));
        assert!(rel_ok(
            StudentT::new(0.3).sf(f64::MAX),
            1.166_899_385_112_180_4e-93,
            1e-11
        ));
        // Still exactly 0 / 1 at infinity, and monotone across the switch.
        assert_eq!(cauchy.sf(f64::INFINITY), 0.0);
        assert_eq!(cauchy.cdf(f64::INFINITY), 1.0);
        assert!(cauchy.sf(1e99) > cauchy.sf(1e101));
    }

    #[test]
    // Ignored under Miri: `sample` goes through the quantile (AS 241 / ln), and
    // Miri deliberately randomizes the last ULPs of transcendental float
    // intrinsics per call, so lockstep bit-identity cannot hold under the
    // interpreter. On real hardware the property holds and stays enforced by
    // the native Build & Test jobs. (SplitMix64 itself is integer-only and
    // stays Miri-checked via the rng tests.)
    #[cfg_attr(miri, ignore)]
    fn sampling_is_deterministic_and_plausible() {
        let n = Normal::new(5.0, 2.0);
        let mut r1 = SplitMix64::new(123);
        let mut r2 = SplitMix64::new(123);
        let s: Vec<f64> = (0..50_000).map(|_| n.sample(&mut r1)).collect();
        // reproducible
        for _ in 0..50_000
        {
            // consume r2 in lockstep — bit identical
        }
        let mut r2b = SplitMix64::new(123);
        for &x in s.iter().take(1000)
        {
            let y = n.sample(&mut r2b);
            assert_eq!(x.to_bits(), y.to_bits());
        }
        let _ = r2.next_f64();
        // sample mean/var near the true values
        let m: f64 = s.iter().sum::<f64>() / s.len() as f64;
        let v: f64 = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (s.len() - 1) as f64;
        assert!((m - 5.0).abs() < 0.05, "mean {m}");
        assert!((v - 4.0).abs() < 0.15, "var {v}");
    }
}

/// Property-based tests for the `Distribution` impls: invariants that must
/// hold for *any* parameter values and *any* point in the support, checked
/// against hundreds of randomly generated inputs rather than a handful of
/// hand-picked reference points.
// Excluded from Miri: proptest runs hundreds of randomized cases per property
// — impractically slow under the interpreter — and its harness is not designed
// to run under Miri. The native Build & Test jobs exercise these properties.
#[cfg(all(test, not(miri)))]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// CDF monotonicity: `x1 <= x2 ⇒ cdf(x1) <= cdf(x2)`. General for any
    /// distribution, independent of how `quantile` is implemented, and a
    /// real bug catcher — a sign error or a bad branch condition in a CDF
    /// formula routinely produces a non-monotone (or out-of-[0,1]) curve.
    fn assert_cdf_monotonic_and_bounded(d: &impl Distribution, lo: f64, hi: f64) {
        let c_lo = d.cdf(lo);
        let c_hi = d.cdf(hi);
        assert!(!c_lo.is_nan() && !c_hi.is_nan(), "cdf produced NaN");
        assert!(
            (-1e-12..=1.0 + 1e-12).contains(&c_lo) && (-1e-12..=1.0 + 1e-12).contains(&c_hi),
            "cdf out of [0,1]: cdf({lo})={c_lo}, cdf({hi})={c_hi}"
        );
        assert!(
            c_lo <= c_hi + 1e-9,
            "cdf not monotone: cdf({lo})={c_lo} > cdf({hi})={c_hi}"
        );
    }

    /// `pdf` must be non-negative and finite everywhere it's defined.
    fn assert_pdf_nonnegative(d: &impl Distribution, x: f64) {
        let p = d.pdf(x);
        assert!(!p.is_nan(), "pdf({x}) is NaN");
        assert!(p >= -1e-12, "pdf({x}) = {p} is negative");
    }

    proptest! {
        #[test]
        fn normal_cdf_monotonic_pdf_nonneg_and_quantile_round_trips(
            mean in -1e3f64..1e3, sd in 1e-2f64..1e3,
            x1 in -1e3f64..1e3, x2 in -1e3f64..1e3,
            p in 0.001f64..0.999,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let n = Normal::new(mean, sd);
            assert_cdf_monotonic_and_bounded(&n, lo, hi);
            assert_pdf_nonnegative(&n, x1);
            // Normal's quantile is a closed form (AS 241), entirely
            // independent of cdf's own erfc-based formula, so this
            // round-trip genuinely cross-checks the two.
            let x = n.quantile(p);
            prop_assert!(rel_close(n.cdf(x), p, 1e-6), "p={p} x={x} cdf(x)={}", n.cdf(x));
        }

        #[test]
        fn exponential_cdf_monotonic_and_pdf_nonneg(
            rate in 1e-3f64..1e3, x1 in 0.0f64..1e4, x2 in 0.0f64..1e4,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let e = Exponential::new(rate);
            assert_cdf_monotonic_and_bounded(&e, lo, hi);
            assert_pdf_nonnegative(&e, x1);
        }

        #[test]
        fn uniform_cdf_monotonic_and_quantile_round_trips(
            a in -1e3f64..1e3, width in 1e-3f64..1e3,
            x1 in -2e3f64..2e3, x2 in -2e3f64..2e3,
            p in 0.0f64..1.0,
        ) {
            let b = a + width;
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let u = Uniform::new(a, b);
            assert_cdf_monotonic_and_bounded(&u, lo, hi);
            let x = u.quantile(p);
            prop_assert!(rel_close(u.cdf(x), p, 1e-9), "p={p} x={x} cdf(x)={}", u.cdf(x));
        }

        /// Gamma's `quantile` bisects its own `cdf`, so `cdf(quantile(p))`
        /// mostly re-confirms bisection converged — still worth checking
        /// (a non-monotone `cdf` would make bisection silently converge to
        /// the wrong root), but the independent cross-check is
        /// `ChiSquared(k) = Gamma(k/2, 2)` below.
        #[test]
        fn gamma_cdf_monotonic_pdf_nonneg_and_quantile_round_trips(
            shape in 1e-2f64..1e3, scale in 1e-2f64..1e3,
            x1 in 0.0f64..1e4, x2 in 0.0f64..1e4,
            p in 0.01f64..0.99,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let g = Gamma::new(shape, scale);
            assert_cdf_monotonic_and_bounded(&g, lo, hi);
            assert_pdf_nonnegative(&g, x1.max(1e-9));
            let p_hat = g.cdf(g.quantile(p));
            prop_assert!(rel_close(p_hat, p, 1e-5), "p={p} p_hat={p_hat}");
        }

        /// Independent cross-check: χ²(k) is defined to be Gamma(k/2, 2),
        /// but `ChiSquared::cdf` calls `regularized_gamma_p` directly
        /// instead of delegating to `Gamma::cdf` — two separately written
        /// expressions that must agree. Catches a parameter-transcription
        /// bug (e.g. `k` vs `k/2`, or a wrong scale) that a self-consistency
        /// check within `ChiSquared` alone could never see.
        #[test]
        fn chi_squared_matches_the_equivalent_gamma_distribution(
            k in 0.1f64..500.0, x in 0.0f64..2000.0,
        ) {
            let c = ChiSquared::new(k);
            let g = Gamma::new(k / 2.0, 2.0);
            prop_assert!(rel_close(c.cdf(x), g.cdf(x), 1e-9), "k={k} x={x} chi2={} gamma={}", c.cdf(x), g.cdf(x));
        }

        #[test]
        fn chi_squared_cdf_monotonic_and_pdf_nonneg(
            k in 0.1f64..500.0, x1 in 0.0f64..2000.0, x2 in 0.0f64..2000.0,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let c = ChiSquared::new(k);
            assert_cdf_monotonic_and_bounded(&c, lo, hi);
            assert_pdf_nonnegative(&c, x1.max(1e-9));
        }

        #[test]
        fn student_t_cdf_monotonic_and_symmetric(
            nu in 0.5f64..500.0, x1 in -100.0f64..100.0, x2 in -100.0f64..100.0,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let t = StudentT::new(nu);
            assert_cdf_monotonic_and_bounded(&t, lo, hi);
            assert_pdf_nonnegative(&t, x1);
            // Genuine symmetry check: cdf(-x) and cdf(x) take different
            // branches of the `if t >= 0.0` split in `StudentT::cdf`, so
            // this is not definitionally 1 − cdf(x).
            prop_assert!(
                rel_close(t.cdf(-x1) + t.cdf(x1), 1.0, 1e-6),
                "nu={nu} x1={x1} cdf(-x1)={} cdf(x1)={}", t.cdf(-x1), t.cdf(x1)
            );
        }

        #[test]
        fn fisher_f_cdf_monotonic_pdf_nonneg_and_complementary(
            d1 in 0.5f64..200.0, d2 in 0.5f64..200.0, x1 in 0.001f64..1e3, x2 in 0.001f64..1e3,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let f = FisherF::new(d1, d2);
            assert_cdf_monotonic_and_bounded(&f, lo, hi);
            assert_pdf_nonnegative(&f, x1);
            // `sf` is its own regularized_incomplete_beta call with swapped
            // shape parameters and a different argument, not `1 - cdf`, so
            // this is a genuine cross-check (mirrors the incomplete-beta
            // symmetry identity in scirust-special).
            prop_assert!(
                rel_close(f.cdf(x1) + f.sf(x1), 1.0, 1e-6),
                "d1={d1} d2={d2} x1={x1} cdf={} sf={}", f.cdf(x1), f.sf(x1)
            );
        }

        #[test]
        fn beta_cdf_monotonic_and_pdf_nonneg(
            a in 1e-2f64..500.0, b in 1e-2f64..500.0,
            x1 in 0.0f64..1.0, x2 in 0.0f64..1.0,
        ) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let beta = Beta::new(a, b);
            assert_cdf_monotonic_and_bounded(&beta, lo, hi);
            assert_pdf_nonnegative(&beta, x1);
        }

        /// Quantile round trip, restricted to `a, b >= 0.5`: this property
        /// test itself found that below that, `Beta(a, b)`'s mass can
        /// concentrate within a single `f64` ULP of an endpoint (e.g.
        /// `Beta(390, 0.01)` needs `x` resolved to better than 1e-16 near
        /// `x = 1` to tell `p = 0.5` from `p = 0.99` apart) — a genuine
        /// floating-point representability limit, not a bug `invert_cdf`
        /// can bisect its way around. See `beta_quantile_round_trip_at_the_
        /// edge_of_f64_resolution` below for what `invert_cdf`'s tolerance
        /// fix actually improved in that regime.
        #[test]
        fn beta_quantile_round_trips_away_from_the_f64_resolution_limit(
            a in 0.5f64..500.0, b in 0.5f64..500.0,
            p in 0.01f64..0.99,
        ) {
            let beta = Beta::new(a, b);
            let p_hat = beta.cdf(beta.quantile(p));
            prop_assert!(rel_close(p_hat, p, 1e-5), "a={a} b={b} p={p} p_hat={p_hat}");
        }
    }
}
