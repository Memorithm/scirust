//! Discrete probability distributions with a unified [`DiscreteDistribution`]
//! trait.
//!
//! Tail probabilities reuse the audited `scirust-special` primitives wherever
//! a closed identity exists — the binomial CDF through the regularized
//! incomplete beta, the Poisson CDF through the regularized incomplete gamma —
//! so no pmf summation loop is needed for the two workhorse laws. The
//! hypergeometric CDF sums its (finite, at most `draws + 1` term) support
//! directly from exact log-space pmfs.
//!
//! Conventions match SciPy: `cdf(k) = P(X ≤ k)`, `sf(k) = P(X > k)`, and
//! `quantile(p)` is the smallest `k` with `cdf(k) ≥ p`. [`Geometric`] counts
//! the number of trials up to and including the first success (support
//! `k ≥ 1`, SciPy's `geom`), not the number of failures (R's `dgeom`);
//! [`NegativeBinomial`] counts the failures before the `r`-th success
//! (SciPy's `nbinom`). [`Skellam`] lives on all of ℤ, so it exposes its own
//! `i64` methods instead of the non-negative-integer trait.

use crate::comb::ln_binomial;
use crate::rng::SplitMix64;
use scirust_special::{
    binom_deviance, ln_binomial_pmf, ln_poisson_pmf, regularized_gamma_p, regularized_gamma_q,
    regularized_incomplete_beta, riemann_zeta, riemann_zeta_tail, stirling_error,
};

/// `ln(2π)`, the constant of Stirling's formula.
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// A univariate distribution on the non-negative integers.
///
/// `pmf` defaults to `exp(ln_pmf)`; `quantile` defaults to a deterministic
/// bracket-and-bisect on the CDF (smallest `k` with `cdf(k) ≥ p`, the SciPy
/// `ppf` convention); `sample` draws by inverse-CDF transform from a seeded
/// [`SplitMix64`], so every draw is reproducible bit-for-bit.
pub trait DiscreteDistribution {
    /// Natural log of the probability mass at `k` (`−∞` outside the support).
    fn ln_pmf(&self, k: u64) -> f64;
    /// Cumulative distribution `P(X ≤ k)`.
    fn cdf(&self, k: u64) -> f64;
    /// Distribution mean.
    fn mean(&self) -> f64;
    /// Distribution variance.
    fn variance(&self) -> f64;

    /// Probability mass `P(X = k)`.
    fn pmf(&self, k: u64) -> f64 {
        self.ln_pmf(k).exp()
    }
    /// Survival function `P(X > k) = 1 − cdf(k)`. Override for tail accuracy.
    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// Log of the CDF, `ln P(X ≤ k)` (SciPy's `logcdf`). Defaults to
    /// `ln(cdf)`; since every override computes `cdf` directly this stays
    /// accurate in the lower tail where `cdf` itself does not cancel.
    fn logcdf(&self, k: u64) -> f64 {
        self.cdf(k).ln()
    }
    /// Log of the survival function, `ln P(X > k)` (SciPy's `logsf`).
    /// Defaults to `ln(sf)`; because `sf` is overridden to a direct upper-tail
    /// form on every distribution here, this avoids the `ln(1 − cdf)`
    /// catastrophic cancellation of the far tail.
    fn logsf(&self, k: u64) -> f64 {
        self.sf(k).ln()
    }
    /// Inverse survival function: smallest `k` with `sf(k) ≤ p`, i.e.
    /// `quantile(1 − p)` evaluated through the direct `sf` (SciPy's `isf`).
    ///
    /// More accurate than `quantile(1 − p)` for tiny `p`, where forming
    /// `1 − p` loses precision. Deterministic bracket-and-bisect on `sf`.
    fn isf(&self, p: f64) -> u64 {
        let pt = p.clamp(0.0, 1.0);
        // sf is non-increasing; want the smallest k with sf(k) <= pt.
        if self.sf(0) <= pt
        {
            return 0;
        }
        let guess = self.mean() + 10.0 * self.std_dev();
        let mut hi: u64 = if guess.is_finite() && guess >= 1.0
        {
            guess.ceil() as u64
        }
        else
        {
            1
        };
        let mut guard = 0;
        while self.sf(hi) > pt && guard < 200
        {
            hi = hi.saturating_mul(2);
            guard += 1;
        }
        let mut lo: u64 = 0;
        while lo < hi
        {
            let mid = lo + (hi - lo) / 2;
            if self.sf(mid) > pt
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        lo
    }
    /// Standard deviation, `sqrt(variance)`.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Smallest `k` such that `cdf(k) ≥ p` (percent-point function).
    ///
    /// `p` is clamped to `[0, 1]`; `p ≤ cdf(0)` gives `0`. Deterministic:
    /// exponential bracket expansion, then binary search.
    fn quantile(&self, p: f64) -> u64 {
        let pt = p.clamp(0.0, 1.0);
        if self.cdf(0) >= pt
        {
            return 0;
        }
        // Expand an upper bracket from a moment-based guess. On finite
        // supports the CDF reaches exactly 1.0, so the loop terminates.
        let guess = self.mean() + 10.0 * self.std_dev();
        let mut hi: u64 = if guess.is_finite() && guess >= 1.0
        {
            guess.ceil() as u64
        }
        else
        {
            1
        };
        let mut guard = 0;
        while self.cdf(hi) < pt && guard < 200
        {
            hi = hi.saturating_mul(2);
            guard += 1;
        }
        // Smallest k in (0, hi] with cdf(k) >= pt.
        let mut lo: u64 = 0;
        while lo < hi
        {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) < pt
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        lo
    }
    /// Equal-tailed `confidence`-level interval `(low, high)` such that
    /// `P(X < low) ≤ (1−c)/2` and `P(X > high) ≤ (1−c)/2` (SciPy's
    /// `interval`): `low = quantile((1−c)/2)`, `high = quantile((1+c)/2)`.
    fn interval(&self, confidence: f64) -> (u64, u64) {
        let c = confidence.clamp(0.0, 1.0);
        (
            self.quantile((1.0 - c) / 2.0),
            self.quantile((1.0 + c) / 2.0),
        )
    }
    /// Expectation `E[f(X)] = Σ_k f(k)·pmf(k)` over the support (SciPy's
    /// `expect`). Deterministic finite sum: accumulates until the remaining
    /// tail mass `sf(k)` is negligible, with a hard term cap as a backstop.
    ///
    /// Assumes `f` grows slower than the tail decays (true for moments of the
    /// light-tailed laws); for a heavy-tailed law whose moment diverges the
    /// truncated sum is only a partial sum, by construction.
    fn expect(&self, f: &dyn Fn(u64) -> f64) -> f64 {
        let mut acc = 0.0;
        let mut k: u64 = 0;
        loop
        {
            acc += self.pmf(k) * f(k);
            if (k > 0 && self.sf(k) < 1e-16) || k >= 10_000_000
            {
                break;
            }
            k += 1;
        }
        acc
    }
    /// One deterministic draw via inverse-CDF from a seeded uniform source.
    fn sample(&self, rng: &mut SplitMix64) -> u64 {
        let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
        self.quantile(u)
    }
}

// ============================================================ //
//  Binomial                                                    //
// ============================================================ //

/// Binomial distribution: number of successes in `n` independent trials with
/// success probability `p`.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// `n` trials with success probability `p ∈ [0, 1]`.
    pub fn new(n: u64, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Binomial: p must be within [0, 1]"
        );
        Self { n, p }
    }
}

impl DiscreteDistribution for Binomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        // Loader's saddle-point form: full relative accuracy at large n and
        // exact `p = 0`/`p = 1`/`k = 0`/`k = n` endpoints (see scirust-special).
        ln_binomial_pmf(k, self.n, self.p)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 1.0;
        }
        // P(X ≤ k) = I_{1−p}(n − k, k + 1).
        regularized_incomplete_beta((self.n - k) as f64, k as f64 + 1.0, 1.0 - self.p)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 0.0;
        }
        // P(X > k) = I_p(k + 1, n − k) — direct form, no 1 − cdf cancellation.
        regularized_incomplete_beta(k as f64 + 1.0, (self.n - k) as f64, self.p)
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

// ============================================================ //
//  Poisson                                                     //
// ============================================================ //

/// Poisson distribution: count of events at mean rate `λ > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Mean rate `λ > 0`.
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda > 0.0 && lambda.is_finite(),
            "Poisson: λ must be finite and > 0"
        );
        Self { lambda }
    }

    /// Fit by the method of moments: `λ̂ = mean(data)` (which is also the MLE).
    /// `None` for an empty sample or a non-positive mean.
    pub fn fit_mom(data: &[f64]) -> Option<Self> {
        if data.is_empty()
        {
            return None;
        }
        let m = data.iter().sum::<f64>() / data.len() as f64;
        (m > 0.0 && m.is_finite()).then(|| Self::new(m))
    }
}

impl DiscreteDistribution for Poisson {
    fn ln_pmf(&self, k: u64) -> f64 {
        // Loader's saddle-point form: full relative accuracy at large λ.
        ln_poisson_pmf(k, self.lambda)
    }
    fn cdf(&self, k: u64) -> f64 {
        // P(X ≤ k) = Q(k + 1, λ).
        regularized_gamma_q(k as f64 + 1.0, self.lambda)
    }
    fn sf(&self, k: u64) -> f64 {
        // P(X > k) = P(k + 1, λ) — direct lower tail, accurate far out.
        regularized_gamma_p(k as f64 + 1.0, self.lambda)
    }
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
}

// ============================================================ //
//  Hypergeometric                                              //
// ============================================================ //

/// Hypergeometric distribution: number of marked items in a sample of
/// `draws` taken *without replacement* from a population of size `population`
/// containing `successes` marked items.
///
/// This is the law that governs lottery matches (see [`crate::lottery`]),
/// acceptance sampling, and capture–recapture estimates.
#[derive(Debug, Clone, Copy)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
}

impl Hypergeometric {
    /// Population `N ≥ 1` containing `K ≤ N` marked items, sampled `n ≤ N`
    /// times without replacement.
    pub fn new(population: u64, successes: u64, draws: u64) -> Self {
        assert!(population >= 1, "Hypergeometric: population must be ≥ 1");
        assert!(
            successes <= population && draws <= population,
            "Hypergeometric: require successes ≤ population and draws ≤ population"
        );
        Self {
            population,
            successes,
            draws,
        }
    }
    /// Smallest attainable count, `max(0, draws + successes − population)`.
    pub fn support_min(&self) -> u64 {
        (self.draws + self.successes).saturating_sub(self.population)
    }
    /// Largest attainable count, `min(draws, successes)`.
    pub fn support_max(&self) -> u64 {
        self.draws.min(self.successes)
    }
}

impl DiscreteDistribution for Hypergeometric {
    fn ln_pmf(&self, k: u64) -> f64 {
        // Outside the support one of the two ln C(·,·) terms is −∞.
        if k > self.draws
        {
            return f64::NEG_INFINITY;
        }
        ln_binomial(self.successes, k)
            + ln_binomial(self.population - self.successes, self.draws - k)
            - ln_binomial(self.population, self.draws)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.support_max()
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for i in self.support_min()..=k
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        // Sum the (often much shorter) upper tail directly.
        let hi = self.support_max();
        if k >= hi
        {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in (k + 1).max(self.support_min())..=hi
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn mean(&self) -> f64 {
        self.draws as f64 * self.successes as f64 / self.population as f64
    }
    fn variance(&self) -> f64 {
        let (nn, kk, n) = (
            self.population as f64,
            self.successes as f64,
            self.draws as f64,
        );
        if self.population == 1
        {
            return 0.0;
        }
        n * (kk / nn) * (1.0 - kk / nn) * (nn - n) / (nn - 1.0)
    }
}

// ============================================================ //
//  Geometric                                                   //
// ============================================================ //

/// Geometric distribution: number of Bernoulli(`p`) trials up to and
/// including the first success. Support `k ≥ 1` (SciPy's `geom` convention).
#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Per-trial success probability `p ∈ (0, 1]`.
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p <= 1.0, "Geometric: p must be within (0, 1]");
        Self { p }
    }
    /// `ln(1 − p)`, computed as `ln_1p(−p)` for accuracy at small `p`.
    fn ln_q(&self) -> f64 {
        (-self.p).ln_1p()
    }

    /// Fit by the method of moments (support `k ≥ 1`): `p̂ = 1/mean(data)`.
    /// `None` unless the sample mean is `≥ 1` (the support's minimum).
    pub fn fit_mom(data: &[f64]) -> Option<Self> {
        if data.is_empty()
        {
            return None;
        }
        let m = data.iter().sum::<f64>() / data.len() as f64;
        (m >= 1.0 && m.is_finite()).then(|| Self::new(1.0 / m))
    }
}

impl DiscreteDistribution for Geometric {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        // Guard k = 1 so p = 1 avoids 0·ln(0).
        let tail = if k > 1
        {
            (k - 1) as f64 * self.ln_q()
        }
        else
        {
            0.0
        };
        self.p.ln() + tail
    }
    fn cdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        // 1 − (1 − p)^k without cancellation.
        -(k as f64 * self.ln_q()).exp_m1()
    }
    fn sf(&self, k: u64) -> f64 {
        // Guard k = 0 so p = 1 (ln q = −∞) avoids 0·(−∞) = NaN: the whole
        // mass lies at k ≥ 1, so P(X > 0) = 1 for every p.
        if k == 0
        {
            return 1.0;
        }
        (k as f64 * self.ln_q()).exp()
    }
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}

// ============================================================ //
//  Negative binomial                                           //
// ============================================================ //

/// Negative binomial distribution: number of **failures** before the `r`-th
/// success of independent Bernoulli(`p`) trials (SciPy's `nbinom` convention;
/// R's `dnbinom` counts the same way). `r` may be real-valued (the
/// Pólya / overdispersed-Poisson parametrization used in count regression).
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

impl NegativeBinomial {
    /// `r > 0` successes (possibly non-integer), per-trial success
    /// probability `p ∈ (0, 1]`.
    pub fn new(r: f64, p: f64) -> Self {
        assert!(
            r > 0.0 && r.is_finite(),
            "NegativeBinomial: r must be finite and > 0"
        );
        assert!(
            p > 0.0 && p <= 1.0,
            "NegativeBinomial: p must be within (0, 1]"
        );
        Self { r, p }
    }

    /// Fit by the method of moments from the sample mean `m` and unbiased
    /// sample variance `v`: `p̂ = m/v`, `r̂ = m²/(v − m)`.
    ///
    /// Defined only under **overdispersion** (`v > m`, the regime the negative
    /// binomial models); returns `None` otherwise (including `v ≤ m`, which a
    /// Poisson fits better) and for fewer than two observations.
    pub fn fit_mom(data: &[f64]) -> Option<Self> {
        if data.len() < 2
        {
            return None;
        }
        let n = data.len() as f64;
        let m = data.iter().sum::<f64>() / n;
        let v = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        // Defined only under overdispersion (v > m) with a positive mean.
        (m > 0.0 && v.is_finite() && v > m).then(|| Self::new(m * m / (v - m), m / v))
    }
}

impl DiscreteDistribution for NegativeBinomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        let (r, p) = (self.r, self.p);
        if p == 1.0
        {
            // All mass at k = 0.
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if k == 0
        {
            return r * p.ln();
        }
        // pmf(k) = r/(r+k) · C(r+k, r)·p^r·q^k, the binomial term written in
        // Loader's saddle-point form (as R's `dnbinom`). Summing
        // `ln Γ(k+r) − ln Γ(r) − ln k! + r·ln p + k·ln q` directly cancels
        // terms of size ~k·ln k and loses every digit once k or r nears 1e15.
        let kf = k as f64;
        let n = r + kf;
        let q = 1.0 - p;
        let lc = stirling_error(n)
            - stirling_error(r)
            - stirling_error(kf)
            - binom_deviance(r, n * p)
            - binom_deviance(kf, n * q);
        // ln(r/n) − ½·ln(2π·r·k/n).
        (r / n).ln() + lc - 0.5 * (LN_2PI + r.ln() + kf.ln() - n.ln())
    }
    fn cdf(&self, k: u64) -> f64 {
        // P(X ≤ k) = I_p(r, k + 1).
        regularized_incomplete_beta(self.r, k as f64 + 1.0, self.p)
    }
    fn sf(&self, k: u64) -> f64 {
        // P(X > k) = I_{1−p}(k + 1, r) — direct upper tail.
        regularized_incomplete_beta(k as f64 + 1.0, self.r, 1.0 - self.p)
    }
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }
    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }
}

// ============================================================ //
//  Beta-binomial                                               //
// ============================================================ //

/// Beta-binomial distribution: a Binomial(`n`, `p`) whose `p` is itself
/// Beta(`a`, `b`)-distributed — the standard model for overdispersed
/// proportions (defect rates varying batch to batch, per-site response
/// rates…). `a = b = 1` reduces to the discrete uniform on `0..=n`.
#[derive(Debug, Clone, Copy)]
pub struct BetaBinomial {
    n: u64,
    a: f64,
    b: f64,
}

impl BetaBinomial {
    /// `n` trials, Beta shape parameters `a > 0`, `b > 0`.
    pub fn new(n: u64, a: f64, b: f64) -> Self {
        assert!(
            a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite(),
            "BetaBinomial: shapes must be finite and > 0"
        );
        Self { n, a, b }
    }
}

impl DiscreteDistribution for BetaBinomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k > self.n
        {
            return f64::NEG_INFINITY;
        }
        // The two-category Pólya form; summing `ln C(n,k) + ln B(k+a, n−k+b)
        // − ln B(a,b)` cancels terms of size ~n·ln 2 at large n.
        ln_polya_pmf(&[k, self.n - k], &[self.a, self.b], self.n, self.a + self.b)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for i in 0..=k
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 0.0;
        }
        // Direct upper-tail sum over the finite support.
        let mut acc = 0.0;
        for i in (k + 1)..=self.n
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let (n, a, b) = (self.n as f64, self.a, self.b);
        let s = a + b;
        n * a * b * (s + n) / (s * s * (s + 1.0))
    }
}

// ============================================================ //
//  Zipfian (finite)                                            //
// ============================================================ //

/// Finite Zipfian distribution on ranks `1..=n`: `pmf(k) ∝ k^(−s)`
/// (SciPy's `zipfian`). The rank-frequency law of natural language, city
/// sizes, and access patterns; `s = 0` is the discrete uniform on `1..=n`.
///
/// The infinite-support zeta distribution (SciPy's `zipf`) needs the Riemann
/// ζ function and is deliberately not approximated here.
#[derive(Debug, Clone, Copy)]
pub struct Zipfian {
    s: f64,
    n: u64,
    /// Generalized harmonic normalizer `H(n, s) = Σ_{j=1..n} j^(−s)`,
    /// pre-summed smallest-terms-first in a fixed order (deterministic).
    h: f64,
}

impl Zipfian {
    /// Exponent `s ≥ 0` over ranks `1..=n`, `n ≥ 1`.
    pub fn new(s: f64, n: u64) -> Self {
        assert!(
            s >= 0.0 && s.is_finite(),
            "Zipfian: s must be finite and ≥ 0"
        );
        assert!(n >= 1, "Zipfian: n must be ≥ 1");
        Self {
            s,
            n,
            h: Self::harmonic(s, n),
        }
    }
    /// `H(n, s) = Σ_{j=1..n} j^(−s)` summed descending (small terms first).
    fn harmonic(s: f64, n: u64) -> f64 {
        let mut acc = 0.0;
        for j in (1..=n).rev()
        {
            acc += (j as f64).powf(-s);
        }
        acc
    }
    /// `Σ_{j=1..n} j^(power)` with the same deterministic order.
    fn power_sum(&self, power: f64) -> f64 {
        let mut acc = 0.0;
        for j in (1..=self.n).rev()
        {
            acc += (j as f64).powf(power);
        }
        acc
    }
}

impl DiscreteDistribution for Zipfian {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0 || k > self.n
        {
            return f64::NEG_INFINITY;
        }
        -self.s * (k as f64).ln() - self.h.ln()
    }
    fn pmf(&self, k: u64) -> f64 {
        if k == 0 || k > self.n
        {
            return 0.0;
        }
        (k as f64).powf(-self.s) / self.h
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for j in (1..=k.min(self.n)).rev()
        {
            acc += (j as f64).powf(-self.s);
        }
        (acc / self.h).min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 0.0;
        }
        let mut acc = 0.0;
        for j in ((k + 1)..=self.n).rev()
        {
            acc += (j as f64).powf(-self.s);
        }
        (acc / self.h).min(1.0)
    }
    fn mean(&self) -> f64 {
        // Σ k·k^(−s) / H = H(n, s−1) / H(n, s).
        self.power_sum(1.0 - self.s) / self.h
    }
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.power_sum(2.0 - self.s) / self.h - m * m
    }
}

// ============================================================ //
//  Skellam (support ℤ — outside the u64 trait)                 //
// ============================================================ //

/// Skellam distribution: the difference `X₁ − X₂` of two independent Poisson
/// counts with rates `μ₁` and `μ₂` (score differences, detector count
/// differences, queue drift…).
///
/// Its support is **all of ℤ**, so it deliberately does not implement
/// [`DiscreteDistribution`] (which lives on the non-negative integers);
/// the same method names are provided over `i64`. The pmf is evaluated by
/// the defining convolution `Σ_j pois₁(k + j)·pois₂(j)` with a fixed
/// deterministic truncation rule (stop once terms fall below 1e-18 of the
/// running peak past the summand's mode) rather than via Bessel `I_k`, so it
/// stays on the audited `scirust-special` base; accuracy vs SciPy is ~1e-12.
#[derive(Debug, Clone, Copy)]
pub struct Skellam {
    mu1: f64,
    mu2: f64,
}

impl Skellam {
    /// Rates `μ₁ > 0`, `μ₂ > 0` of the two Poisson components.
    pub fn new(mu1: f64, mu2: f64) -> Self {
        assert!(
            mu1 > 0.0 && mu1.is_finite() && mu2 > 0.0 && mu2.is_finite(),
            "Skellam: both rates must be finite and > 0"
        );
        Self { mu1, mu2 }
    }

    /// Convolution engine: `Σ_{j ≥ j0} w(j)` where `w` climbs to a single
    /// peak then decays super-exponentially. Deterministic truncation.
    fn convolve(&self, j0: u64, term: impl Fn(u64) -> f64) -> f64 {
        let mut acc = 0.0;
        let mut peak = 0.0_f64;
        let mut j = j0;
        loop
        {
            let t = term(j);
            acc += t;
            peak = peak.max(t);
            // Past the peak and negligible: stop. The +8 floor makes the
            // rule fixed for tiny rates too.
            if (t < peak * 1e-18 && j > j0 + 8) || j > j0 + 100_000
            {
                break;
            }
            j += 1;
        }
        acc
    }

    /// Probability mass `P(X₁ − X₂ = k)`, `k ∈ ℤ`.
    pub fn pmf(&self, k: i64) -> f64 {
        let p1 = Poisson::new(self.mu1);
        let p2 = Poisson::new(self.mu2);
        // X₁ = k + j, X₂ = j, j ≥ max(0, −k).
        let j0 = (-k).max(0) as u64;
        self.convolve(j0, |j| {
            (p1.ln_pmf((k + j as i64) as u64) + p2.ln_pmf(j)).exp()
        })
    }

    /// Cumulative distribution `P(X₁ − X₂ ≤ k)`.
    pub fn cdf(&self, k: i64) -> f64 {
        let p1 = Poisson::new(self.mu1);
        let p2 = Poisson::new(self.mu2);
        // Condition on X₂ = j: P(X₁ ≤ k + j); zero until k + j ≥ 0.
        let j0 = (-k).max(0) as u64;
        self.convolve(j0, |j| p2.pmf(j) * p1.cdf((k + j as i64) as u64))
            .min(1.0)
    }

    /// Survival function `P(X₁ − X₂ > k)`, summed directly (no `1 − cdf`).
    pub fn sf(&self, k: i64) -> f64 {
        let p1 = Poisson::new(self.mu1);
        let p2 = Poisson::new(self.mu2);
        // Condition on X₂ = j: P(X₁ > k + j), which is 1 until k + j ≥ 0.
        let mut acc = 0.0;
        // Terms with k + j < 0 contribute pois₂(j) whole.
        if k < 0
        {
            for j in 0..((-k) as u64)
            {
                acc += p2.pmf(j);
            }
        }
        let j0 = (-k).max(0) as u64;
        acc + self.convolve(j0, |j| p2.pmf(j) * p1.sf((k + j as i64) as u64))
    }

    /// Mean `μ₁ − μ₂`.
    pub fn mean(&self) -> f64 {
        self.mu1 - self.mu2
    }
    /// Variance `μ₁ + μ₂`.
    pub fn variance(&self) -> f64 {
        self.mu1 + self.mu2
    }
    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// One deterministic draw as the difference of two inverse-CDF Poisson
    /// draws consuming the rng in a fixed order (X₁ first, then X₂).
    pub fn sample(&self, rng: &mut SplitMix64) -> i64 {
        let x1 = Poisson::new(self.mu1).sample(rng) as i64;
        let x2 = Poisson::new(self.mu2).sample(rng) as i64;
        x1 - x2
    }
}

// ============================================================ //
//  Zeta (infinite Zipf)                                        //
// ============================================================ //

/// Zeta distribution on `k ≥ 1`: `pmf(k) = k^(−s) / ζ(s)`, `s > 1` (SciPy's
/// `zipf`). The infinite-support limit of [`Zipfian`], now expressible since
/// `scirust-special` provides `riemann_zeta`.
///
/// The far tail is handled without `ζ(s) − partial-sum` cancellation via the
/// Euler–Maclaurin tail (`riemann_zeta_tail`), so `sf`/`cdf` are O(1) for
/// `k ≥ 19` — which keeps the default bracket-and-bisect `quantile` usable
/// even in the heavy-tail regime `s ≤ 2` where the mean is infinite.
#[derive(Debug, Clone, Copy)]
pub struct Zeta {
    s: f64,
    zeta_s: f64,
}

impl Zeta {
    /// Exponent `s > 1` (the pmf is not normalizable at `s ≤ 1`).
    pub fn new(s: f64) -> Self {
        assert!(s > 1.0 && s.is_finite(), "Zeta: s must be finite and > 1");
        Self {
            s,
            zeta_s: riemann_zeta(s),
        }
    }
}

impl DiscreteDistribution for Zeta {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        -self.s * (k as f64).ln() - self.zeta_s.ln()
    }
    fn pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        (k as f64).powf(-self.s) / self.zeta_s
    }
    fn cdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        if k < 20
        {
            // Short head: direct sum, no cancellation.
            let mut acc = 0.0;
            for j in (1..=k).rev()
            {
                acc += (j as f64).powf(-self.s);
            }
            return (acc / self.zeta_s).min(1.0);
        }
        // cdf ≈ 1 here; the tiny complement carries the accuracy.
        1.0 - self.sf(k)
    }
    fn sf(&self, k: u64) -> f64 {
        let t = if k < 19
        {
            // Tail = the few explicit terms up to 19 plus the E–M remainder.
            let mut acc = riemann_zeta_tail(self.s, 20.0);
            for j in ((k + 1)..20).rev()
            {
                acc += (j as f64).powf(-self.s);
            }
            acc
        }
        else
        {
            riemann_zeta_tail(self.s, k as f64 + 1.0)
        };
        (t / self.zeta_s).min(1.0)
    }
    fn mean(&self) -> f64 {
        if self.s > 2.0
        {
            riemann_zeta(self.s - 1.0) / self.zeta_s
        }
        else
        {
            f64::INFINITY
        }
    }
    fn variance(&self) -> f64 {
        if self.s > 3.0
        {
            let m = self.mean();
            riemann_zeta(self.s - 2.0) / self.zeta_s - m * m
        }
        else
        {
            f64::INFINITY
        }
    }
}

// ============================================================ //
//  Poisson-binomial                                            //
// ============================================================ //

/// Poisson-binomial distribution: number of successes among `n` independent
/// Bernoulli trials with **heterogeneous** probabilities `p₁ … pₙ` — the
/// exact law of "how many of these n distinct risky events occur" (system
/// reliability, portfolio defaults, per-lot defect counts).
///
/// The full mass vector is computed once at construction by the standard
/// O(n²) convolution recurrence — exact, deterministic, no FFT round-off —
/// so `pmf`/`cdf`/`sf` are table lookups afterwards.
#[derive(Debug, Clone)]
pub struct PoissonBinomial {
    mass: Vec<f64>,
    mean: f64,
    var: f64,
}

impl PoissonBinomial {
    /// Success probabilities, each in `[0, 1]`; at least one trial.
    pub fn new(probs: &[f64]) -> Self {
        assert!(
            !probs.is_empty(),
            "PoissonBinomial: need at least one trial"
        );
        assert!(
            probs.iter().all(|&p| (0.0..=1.0).contains(&p)),
            "PoissonBinomial: every probability must be within [0, 1]"
        );
        let mut mass = vec![0.0; probs.len() + 1];
        mass[0] = 1.0;
        for (i, &p) in probs.iter().enumerate()
        {
            for k in (1..=i + 1).rev()
            {
                mass[k] = mass[k] * (1.0 - p) + mass[k - 1] * p;
            }
            mass[0] *= 1.0 - p;
        }
        let mean = probs.iter().sum();
        let var = probs.iter().map(|&p| p * (1.0 - p)).sum();
        Self { mass, mean, var }
    }
    /// Number of trials `n`.
    pub fn trials(&self) -> u64 {
        (self.mass.len() - 1) as u64
    }
}

impl DiscreteDistribution for PoissonBinomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        self.pmf(k).ln()
    }
    fn pmf(&self, k: u64) -> f64 {
        usize::try_from(k)
            .ok()
            .and_then(|i| self.mass.get(i))
            .copied()
            .unwrap_or(0.0)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.trials()
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for i in 0..=k as usize
        {
            acc += self.mass[i];
        }
        acc.min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.trials()
        {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in (k as usize + 1)..self.mass.len()
        {
            acc += self.mass[i];
        }
        acc.min(1.0)
    }
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.var
    }
}

// ============================================================ //
//  Multinomial (vector-valued — outside the univariate trait)  //
// ============================================================ //

/// Multinomial distribution: `n` independent trials, each landing in one of
/// `m ≥ 2` categories with probabilities `p₁ … pₘ`; the outcome is the vector
/// of category counts. Vector-valued, so it exposes its own slice-based API
/// instead of the univariate [`DiscreteDistribution`] trait.
#[derive(Debug, Clone)]
pub struct Multinomial {
    n: u64,
    probs: Vec<f64>,
}

impl Multinomial {
    /// `n` trials over `probs.len() ≥ 2` categories; probabilities must be
    /// non-negative and sum to 1 within 1e-9 (they are renormalized exactly).
    pub fn new(n: u64, probs: &[f64]) -> Self {
        assert!(
            probs.len() >= 2,
            "Multinomial: need at least two categories"
        );
        assert!(
            probs.iter().all(|&p| p >= 0.0 && p.is_finite()),
            "Multinomial: probabilities must be finite and ≥ 0"
        );
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-9,
            "Multinomial: probabilities must sum to 1"
        );
        Self {
            n,
            probs: probs.iter().map(|&p| p / total).collect(),
        }
    }

    /// Natural log of `P(counts)`; `−∞` unless `Σ counts = n` (and every
    /// zero-probability category has a zero count). Panics if `counts` has
    /// the wrong length.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.probs.len(),
            "Multinomial: counts length must match the number of categories"
        );
        if counts.iter().sum::<u64>() != self.n
        {
            return f64::NEG_INFINITY;
        }
        if counts
            .iter()
            .zip(&self.probs)
            .any(|(&k, &p)| k > 0 && p == 0.0)
        {
            return f64::NEG_INFINITY;
        }
        if let Some(j) = counts.iter().position(|&k| k == self.n)
        {
            // One category holds every trial (also covers n = 0).
            return if self.n == 0
            {
                0.0
            }
            else
            {
                self.n as f64 * self.probs[j].ln()
            };
        }
        // Loader's saddle-point form, the multivariate analogue of the
        // binomial pmf. With δ the Stirling remainder, D₀ the binomial
        // deviance and P₀ the total probability of the empty categories,
        // `ln n! − Σ ln kᵢ! + Σ kᵢ ln pᵢ` equals
        // `δ(n) − Σ_{kᵢ>0}[δ(kᵢ) + D₀(kᵢ, n·pᵢ) + ½ln kᵢ] + ½ln n
        //  − (m₊ − 1)·½ln 2π − n·P₀`, where m₊ counts the non-empty
        // categories. Every term stays small near the mode, whereas the
        // factorial sum cancels values of size ~n·ln n.
        let n = self.n as f64;
        let mut acc = stirling_error(n) + 0.5 * n.ln() + 0.5 * LN_2PI;
        let mut empty_mass = 0.0;
        for (&k, &p) in counts.iter().zip(&self.probs)
        {
            if k == 0
            {
                empty_mass += p;
                continue;
            }
            let kf = k as f64;
            acc -= stirling_error(kf) + binom_deviance(kf, n * p) + 0.5 * (kf.ln() + LN_2PI);
        }
        acc - n * empty_mass
    }
    /// Probability mass `P(counts)`.
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
    /// Mean vector `n·pᵢ`.
    pub fn mean(&self) -> Vec<f64> {
        self.probs.iter().map(|&p| self.n as f64 * p).collect()
    }
    /// Covariance matrix: `n·pᵢ(1−pᵢ)` on the diagonal, `−n·pᵢpⱼ` off it.
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.n as f64;
        self.probs
            .iter()
            .enumerate()
            .map(|(i, &pi)| {
                self.probs
                    .iter()
                    .enumerate()
                    .map(|(j, &pj)| {
                        if i == j
                        {
                            n * pi * (1.0 - pi)
                        }
                        else
                        {
                            -n * pi * pj
                        }
                    })
                    .collect()
            })
            .collect()
    }
    /// One deterministic draw: sequential conditional binomials, one uniform
    /// consumed per category except the last (fixed order ⇒ reproducible).
    pub fn sample(&self, rng: &mut SplitMix64) -> Vec<u64> {
        let m = self.probs.len();
        let mut out = Vec::with_capacity(m);
        let mut remaining = self.n;
        let mut rest = 1.0_f64;
        for (i, &p) in self.probs.iter().enumerate()
        {
            if i + 1 == m
            {
                out.push(remaining);
                break;
            }
            let cond = if rest > 0.0
            {
                (p / rest).clamp(0.0, 1.0)
            }
            else
            {
                1.0
            };
            let k = Binomial::new(remaining, cond).sample(rng);
            out.push(k);
            remaining -= k;
            rest -= p;
        }
        out
    }
}

// ============================================================ //
//  Multivariate hypergeometric (vector-valued)                 //
// ============================================================ //

/// Multivariate hypergeometric distribution: draw `draws` items without
/// replacement from an urn holding `colors[i]` items of each of `m ≥ 2`
/// colors; the outcome is the vector of per-color counts (stratified lot
/// sampling, multi-tier lottery pools, capture panels).
#[derive(Debug, Clone)]
pub struct MultivariateHypergeometric {
    colors: Vec<u64>,
    total: u64,
    draws: u64,
}

impl MultivariateHypergeometric {
    /// Urn composition (`≥ 2` colors) and number of draws `≤ Σ colors`.
    pub fn new(colors: &[u64], draws: u64) -> Self {
        assert!(
            colors.len() >= 2,
            "MultivariateHypergeometric: need at least two colors"
        );
        let total: u64 = colors.iter().sum();
        assert!(
            draws <= total,
            "MultivariateHypergeometric: draws must not exceed the urn size"
        );
        Self {
            colors: colors.to_vec(),
            total,
            draws,
        }
    }

    /// Natural log of `P(counts)`; `−∞` unless `Σ counts = draws` with every
    /// `counts[i] ≤ colors[i]`. Panics if `counts` has the wrong length.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.colors.len(),
            "MultivariateHypergeometric: counts length must match the number of colors"
        );
        if counts.iter().sum::<u64>() != self.draws
        {
            return f64::NEG_INFINITY;
        }
        let mut acc = -ln_binomial(self.total, self.draws);
        for (&k, &c) in counts.iter().zip(&self.colors)
        {
            acc += ln_binomial(c, k); // −∞ when k > c
        }
        acc
    }
    /// Probability mass `P(counts)`.
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
    /// Mean vector `draws·colorsᵢ/total`.
    pub fn mean(&self) -> Vec<f64> {
        self.colors
            .iter()
            .map(|&c| self.draws as f64 * c as f64 / self.total as f64)
            .collect()
    }
    /// One deterministic draw: sequential conditional univariate
    /// hypergeometrics over the remaining urn (fixed order ⇒ reproducible).
    pub fn sample(&self, rng: &mut SplitMix64) -> Vec<u64> {
        let m = self.colors.len();
        let mut out = Vec::with_capacity(m);
        let mut pop = self.total;
        let mut remaining = self.draws;
        for (i, &c) in self.colors.iter().enumerate()
        {
            if i + 1 == m
            {
                out.push(remaining);
                break;
            }
            let k = Hypergeometric::new(pop.max(1), c, remaining).sample(rng);
            out.push(k);
            pop -= c;
            remaining -= k;
        }
        out
    }
}

/// `ln P(counts)` of the Pólya (Dirichlet-multinomial) law with `n = Σ kᵢ`
/// trials and concentrations `αᵢ > 0` summing to `A`; the beta-binomial is
/// the two-category case. The caller has checked `Σ counts = n`.
///
/// With `T(k, α) = ln Γ(k+α) − ln Γ(α) − ln k!` the pmf is
/// `Σᵢ T(kᵢ, αᵢ) − T(n, A)`. Summing `ln Γ` values directly cancels terms of
/// size `~k·ln k` (or `~α·ln α`) and loses every digit once a count or a
/// concentration nears `1e15`. Writing `ln Γ(x+1) = (x+½)ln x − x + ½ln 2π + δ(x)`
/// (Stirling, with the remainder `δ` from `stirling_error`) splits each `T`
/// into `k·ln(1+α/k) + α·ln(1+k/α)`, a few `½ln` terms and `δ` values. The
/// first parts of all the `T` combine exactly into
/// `−Σᵢ [D₀(kᵢ, wᵢP) + D₀(αᵢ, wᵢQ)]` with `wᵢ = kᵢ+αᵢ`, `P = n/(n+A)`,
/// `Q = A/(n+A)` and the binomial deviance `D₀(x, μ) = x·ln(x/μ) + μ − x ≥ 0`,
/// the same device as Loader's binomial pmf. Every remaining term is at most
/// of size `~ln n`, so no large values cancel.
fn ln_polya_pmf(counts: &[u64], alpha: &[f64], n: u64, alpha_sum: f64) -> f64 {
    use scirust_special::{binom_deviance, stirling_error};
    if n == 0
    {
        // Only the all-zero vector is possible.
        return 0.0;
    }
    let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let nf = n as f64;
    let total = nf + alpha_sum;
    let (share_k, share_a) = (nf / total, alpha_sum / total);
    let (ln_share_k, ln_share_a) = (nf.ln() - total.ln(), alpha_sum.ln() - total.ln());
    // D₀(x, μ) with μ = w·share. D₀(0, μ) = μ; `binom_deviance` needs x > 0.
    // When the product w·share would underflow (tiny concentrations), μ is not
    // materialised: D₀ = x·(ln x − ln w − ln share) + μ − x, where every term
    // is tiny or the logarithm dominates, so nothing cancels.
    let deviance = |x: f64, w: f64, share: f64, ln_share: f64| {
        let mu = w * share;
        if x == 0.0
        {
            mu
        }
        else if mu >= f64::MIN_POSITIVE
        {
            binom_deviance(x, mu)
        }
        else
        {
            x * (x.ln() - w.ln() - ln_share) + mu - x
        }
    };
    // ½ln(α/(k(k+α))) − ½ln 2π + δ(k+α) − δ(α) − δ(k), the non-deviance part
    // of T(k, α) for k ≥ 1.
    let remainder = |k: f64, a: f64| {
        // ½ln(α/(k(k+α))) = −½[ln k + ln(1 + k/α)], which cannot overflow.
        -0.5 * (k.ln() + (k / a).ln_1p()) - half_ln_2pi + stirling_error(k + a)
            - stirling_error(a)
            - stirling_error(k)
    };
    let mut dev = 0.0;
    let mut rest = -remainder(nf, alpha_sum);
    for (&k, &a) in counts.iter().zip(alpha)
    {
        let kf = k as f64;
        let w = kf + a;
        dev += deviance(kf, w, share_k, ln_share_k) + deviance(a, w, share_a, ln_share_a);
        if k > 0
        {
            rest += remainder(kf, a);
        }
    }
    rest - dev
}

// ============================================================ //
//  Dirichlet-multinomial (vector-valued)                       //
// ============================================================ //

/// Dirichlet-multinomial (multivariate Pólya) distribution: a
/// [`Multinomial`] whose category probabilities are themselves Dirichlet(`α`)
/// distributed — the multivariate generalization of [`BetaBinomial`] and the
/// standard model for **overdispersed count vectors** (topic/word counts,
/// repeated categorical trials with batch-to-batch drift). `m = 2` categories
/// reduce to the beta-binomial; `α → ∞` (with fixed ratios) recovers the
/// multinomial.
#[derive(Debug, Clone)]
pub struct DirichletMultinomial {
    n: u64,
    alpha: Vec<f64>,
    alpha_sum: f64,
}

impl DirichletMultinomial {
    /// `n` trials over `alpha.len() ≥ 2` categories with concentration
    /// parameters `alpha[i] > 0`.
    pub fn new(n: u64, alpha: &[f64]) -> Self {
        assert!(
            alpha.len() >= 2,
            "DirichletMultinomial: need at least two categories"
        );
        assert!(
            alpha.iter().all(|&a| a > 0.0 && a.is_finite()),
            "DirichletMultinomial: concentrations must be finite and > 0"
        );
        Self {
            n,
            alpha: alpha.to_vec(),
            alpha_sum: alpha.iter().sum(),
        }
    }

    /// Natural log of `P(counts)`; `−∞` unless `Σ counts = n`. Panics if
    /// `counts` has the wrong length.
    ///
    /// The value is `ln Γ(A) − ln Γ(n+A) + ln n! + Σ[ln Γ(kᵢ+αᵢ) − ln Γ(αᵢ) − ln kᵢ!]`
    /// with `A = Σ αᵢ`, evaluated in a saddle-point form (binomial deviances
    /// plus Stirling remainders) that keeps its relative precision when the
    /// counts or the concentrations are huge.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.alpha.len(),
            "DirichletMultinomial: counts length must match the number of categories"
        );
        if counts.iter().sum::<u64>() != self.n
        {
            return f64::NEG_INFINITY;
        }
        ln_polya_pmf(counts, &self.alpha, self.n, self.alpha_sum)
    }
    /// Probability mass `P(counts)`.
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
    /// Mean vector `n·αᵢ/A`.
    pub fn mean(&self) -> Vec<f64> {
        self.alpha
            .iter()
            .map(|&a| self.n as f64 * a / self.alpha_sum)
            .collect()
    }
    /// Covariance matrix. Each entry carries the multinomial value times the
    /// overdispersion factor `ρ = (n+A)/(1+A)`:
    /// `Var(Xᵢ) = n·pᵢ(1−pᵢ)·ρ`, `Cov(Xᵢ,Xⱼ) = −n·pᵢpⱼ·ρ` with `pᵢ = αᵢ/A`.
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.n as f64;
        let a = self.alpha_sum;
        let rho = (n + a) / (1.0 + a);
        let p: Vec<f64> = self.alpha.iter().map(|&ai| ai / a).collect();
        p.iter()
            .enumerate()
            .map(|(i, &pi)| {
                p.iter()
                    .enumerate()
                    .map(|(j, &pj)| {
                        if i == j
                        {
                            n * pi * (1.0 - pi) * rho
                        }
                        else
                        {
                            -n * pi * pj * rho
                        }
                    })
                    .collect()
            })
            .collect()
    }
    /// One deterministic draw: sequential conditional beta-binomials — the
    /// exact stick-breaking of a Dirichlet-multinomial — consuming one
    /// uniform per category except the last (fixed order ⇒ reproducible).
    pub fn sample(&self, rng: &mut SplitMix64) -> Vec<u64> {
        let m = self.alpha.len();
        let mut out = Vec::with_capacity(m);
        let mut remaining = self.n;
        let mut rest_alpha = self.alpha_sum;
        for (i, &a) in self.alpha.iter().enumerate()
        {
            if i + 1 == m
            {
                out.push(remaining);
                break;
            }
            // Xᵢ | rest ~ BetaBinomial(remaining, αᵢ, A_rest − αᵢ).
            let b = rest_alpha - a;
            let k = if b > 0.0
            {
                BetaBinomial::new(remaining, a, b).sample(rng)
            }
            else
            {
                remaining
            };
            out.push(k);
            remaining -= k;
            rest_alpha -= a;
        }
        out
    }
}

// ============================================================ //
//  Yule–Simon                                                  //
// ============================================================ //

/// Yule–Simon distribution: a **heavy-tailed** law on `k ≥ 1` with
/// `pmf(k) = α·B(k, α+1)` (`B` the beta function), arising from
/// preferential-attachment / "rich-get-richer" processes (word frequencies,
/// citation counts, species-per-genus). The tail decays as a power law
/// `k^(−(α+1))`, so the mean is finite only for `α > 1` and the variance only
/// for `α > 2`; the survival function has the closed form `sf(k) = k·B(k, α+1)`.
///
/// Tails are evaluated from `ln sf(k) = −Σ_{j=1}^{k} ln(1 + α/j)` (the product
/// form `sf(k) = Π_{j≤k} j/(j+α)`), summed directly for `k ≤ 32` and continued
/// by a Stirling series for `ln Γ(x+α) − ln Γ(x)` beyond, so neither `sf` far
/// in the power-law tail nor `cdf = −expm1(ln sf)` at small `α` loses
/// precision to the cancellation of large `ln Γ` values.
#[derive(Debug, Clone, Copy)]
pub struct YuleSimon {
    alpha: f64,
}

impl YuleSimon {
    /// Shape `α > 0` (larger `α` ⇒ lighter tail).
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha.is_finite(),
            "YuleSimon: α must be finite and > 0"
        );
        Self { alpha }
    }

    /// `ln sf(k) = ln P(X > k)` for `k ≥ 1`, i.e. `−Σ_{j=1}^{k} ln(1 + α/j)`.
    ///
    /// The first `YULE_SIMON_DIRECT_TERMS` factors are summed with `ln_1p`;
    /// the rest of the product is `Γ(k+1)Γ(m+1+α) / (Γ(k+1+α)Γ(m+1))`, taken
    /// from `ln_gamma_shift_diff`, which never forms a full `ln Γ` value.
    fn ln_sf_pos(&self, k: u64) -> f64 {
        let a = self.alpha;
        let m = k.min(YULE_SIMON_DIRECT_TERMS);
        let mut s = 0.0;
        for j in 1..=m
        {
            s -= (a / j as f64).ln_1p();
        }
        if k > m
        {
            s -= ln_gamma_shift_diff(k as f64 + 1.0, m as f64 + 1.0, a);
        }
        s
    }
}

/// Number of leading factors of the Yule–Simon product `Π j/(j+α)` that are
/// summed term by term before switching to the Stirling form.
const YULE_SIMON_DIRECT_TERMS: u64 = 32;

/// `[ln Γ(x₂+a) − ln Γ(x₂)] − [ln Γ(x₁+a) − ln Γ(x₁)]` for `x₂ ≥ x₁ ≥ 33` and
/// `a ≥ 0`, without forming any `ln Γ` value.
///
/// From Stirling's series `ln Γ(z) = (z−½)ln z − z + ½ln 2π + C(z)` with
/// `C(z) = Σ B₂ₙ / (2n(2n−1) z^(2n−1))`, one shift is
/// `ln Γ(x+a) − ln Γ(x) = (x−½)·ln_1p(a/x) + a·(ln(x+a) − 1) + C(x+a) − C(x)`.
/// In the difference of two shifts the `a·ln(x+a)` terms, which are large
/// when `a` is large, combine into `a·ln_1p((x₂−x₁)/(x₁+a))`, and each
/// correction difference is `cₙ x^(−p)·expm1(−p·ln_1p(a/x))`. The result
/// therefore keeps its relative precision for tiny `a`, for huge `a`, and for
/// `x₂` up to `u64::MAX`. Five correction terms leave a truncation error below
/// `1e-19` at `x = 33`.
fn ln_gamma_shift_diff(x2: f64, x1: f64, a: f64) -> f64 {
    debug_assert!(x2 >= x1 && x1 >= 33.0 && a >= 0.0);
    // (cₙ, power p = 2n − 1) for B₂ … B₁₀.
    const C: [(f64, i32); 5] = [
        (1.0 / 12.0, 1),
        (-1.0 / 360.0, 3),
        (1.0 / 1260.0, 5),
        (-1.0 / 1680.0, 7),
        (1.0 / 1188.0, 9),
    ];
    // (x−½)·ln_1p(a/x) + C(x+a) − C(x) for one shift.
    let part = |x: f64| {
        let r = (a / x).ln_1p();
        let mut corr = 0.0;
        for &(c, p) in C.iter().rev()
        {
            corr += c * x.powi(-p) * (-(p as f64) * r).exp_m1();
        }
        (x - 0.5) * r + corr
    };
    part(x2) - part(x1) + a * ((x2 - x1) / (x1 + a)).ln_1p()
}

impl DiscreteDistribution for YuleSimon {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        // pmf(k) = α·B(k, α+1) = (α/k)·sf(k): reuses the cancellation-free tail.
        self.alpha.ln() - (k as f64).ln() + self.ln_sf_pos(k)
    }
    fn sf(&self, k: u64) -> f64 {
        // P(X > k) = k·B(k, α+1); at k = 0 the whole mass (support k ≥ 1) is above.
        if k == 0
        {
            return 1.0;
        }
        // ln sf ≤ 0; the clamp only guards against a last-ulp libm overshoot.
        self.ln_sf_pos(k).exp().min(1.0)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        // 1 − sf without cancellation when sf ≈ 1 (small α or small k).
        (-self.ln_sf_pos(k).exp_m1()).min(1.0)
    }
    fn logsf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        // Stays finite where sf itself underflows (large α or far tail).
        self.ln_sf_pos(k)
    }
    fn logcdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        let l = self.ln_sf_pos(k);
        if l < -std::f64::consts::LN_2
        {
            (-l.exp()).ln_1p()
        }
        else
        {
            (-l.exp_m1()).ln()
        }
    }
    fn mean(&self) -> f64 {
        if self.alpha > 1.0
        {
            self.alpha / (self.alpha - 1.0)
        }
        else
        {
            f64::INFINITY
        }
    }
    fn variance(&self) -> f64 {
        if self.alpha > 2.0
        {
            let a = self.alpha;
            a * a / ((a - 1.0) * (a - 1.0) * (a - 2.0))
        }
        else
        {
            f64::INFINITY
        }
    }
}

/// Below this rate the Boltzmann moments switch to the pole-free series
/// forms (their Bernoulli series converge with ratio `(x/2π)² < 0.007` here).
const SMALL_RATE: f64 = 0.5;

/// `q/(1 − q)²` with `q = e^(−x)`, i.e. `1/(4 sinh²(x/2))`, for `x > 0`:
/// the variance of a geometric count of failures with continuation
/// probability `q`. No `1 − q` cancellation at small `x`; `0` once
/// `sinh` overflows.
fn geometric_variance(x: f64) -> f64 {
    let s = (0.5 * x).sinh();
    0.25 / (s * s)
}

/// `g(x) = 1/(e^x − 1) − 1/x`, the Bose function with its pole removed.
/// Bernoulli series `−1/2 + Σ B₂ₖ x^(2k−1)/(2k)!` below [`SMALL_RATE`]
/// (truncation error < 1e-17 there), direct form above.
fn bose_regular(x: f64) -> f64 {
    if x >= SMALL_RATE
    {
        return 1.0 / x.exp_m1() - 1.0 / x;
    }
    let x2 = x * x;
    // B₂ₖ/(2k)! for k = 1..=7, innermost (highest order) first.
    let c = [
        1.0 / 74_724_249_600.0,
        -691.0 / 1_307_674_368_000.0,
        1.0 / 47_900_160.0,
        -1.0 / 1_209_600.0,
        1.0 / 30_240.0,
        -1.0 / 720.0,
        1.0 / 12.0,
    ];
    let odd = c.iter().fold(0.0, |acc, &ck| acc * x2 + ck);
    -0.5 + x * odd
}

/// `k(x) = 1/(4 sinh²(x/2)) − 1/x²`, the geometric variance with its double
/// pole removed. Series `−Σ (2k−1)·B₂ₖ x^(2k−2)/(2k)!` (that is
/// `−1/12 + x²/240 − x⁴/6048 + …`) below [`SMALL_RATE`], direct form above.
fn geometric_variance_regular(x: f64) -> f64 {
    if x >= SMALL_RATE
    {
        return geometric_variance(x) - 1.0 / (x * x);
    }
    let x2 = x * x;
    // −(2k−1)·B₂ₖ/(2k)! for k = 1..=7, innermost (highest order) first.
    let c = [
        -13.0 / 74_724_249_600.0,
        7_601.0 / 1_307_674_368_000.0,
        -1.0 / 5_322_240.0,
        1.0 / 172_800.0,
        -1.0 / 6_048.0,
        1.0 / 240.0,
        -1.0 / 12.0,
    ];
    c.iter().fold(0.0, |acc, &ck| acc * x2 + ck)
}

// ============================================================ //
//  Boltzmann (truncated Planck)                                //
// ============================================================ //

/// Boltzmann distribution — a geometric law truncated to `0..=n−1`
/// (SciPy's `boltzmann`, the "truncated Planck"):
/// `pmf(k) = (1−e^(−λ))·e^(−λk) / (1−e^(−λN))`. Models discrete energy-level
/// occupation and any exponentially-decaying count capped at `n` levels.
#[derive(Debug, Clone, Copy)]
pub struct Boltzmann {
    lambda: f64,
    n: u64,
}

impl Boltzmann {
    /// Decay rate `λ > 0` over `n ≥ 1` levels (support `0..=n−1`).
    pub fn new(lambda: f64, n: u64) -> Self {
        assert!(
            lambda > 0.0 && lambda.is_finite(),
            "Boltzmann: λ must be finite and > 0"
        );
        assert!(n >= 1, "Boltzmann: n must be ≥ 1");
        Self { lambda, n }
    }
    /// `1 − e^(−λN)`, the normalizer, via `−expm1` for accuracy at small `λN`.
    fn denom(&self) -> f64 {
        -(-self.lambda * self.n as f64).exp_m1()
    }
}

impl DiscreteDistribution for Boltzmann {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return f64::NEG_INFINITY;
        }
        // ln(1−e^(−λ)) − λk − ln(1−e^(−λN)).
        (-(-self.lambda).exp_m1()).ln() - self.lambda * k as f64 - self.denom().ln()
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n - 1
        {
            return 1.0;
        }
        // (1 − e^(−λ(k+1))) / (1 − e^(−λN)).
        -(-self.lambda * (k as f64 + 1.0)).exp_m1() / self.denom()
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n - 1
        {
            return 0.0;
        }
        // (e^(−λ(k+1)) − e^(−λN)) / (1 − e^(−λN)), with the numerator
        // factored as e^(−λ(k+1))·(1 − e^(−λ(N−k−1))) so that the two nearly
        // equal exponentials of a small-λ law never get subtracted.
        let head = (-self.lambda * (k as f64 + 1.0)).exp();
        let gap = -(-self.lambda * (self.n - k - 1) as f64).exp_m1();
        head * gap / self.denom()
    }
    fn mean(&self) -> f64 {
        // E[X] = 1/(e^λ − 1) − N/(e^(λN) − 1). For small λ both terms are
        // ≈ 1/λ and cancel, so there the 1/x poles are removed analytically:
        // with g(x) = 1/(e^x − 1) − 1/x, E[X] = g(λ) − N·g(λN).
        let (l, nn) = (self.lambda, self.n as f64);
        if l >= SMALL_RATE
        {
            1.0 / l.exp_m1() - nn / (l * nn).exp_m1()
        }
        else
        {
            bose_regular(l) - nn * bose_regular(l * nn)
        }
    }
    fn variance(&self) -> f64 {
        // Var[X] = h(λ) − N²·h(λN) with h(x) = e^x/(e^x − 1)² = 1/(4 sinh²(x/2)).
        // For small λ both terms are ≈ 1/λ² and cancel, so there the double
        // pole is removed analytically: with k(x) = h(x) − 1/x²,
        // Var[X] = k(λ) − N²·k(λN).
        let (l, nn) = (self.lambda, self.n as f64);
        if l >= SMALL_RATE
        {
            geometric_variance(l) - nn * nn * geometric_variance(l * nn)
        }
        else
        {
            geometric_variance_regular(l) - nn * nn * geometric_variance_regular(l * nn)
        }
    }
}

// ============================================================ //
//  Logarithmic (log-series)                                    //
// ============================================================ //

/// Logarithmic (log-series) distribution on `k ≥ 1`:
/// `pmf(k) = −pᵏ / (k·ln(1−p))` (SciPy's `logser`). Fisher's model for
/// species abundance and a common law for the number of items per purchase;
/// the term-frequency companion to the [`Zeta`]/[`YuleSimon`] family.
#[derive(Debug, Clone, Copy)]
pub struct Logarithmic {
    p: f64,
    /// `−1/ln(1−p) > 0`, the normalizing constant, precomputed.
    c: f64,
}

impl Logarithmic {
    /// Parameter `p ∈ (0, 1)`.
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p < 1.0, "Logarithmic: p must be within (0, 1)");
        Self {
            p,
            c: -1.0 / (-p).ln_1p(),
        }
    }

    /// Head sum `P(X ≤ k) = Σ_{i=1..k} pmf(i)` (O(k) terms).
    fn head(&self, k: u64) -> f64 {
        let mut acc = 0.0;
        for i in 1..=k
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }

    /// Direct upper tail `P(X > k) = c·Σ_{i>k} pⁱ/i` for `k ≥ 1`.
    ///
    /// Summed as `t₀·Σ_j r_j` with `t₀ = p^(k+1)/(k+1)` kept in log space
    /// (so a tiny leading term does not underflow before scaling) and the
    /// ratio `r_{j+1}/r_j = p·(k+1+j)/(k+2+j) < p`. The remainder after term
    /// `r_j` is at most `r_j·p/(1−p)`, which is the stopping rule. Returns
    /// `None` if that bound is not met within a fixed term budget (only when
    /// `1 − p` is below about `5e-6`), so callers fall back to the head sum.
    fn upper_tail(&self, k: u64) -> Option<f64> {
        const MAX_TERMS: u64 = 10_000_000;
        let i0 = k as f64 + 1.0;
        let ln_t0 = i0 * self.p.ln() - i0.ln();
        let geo = self.p / (1.0 - self.p);
        let (mut r, mut acc) = (1.0_f64, 0.0_f64);
        for j in 0..MAX_TERMS
        {
            acc += r;
            if r * geo <= 1e-17 * acc
            {
                let t = (self.c.ln() + ln_t0 + acc.ln()).exp();
                return Some(t.min(1.0));
            }
            let i = i0 + j as f64;
            r *= self.p * i / (i + 1.0);
        }
        None
    }
}

impl DiscreteDistribution for Logarithmic {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        // ln pmf = k·ln p − ln k + ln c.
        k as f64 * self.p.ln() - (k as f64).ln() + self.c.ln()
    }
    fn cdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        // Upper half: 1 − (direct tail), so cdf reaches exactly 1.0 once the
        // tail underflows instead of stalling one ulp short of it.
        match self.upper_tail(k)
        {
            Some(t) if t <= 0.5 => 1.0 - t,
            _ => self.head(k),
        }
    }
    fn sf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 1.0;
        }
        // Direct tail sum: no 1 − cdf cancellation, so far-tail values keep
        // full relative precision instead of collapsing to 0 or 1.1e-16.
        self.upper_tail(k)
            .unwrap_or_else(|| (1.0 - self.head(k)).max(0.0))
    }
    fn mean(&self) -> f64 {
        // −p / ((1−p)·ln(1−p)) = c·p/(1−p).
        self.c * self.p / (1.0 - self.p)
    }
    fn variance(&self) -> f64 {
        let r = (-self.p).ln_1p(); // ln(1−p) < 0
        -self.p * (self.p + r) / ((1.0 - self.p) * (1.0 - self.p) * r * r)
    }
}

// ============================================================ //
//  Planck (non-truncated)                                      //
// ============================================================ //

/// Planck distribution on `k ≥ 0`: `pmf(k) = (1−e^(−λ))·e^(−λk)` (SciPy's
/// `planck`). The geometric law counting *failures* (support from 0), i.e.
/// the `n → ∞` limit of [`Boltzmann`]; models unbounded discrete
/// energy-level occupation.
#[derive(Debug, Clone, Copy)]
pub struct Planck {
    lambda: f64,
}

impl Planck {
    /// Decay rate `λ > 0`.
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda > 0.0 && lambda.is_finite(),
            "Planck: λ must be finite and > 0"
        );
        Self { lambda }
    }
}

impl DiscreteDistribution for Planck {
    fn ln_pmf(&self, k: u64) -> f64 {
        // ln(1−e^(−λ)) − λk.
        (-(-self.lambda).exp_m1()).ln() - self.lambda * k as f64
    }
    fn cdf(&self, k: u64) -> f64 {
        // 1 − e^(−λ(k+1)).
        -(-self.lambda * (k as f64 + 1.0)).exp_m1()
    }
    fn sf(&self, k: u64) -> f64 {
        // e^(−λ(k+1)) — direct upper tail.
        (-self.lambda * (k as f64 + 1.0)).exp()
    }
    fn mean(&self) -> f64 {
        1.0 / self.lambda.exp_m1()
    }
    fn variance(&self) -> f64 {
        // q/(1 − q)² with q = e^(−λ), evaluated without forming 1 − q.
        geometric_variance(self.lambda)
    }
}

// ============================================================ //
//  Discrete Laplace (support ℤ — outside the u64 trait)        //
// ============================================================ //

/// Discrete Laplace (two-sided geometric) distribution on all of ℤ:
/// `pmf(k) = tanh(a/2)·e^(−a·|k|)` (SciPy's `dlaplace`). It is the difference
/// of two i.i.d. geometric variables and the law behind the **geometric
/// mechanism** of differential privacy (integer-valued additive noise with a
/// pure-DP guarantee). Symmetric about 0, so the mean is 0.
///
/// Like [`Skellam`], its support is ℤ, so it exposes its own `i64` methods
/// rather than the non-negative-integer [`DiscreteDistribution`] trait.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteLaplace {
    a: f64,
}

impl DiscreteLaplace {
    /// Rate `a > 0` (larger `a` ⇒ tighter concentration around 0). For the
    /// DP geometric mechanism with sensitivity 1 and budget ε, take `a = ε`.
    pub fn new(a: f64) -> Self {
        assert!(
            a > 0.0 && a.is_finite(),
            "DiscreteLaplace: a must be finite and > 0"
        );
        Self { a }
    }

    /// Probability mass `P(X = k)`, `k ∈ ℤ`.
    pub fn pmf(&self, k: i64) -> f64 {
        // tanh(a/2) = (1−e^(−a))/(1+e^(−a)), the normalizer.
        (self.a * 0.5).tanh() * (-self.a * k.unsigned_abs() as f64).exp()
    }
    /// Natural log of the probability mass.
    pub fn ln_pmf(&self, k: i64) -> f64 {
        (self.a * 0.5).tanh().ln() - self.a * k.unsigned_abs() as f64
    }
    /// Cumulative distribution `P(X ≤ k)`.
    pub fn cdf(&self, k: i64) -> f64 {
        // Geometric tail sum in closed form, split at 0 by symmetry.
        // For k < 0, P(X ≤ k) = Σ_{m ≥ |k|} tanh(a/2)e^(−am) = e^(a·k)/(1+e^(−a)).
        let denom = 1.0 + (-self.a).exp();
        if k < 0
        {
            (self.a * k as f64).exp() / denom
        }
        else
        {
            1.0 - (-self.a * (k as f64 + 1.0)).exp() / denom
        }
    }
    /// Survival function `P(X > k)`.
    pub fn sf(&self, k: i64) -> f64 {
        let denom = 1.0 + (-self.a).exp();
        if k < 0
        {
            1.0 - (self.a * k as f64).exp() / denom
        }
        else
        {
            (-self.a * (k as f64 + 1.0)).exp() / denom
        }
    }
    /// Mean, `0` by symmetry.
    pub fn mean(&self) -> f64 {
        0.0
    }
    /// Variance `2·e^(−a) / (1 − e^(−a))²`.
    pub fn variance(&self) -> f64 {
        // 2q/(1 − q)² with q = e^(−a), evaluated without forming 1 − q.
        2.0 * geometric_variance(self.a)
    }
    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// One deterministic draw as the difference of two geometric variables
    /// (support `k ≥ 0`), consuming the rng in a fixed order.
    pub fn sample(&self, rng: &mut SplitMix64) -> i64 {
        // Geometric(1−e^(−a)) counting failures ⇒ shift the k≥1 Geometric by 1.
        let p = -(-self.a).exp_m1(); // 1 − e^(−a)
        let g = Geometric::new(p);
        (g.sample(rng) as i64 - 1) - (g.sample(rng) as i64 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// Purely relative comparison, for tail values far below 1.
    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs()
    }

    // Oracle values: SciPy 1.17.1 (binom, poisson, hypergeom, geom) and exact
    // fractions via Python `math.comb` — see the values quoted inline.

    #[test]
    fn binomial_matches_scipy() {
        let b = Binomial::new(20, 0.3);
        assert!(close(b.pmf(0), 0.000_797_922_662_976_117_1, 1e-12));
        assert!(close(b.pmf(3), 0.071_603_672_205_262_1, 1e-12));
        assert!(close(b.pmf(6), 0.191_638_982_753_442_54, 1e-12));
        assert!(close(b.pmf(20), 3.486_784_400_999_997_5e-11, 1e-11));
        assert_eq!(b.pmf(21), 0.0);
        assert!(close(b.cdf(6), 0.608_009_812_200_924_4, 1e-12));
        assert!(close(b.sf(6), 0.391_990_187_799_075_6, 1e-12));
        assert!(close(b.cdf(10), 0.982_855_183_568_741_6, 1e-12));
        assert!(close(b.sf(10), 0.017_144_816_431_258_418, 1e-12));
        assert_eq!(b.quantile(0.5), 6);
        assert_eq!(b.quantile(0.95), 9);
        assert!(close(b.mean(), 6.0, 1e-15));
        assert!(close(b.variance(), 4.2, 1e-15));
        // Small-p regime (ln_1p path).
        let b2 = Binomial::new(1000, 0.001);
        assert!(close(b2.pmf(0), 0.367_695_424_770_963_9, 1e-12));
        assert!(close(b2.cdf(2), 0.919_790_657_159_799, 1e-12));
    }

    #[test]
    // Ignored under Miri: the degenerate edges reduce `pmf` to `exp(0.0)`,
    // which every real libm returns as exactly 1.0 but Miri deliberately
    // perturbs (it models the platform freedom of the transcendental
    // intrinsics), breaking the exact 1e-15 checks. Native Build & Test jobs
    // enforce these values.
    #[cfg_attr(miri, ignore)]
    fn binomial_degenerate_edges() {
        let zero = Binomial::new(10, 0.0);
        assert!(close(zero.pmf(0), 1.0, 1e-15));
        assert_eq!(zero.pmf(1), 0.0);
        assert!(close(zero.cdf(0), 1.0, 1e-15));
        let one = Binomial::new(10, 1.0);
        assert!(close(one.pmf(10), 1.0, 1e-15));
        assert_eq!(one.pmf(9), 0.0);
        assert_eq!(one.quantile(0.99), 10);
        let point = Binomial::new(0, 0.5);
        assert!(close(point.pmf(0), 1.0, 1e-15));
        assert_eq!(point.quantile(0.7), 0);
    }

    #[test]
    fn poisson_matches_scipy() {
        let p = Poisson::new(4.2);
        assert!(close(p.pmf(0), 0.014_995_576_820_477_703, 1e-12));
        assert!(close(p.pmf(3), 0.185_165_382_579_258_7, 1e-12));
        assert!(close(p.pmf(7), 0.068_592_664_322_660_6, 1e-12));
        assert!(close(p.cdf(3), 0.395_403_369_602_356_17, 1e-12));
        assert!(close(p.sf(3), 0.604_596_630_397_643_8, 1e-12));
        assert!(close(p.cdf(7), 0.936_056_660_272_578_9, 1e-12));
        assert!(close(p.sf(7), 0.063_943_339_727_421_1, 1e-12));
        assert_eq!(p.quantile(0.95), 8);
        assert!(close(p.mean(), 4.2, 1e-15));
        assert!(close(p.variance(), 4.2, 1e-15));
        // cdf + sf = 1 across the range.
        for k in 0..30
        {
            assert!(close(p.cdf(k) + p.sf(k), 1.0, 1e-13), "k = {k}");
        }
    }

    #[test]
    fn hypergeometric_matches_exact_6_of_49() {
        // Classic 6/49: population 49, 6 winning numbers, player draws 6.
        // Exact fractions: pmf(k) = C(6,k)·C(43,6−k)/C(49,6).
        let h = Hypergeometric::new(49, 6, 6);
        assert!(close(h.pmf(0), 0.435_964_975_511_691_5, 1e-12));
        assert!(close(h.pmf(1), 0.413_019_450_484_760_4, 1e-12));
        assert!(close(h.pmf(2), 0.132_378_029_001_525_76, 1e-12));
        assert!(close(h.pmf(3), 0.017_650_403_866_870_102, 1e-12));
        assert!(close(h.pmf(4), 0.000_968_619_724_401_408, 1e-12));
        assert!(close(h.pmf(5), 1.844_989_951_240_777_2e-5, 1e-12));
        assert!(close(h.pmf(6), 1.0 / 13_983_816.0, 1e-12));
        assert!(close(h.cdf(2), 0.981_362_454_997_977_7, 1e-12));
        assert!(close(h.sf(2), 0.018_637_545_002_022_343, 1e-12));
        assert!(close(h.mean(), 0.734_693_877_551_020_4, 1e-13));
        assert!(close(h.variance(), 0.577_571_845_064_556_4, 1e-13));
        // Total mass is 1.
        let total: f64 = (0..=6).map(|k| h.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-13));
    }

    #[test]
    fn hypergeometric_larger_case_and_support() {
        let h = Hypergeometric::new(500, 50, 60);
        assert!(close(h.pmf(5), 0.173_200_819_493_689_73, 1e-11));
        assert!(close(h.cdf(5), 0.427_334_645_995_490_37, 1e-11));
        assert_eq!(h.quantile(0.5), 6);
        // Truncated support: draw 8 from 10 with 9 marked ⇒ at least 7 marked.
        let t = Hypergeometric::new(10, 9, 8);
        assert_eq!(t.support_min(), 7);
        assert_eq!(t.support_max(), 8);
        assert_eq!(t.pmf(6), 0.0);
        assert!(close(t.pmf(7) + t.pmf(8), 1.0, 1e-13));
    }

    #[test]
    fn geometric_matches_scipy() {
        let g = Geometric::new(0.25);
        assert_eq!(g.pmf(0), 0.0);
        assert!(close(g.pmf(1), 0.25, 1e-15));
        assert!(close(g.pmf(3), 0.140_625, 1e-15));
        assert!(close(g.pmf(8), 0.033_370_971_679_687_5, 1e-14));
        assert!(close(g.cdf(3), 0.578_125, 1e-15));
        assert!(close(g.sf(3), 0.421_875, 1e-14));
        assert_eq!(g.quantile(0.99), 17);
        assert!(close(g.mean(), 4.0, 1e-15));
        assert!(close(g.variance(), 12.0, 1e-14));
        // p = 1: certain success on the first trial.
        let sure = Geometric::new(1.0);
        assert!(close(sure.pmf(1), 1.0, 1e-15));
        assert_eq!(sure.pmf(2), 0.0);
        assert_eq!(sure.quantile(0.999), 1);
    }

    #[test]
    // Miri deliberately randomizes the last ULPs of float intrinsics, so the
    // lockstep bit-identity below cannot hold under the interpreter (same
    // rationale as the continuous sampling test in `dist`).
    #[cfg_attr(miri, ignore)]
    fn sampling_is_deterministic_and_plausible() {
        let b = Binomial::new(40, 0.35);
        let mut r1 = SplitMix64::new(42);
        let mut r2 = SplitMix64::new(42);
        let s1: Vec<u64> = (0..20_000).map(|_| b.sample(&mut r1)).collect();
        let s2: Vec<u64> = (0..20_000).map(|_| b.sample(&mut r2)).collect();
        assert_eq!(s1, s2);
        let m = s1.iter().sum::<u64>() as f64 / s1.len() as f64;
        assert!((m - 14.0).abs() < 0.1, "mean {m}");
        // Poisson sample moments near λ.
        let p = Poisson::new(6.5);
        let mut r = SplitMix64::new(7);
        let sp: Vec<u64> = (0..20_000).map(|_| p.sample(&mut r)).collect();
        let mp = sp.iter().sum::<u64>() as f64 / sp.len() as f64;
        assert!((mp - 6.5).abs() < 0.1, "mean {mp}");
    }

    #[test]
    // Ignored under Miri: the pmf/cdf go through `ln_gamma`/`exp`, whose last
    // bits Miri deliberately perturbs (it models the platform freedom of the
    // transcendental intrinsics), so the 1e-12 SciPy checks can miss. Native
    // Build & Test jobs enforce these values.
    #[cfg_attr(miri, ignore)]
    fn negative_binomial_matches_scipy() {
        // SciPy nbinom(5, 0.4): failures before the 5th success.
        let nb = NegativeBinomial::new(5.0, 0.4);
        assert!(close(nb.pmf(0), 0.010_239_999_999_999_996, 1e-12));
        assert!(close(nb.pmf(4), 0.092_897_280_000_000_03, 1e-12));
        assert!(close(nb.pmf(10), 0.061_979_281_588_224_036, 1e-12));
        assert!(close(nb.cdf(7), 0.561_821_777_92, 1e-12));
        assert!(close(nb.sf(7), 0.438_178_222_08, 1e-12));
        assert_eq!(nb.quantile(0.5), 7);
        assert_eq!(nb.quantile(0.95), 16);
        assert!(close(nb.mean(), 7.5, 1e-14));
        assert!(close(nb.variance(), 18.75, 1e-13));
        // Real-valued r (Pólya).
        let nb2 = NegativeBinomial::new(2.5, 0.3);
        assert!(close(nb2.pmf(3), 0.110_960_031_985_585_6, 1e-12));
        assert!(close(nb2.cdf(5), 0.556_183_734_708_268_1, 1e-12));
        // r = 1 is Geometric shifted to failures: pmf(k) = p(1−p)^k.
        let nb1 = NegativeBinomial::new(1.0, 0.25);
        assert!(close(nb1.pmf(2), 0.25 * 0.75 * 0.75, 1e-14));
        // p = 1: point mass at zero failures.
        let sure = NegativeBinomial::new(3.0, 1.0);
        assert!(close(sure.pmf(0), 1.0, 1e-15));
        assert_eq!(sure.pmf(1), 0.0);
    }

    #[test]
    fn beta_binomial_matches_scipy() {
        // SciPy betabinom(10, 2, 3).
        let bb = BetaBinomial::new(10, 2.0, 3.0);
        assert!(close(bb.pmf(0), 0.065_934_065_934_065_95, 1e-12));
        assert!(close(bb.pmf(4), 0.139_860_139_860_139_76, 1e-12));
        assert!(close(bb.pmf(10), 0.010_989_010_989_010_992, 1e-12));
        assert!(close(bb.cdf(4), 0.594_405_594_405_594_4, 1e-12));
        assert!(close(bb.sf(4), 0.405_594_405_594_405_6, 1e-12));
        assert_eq!(bb.quantile(0.5), 4);
        assert!(close(bb.mean(), 4.0, 1e-14));
        assert!(close(bb.variance(), 6.0, 1e-13));
        // a = b = 1 is the discrete uniform on 0..=n.
        let u = BetaBinomial::new(6, 1.0, 1.0);
        assert!(close(u.pmf(3), 1.0 / 7.0, 1e-13));
        // Total mass 1.
        let total: f64 = (0..=10).map(|k| bb.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-13));
    }

    #[test]
    fn zipfian_matches_scipy() {
        // SciPy zipfian(1.5, 20), support 1..=20.
        let z = Zipfian::new(1.5, 20);
        assert!(close(z.pmf(1), 0.460_684_691_303_022_2, 1e-13));
        assert!(close(z.pmf(2), 0.162_876_634_604_599_42, 1e-13));
        assert!(close(z.pmf(5), 0.041_204_891_437_882_57, 1e-13));
        assert!(close(z.pmf(20), 0.005_150_611_429_735_321, 1e-13));
        assert_eq!(z.pmf(0), 0.0);
        assert_eq!(z.pmf(21), 0.0);
        assert!(close(z.cdf(5), 0.811_010_613_936_828_4, 1e-13));
        assert!(close(z.sf(5), 0.188_989_386_063_171_64, 1e-13));
        assert_eq!(z.quantile(0.5), 2);
        assert!(close(z.mean(), 3.499_017_716_693_377, 1e-13));
        assert!(close(z.variance(), 16.165_446_970_218_827, 1e-12));
        // s = 0 is the discrete uniform on 1..=n.
        let u = Zipfian::new(0.0, 10);
        assert!(close(u.pmf(7), 0.1, 1e-14));
        assert!(close(u.cdf(10), 1.0, 1e-15));
    }

    #[test]
    fn skellam_matches_scipy() {
        // SciPy skellam(3.2, 1.5).
        let s = Skellam::new(3.2, 1.5);
        assert!(close(s.pmf(-4), 0.004_693_621_474_905_621, 1e-11));
        assert!(close(s.pmf(-1), 0.086_025_279_807_399_57, 1e-11));
        assert!(close(s.pmf(0), 0.143_310_965_409_640_56, 1e-11));
        assert!(close(s.pmf(2), 0.183_382_994_925_598_12, 1e-11));
        assert!(close(s.pmf(6), 0.026_216_209_058_590_838, 1e-11));
        assert!(close(s.cdf(0), 0.291_039_386_736_692_55, 1e-11));
        assert!(close(s.cdf(3), 0.804_942_925_451_844, 1e-11));
        assert!(close(s.sf(3), 0.195_057_074_548_156_02, 1e-11));
        assert!(close(s.mean(), 1.7, 1e-14));
        assert!(close(s.variance(), 4.7, 1e-14));
        // cdf + sf = 1 across ℤ, both tails included.
        for k in -8..=10_i64
        {
            assert!(close(s.cdf(k) + s.sf(k), 1.0, 1e-12), "k = {k}");
        }
        // Equal rates ⇒ symmetric about 0.
        let sym = Skellam::new(2.0, 2.0);
        assert!(close(sym.pmf(0), 0.207_001_921_223_986_64, 1e-11));
        assert!(close(sym.pmf(1), sym.pmf(-1), 1e-13));
        assert!(close(sym.pmf(1), 0.178_750_839_502_435_3, 1e-11));
        // Total mass over a wide window is 1.
        let total: f64 = (-40..=40_i64).map(|k| s.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-11));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn skellam_sampling_is_deterministic_and_plausible() {
        let s = Skellam::new(3.2, 1.5);
        let mut r1 = SplitMix64::new(99);
        let mut r2 = SplitMix64::new(99);
        let a: Vec<i64> = (0..20_000).map(|_| s.sample(&mut r1)).collect();
        let b: Vec<i64> = (0..20_000).map(|_| s.sample(&mut r2)).collect();
        assert_eq!(a, b);
        let m = a.iter().sum::<i64>() as f64 / a.len() as f64;
        assert!((m - 1.7).abs() < 0.06, "mean {m}");
    }

    #[test]
    fn zeta_matches_scipy() {
        // SciPy zipf(2.5) — infinite-support zeta law.
        let z = Zeta::new(2.5);
        assert_eq!(z.pmf(0), 0.0);
        assert!(close(z.pmf(1), 0.745_441_296_288_777, 1e-13));
        assert!(close(z.pmf(2), 0.131_776_648_895_571_14, 1e-13));
        assert!(close(z.pmf(3), 0.047_820_081_453_043_214, 1e-13));
        assert!(close(z.pmf(10), 0.002_357_292_358_220_957_2, 1e-13));
        assert!(close(z.cdf(5), 0.961_667_926_440_313_7, 1e-13));
        assert!(close(z.sf(5), 0.038_332_073_559_686_264, 1e-13));
        assert_eq!(z.quantile(0.9), 3);
        assert_eq!(z.quantile(0.99), 14);
        // mean = ζ(1.5)/ζ(2.5); variance diverges at s ≤ 3.
        assert!(close(z.mean(), 1.947_372_466_316_956, 1e-13));
        assert!(z.variance().is_infinite());
        // s = 4: both moments finite.
        let z4 = Zeta::new(4.0);
        assert!(close(z4.pmf(2), 0.057_746_150_182_599_39, 1e-13));
        assert!(close(z4.mean(), 1.110_626_535_326_148, 1e-13));
        assert!(close(z4.variance(), 0.286_326_453_664_503_4, 1e-12));
        // cdf + sf = 1 across the head/tail split at k = 19/20.
        for k in [1u64, 5, 18, 19, 20, 50, 1000]
        {
            assert!(close(z.cdf(k) + z.sf(k), 1.0, 1e-12), "k = {k}");
        }
        // Far-tail sf stays accurate and O(1): sf(k) ~ k^(1−s)/((s−1)ζ(s)).
        let k = 1_000_000_u64;
        let approx = (k as f64).powf(-1.5) / (1.5 * 1.341_487_257_250_917_3);
        assert!(close(z.sf(k), approx, 1e-3));
    }

    #[test]
    fn poisson_binomial_matches_scipy() {
        // SciPy poisson_binom([0.1, 0.4, 0.75, 0.5, 0.9]).
        let pb = PoissonBinomial::new(&[0.1, 0.4, 0.75, 0.5, 0.9]);
        assert!(close(pb.pmf(0), 0.006_749_999_999_999_999, 1e-13));
        assert!(close(pb.pmf(1), 0.093, 1e-13));
        assert!(close(pb.pmf(2), 0.332, 1e-13));
        assert!(close(pb.pmf(3), 0.393_5, 1e-13));
        assert!(close(pb.pmf(4), 0.161_25, 1e-13));
        assert!(close(pb.pmf(5), 0.013_5, 1e-13));
        assert_eq!(pb.pmf(6), 0.0);
        assert!(close(pb.cdf(2), 0.431_75, 1e-13));
        assert!(close(pb.sf(2), 0.568_25, 1e-13));
        assert_eq!(pb.quantile(0.5), 3);
        assert!(close(pb.mean(), 2.65, 1e-14));
        assert!(close(pb.variance(), 0.857_5, 1e-14));
        assert_eq!(pb.trials(), 5);
        // Homogeneous probabilities collapse to the Binomial.
        let pb_h = PoissonBinomial::new(&[0.3; 20]);
        let b = Binomial::new(20, 0.3);
        for k in [0u64, 3, 6, 10, 20]
        {
            assert!(close(pb_h.pmf(k), b.pmf(k), 1e-12), "k = {k}");
        }
        // Total mass 1.
        let total: f64 = (0..=5).map(|k| pb.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-14));
    }

    #[test]
    fn multinomial_matches_scipy() {
        // SciPy multinomial(8, [0.2, 0.3, 0.5]).
        let m = Multinomial::new(8, &[0.2, 0.3, 0.5]);
        assert!(close(m.pmf(&[2, 3, 3]), 0.075_599_999_999_999_96, 1e-12));
        assert!(close(m.pmf(&[1, 2, 5]), 0.094_500_000_000_000_2, 1e-12));
        assert!(close(m.pmf(&[8, 0, 0]), 2.560_000_000_000_001_7e-6, 1e-12));
        assert!(close(m.ln_pmf(&[2, 3, 3]), -2.582_298_995_796_650_2, 1e-12));
        // Wrong total ⇒ impossible outcome.
        assert_eq!(m.pmf(&[2, 3, 4]), 0.0);
        // Moments.
        let mean = m.mean();
        assert!(close(mean[0], 1.6, 1e-14) && close(mean[1], 2.4, 1e-14));
        let cov = m.covariance();
        assert!(close(cov[0][0], 1.28, 1e-14));
        assert!(close(cov[0][1], -0.48, 1e-14));
        assert!(close(cov[2][2], 2.0, 1e-14));
        // Zero-probability category: only zero counts allowed.
        let z = Multinomial::new(4, &[0.5, 0.5, 0.0]);
        assert_eq!(z.pmf(&[2, 1, 1]), 0.0);
        assert!(z.pmf(&[2, 2, 0]) > 0.0);
        // Two categories degenerate to the Binomial.
        let m2 = Multinomial::new(20, &[0.3, 0.7]);
        let b = Binomial::new(20, 0.3);
        assert!(close(m2.pmf(&[6, 14]), b.pmf(6), 1e-12));
    }

    #[test]
    fn multivariate_hypergeometric_matches_scipy() {
        // SciPy multivariate_hypergeom(m=[10, 5, 15], n=8);
        // exact pmf([3,1,4]) = C(10,3)·C(5,1)·C(15,4)/C(30,8) = 280/2001.
        let mh = MultivariateHypergeometric::new(&[10, 5, 15], 8);
        assert!(close(mh.pmf(&[3, 1, 4]), 280.0 / 2001.0, 1e-12));
        assert!(close(
            mh.pmf(&[0, 0, 8]),
            0.001_099_450_274_862_565_7,
            1e-12
        ));
        // Wrong total or over-drawing a color ⇒ impossible.
        assert_eq!(mh.pmf(&[3, 1, 3]), 0.0);
        assert_eq!(mh.pmf(&[0, 6, 2]), 0.0);
        let mean = mh.mean();
        assert!(close(mean[0], 8.0 / 3.0, 1e-14));
        assert!(close(mean[1], 4.0 / 3.0, 1e-14));
        assert!(close(mean[2], 4.0, 1e-14));
        // Two colors degenerate to the univariate Hypergeometric.
        let mh2 = MultivariateHypergeometric::new(&[6, 43], 6);
        let h = Hypergeometric::new(49, 6, 6);
        for k in 0..=6u64
        {
            assert!(close(mh2.pmf(&[k, 6 - k]), h.pmf(k), 1e-12), "k = {k}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn vector_sampling_is_deterministic_and_consistent() {
        // Multinomial draws always sum to n and reproduce bit-for-bit.
        let m = Multinomial::new(50, &[0.2, 0.3, 0.5]);
        let mut r1 = SplitMix64::new(11);
        let mut r2 = SplitMix64::new(11);
        let mut totals = [0u64; 3];
        for _ in 0..2_000
        {
            let a = m.sample(&mut r1);
            let b = m.sample(&mut r2);
            assert_eq!(a, b);
            assert_eq!(a.iter().sum::<u64>(), 50);
            for (t, &x) in totals.iter_mut().zip(&a)
            {
                *t += x;
            }
        }
        // Empirical means near n·p = [10, 15, 25].
        assert!((totals[0] as f64 / 2_000.0 - 10.0).abs() < 0.2);
        assert!((totals[2] as f64 / 2_000.0 - 25.0).abs() < 0.3);
        // Multivariate hypergeometric draws sum to `draws` and respect caps.
        let mh = MultivariateHypergeometric::new(&[10, 5, 15], 8);
        let mut r = SplitMix64::new(23);
        for _ in 0..2_000
        {
            let d = mh.sample(&mut r);
            assert_eq!(d.iter().sum::<u64>(), 8);
            assert!(d[0] <= 10 && d[1] <= 5 && d[2] <= 15);
        }
    }

    #[test]
    fn dirichlet_multinomial_matches_scipy() {
        // SciPy dirichlet_multinomial(alpha=[1, 2, 3], n=10).
        let dm = DirichletMultinomial::new(10, &[1.0, 2.0, 3.0]);
        assert!(close(dm.pmf(&[2, 3, 5]), 0.027_972_027_972_027_96, 1e-12));
        assert!(close(dm.pmf(&[0, 0, 10]), 0.021_978_021_978_021_907, 1e-12));
        assert!(close(
            dm.pmf(&[10, 0, 0]),
            0.000_333_000_333_000_332_7,
            1e-12
        ));
        assert!(close(dm.ln_pmf(&[3, 3, 4]), -3.913_022_505_761_23, 1e-12));
        // Wrong total ⇒ impossible.
        assert_eq!(dm.pmf(&[1, 1, 1]), 0.0);
        let mean = dm.mean();
        assert!(close(mean[0], 10.0 / 6.0, 1e-14));
        assert!(close(mean[1], 10.0 / 3.0, 1e-14));
        assert!(close(mean[2], 5.0, 1e-14));
        // Covariance vs SciPy .cov().
        let cov = dm.covariance();
        assert!(close(cov[0][0], 3.174_603_174_603_17, 1e-12));
        assert!(close(cov[1][1], 5.079_365_079_365_08, 1e-12));
        assert!(close(cov[2][2], 5.714_285_714_285_71, 1e-12));
        assert!(close(cov[0][1], -1.269_841_269_841_27, 1e-12));
        assert!(close(cov[1][2], -3.809_523_809_523_81, 1e-12));
        // Total mass 1 over the simplex Σ = n.
        let mut total = 0.0;
        for i in 0..=10
        {
            for j in 0..=(10 - i)
            {
                total += dm.pmf(&[i, j, 10 - i - j]);
            }
        }
        assert!(close(total, 1.0, 1e-12));
        // Two categories reduce to the beta-binomial; α = [1,1] ⇒ uniform.
        let dm2 = DirichletMultinomial::new(5, &[1.0, 1.0]);
        for k in 0..=5u64
        {
            assert!(close(dm2.pmf(&[k, 5 - k]), 1.0 / 6.0, 1e-13), "k = {k}");
        }
        let bb = BetaBinomial::new(5, 2.0, 3.0);
        let dm3 = DirichletMultinomial::new(5, &[2.0, 3.0]);
        for k in 0..=5u64
        {
            assert!(close(dm3.pmf(&[k, 5 - k]), bb.pmf(k), 1e-12), "k = {k}");
        }
        // Exact rational: alpha=[2,3,5], n=4, counts=[1,1,2] = 18/143.
        let dm4 = DirichletMultinomial::new(4, &[2.0, 3.0, 5.0]);
        assert!(close(dm4.pmf(&[1, 1, 2]), 18.0 / 143.0, 1e-12));
    }

    /// `|got − want| ≤ tol·max(1, |want|)` on a log-pmf: absolute near 0
    /// (relative precision of the pmf itself), relative for large magnitudes.
    fn assert_polya_ln_close(got: f64, want: f64, what: &str) {
        let tol = 1e-12 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
    }

    // Oracle: mpmath 1.4.1 at 360 digits,
    // Σ[lnΓ(kᵢ+αᵢ) − lnΓ(αᵢ) − lnΓ(kᵢ+1)] − [lnΓ(n+A) − lnΓ(A) − lnΓ(n+1)].
    #[test]
    fn dirichlet_multinomial_ln_pmf_keeps_precision_at_huge_arguments() {
        // Huge counts near the mean: the direct ln Γ sum returned −14336.
        let n = 1u64 << 60;
        let dm = DirichletMultinomial::new(n, &[1e15, 2e15, 3e15]);
        assert_polya_ln_close(
            dm.ln_pmf(&[
                192_153_584_101_141_162,
                384_307_168_202_282_325,
                576_460_752_303_423_489,
            ]),
            -46.898_434_075_768_57,
            "DM(2^60, [1e15,2e15,3e15])",
        );
        // Small counts, huge concentrations (was −61.27).
        let dm = DirichletMultinomial::new(10, &[1e15, 2e15, 3e15]);
        assert_polya_ln_close(
            dm.ln_pmf(&[1, 3, 6]),
            -2.513_077_526_754_697_6,
            "DM(10, [1e15,2e15,3e15])",
        );
        // All mass in the dominant category: the old form returned 0 (pmf 1).
        let dm = DirichletMultinomial::new(n, &[1e-8, 2.0, 3000.0]);
        assert_polya_ln_close(
            dm.ln_pmf(&[0, 0, n]),
            -67.164_593_589_922_46,
            "DM(2^60, [1e-8,2,3000]) one-hot",
        );
        assert_polya_ln_close(
            dm.ln_pmf(&[1, 1, n - 2]),
            -84.892_127_153_314_89,
            "DM(2^60, [1e-8,2,3000]) skewed",
        );
        // α → ∞ is the multinomial: (1/3)^10. The old form gave +8.9e-16.
        let dm = DirichletMultinomial::new(10, &[1e300, 2e300]);
        assert_polya_ln_close(
            dm.ln_pmf(&[10, 0]),
            -10.986_122_886_681_097,
            "DM(10, [1e300,2e300])",
        );
        assert_polya_ln_close(
            dm.ln_pmf(&[3, 7]),
            -1.346_600_879_979_433_8,
            "DM(10, [1e300,2e300]) mixed",
        );
        // n = 0: the empty vector is certain.
        let dm = DirichletMultinomial::new(0, &[0.5, 2.0]);
        assert_eq!(dm.ln_pmf(&[0, 0]), 0.0);
    }

    #[test]
    fn beta_binomial_ln_pmf_keeps_precision_at_huge_arguments() {
        // a = b = 1 is uniform on 0..=n: exactly −ln(n+1) (was −40960).
        let n = 1u64 << 62;
        let bb = BetaBinomial::new(n, 1.0, 1.0);
        for k in [0, 1, n / 3, n / 2, n - 1, n]
        {
            assert_polya_ln_close(bb.ln_pmf(k), -42.975_125_194_716_61, "BB(2^62,1,1)");
        }
        // Huge shapes and counts (was −50).
        let bb = BetaBinomial::new(1_000_000_000_000_000, 1e15, 1e15);
        assert_polya_ln_close(
            bb.ln_pmf(500_000_000_000_000),
            -17.697_912_104_154_153,
            "BB(1e15, 1e15, 1e15)",
        );
        // Lopsided shapes (was +55411.8, a pmf far above 1).
        let bb = BetaBinomial::new(n, 3.0, 1e14);
        assert_polya_ln_close(
            bb.ln_pmf(138_350),
            -12.234_863_141_759_818,
            "BB(2^62, 3, 1e14)",
        );
        // Small arguments still agree with SciPy betabinom.
        let bb = BetaBinomial::new(10, 2.0, 3.0);
        let total: f64 = (0..=10).map(|k| bb.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-13));
    }

    #[test]
    fn beta_binomial_ln_pmf_stays_finite_at_tiny_concentrations() {
        // a = b = ε → 0: the mass concentrates on {0, n} with 1/2 each. The
        // deviance mean w·A/(n+A) underflows here and must not be materialised.
        let d = BetaBinomial::new(10, 1.0e-300, 1.0e-300);
        for k in [0, 10]
        {
            let got = d.ln_pmf(k);
            assert!(got.is_finite(), "ln_pmf({k}) = {got}");
            assert!((got - 0.5f64.ln()).abs() < 1.0e-12, "ln_pmf({k}) = {got}");
        }
        let total: f64 = (0..=10).map(|k| d.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1.0e-12, "total = {total}");
        let dm = DirichletMultinomial::new(10, &[1.0e-300, 1.0e-300]);
        assert!((dm.ln_pmf(&[0, 10]) - 0.5f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn dirichlet_multinomial_sampling_is_deterministic_and_consistent() {
        let dm = DirichletMultinomial::new(30, &[1.0, 2.0, 3.0]);
        let mut r1 = SplitMix64::new(77);
        let mut r2 = SplitMix64::new(77);
        let mut totals = [0u64; 3];
        for _ in 0..3_000
        {
            let a = dm.sample(&mut r1);
            let b = dm.sample(&mut r2);
            assert_eq!(a, b);
            assert_eq!(a.iter().sum::<u64>(), 30);
            for (t, &x) in totals.iter_mut().zip(&a)
            {
                *t += x;
            }
        }
        // Empirical means near n·α/A = [5, 10, 15].
        assert!((totals[0] as f64 / 3_000.0 - 5.0).abs() < 0.4);
        assert!((totals[2] as f64 / 3_000.0 - 15.0).abs() < 0.6);
    }

    #[test]
    fn log_tail_and_isf_methods() {
        // logcdf / logsf / isf against SciPy.
        let b = Binomial::new(20, 0.3);
        assert!(close(b.logcdf(6), -0.497_564_258_657_831_5, 1e-12));
        assert!(close(b.logsf(10), -4.066_059_399_962_81, 1e-12));
        assert_eq!(b.isf(0.05), 9);
        let p = Poisson::new(4.2);
        assert!(close(p.logsf(15), -11.632_281_509_965_878, 1e-11));
        assert_eq!(p.isf(1e-6), 17);
        // Zeta: logsf stays finite deep in the heavy tail (no ln(1−cdf) blowup).
        let z = Zeta::new(2.5);
        assert!(close(z.logsf(5), -3.261_468_303_487_377, 1e-10));
        assert_eq!(z.isf(0.01), 14);
        // Consistency: exp(logcdf) == cdf, exp(logsf) == sf, isf∘sf round-trip.
        assert!(close(b.logcdf(6).exp(), b.cdf(6), 1e-13));
        assert!(close(p.logsf(7).exp(), p.sf(7), 1e-13));
        // isf(p) is the smallest k with sf(k) ≤ p.
        let k = p.isf(0.1);
        assert!(p.sf(k) <= 0.1 && (k == 0 || p.sf(k - 1) > 0.1));
    }

    #[test]
    fn interval_and_expect_match_scipy() {
        let b = Binomial::new(20, 0.3);
        assert_eq!(b.interval(0.9), (3, 9));
        assert_eq!(b.interval(0.95), (2, 10));
        let p = Poisson::new(4.2);
        assert_eq!(p.interval(0.9), (1, 8));
        // E[X] = mean, E[X²] = var + mean².
        assert!(close(p.expect(&|k| k as f64), 4.2, 1e-12));
        assert!(close(p.expect(&|k| (k * k) as f64), 21.84, 1e-11));
        assert!(close(b.expect(&|k| k as f64), 6.0, 1e-12));
        // E[1] = 1 (total mass).
        assert!(close(p.expect(&|_| 1.0), 1.0, 1e-13));
        let y = YuleSimon::new(2.5);
        assert_eq!(y.interval(0.8), (1, 3));
    }

    #[test]
    fn yule_simon_matches_scipy() {
        // SciPy yulesimon(2.5), support k ≥ 1.
        let y = YuleSimon::new(2.5);
        assert!(close(y.pmf(1), 0.714_285_714_285_714_4, 1e-12));
        assert!(close(y.pmf(2), 0.158_730_158_730_158_75, 1e-12));
        assert!(close(y.pmf(3), 0.057_720_057_720_057_74, 1e-12));
        assert!(close(y.pmf(10), 0.001_762_566_414_605_72, 1e-12));
        assert_eq!(y.pmf(0), 0.0);
        assert!(close(y.cdf(3), 0.930_735_930_735_930_7, 1e-12));
        assert!(close(y.sf(3), 0.069_264_069_264_069_28, 1e-12));
        assert!(close(y.sf(10), 0.007_050_265_658_422_88, 1e-11));
        assert!(close(y.mean(), 5.0 / 3.0, 1e-13));
        assert!(close(y.variance(), 50.0 / 9.0, 1e-12));
        // α = 2: pmf(k) = 4/(k(k+1)(k+2)) exactly.
        let y2 = YuleSimon::new(2.0);
        for k in 1..=6u64
        {
            let exact = 4.0 / (k * (k + 1) * (k + 2)) as f64;
            assert!(close(y2.pmf(k), exact, 1e-12), "k = {k}");
        }
        assert!(close(y2.mean(), 2.0, 1e-13));
        // Heavy tail: mean/variance diverge for α ≤ 1.
        let y3 = YuleSimon::new(0.8);
        assert_eq!(y3.mean(), f64::INFINITY);
        assert!(close(y3.pmf(1), 0.444_444_444_444_444_5, 1e-12));
    }

    #[test]
    fn yule_simon_lower_tail_small_alpha() {
        // cdf = 1 − sf used to cancel when sf ≈ 1. References from mpmath
        // 1.4.1 (60 digits): cdf(k) = −expm1(lnΓ(k+1) + lnΓ(α+1) − lnΓ(k+α+1)).
        let y = YuleSimon::new(1e-10);
        // Was 1.0000178e-10 (relative error 1.8e-5).
        assert!(rel_close(y.cdf(1), 9.999_999_999_000_001e-11, 1e-13));
        assert!(rel_close(y.cdf(10), 2.928_968_253_461_823e-10, 1e-13));
        assert!(rel_close(y.cdf(1000), 7.485_470_857_666_534e-10, 1e-13));
        assert!(rel_close(y.logcdf(1), -23.025_850_930_040_455, 1e-14));
        let y = YuleSimon::new(1e-12);
        // Was 1.7e-13 (84× too small).
        assert!(rel_close(
            y.cdf(1_000_000),
            1.439_272_672_276_132_6e-11,
            1e-12
        ));
        // Was 0.768 instead of 3.5e-11.
        assert!(rel_close(
            y.cdf(1_000_000_000_000_000),
            3.511_599_205_919_483e-11,
            1e-12
        ));
        // Smallest k with cdf(k) ≥ 1e-11 (cdf(12366) = 9.999962e-12,
        // cdf(12367) = 1.0000043e-11). Was 4092.
        assert_eq!(y.quantile(1e-11), 12_367);
        // α far below the f64 epsilon: cdf(5) = α·H₅ to relative order α.
        let y = YuleSimon::new(1e-300);
        assert!(rel_close(y.cdf(5), 1e-300 * 137.0 / 60.0, 1e-14));
        // At the top of the u64 range the old sf exceeded 1 and cdf was negative.
        let y = YuleSimon::new(1e-3);
        let k = u64::MAX;
        assert!(rel_close(y.cdf(k), 0.043_943_065_850_278_68, 1e-12));
        assert!(rel_close(y.sf(k), 0.956_056_934_149_721_3, 1e-13));
    }

    #[test]
    fn yule_simon_power_law_tail() {
        // sf(k) ~ Γ(α+1)·k^(−α) far in the tail; the old ln-gamma difference
        // lost all digits at k = 1e15 (sf was 2.57e-41, pmf similarly off).
        let y = YuleSimon::new(2.5);
        let k = 1_000_000_000_000_000;
        assert!(rel_close(y.sf(k), 1.050_935_853_074_607e-37, 1e-11));
        assert!(rel_close(y.pmf(k), 2.627_339_632_686_517_6e-52, 1e-11));
        assert!(rel_close(y.logsf(k), -85.145_967_384_929_65, 1e-13));
        // isf(1e-37) is 1_020_071_193_429_162 per mpmath; sf changes by only
        // 2.4e-15 relative per step there, so compare k to a relative 1e-9.
        // The old value was 5.59e14.
        let k = y.isf(1e-37) as f64;
        assert!(rel_close(k, 1_020_071_193_429_162.0, 1e-9), "isf = {k}");
        let y = YuleSimon::new(0.7);
        assert!(rel_close(
            y.sf(1_000_000_000),
            4.553_981_326_143_692_6e-7,
            1e-11
        ));
        assert!(rel_close(y.cdf(40), 0.932_305_570_568_456_9, 1e-13));
        assert!(rel_close(y.pmf(40), 0.001_184_652_515_052_004_7, 1e-12));
    }

    #[test]
    fn yule_simon_large_alpha_and_branch_switch() {
        // logsf stays finite after sf underflows: ln sf(100) = −787.6 at α = 1e5.
        let y = YuleSimon::new(1e5);
        assert!(y.sf(100) < f64::MIN_POSITIVE);
        assert!(rel_close(y.logsf(100), -787.603_654_032_455_1, 1e-13));
        assert!(rel_close(y.sf(40), 8.092_530_298_335_503e-153, 1e-12));
        // Extreme α: no NaN, the whole mass sits at k = 1.
        let y = YuleSimon::new(1e306);
        assert!(y.sf(40) < f64::MIN_POSITIVE);
        assert!(rel_close(y.cdf(40), 1.0, 1e-14));
        assert!(y.logsf(40).is_finite());
        // The direct sum (k ≤ 32) and the Stirling continuation (k ≥ 33)
        // agree with mpmath on both sides of the switch, and the pmf matches
        // the drop in sf across it.
        let y = YuleSimon::new(2.5);
        assert!(rel_close(y.sf(32), 5.024_586_513_153_711e-4, 1e-13));
        assert!(rel_close(y.sf(33), 4.670_742_392_509_083_8e-4, 1e-13));
        assert!(rel_close(y.pmf(33), 3.538_441_206_446_276e-5, 1e-12));
        assert!(rel_close(y.sf(32) - y.sf(33), y.pmf(33), 1e-11));
        let mass: f64 = (1..=40).map(|k| y.pmf(k)).sum::<f64>() + y.sf(40);
        assert!(rel_close(mass, 1.0, 1e-13));
    }

    #[test]
    fn boltzmann_matches_scipy() {
        // SciPy boltzmann(1.4, 10), support 0..=9.
        let b = Boltzmann::new(1.4, 10);
        assert!(close(b.pmf(0), 0.753_403_662_535_176, 1e-12));
        assert!(close(b.pmf(1), 0.185_787_055_803_661_06, 1e-12));
        assert!(close(b.pmf(5), 0.000_687_015_212_648_547_8, 1e-12));
        assert!(close(b.pmf(9), 2.540_488_627_524_870_7e-6, 1e-11));
        assert_eq!(b.pmf(10), 0.0);
        assert!(close(b.cdf(3), 0.996_302_964_738_045_1, 1e-12));
        assert!(close(b.sf(3), 0.003_697_035_261_954_862, 1e-11));
        assert!(close(b.mean(), 0.327_302_502_607_209_3, 1e-11));
        assert!(close(b.variance(), 0.434_360_036_406_343_8, 1e-11));
        // Total mass 1 and cdf reaches exactly 1 at the top level.
        let total: f64 = (0..10).map(|k| b.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-13));
        assert_eq!(b.cdf(9), 1.0);
        assert_eq!(b.sf(9), 0.0);
        // cdf + sf = 1 across the support.
        for k in 0..10
        {
            assert!(close(b.cdf(k) + b.sf(k), 1.0, 1e-12), "k = {k}");
        }
    }

    #[test]
    fn logarithmic_matches_scipy() {
        // SciPy logser(0.6), support k ≥ 1.
        let l = Logarithmic::new(0.6);
        assert!(close(l.pmf(1), 0.654_814_000_762_375, 1e-12));
        assert!(close(l.pmf(2), 0.196_444_200_228_712_48, 1e-12));
        assert!(close(l.pmf(3), 0.078_577_680_091_484_98, 1e-12));
        assert!(close(l.pmf(10), 0.000_659_901_643_622_698, 1e-12));
        assert_eq!(l.pmf(0), 0.0);
        assert!(close(l.cdf(3), 0.929_835_881_082_572_4, 1e-12));
        assert!(close(l.sf(3), 0.070_164_118_917_427_7, 1e-11));
        assert!(close(l.mean(), 1.637_035_001_905_937, 1e-12));
        assert!(close(l.variance(), 1.412_703_907_299_671_4, 1e-11));
        // Total head mass converges to 1.
        let total: f64 = (1..=200).map(|k| l.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-12));
    }

    #[test]
    fn planck_matches_scipy_and_equals_geometric_shift() {
        // SciPy planck(0.9), support k ≥ 0.
        let pl = Planck::new(0.9);
        assert!(close(pl.pmf(0), 0.593_430_340_259_400_9, 1e-12));
        assert!(close(pl.pmf(1), 0.241_270_771_519_012_58, 1e-12));
        assert!(close(pl.pmf(2), 0.098_093_375_481_836_78, 1e-12));
        assert!(close(pl.pmf(5), 0.006_592_415_595_629_638, 1e-12));
        assert!(close(pl.cdf(3), 0.972_676_277_552_707_5, 1e-12));
        assert!(close(pl.sf(3), 0.027_323_722_447_292_56, 1e-11));
        assert!(close(pl.mean(), 0.685_117_750_405_007_9, 1e-12));
        assert!(close(pl.variance(), 1.154_504_082_325_026_5, 1e-11));
        // Planck(λ) is the failures-count geometric: pmf(k) = (1−q)qᵏ with
        // q = e^(−λ); relates to Geometric(1−q) shifted by one trial.
        let g = Geometric::new(1.0 - (-0.9_f64).exp());
        for k in 0..8u64
        {
            assert!(close(pl.pmf(k), g.pmf(k + 1), 1e-12), "k = {k}");
        }
        // cdf + sf = 1.
        for k in 0..20
        {
            assert!(close(pl.cdf(k) + pl.sf(k), 1.0, 1e-12), "k = {k}");
        }
    }

    #[test]
    fn binomial_poisson_loader_accurate_at_large_n() {
        // The Loader path keeps full relative accuracy where the old
        // exp(Σ lnΓ) form drifted (values from SciPy, which uses Loader).
        let b = Binomial::new(100_000, 0.3);
        assert!(close(b.pmf(30_000), 0.002_752_954_648_397_429, 1e-12));
        assert!(close(b.pmf(31_000), 1.444_411_387_472_601_2e-13, 1e-11));
        let p = Poisson::new(10_000.0);
        assert!(close(p.pmf(10_000), 0.003_989_389_558_963_281, 1e-12));
        // Small-n values (unchanged behaviour) still match exactly.
        let s = Binomial::new(20, 0.3);
        assert!(close(s.pmf(6), 0.191_638_982_753_442_54, 1e-12));
        assert!(close(
            Poisson::new(4.2).pmf(3),
            0.185_165_382_579_258_7,
            1e-12
        ));
    }

    #[test]
    fn discrete_laplace_matches_scipy() {
        // SciPy dlaplace(0.8), support ℤ.
        let d = DiscreteLaplace::new(0.8);
        assert!(close(d.pmf(0), 0.379_948_962_255_224_9, 1e-12));
        assert!(close(d.pmf(1), 0.170_722_073_627_553_5, 1e-12));
        assert!(close(d.pmf(-1), 0.170_722_073_627_553_5, 1e-12));
        assert!(close(d.pmf(-3), 0.034_468_192_210_230_23, 1e-12));
        assert!(close(d.pmf(4), 0.015_487_557_100_816_039, 1e-12));
        assert!(close(d.cdf(0), 0.689_974_481_127_612_5, 1e-12));
        assert!(close(d.cdf(2), 0.937_406_927_250_178_6, 1e-12));
        assert!(close(d.cdf(-1), 0.310_025_518_872_387_55, 1e-12));
        assert!(close(d.sf(2), 0.062_593_072_749_821_4, 1e-11));
        assert!(close(d.mean(), 0.0, 1e-15));
        assert!(close(d.variance(), 2.963_534_189_184_372, 1e-11));
        // Symmetry and total mass.
        assert!(close(d.pmf(5), d.pmf(-5), 1e-13));
        let total: f64 = (-60..=60).map(|k| d.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-12));
        // cdf + sf = 1 on both sides of 0.
        for k in -5..=5_i64
        {
            assert!(close(d.cdf(k) + d.sf(k), 1.0, 1e-12), "k = {k}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn discrete_laplace_sampling_is_deterministic_and_symmetric() {
        let d = DiscreteLaplace::new(0.8);
        let mut r1 = SplitMix64::new(5);
        let mut r2 = SplitMix64::new(5);
        let a: Vec<i64> = (0..30_000).map(|_| d.sample(&mut r1)).collect();
        let b: Vec<i64> = (0..30_000).map(|_| d.sample(&mut r2)).collect();
        assert_eq!(a, b);
        let m = a.iter().sum::<i64>() as f64 / a.len() as f64;
        assert!(m.abs() < 0.06, "mean {m}");
    }

    #[test]
    fn method_of_moments_fitting() {
        // Poisson: λ̂ = mean.
        let dp = [3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 4.0, 5.0];
        let p = Poisson::fit_mom(&dp).unwrap();
        assert!(close(p.mean(), 4.0, 1e-13));
        assert!(Poisson::fit_mom(&[]).is_none());
        // Geometric (k ≥ 1): p̂ = 1/mean.
        let dg = [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let g = Geometric::fit_mom(&dg).unwrap();
        assert!(close(g.mean(), 1.875, 1e-13)); // 1/p̂ = mean
        assert!(Geometric::fit_mom(&[0.4, 0.6]).is_none()); // mean < 1
        // Negative binomial: overdispersed data ⇒ recovers sample mean/var.
        let dnb = [0.0, 1.0, 0.0, 5.0, 2.0, 10.0, 0.0, 3.0, 1.0, 8.0, 0.0, 4.0];
        let nb = NegativeBinomial::fit_mom(&dnb).unwrap();
        assert!(close(nb.mean(), 2.833_333_333_333_333_5, 1e-12));
        assert!(close(nb.variance(), 11.242_424_242_424_242, 1e-11));
        // Underdispersed (var ≤ mean) ⇒ no negative-binomial fit.
        assert!(NegativeBinomial::fit_mom(&[2.0, 2.0, 2.0, 3.0]).is_none());
    }

    // Regression tests for small-rate cancellation and far-tail collapse.
    // Oracle values: mpmath 1.4.1 at 50+ digits: exact pmf sums for the
    // finite laws, closed forms for Planck/discrete Laplace, and
    // `c·p^(k+1)·lerchphi(p, 1, k+1)` for the log-series tail.

    /// Pure relative closeness, so tiny tail values are actually checked
    /// (`close` degrades to an absolute test below 1).
    fn rel(a: f64, b: f64, tol: f64) -> bool {
        ((a - b) / b).abs() <= tol
    }

    #[test]
    fn boltzmann_moments_survive_small_rates() {
        // λ = 1e-10, N = 1000: the old 1/(1 − e^(−λ)) form returned a
        // negative mean (−333.3) and variance (−1.67e13).
        let b = Boltzmann::new(1e-10, 1000);
        assert!(rel(b.mean(), 499.499_991_666_675, 1e-12));
        assert!(rel(b.variance(), 83_333.249_999_999_96, 1e-12));
        // λ = 1e-8, N = 10: old variance 1.18e7 instead of 8.25.
        let b = Boltzmann::new(1e-8, 10);
        assert!(rel(b.mean(), 4.499_999_917_5, 1e-13));
        assert!(rel(b.variance(), 8.249_999_999_999_996, 1e-13));
        // λ = 1e-6, N = 100: old variance 865.0 (3.8 % high).
        let b = Boltzmann::new(1e-6, 100);
        assert!(rel(b.mean(), 49.499_166_750_000_139, 1e-13));
        assert!(rel(b.variance(), 833.249_999_583_333_3, 1e-13));
        // Large-rate branch unchanged.
        let b = Boltzmann::new(0.5, 4);
        assert!(rel(b.mean(), 0.915_423_511_538_135_7, 1e-14));
        assert!(rel(b.variance(), 1.021_451_445_167_521_9, 1e-14));
        let b = Boltzmann::new(1e-3, 5);
        assert!(rel(b.mean(), 1.998_000_000_866_666, 1e-14));
        assert!(rel(b.variance(), 1.999_997_400_002_583_3, 1e-14));
    }

    #[test]
    fn boltzmann_moment_branches_agree_at_switch() {
        // The series and direct forms must meet continuously at λ = 0.5.
        for &n in &[1u64, 2, 7, 50, 1000]
        {
            let lo = Boltzmann::new(SMALL_RATE * (1.0 - 1e-12), n);
            let hi = Boltzmann::new(SMALL_RATE, n);
            assert!(close(lo.mean(), hi.mean(), 1e-11), "mean n={n}");
            assert!(close(lo.variance(), hi.variance(), 1e-11), "var n={n}");
        }
        // N = 1 is a point mass at 0.
        let b = Boltzmann::new(1e-9, 1);
        assert!(b.mean().abs() < 1e-15 && b.variance().abs() < 1e-15);
    }

    #[test]
    fn boltzmann_sf_no_cancellation_near_top_level() {
        // sf(N−2) = pmf(N−1); the old e^(−λ(k+1)) − e^(−λN) form lost ~1e-6
        // relative precision here (9.999990225e-4).
        let b = Boltzmann::new(1e-10, 1000);
        assert!(rel(b.sf(998), 9.999_999_500_500_008e-4, 1e-12));
        assert!(rel(b.sf(0), 0.998_999_999_950_05, 1e-12));
        let b = Boltzmann::new(1e-8, 10);
        assert!(rel(b.sf(8), 0.099_999_995_500_000_06, 1e-13));
    }

    #[test]
    fn planck_and_dlaplace_variance_small_rate() {
        // 1 − e^(−λ) cancellation used to give 1.0000000022e16 at λ = 1e-8.
        let v = Planck::new(1e-8).variance();
        assert!(rel(v, 9_999_999_999_999_999.5, 1e-14));
        assert!(close(
            Planck::new(1e-5).variance(),
            9_999_999_999.916_665,
            1e-14
        ));
        let v = DiscreteLaplace::new(1e-8).variance();
        assert!(rel(v, 19_999_999_999_999_999.0, 1e-14));
    }

    #[test]
    fn logarithmic_far_tail_is_direct() {
        // The default 1 − cdf stalled at 1.1e-16 (p = 0.5) or returned 0
        // (p = 1e-3) instead of the true tiny tails.
        let l = Logarithmic::new(0.5);
        assert!(rel(l.sf(10), 1.187_690_168_272_118_9e-4, 1e-13));
        assert!(rel(l.sf(60), 2.019_291_821_716_591e-20, 1e-13));
        assert!(rel(l.sf(200), 4.444_726_234_643_462e-63, 1e-12));
        assert!(rel(l.sf(3), 0.038_203_306_074_024_4, 1e-14));
        assert!(rel(l.cdf(3), 0.961_796_693_925_975_6, 1e-14));
        assert_eq!(l.cdf(2000), 1.0);
        let l = Logarithmic::new(1e-3);
        assert!(rel(l.sf(10), 9.094_699_739_987_579e-32, 1e-13));
        assert!(rel(l.sf(60), 1.640_138_138_194_345e-182, 1e-12));
        let l = Logarithmic::new(0.9);
        assert!(rel(l.sf(10), 0.079_839_611_018_923_84, 1e-13));
        assert!(rel(l.sf(200), 1.315_613_176_104_763_6e-11, 1e-12));
        let l = Logarithmic::new(0.999);
        assert!(rel(l.sf(100), 0.263_176_042_357_603, 1e-12));
        assert!(rel(l.sf(20_000), 1.408_248_794_787_243_4e-11, 1e-11));
    }

    #[test]
    fn logarithmic_isf_and_quantile_terminate() {
        // With sf = 1 − cdf stuck at 1.1e-16, isf(1e-20) doubled its bracket
        // to u64::MAX and then summed 2^64 pmf terms (never returned).
        let l = Logarithmic::new(0.5);
        assert_eq!(l.isf(1e-20), 61);
        assert!(l.quantile(1.0) < 2_000);
    }

    #[test]
    fn geometric_p_one_survival_at_zero() {
        // p = 1 used to give sf(0) = exp(0·(−∞)) = NaN, so isf(0.5) was 0
        // (outside the k ≥ 1 support) instead of 1.
        let g = Geometric::new(1.0);
        assert_eq!(g.sf(0), 1.0);
        assert_eq!(g.logsf(0), 0.0);
        assert_eq!(g.sf(1), 0.0);
        assert_eq!(g.isf(0.5), 1);
        assert_eq!(g.isf(0.999), 1);
        assert_eq!(Geometric::new(0.3).sf(0), 1.0);
    }

    // Oracle values: mpmath 1.4.1 at 50 digits from
    // `loggamma(k+r) − loggamma(r) − loggamma(k+1) + r·log(p) + k·log1p(−p)`
    // and `loggamma(n+1) − Σ loggamma(kᵢ+1) + Σ kᵢ·log(pᵢ)`, with every f64
    // input taken at its exact binary value.
    fn assert_ln_close(got: f64, want: f64, what: &str) {
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(err < 1e-12, "{what}: got {got}, want {want} (rel {err:e})");
    }

    #[test]
    fn negative_binomial_ln_pmf_keeps_precision_at_huge_counts() {
        // The ln Γ sum used to return −24.347 (pmf ~9e4 times too large).
        let nb = NegativeBinomial::new(2.5, 1e-15);
        assert_ln_close(
            nb.ln_pmf(2_000_000_000_000_000),
            -35.783_738_494_543_686,
            "r=2.5",
        );
        // Used to return −14.5 (pmf ~57 times too large).
        let nb = NegativeBinomial::new(1e15, 0.5);
        assert_ln_close(
            nb.ln_pmf(1_000_000_000_000_000),
            -18.534_900_320_939_99,
            "r=1e15",
        );
        // Used to return −45.03 (pmf ~13 times too small).
        let nb = NegativeBinomial::new(1e-3, 1e-15);
        assert_ln_close(
            nb.ln_pmf(1_000_000_000_000_000),
            -42.445_955_280_294_54,
            "r=1e-3",
        );
        // Used to be 8.4 off in the log.
        let nb = NegativeBinomial::new(1e15, 0.999_999);
        assert_ln_close(nb.ln_pmf(3), -1_000_000_439.651_051, "k=3");
        assert_ln_close(
            NegativeBinomial::new(0.5, 1e-12).ln_pmf(100_000_000_000),
            -27.152_093_512_357_525,
            "r=0.5",
        );
    }

    #[test]
    fn negative_binomial_ln_pmf_small_arguments_and_endpoints() {
        assert_ln_close(
            NegativeBinomial::new(4.0, 0.3).ln_pmf(7),
            -2.525_124_082_092_825,
            "k=7",
        );
        assert_ln_close(
            NegativeBinomial::new(3.0, 0.2).ln_pmf(0),
            -4.828_313_737_302_301,
            "k=0",
        );
        let degenerate = NegativeBinomial::new(2.0, 1.0);
        assert_eq!(degenerate.ln_pmf(0), 0.0);
        assert_eq!(degenerate.ln_pmf(5), f64::NEG_INFINITY);
        // The pmf still sums to one.
        let nb = NegativeBinomial::new(0.7, 0.4);
        let total: f64 = (0..400).map(|k| nb.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-12, "sum = {total}");
    }

    #[test]
    fn multinomial_ln_pmf_keeps_precision_at_huge_n() {
        // The ln n! − Σ ln kᵢ! sum used to return −20.625 (pmf ~23 times too small).
        let half = 500_000_000_000_000;
        let m = Multinomial::new(2 * half, &[0.5, 0.5]);
        assert_ln_close(
            m.ln_pmf(&[half, half]),
            -17.495_179_550_100_07,
            "two halves",
        );
        let m = Multinomial::new(1_000_000_000_000_000, &[0.2, 0.3, 0.5]);
        assert_ln_close(
            m.ln_pmf(&[
                200_000_000_000_000,
                300_000_000_000_000,
                500_000_000_000_000,
            ]),
            -34.623_374_512_660_04,
            "three categories",
        );
    }

    #[test]
    fn multinomial_ln_pmf_empty_categories_and_endpoints() {
        let m = Multinomial::new(1_000_000, &[0.1, 0.4, 0.5]);
        assert_ln_close(
            m.ln_pmf(&[0, 400_000, 600_000]),
            -109_400.047_212_271,
            "empty category",
        );
        let m = Multinomial::new(10, &[0.2, 0.3, 0.5]);
        assert_ln_close(m.ln_pmf(&[2, 3, 5]), -2.464_515_960_140_266_3, "n=10");
        // Probabilities sum to one over all compositions of n = 6.
        let mut total = 0.0;
        for a in 0..=6_u64
        {
            for b in 0..=6 - a
            {
                total += m_six().pmf(&[a, b, 6 - a - b]);
            }
        }
        assert!((total - 1.0).abs() < 1e-13, "sum = {total}");
        let m = Multinomial::new(1_000_000_000_000_000, &[0.999, 0.001]);
        assert_ln_close(
            m.ln_pmf(&[1_000_000_000_000_000, 0]),
            -1_000_500_333_583.534_4,
            "all in one",
        );
        assert_eq!(Multinomial::new(0, &[0.5, 0.5]).ln_pmf(&[0, 0]), 0.0);
        let zero_p = Multinomial::new(4, &[0.0, 1.0]);
        assert_eq!(zero_p.ln_pmf(&[1, 3]), f64::NEG_INFINITY);
        assert_eq!(zero_p.ln_pmf(&[0, 4]), 0.0);
    }

    fn m_six() -> Multinomial {
        Multinomial::new(6, &[0.15, 0.6, 0.25])
    }
}
