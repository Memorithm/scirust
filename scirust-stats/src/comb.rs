//! Exact and logarithmic combinatorics.
//!
//! Exact counts are computed in `u128` and return `None` on overflow rather
//! than panicking or silently saturating. The approximate `ln_*` forms avoid
//! floating-point overflow and support [`crate::discrete`] and [`crate::lottery`].
//!
//! `binomial` uses the multiplicative recurrence `C(m, i) = C(m−1, i−1)·m/i`
//! (exact integer division at every step) with gcd pre-reduction, so it stays
//! exact up to genuinely astronomical counts (e.g. `C(130, 65) ≈ 9.5 × 10³⁷`)
//! instead of failing at the first intermediate overflow.

use scirust_special::ln_gamma;

/// Greatest common divisor (Euclid), used to pre-reduce factors in `binomial`.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0
    {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Exact factorial `n!` as `u128`; `None` for `n > 34` (`35!` overflows u128).
///
/// # Examples
///
/// ```
/// use scirust_stats::comb::factorial;
/// assert_eq!(factorial(5), Some(120));
/// assert_eq!(factorial(0), Some(1));
/// ```
///
/// ```
/// use scirust_stats::comb::factorial;
/// assert!(factorial(34).is_some());
/// assert_eq!(factorial(35), None);
/// ```
pub fn factorial(n: u64) -> Option<u128> {
    if n > 34
    {
        return None;
    }
    let mut acc: u128 = 1;
    for i in 2..=u128::from(n)
    {
        acc *= i;
    }
    Some(acc)
}

/// Approximate `ln(n!)` via `ln_gamma(n + 1)`; finite for every `u64`.
///
/// The conversion to `f64` and the log-gamma approximation are not exact.
/// Subtracting these results can lose small differences; use [`ln_binomial`]
/// instead of manually subtracting three log-factorials for a coefficient.
///
/// # Examples
///
/// ```
/// use scirust_stats::comb::ln_factorial;
/// assert!((ln_factorial(5) - 120.0_f64.ln()).abs() < 1e-12);
/// ```
///
/// ```
/// use scirust_stats::comb::ln_factorial;
/// assert!(ln_factorial(0).abs() < 1e-12);
/// assert!(ln_factorial(u64::MAX).is_finite());
/// ```
pub fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// Exact binomial coefficient `C(n, k)` as `u128`.
///
/// Returns `Some(0)` when `k > n` (the standard convention) and `None` only
/// when the exact value cannot be represented even after gcd reduction of the
/// intermediate products.
///
/// # Examples
///
/// ```
/// use scirust_stats::comb::binomial;
/// assert_eq!(binomial(49, 6), Some(13_983_816));
/// assert_eq!(binomial(49, 43), binomial(49, 6));
/// ```
///
/// ```
/// use scirust_stats::comb::binomial;
/// assert_eq!(binomial(5, 7), Some(0));
/// assert_eq!(binomial(1_000, 500), None);
/// ```
pub fn binomial(n: u64, k: u64) -> Option<u128> {
    if k > n
    {
        return Some(0);
    }
    let k = k.min(n - k);
    // Invariant after step i: acc = C(n − k + i, i), an exact integer.
    let mut acc: u128 = 1;
    for i in 1..=k
    {
        let f = u128::from(n - k + i);
        let mut den = u128::from(i);
        // acc·f is divisible by den; strip shared factors before multiplying
        // so the intermediate stays as small as possible.
        let g1 = gcd(acc, den);
        let acc_r = acc / g1;
        den /= g1;
        let g2 = gcd(f, den);
        let f_r = f / g2;
        den /= g2;
        acc = acc_r.checked_mul(f_r)? / den;
    }
    Some(acc)
}

/// Approximate `ln C(n, k)`; `−∞` when `k > n` (an impossible selection).
///
/// Uses [`binomial`] when the exact coefficient fits `u128`, avoiding the
/// subtraction of large, nearly equal log-factorials. The subsequent `f64`
/// conversion and logarithm are rounded; no bitwise reproducibility is promised.
///
/// Larger coefficients are evaluated without subtracting `ln n!` from
/// `ln (n−k)!`: with `j = min(k, n−k)` and `m = n − j`, the difference
/// `ln Γ(n+1) − ln Γ(m+1)` is expanded with Stirling's series as
/// `j·ln n + m·(ln1p(j/m) − j/m) + ½·ln1p(j/m)` plus the difference of the
/// series tails, and `ln j!` is then subtracted. Each term is at most of the
/// size of the result, so the error stays a small multiple of the rounding
/// unit relative to `ln C(n, k)` instead of scaling with `ln n!` (which made,
/// for example, `ln C(u64::MAX, 40)` evaluate to `0` instead of `≈ 1664.14`).
/// No heap allocation is used.
///
/// # Examples
///
/// ```
/// use scirust_stats::comb::ln_binomial;
/// assert!((ln_binomial(49, 6) - 13_983_816.0_f64.ln()).abs() < 1e-12);
/// ```
///
/// ```
/// use scirust_stats::comb::ln_binomial;
/// assert_eq!(ln_binomial(u64::MAX, 0), 0.0);
/// assert!((ln_binomial(u64::MAX, 1) - (u64::MAX as f64).ln()).abs() < 1e-12);
/// assert_eq!(ln_binomial(5, 7), f64::NEG_INFINITY);
/// ```
pub fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n
    {
        return f64::NEG_INFINITY;
    }
    if let Some(count) = binomial(n, k)
    {
        // Keep exact endpoint identities independent of transcendental rounding.
        if count == 1
        {
            return 0.0;
        }
        return (count as f64).ln();
    }
    ln_binomial_beyond_u128(n, k.min(n - k))
}

/// Tail `ln Γ(x+1) − [(x+½)·ln x − x + ½·ln 2π]` of Stirling's series.
///
/// Five terms; for `x ≥ 20` the first omitted term is below `1e-19`.
fn stirling_tail(x: f64) -> f64 {
    let r = x.recip();
    let r2 = r * r;
    r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 * (1.0 / 1680.0 - r2 / 1188.0))))
}

/// `ln C(n, j)` for `j ≤ n − j` when the exact coefficient exceeds `u128`.
fn ln_binomial_beyond_u128(n: u64, j: u64) -> f64 {
    let m = n - j;
    if m < 20
    {
        // Unreachable in practice (C(n, j) > u128::MAX needs n ≥ 130), kept so
        // the Stirling tail is never used outside its accurate range.
        return ln_factorial(n) - ln_factorial(j) - ln_factorial(m);
    }
    let jf = j as f64;
    let mf = m as f64;
    let t = jf / mf; // j/m ∈ (0, 1]
    let l = t.ln_1p(); // ln(n/m), computed without forming n/m
    // ln Γ(n+1) − ln Γ(m+1) = (m+½)·ln(n/m) + j·ln n − j + tail(n) − tail(m).
    let shifted = jf * (n as f64).ln()
        + mf * (l - t)
        + 0.5 * l
        + (stirling_tail(n as f64) - stirling_tail(mf));
    shifted - ln_factorial(j)
}

/// Exact number of ordered `k`-arrangements `P(n, k) = n!/(n−k)!` as `u128`;
/// `Some(0)` when `k > n`, `None` on result overflow.
///
/// # Examples
///
/// ```
/// use scirust_stats::comb::permutations;
/// assert_eq!(permutations(5, 3), Some(60));
/// assert_eq!(permutations(3, 5), Some(0));
/// ```
///
/// ```
/// use scirust_stats::comb::permutations;
/// assert_eq!(permutations(u64::MAX, 0), Some(1));
/// assert_eq!(permutations(u64::MAX, 3), None);
/// ```
pub fn permutations(n: u64, k: u64) -> Option<u128> {
    if k == 0
    {
        return Some(1);
    }
    if k > n
    {
        return Some(0);
    }
    let mut acc: u128 = 1;
    for i in (n - k + 1)..=n
    {
        acc = acc.checked_mul(u128::from(i))?;
    }
    Some(acc)
}

/// Combinations with repetition ("multichoose"): `C(n + k − 1, k)`.
///
/// The number of size-`k` multisets drawn from `n` distinct items; `Some(1)`
/// for `k = 0`, `Some(0)` for `n = 0, k > 0`. Returns `None` if `n + k − 1`
/// cannot fit `u64`, or if the exact coefficient cannot fit `u128`.
///
/// # Examples
///
/// ```
/// use scirust_stats::comb::multichoose;
/// assert_eq!(multichoose(5, 3), Some(35));
/// assert_eq!(multichoose(0, 2), Some(0));
/// ```
///
/// ```
/// use scirust_stats::comb::multichoose;
/// assert_eq!(multichoose(u64::MAX, 0), Some(1));
/// assert_eq!(multichoose(u64::MAX, 1), Some(u128::from(u64::MAX)));
/// assert_eq!(multichoose(u64::MAX, 2), None);
/// ```
pub fn multichoose(n: u64, k: u64) -> Option<u128> {
    if k == 0
    {
        return Some(1);
    }
    if n == 0
    {
        return Some(0);
    }
    binomial(n.checked_add(k - 1)?, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorial_exact_and_bounds() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
        assert_eq!(factorial(12), Some(479_001_600));
        // 34! is the largest factorial representable in u128.
        assert_eq!(
            factorial(34),
            Some(295_232_799_039_604_140_847_618_609_643_520_000_000)
        );
        assert_eq!(factorial(35), None);
    }

    #[test]
    fn binomial_exact_reference_values() {
        assert_eq!(binomial(0, 0), Some(1));
        assert_eq!(binomial(5, 7), Some(0));
        assert_eq!(binomial(49, 6), Some(13_983_816)); // classic 6/49
        assert_eq!(binomial(50, 5), Some(2_118_760)); // EuroMillions main
        assert_eq!(binomial(69, 5), Some(11_238_513)); // Powerball main
        assert_eq!(binomial(52, 5), Some(2_598_960)); // poker hands
        // Symmetry.
        assert_eq!(binomial(49, 6), binomial(49, 43));
        // Survives intermediate magnitudes near the u128 ceiling.
        assert_eq!(
            binomial(130, 65),
            Some(95_067_625_827_960_698_145_584_333_020_095_113_100)
        );
    }

    #[test]
    fn ln_forms_match_exact() {
        let exact = binomial(49, 6).unwrap() as f64;
        assert!((ln_binomial(49, 6) - exact.ln()).abs() < 1e-12);
        assert!(ln_binomial(5, 7) == f64::NEG_INFINITY);
        assert!((ln_factorial(12) - (479_001_600.0_f64).ln()).abs() < 1e-10);
        // Stirling territory: ln(1000!) = 5912.128178... (published value).
        assert!((ln_factorial(1000) - 5_912.128_178_488_163).abs() < 1e-8);
    }

    #[test]
    fn permutations_and_multichoose() {
        assert_eq!(permutations(49, 6), Some(10_068_347_520));
        assert_eq!(permutations(5, 0), Some(1));
        assert_eq!(permutations(3, 5), Some(0));
        assert_eq!(multichoose(5, 3), Some(35)); // C(7,3)
        assert_eq!(multichoose(0, 0), Some(1));
        assert_eq!(multichoose(0, 2), Some(0));
    }

    #[test]
    fn audit_ln_binomial_large_n_small_k_does_not_cancel() {
        for n in [1_u64 << 53, u64::MAX]
        {
            assert_eq!(ln_binomial(n, 0), 0.0);
            assert_eq!(ln_binomial(n, n), 0.0);
            for k in [1, 2]
            {
                let expected = (binomial(n, k).unwrap() as f64).ln();
                let tolerance = 16.0 * f64::EPSILON * expected.abs().max(1.0);
                assert!((ln_binomial(n, k) - expected).abs() <= tolerance);
                assert!((ln_binomial(n, n - k) - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn audit_ln_binomial_matches_small_exact_counts() {
        for n in 0..=24
        {
            for k in 0..=n
            {
                let expected = (binomial(n, k).unwrap() as f64).ln();
                let tolerance = 16.0 * f64::EPSILON * expected.abs().max(1.0);
                assert!((ln_binomial(n, k) - expected).abs() <= tolerance);
            }
            assert_eq!(ln_binomial(n, n + 1), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn audit_ln_binomial_overflow_falls_back_to_log_gamma() {
        assert_eq!(binomial(1_000, 500), None);
        // ln(math.comb(1000, 500)), independently evaluated with Decimal precision 80.
        let expected = 689.467_261_567_851_2;
        assert!((ln_binomial(1_000, 500) - expected).abs() < 1e-9);
    }

    /// Regression: beyond `u128`, `ln C(n, k)` was `ln n! − ln k! − ln (n−k)!`,
    /// whose rounding error scales with `ln n!`. References are
    /// `mpmath.log(mpmath.binomial(n, k))` at 50 significant digits.
    #[test]
    fn ln_binomial_beyond_u128_is_relatively_accurate() {
        let cases: [(u64, u64, f64); 9] = [
            (1_000_000_000_000_000, 50, 1578.46105279376),
            (1_000_000_000_000, 50, 1233.0732888434293),
            (u64::MAX, 40, 1664.1361425187026),
            (u64::MAX, 1 << 63, 1.2786308645202655e+19),
            (1_000_000_000_000_000, 500, 14658.057738995061),
            (1_000_000_000_000_000_000, 1_000, 35534.40349540466),
            (1_000_000, 500_000, 693140.0470130637),
            (200, 100, 135.7532360812785),
            (1_000_000_000, 100_000_000, 325082963.3148496),
        ];
        for (n, k, expected) in cases
        {
            assert_eq!(
                binomial(n, k),
                None,
                "case ({n}, {k}) must use the fallback"
            );
            for kk in [k, n - k]
            {
                let got = ln_binomial(n, kk);
                let rel = ((got - expected) / expected).abs();
                assert!(
                    rel < 1e-13,
                    "ln C({n}, {kk}) = {got}, expected {expected}, rel {rel:e}"
                );
            }
        }
    }

    /// Regression: the Hypergeometric law builds its pmf from three `ln C`
    /// terms, so the old cancellation distorted it for large populations
    /// (the first case returned 2980.96, which is not a probability). Reference:
    /// `C(K,k)·C(N−K,n−k)/C(N,n)` in mpmath at 50 digits.
    #[test]
    fn hypergeometric_pmf_with_huge_population() {
        use crate::discrete::{DiscreteDistribution, Hypergeometric};
        let cases: [(u64, u64, u64, u64, f64); 2] = [
            (
                1_000_000_000_000_000,
                500_000_000_000_000,
                100,
                50,
                0.07958923738718274,
            ),
            (
                1_000_000_000_000,
                300_000_000_000,
                60,
                20,
                0.09305760249639095,
            ),
        ];
        for (population, successes, draws, k, expected) in cases
        {
            let got = Hypergeometric::new(population, successes, draws).pmf(k);
            let rel = ((got - expected) / expected).abs();
            assert!(rel < 1e-11, "pmf = {got}, expected {expected}, rel {rel:e}");
        }
    }

    #[test]
    fn audit_permutations_empty_selection_at_u64_boundary() {
        assert_eq!(permutations(u64::MAX, 0), Some(1));
        assert_eq!(permutations(u64::MAX, 1), Some(u128::from(u64::MAX)));
        assert_eq!(permutations(u64::MAX, 3), None);
    }

    #[test]
    fn audit_multichoose_checks_parameter_overflow_without_rejecting_boundary() {
        assert_eq!(multichoose(u64::MAX, 0), Some(1));
        assert_eq!(multichoose(u64::MAX, 1), Some(u128::from(u64::MAX)));
        assert_eq!(multichoose(u64::MAX, 2), None);
        assert_eq!(multichoose(2, u64::MAX), None);
        assert_eq!(multichoose(1, u64::MAX), Some(1));
        assert_eq!(multichoose(0, u64::MAX), Some(0));
    }
}
