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
/// Coefficients larger than `u128` retain the log-gamma fallback, whose
/// cancellation error is not bounded by this API. No heap allocation is used.
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
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
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
