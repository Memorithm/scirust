//! Correlation and rank-based hypothesis tests.
//!
//! - [`pearson_test`]: Pearson product-moment correlation with the exact
//!   Student-t reference distribution (`n − 2` dof) under bivariate normality.
//! - [`spearman_test`]: Spearman rank correlation (Pearson on mid-ranks) with
//!   the same t approximation SciPy uses for `spearmanr`.
//! - [`mann_whitney_u`]: two-sample Mann–Whitney U (Wilcoxon rank-sum) with
//!   the tie-corrected normal approximation and a 0.5 continuity correction.
//! - [`wilcoxon_signed_rank`]: one-sample / paired Wilcoxon signed-rank on
//!   differences, dropping exact zeros (Wilcoxon's convention) and correcting
//!   the variance for tied absolute differences.
//! - [`kruskal_wallis`]: k-sample Kruskal–Wallis H test (rank-based one-way
//!   ANOVA) with the tie correction and the χ²(k − 1) reference distribution.
//!
//! Every function validates its input and returns `None` rather than a
//! meaningless number: mismatched lengths, non-finite values, too few
//! observations, or a degenerate (zero-variance) configuration.
//!
//! The rank tests use the large-sample normal approximation only; for very
//! small samples without ties an exact permutation distribution is more
//! accurate, and callers that need it should compute it explicitly.

use crate::dist::{ChiSquared, Distribution, Normal, StudentT};
use crate::htest::{Tail, TestResult};

/// Outcome of a correlation test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelationTest {
    /// The correlation coefficient in `[-1, 1]` (Pearson's `r` or Spearman's `ρ`).
    pub coefficient: f64,
    /// The t statistic `r √(df / (1 − r²))` (`±∞` for a perfect correlation).
    pub statistic: f64,
    /// Reference degrees of freedom, `n − 2`.
    pub df: f64,
    /// The p-value for the requested [`Tail`]; `Greater` means positive association.
    pub p_value: f64,
}

/// Outcome of a rank test evaluated with the normal approximation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankTestResult {
    /// The rank statistic: `U` of the first sample for [`mann_whitney_u`],
    /// `W⁺` (sum of positive ranks) for [`wilcoxon_signed_rank`].
    pub statistic: f64,
    /// Standardised statistic `(statistic − E[statistic]) / sd`, without the
    /// continuity correction (its sign gives the direction of the effect).
    pub z: f64,
    /// The p-value for the requested [`Tail`].
    pub p_value: f64,
}

/// Mid-ranks (1-based, ties share their average rank) of `data`.
///
/// Returns `None` if any value is non-finite. An empty slice yields an empty
/// vector.
///
/// ```
/// use scirust_stats::nonparam::average_ranks;
/// assert_eq!(average_ranks(&[10.0, 20.0, 10.0, 30.0]).unwrap(), vec![1.5, 3.0, 1.5, 4.0]);
/// ```
pub fn average_ranks(data: &[f64]) -> Option<Vec<f64>> {
    if data.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    Some(ranks_with_ties(data).0)
}

/// Mid-ranks of finite `data`, plus `Σ (t³ − t)` over tie groups.
fn ranks_with_ties(data: &[f64]) -> (Vec<f64>, f64) {
    let n = data.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| data[i].total_cmp(&data[j]));
    let mut ranks = vec![0.0; n];
    let mut tie_term = 0.0;
    let mut start = 0;
    while start < n
    {
        let mut end = start + 1;
        while end < n && data[order[end]] == data[order[start]]
        {
            end += 1;
        }
        // Positions start..end (0-based) share the average 1-based rank.
        let avg = (start + end + 1) as f64 / 2.0;
        for &idx in &order[start..end]
        {
            ranks[idx] = avg;
        }
        let t = (end - start) as f64;
        tie_term += t * t * t - t;
        start = end;
    }
    (ranks, tie_term)
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Pearson correlation of finite, equal-length slices; `None` when either
/// side has zero variance.
fn pearson_r(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y)
    {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if !(sxx > 0.0 && syy > 0.0)
    {
        return None;
    }
    let prod = sxx * syy;
    // `sqrt(sxx·syy)` keeps perfect correlations exactly ±1; fall back to the
    // split form only if the product overflows.
    let denom = if prod.is_finite()
    {
        prod.sqrt()
    }
    else
    {
        sxx.sqrt() * syy.sqrt()
    };
    let r = sxy / denom;
    r.is_finite().then_some(r.clamp(-1.0, 1.0))
}

fn correlation_from_r(r: f64, n: usize, tail: Tail) -> CorrelationTest {
    let df = n as f64 - 2.0;
    let one_minus = (1.0 - r) * (1.0 + r);
    if one_minus <= 0.0
    {
        // Perfect (anti-)correlation: the t statistic is infinite.
        let statistic = if r > 0.0
        {
            f64::INFINITY
        }
        else
        {
            f64::NEG_INFINITY
        };
        let p_value = match tail
        {
            Tail::TwoSided => 0.0,
            Tail::Greater => (r < 0.0) as u8 as f64,
            Tail::Less => (r > 0.0) as u8 as f64,
        };
        return CorrelationTest {
            coefficient: r,
            statistic,
            df,
            p_value,
        };
    }
    let t = r * (df / one_minus).sqrt();
    let dist = StudentT::new(df);
    let p_value = match tail
    {
        Tail::TwoSided => (2.0 * dist.sf(t.abs())).min(1.0),
        Tail::Greater => dist.sf(t),
        Tail::Less => dist.cdf(t),
    };
    CorrelationTest {
        coefficient: r,
        statistic: t,
        df,
        p_value,
    }
}

/// Pearson correlation test of `H₀: ρ = 0`.
///
/// Returns `None` for mismatched lengths, fewer than three pairs, non-finite
/// values, or a constant `x` or `y`.
///
/// ```
/// use scirust_stats::prelude::*;
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let y = [1.2, 1.9, 3.2, 3.8, 5.1, 6.3];
/// let r = pearson_test(&x, &y, Tail::TwoSided).unwrap();
/// assert!(r.coefficient > 0.99 && r.p_value < 1e-4);
/// ```
pub fn pearson_test(x: &[f64], y: &[f64], tail: Tail) -> Option<CorrelationTest> {
    if x.len() != y.len() || x.len() < 3 || !all_finite(x) || !all_finite(y)
    {
        return None;
    }
    let r = pearson_r(x, y)?;
    Some(correlation_from_r(r, x.len(), tail))
}

/// Spearman rank-correlation test of `H₀: ρ_s = 0`.
///
/// `ρ_s` is Pearson's `r` on mid-ranks, so ties are handled exactly; the
/// p-value uses the Student-t approximation with `n − 2` dof (as SciPy's
/// `spearmanr`). Returns `None` under the same conditions as [`pearson_test`].
///
/// ```
/// use scirust_stats::prelude::*;
/// // Monotone but non-linear: Spearman sees a perfect association.
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [1.0, 8.0, 27.0, 64.0, 125.0];
/// let r = spearman_test(&x, &y, Tail::Greater).unwrap();
/// assert_eq!(r.coefficient, 1.0);
/// assert_eq!(r.p_value, 0.0);
/// ```
pub fn spearman_test(x: &[f64], y: &[f64], tail: Tail) -> Option<CorrelationTest> {
    if x.len() != y.len() || x.len() < 3 || !all_finite(x) || !all_finite(y)
    {
        return None;
    }
    let (rx, _) = ranks_with_ties(x);
    let (ry, _) = ranks_with_ties(y);
    let r = pearson_r(&rx, &ry)?;
    Some(correlation_from_r(r, x.len(), tail))
}

/// p-value of a continuity-corrected normal approximation.
///
/// `excess` is `statistic − mean`; `Greater` means the statistic is
/// stochastically larger than under `H₀`.
fn corrected_normal_p(excess: f64, sd: f64, continuity: f64, tail: Tail) -> f64 {
    let z = Normal::standard();
    match tail
    {
        Tail::TwoSided => (2.0 * z.sf((excess.abs() - continuity) / sd)).min(1.0),
        Tail::Greater => z.sf((excess - continuity) / sd),
        Tail::Less => z.sf((-excess - continuity) / sd),
    }
}

/// Mann–Whitney U (Wilcoxon rank-sum) test that `a` and `b` come from the
/// same distribution.
///
/// The statistic is `U₁ = R₁ − n₁(n₁+1)/2` for sample `a`; `Greater` tests
/// whether `a` is stochastically larger than `b`. The p-value uses the normal
/// approximation with the tie-corrected variance
/// `n₁n₂/12 · [(n+1) − Σ(t³−t)/(n(n−1))]` and a 0.5 continuity correction
/// (SciPy's `mannwhitneyu(method="asymptotic")` defaults).
///
/// Returns `None` if either sample is empty, any value is non-finite, or every
/// observation is tied (zero variance).
///
/// ```
/// use scirust_stats::prelude::*;
/// let a = [19.0, 22.0, 16.0, 29.0, 24.0];
/// let b = [20.0, 11.0, 17.0, 12.0];
/// let r = mann_whitney_u(&a, &b, Tail::TwoSided).unwrap();
/// assert_eq!(r.statistic, 17.0);
/// assert!(r.p_value > 0.05);
/// ```
pub fn mann_whitney_u(a: &[f64], b: &[f64], tail: Tail) -> Option<RankTestResult> {
    let (n1, n2) = (a.len(), b.len());
    if n1 == 0 || n2 == 0 || !all_finite(a) || !all_finite(b)
    {
        return None;
    }
    let pooled: Vec<f64> = a.iter().chain(b).copied().collect();
    let (ranks, tie_term) = ranks_with_ties(&pooled);
    let (n1f, n2f) = (n1 as f64, n2 as f64);
    let n = n1f + n2f;
    let r1: f64 = ranks[..n1].iter().sum();
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;
    let mean = n1f * n2f / 2.0;
    let var = n1f * n2f / 12.0 * ((n + 1.0) - tie_term / (n * (n - 1.0)));
    if var.is_nan() || var <= 0.0
    {
        return None;
    }
    let sd = var.sqrt();
    Some(RankTestResult {
        statistic: u1,
        z: (u1 - mean) / sd,
        p_value: corrected_normal_p(u1 - mean, sd, 0.5, tail),
    })
}

/// Wilcoxon signed-rank test of `H₀`: the `differences` are symmetric about 0.
///
/// For a paired design pass `x[i] − y[i]`; for a one-sample location test pass
/// `x[i] − m₀`. Exact zeros are discarded (Wilcoxon's convention) and the
/// remaining `|d|` are mid-ranked. The statistic is `W⁺`, the sum of ranks of
/// positive differences; `Greater` tests a positive shift. The p-value uses the
/// normal approximation with the tie-corrected variance
/// `[n(n+1)(2n+1) − ½Σ(t³−t)] / 24` and no continuity correction (SciPy's
/// `wilcoxon(method="approx")` defaults).
///
/// Returns `None` if any value is non-finite or no non-zero difference remains.
///
/// ```
/// use scirust_stats::prelude::*;
/// let d = [1.5, 2.0, -0.5, 3.0, 2.5, 1.0, 4.0, 0.0, 3.5, -1.0];
/// let r = wilcoxon_signed_rank(&d, Tail::Greater).unwrap();
/// assert!(r.z > 0.0 && r.p_value < 0.05);
/// ```
pub fn wilcoxon_signed_rank(differences: &[f64], tail: Tail) -> Option<RankTestResult> {
    if !all_finite(differences)
    {
        return None;
    }
    let nonzero: Vec<f64> = differences.iter().copied().filter(|&d| d != 0.0).collect();
    if nonzero.is_empty()
    {
        return None;
    }
    let abs: Vec<f64> = nonzero.iter().map(|d| d.abs()).collect();
    let (ranks, tie_term) = ranks_with_ties(&abs);
    let w_plus: f64 = nonzero
        .iter()
        .zip(&ranks)
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| *r)
        .sum();
    let n = nonzero.len() as f64;
    let mean = n * (n + 1.0) / 4.0;
    let var = (n * (n + 1.0) * (2.0 * n + 1.0) - 0.5 * tie_term) / 24.0;
    if var.is_nan() || var <= 0.0
    {
        return None;
    }
    let sd = var.sqrt();
    Some(RankTestResult {
        statistic: w_plus,
        z: (w_plus - mean) / sd,
        p_value: corrected_normal_p(w_plus - mean, sd, 0.0, tail),
    })
}

/// Kruskal–Wallis H test of `H₀`: all `groups` come from the same
/// distribution (the rank-based analogue of [`crate::htest::one_way_anova`]).
///
/// Every observation is mid-ranked in the pooled sample and
/// `H = 12 / (N(N+1)) · Σ Rᵢ²/nᵢ − 3(N+1)` is divided by the tie correction
/// `1 − Σ(t³−t)/(N³−N)`. The p-value is the upper tail of `χ²(k − 1)`, as in
/// SciPy's `kruskal`. Empty groups contain no observations and are ignored,
/// matching [`crate::htest::one_way_anova`].
///
/// Returns `None` if fewer than two non-empty groups remain, any value is
/// non-finite, or every observation is tied (the statistic is undefined).
///
/// ```
/// use scirust_stats::prelude::*;
/// let low = [1.1, 2.0, 1.4, 1.8];
/// let mid = [3.2, 2.9, 3.8, 3.5];
/// let high = [5.0, 6.1, 5.4, 5.9];
/// let r = kruskal_wallis(&[&low, &mid, &high]).unwrap();
/// assert_eq!(r.df, 2.0);
/// assert!(r.p_value < 0.01);
/// ```
pub fn kruskal_wallis(groups: &[&[f64]]) -> Option<TestResult> {
    let groups: Vec<&[f64]> = groups.iter().copied().filter(|g| !g.is_empty()).collect();
    let k = groups.len();
    if k < 2 || !groups.iter().all(|g| all_finite(g))
    {
        return None;
    }
    let pooled: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let (ranks, tie_term) = ranks_with_ties(&pooled);
    let n = pooled.len() as f64;
    let correction = 1.0 - tie_term / (n * n * n - n);
    if correction.is_nan() || correction <= 0.0
    {
        return None;
    }
    let mut offset = 0;
    let mut sum_sq_over_n = 0.0;
    for g in &groups
    {
        let r: f64 = ranks[offset..offset + g.len()].iter().sum();
        sum_sq_over_n += r * r / g.len() as f64;
        offset += g.len();
    }
    let h = (12.0 / (n * (n + 1.0)) * sum_sq_over_n - 3.0 * (n + 1.0)) / correction;
    // Rounding can push an exactly-null configuration a hair below zero.
    let h = h.max(0.0);
    let df = (k - 1) as f64;
    Some(TestResult {
        statistic: h,
        df,
        p_value: ChiSquared::new(df).sf(h),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn ranks_average_ties() {
        let (r, t) = ranks_with_ties(&[3.0, 1.0, 3.0, 2.0, 3.0]);
        assert_eq!(r, vec![4.0, 1.0, 4.0, 2.0, 4.0]);
        assert_eq!(t, 24.0);
        assert!(average_ranks(&[1.0, f64::NAN]).is_none());
        assert_eq!(average_ranks(&[]).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn rejects_bad_input() {
        let x = [1.0, 2.0, 3.0];
        assert!(pearson_test(&x, &[1.0, 2.0], Tail::TwoSided).is_none());
        assert!(pearson_test(&x[..2], &x[..2], Tail::TwoSided).is_none());
        assert!(pearson_test(&x, &[5.0, 5.0, 5.0], Tail::TwoSided).is_none());
        assert!(spearman_test(&x, &[1.0, f64::INFINITY, 2.0], Tail::TwoSided).is_none());
        assert!(mann_whitney_u(&[], &x, Tail::TwoSided).is_none());
        assert!(mann_whitney_u(&[2.0, 2.0], &[2.0], Tail::TwoSided).is_none());
        assert!(wilcoxon_signed_rank(&[0.0, 0.0], Tail::TwoSided).is_none());
        assert!(wilcoxon_signed_rank(&[1.0, f64::NAN], Tail::TwoSided).is_none());
    }

    #[test]
    fn pearson_tails_are_consistent() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = [2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 6.0];
        let two = pearson_test(&x, &y, Tail::TwoSided).unwrap();
        let gt = pearson_test(&x, &y, Tail::Greater).unwrap();
        let lt = pearson_test(&x, &y, Tail::Less).unwrap();
        assert!(close(two.p_value, 2.0 * gt.p_value, 1e-12));
        assert!(close(gt.p_value + lt.p_value, 1.0, 1e-12));
        // Hand-computed r = 23/28.
        assert!(close(two.coefficient, 23.0 / 28.0, 1e-12));
    }

    #[test]
    fn pearson_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [8.0, 6.0, 4.0, 2.0];
        let r = pearson_test(&x, &y, Tail::Less).unwrap();
        assert_eq!(r.coefficient, -1.0);
        assert_eq!(r.statistic, f64::NEG_INFINITY);
        assert_eq!(r.p_value, 0.0);
        assert_eq!(pearson_test(&x, &y, Tail::Greater).unwrap().p_value, 1.0);
    }

    #[test]
    fn spearman_is_rank_invariant() {
        let x = [0.3, 1.7, 2.2, 5.0, 9.1, 3.3];
        let y = [1.0, 2.5, 2.4, 7.0, 9.9, 3.0];
        let raw = spearman_test(&x, &y, Tail::TwoSided).unwrap();
        let ex: Vec<f64> = x.iter().map(|v: &f64| v.exp()).collect();
        let cy: Vec<f64> = y.iter().map(|v| v * v * v).collect();
        let mono = spearman_test(&ex, &cy, Tail::TwoSided).unwrap();
        // Ranks are identical, so the coefficient and df match exactly. The
        // statistic and p-value go through transcendental functions, which
        // Miri deliberately perturbs by a few ULPs, so compare those with a
        // tolerance rather than bit-for-bit.
        assert_eq!(raw.coefficient, mono.coefficient);
        assert_eq!(raw.df, mono.df);
        assert!(close(raw.statistic, mono.statistic, 1e-12));
        assert!(close(raw.p_value, mono.p_value, 1e-12));
    }

    #[test]
    fn mann_whitney_symmetry() {
        let a = [1.1, 2.3, 2.3, 4.0, 5.5, 6.1];
        let b = [3.0, 4.0, 7.2, 8.8, 9.0];
        let ab = mann_whitney_u(&a, &b, Tail::Less).unwrap();
        let ba = mann_whitney_u(&b, &a, Tail::Greater).unwrap();
        assert!(close(ab.statistic + ba.statistic, 30.0, 1e-12));
        assert!(close(ab.p_value, ba.p_value, 1e-12));
        assert!(close(ab.z, -ba.z, 1e-12));
        let two = mann_whitney_u(&a, &b, Tail::TwoSided).unwrap();
        assert!(close(two.p_value, 2.0 * ab.p_value, 1e-12));
    }

    #[test]
    fn wilcoxon_sign_flip_mirrors() {
        let d = [0.4, -1.2, 2.5, 3.1, -0.4, 1.9, 2.2, 0.0, 4.4];
        let neg: Vec<f64> = d.iter().map(|v| -v).collect();
        let g = wilcoxon_signed_rank(&d, Tail::Greater).unwrap();
        let l = wilcoxon_signed_rank(&neg, Tail::Less).unwrap();
        assert!(close(g.p_value, l.p_value, 1e-12));
        assert!(close(g.z, -l.z, 1e-12));
        // Eight non-zero differences: W⁺ + W⁻ = 36.
        let wn = wilcoxon_signed_rank(&neg, Tail::TwoSided).unwrap();
        assert!(close(g.statistic + wn.statistic, 36.0, 1e-12));
    }

    /// Reference values from SciPy 1.18.1 (`pearsonr`, `spearmanr`,
    /// `mannwhitneyu(method="asymptotic")`, `wilcoxon(method="approx")`).
    #[test]
    // `3.14` below is a data point, not an approximation of π.
    #[allow(clippy::approx_constant)]
    fn matches_scipy_oracle() {
        let tails = [Tail::TwoSided, Tail::Greater, Tail::Less];

        let x = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 6.1, 2.8, 4.9, 5.2];
        let y = [1.8, 3.9, 2.5, 5.1, 4.0, 3.3, 6.6, 2.2, 5.5, 4.4];
        let want = [
            7.478216895563361e-05,
            3.7391084477816806e-05,
            0.9999626089155221,
        ];
        for (tail, p) in tails.iter().zip(want)
        {
            let r = pearson_test(&x, &y, *tail).unwrap();
            assert!(close(r.coefficient, 0.9344073469018415, 1e-12));
            assert!(
                close(r.p_value, p, 1e-8),
                "pearson {tail:?}: {} vs {p}",
                r.p_value
            );
        }

        let xs = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0];
        let ys = [2.0, 1.0, 3.0, 3.0, 5.0, 4.0, 6.0, 6.0, 8.0, 7.0, 7.0];
        let want = [
            4.927337286994721e-05,
            2.4636686434973606e-05,
            0.9999753633135651,
        ];
        for (tail, p) in tails.iter().zip(want)
        {
            let r = spearman_test(&xs, &ys, *tail).unwrap();
            assert!(close(r.coefficient, 0.9236210093659822, 1e-12));
            assert!(
                close(r.p_value, p, 1e-8),
                "spearman {tail:?}: {} vs {p}",
                r.p_value
            );
        }

        let a = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 1.62];
        let b = [
            0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29, 1.62, 0.5,
        ];
        let want = [0.07765428098791592, 0.03882714049395796, 0.9667501296799905];
        for (tail, p) in tails.iter().zip(want)
        {
            let r = mann_whitney_u(&a, &b, *tail).unwrap();
            assert_eq!(r.statistic, 80.5);
            assert!(
                close(r.p_value, p, 1e-10),
                "mwu {tail:?}: {} vs {p}",
                r.p_value
            );
        }

        let d = [
            0.952, -0.147, 1.022, 0.43, 0.62, 0.59, 0.49, -0.08, 0.01, 0.0, 1.0,
        ];
        let want = [0.02182427562605354, 0.01091213781302677, 0.9890878621869732];
        for (tail, p) in tails.iter().zip(want)
        {
            let r = wilcoxon_signed_rank(&d, *tail).unwrap();
            assert_eq!(r.statistic, 50.0);
            assert!(close(r.z, 2.293412361469315, 1e-12));
            assert!(
                close(r.p_value, p, 1e-10),
                "wilcoxon {tail:?}: {} vs {p}",
                r.p_value
            );
        }
    }

    #[test]
    fn kruskal_wallis_rejects_bad_input() {
        let g = [1.0, 2.0, 3.0];
        assert!(kruskal_wallis(&[]).is_none());
        assert!(kruskal_wallis(&[&g]).is_none());
        assert!(kruskal_wallis(&[&g, &[]]).is_none());
        assert!(kruskal_wallis(&[&g, &[1.0, f64::NAN]]).is_none());
        assert!(kruskal_wallis(&[&[4.0, 4.0], &[4.0]]).is_none());
    }

    #[test]
    fn kruskal_wallis_invariants() {
        let a = [1.2, 3.4, 2.2, 5.0];
        let b = [2.8, 6.1, 4.4];
        let c = [7.5, 6.6, 8.0, 9.3, 5.5];
        let base = kruskal_wallis(&[&a, &b, &c]).unwrap();
        // Group order and empty groups do not matter.
        let perm = kruskal_wallis(&[&c, &[], &a, &b]).unwrap();
        assert!(close(base.statistic, perm.statistic, 1e-12));
        assert_eq!(perm.df, 2.0);
        // Only ranks matter: a monotone transform leaves H unchanged.
        let ln = |g: &[f64]| g.iter().map(|v| v.ln()).collect::<Vec<_>>();
        let (la, lb, lc) = (ln(&a), ln(&b), ln(&c));
        let mono = kruskal_wallis(&[&la, &lb, &lc]).unwrap();
        assert!(close(base.statistic, mono.statistic, 1e-12));
        // Identical groups have equal rank sums, so H = 0 and p = 1.
        let null = kruskal_wallis(&[&a, &a]).unwrap();
        assert!(null.statistic.abs() < 1e-12);
        assert!(close(null.p_value, 1.0, 1e-9));
        // With two groups and no ties, H equals the squared Mann–Whitney z.
        let z = mann_whitney_u(&a, &c, Tail::TwoSided).unwrap().z;
        let h2 = kruskal_wallis(&[&a, &c]).unwrap();
        assert!(close(h2.statistic, z * z, 1e-12));
    }

    /// Reference values from SciPy 1.18.1 (`kruskal`).
    #[test]
    fn kruskal_wallis_matches_scipy_oracle() {
        let g1 = [2.9, 3.0, 2.5, 2.6, 3.2];
        let g2 = [3.8, 2.7, 4.0, 2.4];
        let g3 = [2.8, 3.4, 3.7, 2.2, 2.0];
        let r = kruskal_wallis(&[&g1, &g2, &g3]).unwrap();
        assert!(close(r.statistic, 0.7714285714285722, 1e-12));
        assert!(close(r.p_value, 0.6799647735788936, 1e-10));

        // Heavily tied integer data exercises the tie correction.
        let h1 = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let h2 = [3.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0];
        let h3 = [2.0, 5.0, 5.0, 9.0, 9.0];
        let r = kruskal_wallis(&[&h1, &h2, &h3]).unwrap();
        assert!(close(r.statistic, 5.412311570330432, 1e-12));
        assert!(close(r.p_value, 0.06679308076510128, 1e-10));
    }
}
