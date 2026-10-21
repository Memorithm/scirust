//! Classical hypothesis tests. Each returns a [`TestResult`] with the test
//! statistic, its reference degrees of freedom, and a p-value computed from the
//! corresponding distribution in [`crate::dist`] — so every p-value traces back
//! to the audited `scirust-special` numeric base.

use crate::describe::{mean, median};
use crate::discrete::DiscreteDistribution;
use crate::dist::{ChiSquared, Distribution, FisherF, StudentT};
use crate::nonparam::{binary_exponent, mul_pow2};

/// Outcome of a hypothesis test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    /// The test statistic (t, F, or χ²).
    pub statistic: f64,
    /// Reference degrees of freedom (for F, this is the numerator dof).
    pub df: f64,
    /// The p-value.
    pub p_value: f64,
}

/// Which tail(s) the alternative hypothesis occupies for a t-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// H₁: mean ≠ reference (default).
    TwoSided,
    /// H₁: mean > reference.
    Greater,
    /// H₁: mean < reference.
    Less,
}

fn t_p_value(t: f64, df: f64, tail: Tail) -> f64 {
    // `StudentT::new` asserts `df > 0`. An undefined dof (NaN, e.g. the
    // Welch dof of two constant samples) leaves the p-value determined by
    // `t` alone only when `t` is infinite: the tails of every Student-t
    // vanish at ±∞. Otherwise there is no p-value.
    if df.is_nan() || df <= 0.0
    {
        return match (t.is_infinite(), tail)
        {
            (false, _) => f64::NAN,
            (true, Tail::TwoSided) => 0.0,
            (true, Tail::Greater) => (t < 0.0) as u8 as f64,
            (true, Tail::Less) => (t > 0.0) as u8 as f64,
        };
    }
    let dist = StudentT::new(df);
    match tail
    {
        Tail::TwoSided => 2.0 * dist.sf(t.abs()),
        Tail::Greater => dist.sf(t),
        Tail::Less => dist.cdf(t),
    }
}

/// Magnitude `2^1022` from which a difference of two values bounded by it,
/// such as a deviation `x − x̄`, can overflow.
const HUGE: f64 = f64::from_bits(0x7fd0_0000_0000_0000);

/// Whether any finite value reaches [`HUGE`].
fn any_huge<'a>(values: impl IntoIterator<Item = &'a f64>) -> bool {
    values.into_iter().any(|x| x.is_finite() && x.abs() >= HUGE)
}

/// Every group multiplied by `1/4` when some finite value reaches [`HUGE`]
/// (or `force` is set); `None` otherwise.
///
/// For finite values bounded by `2^1022`, a difference such as `x − x̄` or
/// `x̄₁ − x̄₂` stays below `2^1023` and cannot overflow. Without this, data near
/// `f64::MAX` turned those deviations into `±∞`, so the tests below returned
/// `t = 0`, `F = NaN`, or `None` for finite samples. Every statistic in this
/// module is invariant under a common positive rescaling, and multiplying by
/// a power of two is exact except for values below `2^-1020`, which are
/// negligible next to a value of magnitude `2^1022`.
fn quartered_if_huge(groups: &[&[f64]], force: bool) -> Option<Vec<Vec<f64>>> {
    (force || any_huge(groups.iter().flat_map(|g| g.iter()))).then(|| {
        groups
            .iter()
            .map(|g| g.iter().map(|x| x * 0.25).collect())
            .collect()
    })
}

/// Exponent `k` such that `2^k` brings the largest absolute deviation of the
/// `(sample, sample mean)` pairs into `[1, 2)`; `0` when that deviation is
/// zero or non-finite (constant, `NaN` or infinite samples), which leaves
/// those cases to ordinary floating-point semantics.
fn deviation_exponent(samples: &[(&[f64], f64)]) -> i32 {
    let largest = samples
        .iter()
        .flat_map(|&(data, m)| data.iter().map(move |&x| (x - m).abs()))
        .fold(0.0_f64, f64::max);
    if largest > 0.0 && largest.is_finite()
    {
        -binary_exponent(largest)
    }
    else
    {
        0
    }
}

/// `variance(data) · 4^k`, from deviations scaled by `2^k` before squaring.
///
/// Scaling by a power of two is exact, so where the unscaled squares neither
/// overflow nor underflow this is exactly `4^k` times [`variance`]; elsewhere
/// it keeps the digits that the unscaled squares lose.
fn scaled_variance(data: &[f64], m: f64, k: i32) -> f64 {
    data.iter()
        .map(|&x| {
            let d = mul_pow2(x - m, k);
            d * d
        })
        .sum::<f64>()
        / (data.len() as f64 - 1.0)
}

/// One-sample t-test of `H₀: mean(data) = mu0`.
///
/// Returns `None` if fewer than two samples (variance undefined).
///
/// The statistic is invariant under a common positive rescaling of `data`
/// and `mu0`, and the computation keeps that property: the deviations are
/// scaled by a power of two before they are squared, so the sample variance
/// does not overflow for a spread above about `1e154` (which gave `t = 0`,
/// `p = 1`) or underflow below about `1e-154` (which gave `t = ±∞`, `p = 0`,
/// or `NaN`). When a finite value of `data` or `mu0` reaches `2^1022`, both
/// are first multiplied by `1/4` (exact), so the deviations and `x̄ − mu0`
/// cannot overflow for finite inputs near `f64::MAX` (which gave `t = 0`,
/// `p = 1`).
///
/// ```
/// use scirust_stats::prelude::*;
/// let m = f64::MAX;
/// // Same statistic as for `[-1, 1, 1]` against `0`: `t = 0.5`.
/// let r = t_test_one_sample(&[-m, m, m], 0.0, Tail::TwoSided).unwrap();
/// assert!((r.statistic - 0.5).abs() < 1e-12);
/// ```
pub fn t_test_one_sample(data: &[f64], mu0: f64, tail: Tail) -> Option<TestResult> {
    let n = data.len();
    if n < 2
    {
        return None;
    }
    let quartered = quartered_if_huge(&[data], any_huge([&mu0]));
    let (data, mu0) = match &quartered
    {
        Some(q) => (q[0].as_slice(), mu0 * 0.25),
        None => (data, mu0),
    };
    let m = mean(data);
    // `s` and the numerator both carry the factor `2^k`, which cancels in `t`.
    let k = deviation_exponent(&[(data, m)]);
    let s = scaled_variance(data, m, k).sqrt();
    let df = n as f64 - 1.0;
    let t = mul_pow2(m - mu0, k) / (s / (n as f64).sqrt());
    Some(TestResult {
        statistic: t,
        df,
        p_value: t_p_value(t, df, tail),
    })
}

/// Two-sample t-test of `H₀: mean(a) = mean(b)`.
///
/// `equal_var = true` uses the pooled-variance Student test; `false` uses
/// Welch's unequal-variance test (the safer default in practice). Returns `None`
/// if either group has fewer than two samples.
///
/// The Welch–Satterthwaite degrees of freedom are undefined (`0/0`) when both
/// samples are constant; `df` is then `NaN`, and the p-value follows from the
/// statistic alone: `t = ±∞` (different constants) gives `0` or `1`, and
/// `t = NaN` (equal constants) gives `NaN`. A `NaN` sample gives a `NaN`
/// p-value in both variants.
///
/// The statistic and the Welch dof are invariant under a common positive
/// rescaling of `a` and `b`. Both samples' deviations are scaled by one power
/// of two before they are squared, so the variances do not overflow for a
/// spread above about `1e154` (which gave `t = 0`, `p = 1`, and a `NaN` Welch
/// dof) or underflow below about `1e-154` (which gave `t = ±∞`, `p = 0`, or
/// `NaN`). When a finite value reaches `2^1022`, both samples are first
/// multiplied by `1/4` (exact), so the deviations and `x̄₁ − x̄₂` cannot
/// overflow for finite inputs near `f64::MAX` (which gave `t = 0`, `p = 1`).
///
/// ```
/// use scirust_stats::prelude::*;
/// let r = t_test_two_sample(&[1.0, 1.0, 1.0], &[2.0, 2.0], false, Tail::TwoSided).unwrap();
/// assert_eq!(r.statistic, f64::NEG_INFINITY);
/// assert!(r.df.is_nan());
/// assert_eq!(r.p_value, 0.0);
/// ```
pub fn t_test_two_sample(a: &[f64], b: &[f64], equal_var: bool, tail: Tail) -> Option<TestResult> {
    let (n1, n2) = (a.len(), b.len());
    if n1 < 2 || n2 < 2
    {
        return None;
    }
    let quartered = quartered_if_huge(&[a, b], false);
    let (a, b) = match &quartered
    {
        Some(q) => (q[0].as_slice(), q[1].as_slice()),
        None => (a, b),
    };
    let (m1, m2) = (mean(a), mean(b));
    // Both variances carry the factor `4^k`, so `se` carries `2^k`, which
    // the scaled mean difference cancels in `t`; the Welch dof below only
    // uses ratios of the `wᵢ`.
    let k = deviation_exponent(&[(a, m1), (b, m2)]);
    let (v1, v2) = (scaled_variance(a, m1, k), scaled_variance(b, m2, k));
    let diff = mul_pow2(m1 - m2, k);
    let (n1f, n2f) = (n1 as f64, n2 as f64);

    let (t, df) = if equal_var
    {
        let sp2 = ((n1f - 1.0) * v1 + (n2f - 1.0) * v2) / (n1f + n2f - 2.0);
        let se = (sp2 * (1.0 / n1f + 1.0 / n2f)).sqrt();
        (diff / se, n1f + n2f - 2.0)
    }
    else
    {
        let (w1, w2) = (v1 / n1f, v2 / n2f);
        let se = (w1 + w2).sqrt();
        // Welch–Satterthwaite degrees of freedom
        // `(w1 + w2)² / (w1²/(n1 − 1) + w2²/(n2 − 1))`, evaluated on `wᵢ`
        // divided by the larger one. The ratio is scale-invariant, but the
        // squares of the raw `wᵢ` underflow for a data spread below about 1e-77 (and
        // overflow above about 1e77), which used to give a wrong dof or `0/0 = NaN`.
        // Two constant samples (`w1 = w2 = 0`) leave it undefined (`NaN`).
        let w_max = w1.max(w2);
        let df = if w_max > 0.0 && w_max.is_finite()
        {
            let (r1, r2) = (w1 / w_max, w2 / w_max);
            (r1 + r2).powi(2) / (r1 * r1 / (n1f - 1.0) + r2 * r2 / (n2f - 1.0))
        }
        else
        {
            f64::NAN
        };
        (diff / se, df)
    };
    Some(TestResult {
        statistic: t,
        df,
        p_value: t_p_value(t, df, tail),
    })
}

/// Between- and within-group sums of squares of a one-way layout, both
/// multiplied by the same factor `4^k`, which cancels in their ratio.
///
/// `2^k` brings the largest of the deviations `xᵢⱼ − x̄ᵢ` and `x̄ᵢ − x̄` into
/// `[1, 2)` before they are squared, so neither sum overflows for a data
/// spread above about `1e154` nor underflows below about `1e-154`. Scaling by
/// a power of two is exact, so in between the two sums are exactly `4^k`
/// times the unscaled ones. `k = 0` when every deviation is zero or one is
/// non-finite, which leaves those cases to ordinary floating-point semantics.
/// Every group must be non-empty.
fn scaled_anova_sums_of_squares(groups: &[&[f64]]) -> (f64, f64) {
    let quartered = quartered_if_huge(groups, false);
    let quartered_refs: Vec<&[f64]>;
    let groups = match &quartered
    {
        Some(q) =>
        {
            quartered_refs = q.iter().map(Vec::as_slice).collect();
            quartered_refs.as_slice()
        },
        None => groups,
    };
    // `mean` keeps the grand mean finite when the plain sum overflows.
    let pooled: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let grand = mean(&pooled);
    let group_means: Vec<f64> = groups.iter().map(|g| mean(g)).collect();
    let largest = groups
        .iter()
        .zip(&group_means)
        .flat_map(|(g, &m)| {
            g.iter()
                .map(move |&x| (x - m).abs())
                .chain([(m - grand).abs()])
        })
        .fold(0.0_f64, f64::max);
    let k = if largest > 0.0 && largest.is_finite()
    {
        -binary_exponent(largest)
    }
    else
    {
        0
    };
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for (g, &m) in groups.iter().zip(&group_means)
    {
        let d = mul_pow2(m - grand, k);
        ss_between += g.len() as f64 * d * d;
        ss_within += g
            .iter()
            .map(|&x| {
                let d = mul_pow2(x - m, k);
                d * d
            })
            .sum::<f64>();
    }
    (ss_between, ss_within)
}

/// One-way ANOVA of `H₀: all group means are equal`.
///
/// `groups` is a slice of samples per group. Returns `None` if fewer than two
/// groups or fewer than one degree of freedom within groups.
///
/// The F statistic is invariant under a common positive rescaling of the
/// data, and the computation keeps that property: the deviations are scaled
/// by a power of two before they are squared, so the sums of squares do not
/// overflow for a data spread above about `1e154` or underflow below about
/// `1e-154` (either gave `F = NaN`). When a finite value reaches `2^1022`,
/// the data are first multiplied by `1/4` (exact) and the grand mean is
/// formed with [`mean`], so finite data near `f64::MAX` no longer make the
/// grand mean or the deviations overflow (which gave `F = NaN`).
///
/// ```
/// use scirust_stats::prelude::*;
/// let a = [1.0e200, 2.0e200, 3.0e200];
/// let b = [4.0e200, 5.0e200, 6.0e200];
/// let r = one_way_anova(&[&a, &b]).unwrap();
/// // F = 13.5 for these data at any scale (SciPy `f_oneway`).
/// assert!((r.statistic - 13.5).abs() < 1e-12);
/// ```
pub fn one_way_anova(groups: &[&[f64]]) -> Option<TestResult> {
    // Empty samples contain no observations and therefore are not factor
    // levels. Counting them in k corrupts both numerator and denominator dof.
    let groups: Vec<&[f64]> = groups.iter().copied().filter(|g| !g.is_empty()).collect();
    let k = groups.len();
    if k < 2
    {
        return None;
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total <= k
    {
        return None; // no within-group degrees of freedom
    }
    let (ss_between, ss_within) = scaled_anova_sums_of_squares(&groups);
    let df1 = (k - 1) as f64;
    let df2 = (n_total - k) as f64;
    let ms_between = ss_between / df1;
    let ms_within = ss_within / df2;
    let f = ms_between / ms_within;
    let p = FisherF::new(df1, df2).sf(f);
    Some(TestResult {
        statistic: f,
        df: df1,
        p_value: p,
    })
}

/// Location used to centre each group in [`levene_test`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeveneCenter {
    /// Group means: Levene's original (1960) test, most powerful for
    /// symmetric, light-tailed data.
    Mean,
    /// Group medians: the Brown–Forsythe (1974) variant, robust to skewed or
    /// heavy-tailed data (SciPy's default for `levene`).
    Median,
}

/// Levene / Brown–Forsythe test of `H₀`: all `groups` have equal variance.
///
/// Each observation is replaced by its absolute deviation from the group
/// centre chosen by `center`, and a one-way ANOVA F statistic is computed on
/// those deviations against `F(k − 1, N − k)`. Empty groups are ignored, as
/// in [`one_way_anova`]. `df` holds the numerator degrees of freedom `k − 1`.
///
/// Returns `None` if fewer than two non-empty groups remain, there are no
/// within-group degrees of freedom (`N ≤ k`), any value is non-finite, or
/// every deviation equals its group mean (the F ratio is undefined).
///
/// The statistic is invariant under a common positive rescaling of the data.
/// The sums of squares are formed as in [`one_way_anova`], so a data spread
/// above about `1e154` or below about `1e-154` no longer makes them overflow
/// or underflow (which returned `None`). When a finite value reaches
/// `2^1022`, the data are first multiplied by `1/4` (exact), so the absolute
/// deviations from the group centres cannot overflow for finite data near
/// `f64::MAX` (which also returned `None`).
///
/// ```
/// use scirust_stats::prelude::*;
/// let tight = [10.0, 10.1, 9.9, 10.0, 10.05, 9.95];
/// let wide = [7.0, 13.0, 9.0, 11.5, 8.0, 12.5];
/// let r = levene_test(&[&tight, &wide], LeveneCenter::Median).unwrap();
/// assert_eq!(r.df, 1.0);
/// assert!(r.p_value < 0.01);
/// ```
pub fn levene_test(groups: &[&[f64]], center: LeveneCenter) -> Option<TestResult> {
    let groups: Vec<&[f64]> = groups.iter().copied().filter(|g| !g.is_empty()).collect();
    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if k < 2 || n_total <= k || groups.iter().any(|g| g.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let quartered = quartered_if_huge(&groups, false);
    let groups: Vec<&[f64]> = match &quartered
    {
        Some(q) => q.iter().map(Vec::as_slice).collect(),
        None => groups,
    };
    let deviations: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let c = match center
            {
                LeveneCenter::Mean => mean(g),
                LeveneCenter::Median => median(g),
            };
            g.iter().map(|x| (x - c).abs()).collect()
        })
        .collect();
    let deviations: Vec<&[f64]> = deviations.iter().map(Vec::as_slice).collect();
    let (ss_between, ss_within) = scaled_anova_sums_of_squares(&deviations);
    if !(ss_within > 0.0 && ss_within.is_finite() && ss_between.is_finite())
    {
        return None;
    }
    let df1 = (k - 1) as f64;
    let df2 = (n_total - k) as f64;
    let w = (ss_between / df1) / (ss_within / df2);
    Some(TestResult {
        statistic: w,
        df: df1,
        p_value: FisherF::new(df1, df2).sf(w),
    })
}

/// Bartlett's test of `H₀`: all `groups` have equal variance.
///
/// The statistic
/// `T = [(N − k) ln s²ₚ − Σ(nᵢ − 1) ln sᵢ²] / [1 + (Σ 1/(nᵢ − 1) − 1/(N − k)) / (3(k − 1))]`
/// is referred to `χ²(k − 1)`, as in SciPy's `bartlett`. It is exact under
/// normality but sensitive to departures from it; prefer [`levene_test`] with
/// [`LeveneCenter::Median`] for non-normal data.
///
/// Returns `None` if fewer than two groups are given, any group has fewer than
/// two observations or zero variance, or any value is non-finite.
///
/// The statistic is invariant under a common positive rescaling of the data,
/// and the computation keeps that property: each group's deviations are
/// scaled by a power of two before they are squared, and the logarithms are
/// taken of the scaled variances, so a data spread above about `1e154` or
/// below about `1e-154` no longer makes the variances overflow or underflow
/// (which returned `None`), including when the groups' spreads differ by
/// more than the `f64` range. When a finite value reaches `2^1022`, the data
/// are first multiplied by `1/4` (exact), so the deviations from the group
/// means cannot overflow for finite data near `f64::MAX` (which returned
/// `None`).
///
/// ```
/// use scirust_stats::prelude::*;
/// let tight = [10.0, 10.1, 9.9, 10.0, 10.05, 9.95];
/// let wide = [7.0, 13.0, 9.0, 11.5, 8.0, 12.5];
/// let r = bartlett_test(&[&tight, &wide]).unwrap();
/// assert!(r.p_value < 0.001);
/// ```
pub fn bartlett_test(groups: &[&[f64]]) -> Option<TestResult> {
    let k = groups.len();
    if k < 2
    {
        return None;
    }
    let quartered = quartered_if_huge(groups, false);
    let quartered_refs: Vec<&[f64]>;
    let groups = match &quartered
    {
        Some(q) =>
        {
            quartered_refs = q.iter().map(Vec::as_slice).collect();
            quartered_refs.as_slice()
        },
        None => groups,
    };
    // Per group: dof, the exponent `kᵢ`, and `sᵢ² · 4^kᵢ`, which lies in
    // `[1/(nᵢ − 1), 4nᵢ/(nᵢ − 1))` because the largest scaled deviation is in
    // `[1, 2)`. `ln sᵢ² = ln(sᵢ² · 4^kᵢ) − 2kᵢ ln 2`.
    let mut scaled = Vec::with_capacity(k);
    for g in groups
    {
        if g.len() < 2 || g.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        let m = mean(g);
        let e = deviation_exponent(&[(g, m)]);
        let v = scaled_variance(g, m, e);
        if !(v > 0.0 && v.is_finite())
        {
            return None;
        }
        scaled.push(((g.len() - 1) as f64, e, v));
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let df_within = (n_total - k) as f64;
    let df = (k - 1) as f64;
    // The pooled variance is formed at the scale `4^e_min` of the widest
    // group; narrower groups are scaled down exactly (contributions below
    // `2^-1100` of it are negligible and flush to zero).
    let e_min = scaled.iter().map(|&(_, e, _)| e).min().unwrap_or(0);
    let mut pooled_ss = 0.0;
    let mut sum_log = 0.0;
    let mut sum_inv = 0.0;
    let mut exponent_sum = 0i64;
    for &(dof, e, v) in &scaled
    {
        pooled_ss += dof * mul_pow2(v, (2 * (e_min - e)).max(-1100));
        sum_log += dof * v.ln();
        sum_inv += 1.0 / dof;
        exponent_sum += (dof as i64) * i64::from(e - e_min);
    }
    let pooled = pooled_ss / df_within;
    // (N − k)·ln s²ₚ − Σ(nᵢ − 1)·ln sᵢ², with the `2 ln 2` terms of the
    // scale factors collected into one exact integer multiple.
    let numerator =
        df_within * pooled.ln() - sum_log + 2.0 * std::f64::consts::LN_2 * exponent_sum as f64;
    let denominator = 1.0 + (sum_inv - 1.0 / df_within) / (3.0 * df);
    // ln is concave, so the numerator is ≥ 0 up to rounding.
    let t = (numerator / denominator).max(0.0);
    Some(TestResult {
        statistic: t,
        df,
        p_value: ChiSquared::new(df).sf(t),
    })
}

/// Pearson's χ² goodness-of-fit test.
///
/// `observed` and `expected` counts must be the same non-empty length with all
/// `expected > 0`. `ddof` is the number of parameters estimated from the data
/// (subtracted from `k − 1` degrees of freedom; pass 0 for a fully-specified
/// model).
///
/// Returns `None` on a shape violation, when an `expected` count is not
/// strictly positive, when any count is NaN or infinite, or when the residual
/// degrees of freedom are non-positive.
pub fn chi_square_gof(observed: &[f64], expected: &[f64], ddof: usize) -> Option<TestResult> {
    if observed.len() != expected.len() || observed.is_empty()
    {
        return None;
    }
    // `e <= 0.0` is false for NaN, so finiteness is checked explicitly.
    if expected.iter().any(|&e| !(e > 0.0 && e.is_finite()))
        || observed.iter().any(|o| !o.is_finite())
    {
        return None;
    }
    // `d · (d / e)` rather than `d² / e`: the square alone overflows once
    // |o − e| exceeds ~1.3e154 even when the term itself is representable.
    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| {
            let d = o - e;
            d * (d / e)
        })
        .sum();
    let df = observed.len() as f64 - 1.0 - ddof as f64;
    if df <= 0.0
    {
        return None;
    }
    let p = ChiSquared::new(df).sf(chi2);
    Some(TestResult {
        statistic: chi2,
        df,
        p_value: p,
    })
}

/// Pearson's χ² goodness-of-fit test for a **discrete distribution**.
///
/// `observed[i]` is the count of observations equal to the value `i`, except
/// the final entry `observed[L−1]`, which counts the whole upper tail
/// (value `≥ L−1`). Expected counts are taken from `dist` — `N·pmf(i)` for the
/// exact bins and `N·sf(L−2)` for the tail bin — so they sum to `N` exactly.
///
/// Adjacent bins are pooled until every pooled expected count reaches
/// `min_expected` (Cochran's rule of thumb is `5`), which also absorbs the
/// zero-probability leading bins of supports that start at 1 (e.g.
/// [`crate::discrete::Geometric`]). `ddof` is the number of parameters
/// estimated from the data — pass `1` when the distribution was fitted with
/// `fit_mom` on one parameter, `0` when fully specified — and is subtracted
/// from `(pooled bins − 1)` degrees of freedom.
///
/// Returns `None` if there is no data, fewer than two bins survive pooling, or
/// the residual degrees of freedom are non-positive.
pub fn chi2_gof_discrete<D: DiscreteDistribution>(
    observed: &[u64],
    dist: &D,
    ddof: usize,
    min_expected: f64,
) -> Option<TestResult> {
    let l = observed.len();
    let n: f64 = observed.iter().map(|&c| c as f64).sum();
    if l < 2 || n <= 0.0
    {
        return None;
    }
    // Expected counts: exact bins 0..L−1, tail bin = N·P(X ≥ L−1).
    let mut expected = vec![0.0_f64; l];
    for (i, e) in expected.iter_mut().enumerate().take(l - 1)
    {
        *e = n * dist.pmf(i as u64);
    }
    expected[l - 1] = n * dist.sf((l - 2) as u64);

    // Pool adjacent bins so each pooled expected ≥ min_expected (> 0 floor so
    // zero-probability bins always merge forward).
    let floor = min_expected.max(1e-12);
    let (mut po, mut pe) = (Vec::new(), Vec::new());
    let (mut co, mut ce) = (0.0_f64, 0.0_f64);
    for i in 0..l
    {
        co += observed[i] as f64;
        ce += expected[i];
        if ce >= floor
        {
            po.push(co);
            pe.push(ce);
            co = 0.0;
            ce = 0.0;
        }
    }
    // Fold any small trailing remainder into the last pooled bin.
    if ce > 0.0 || co > 0.0
    {
        if let (Some(lo), Some(le)) = (po.last_mut(), pe.last_mut())
        {
            *lo += co;
            *le += ce;
        }
        else
        {
            po.push(co);
            pe.push(ce);
        }
    }
    if po.len() < 2
    {
        return None;
    }
    chi_square_gof(&po, &pe, ddof)
}

/// One-sample Kolmogorov–Smirnov test that `data` is drawn from `dist`.
///
/// Returns the D statistic and the asymptotic p-value (Kolmogorov distribution).
/// `df` holds the sample size `n`.
///
/// Returns `None` for an empty sample, for any `NaN` sample, or when
/// `dist.cdf` is `NaN` at a sample point: a `NaN` has no position in the
/// empirical distribution, so no meaningful D exists. Infinite samples are
/// accepted (their CDF is `0` or `1`).
///
/// ```
/// use scirust_stats::dist::Normal;
/// use scirust_stats::htest::ks_test_one_sample;
/// let n = Normal::standard();
/// let sample = [-1.2, -0.4, 0.1, 0.5, 0.9, 1.3, -0.7, 0.2];
/// let r = ks_test_one_sample(&sample, &n).unwrap();
/// assert_eq!(r.df, 8.0);
/// assert!(r.p_value > 0.5);
/// assert!(ks_test_one_sample(&[0.1, f64::NAN], &n).is_none());
/// assert!(ks_test_one_sample(&[], &n).is_none());
/// ```
pub fn ks_test_one_sample<D: Distribution>(data: &[f64], dist: &D) -> Option<TestResult> {
    let n = data.len();
    if n == 0 || data.iter().any(|x| x.is_nan())
    {
        return None;
    }
    let mut v: Vec<f64> = data.to_vec();
    // NaN-free here, so `total_cmp` is the usual numeric order (with −0 < +0,
    // which does not affect the CDF).
    v.sort_by(f64::total_cmp);
    let nf = n as f64;
    let mut d = 0.0_f64;
    for (i, &x) in v.iter().enumerate()
    {
        let f = dist.cdf(x);
        // `f64::max` ignores a NaN operand, so a NaN CDF would otherwise be
        // silently dropped from the supremum.
        if f.is_nan()
        {
            return None;
        }
        let d_plus = (i as f64 + 1.0) / nf - f;
        let d_minus = f - i as f64 / nf;
        d = d.max(d_plus).max(d_minus);
    }
    // Asymptotic p-value with the Stephens small-sample correction.
    let en = nf.sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    Some(TestResult {
        statistic: d,
        df: nf,
        p_value: kolmogorov_q(lambda),
    })
}

/// The complementary Kolmogorov distribution
/// `Q(λ) = 2 Σ_{j≥1} (−1)^{j−1} e^{−2 j² λ²}`, clamped to `[0, 1]`.
fn kolmogorov_q(lambda: f64) -> f64 {
    if lambda.is_nan()
    {
        return f64::NAN;
    }
    if lambda <= 0.0
    {
        return 1.0;
    }
    if lambda.is_infinite()
    {
        return 0.0;
    }

    // The alternating expansion below is catastrophically slow and suffers
    // cancellation near zero. Jacobi's theta transformation gives the CDF
    // there as a rapidly convergent sum of strictly positive terms; Q = 1-CDF.
    if lambda < 1.18
    {
        let scale = std::f64::consts::PI.sqrt() * 2.0_f64.sqrt() / lambda;
        let exponent_scale = -std::f64::consts::PI.powi(2) / (8.0 * lambda * lambda);
        let mut cdf_sum = 0.0;
        for j in 1..=100
        {
            let odd = (2 * j - 1) as f64;
            let term = (exponent_scale * odd * odd).exp();
            cdf_sum += term;
            if term <= f64::EPSILON * cdf_sum.max(1.0)
            {
                break;
            }
        }
        return (1.0 - scale * cdf_sum).clamp(0.0, 1.0);
    }

    let mut sum = 0.0;
    let a = -2.0 * lambda * lambda;
    for j in 1..=10_000
    {
        let jf = j as f64;
        let term = (a * jf * jf).exp();
        sum += if j % 2 == 1 { term } else { -term };
        if term <= f64::EPSILON * sum.abs().max(1.0)
        {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Normal;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn one_sample_t_matches_reference() {
        // Data with mean 5, tested against μ0 = 4.
        let d = [5.1, 4.9, 5.3, 4.7, 5.0, 5.2, 4.8, 5.0];
        let r = t_test_one_sample(&d, 4.0, Tail::TwoSided).unwrap();
        assert!(r.statistic > 0.0);
        assert!(r.df == 7.0);
        // Strongly significant (mean clearly ≠ 4); the exact value is ≈ 2.1e-6.
        assert!(r.p_value < 1e-5, "p = {}", r.p_value);
        // A one-sample test of the data against its own mean gives t ≈ 0, p ≈ 1.
        let r0 = t_test_one_sample(&d, mean(&d), Tail::TwoSided).unwrap();
        assert!(close(r0.statistic, 0.0, 1e-9));
        assert!(close(r0.p_value, 1.0, 1e-9));
    }

    #[test]
    fn two_sample_welch_and_pooled() {
        let a = [23.0, 21.0, 25.0, 22.0, 24.0, 20.0];
        let b = [27.0, 29.0, 26.0, 28.0, 30.0, 25.0];
        let welch = t_test_two_sample(&a, &b, false, Tail::TwoSided).unwrap();
        let pooled = t_test_two_sample(&a, &b, true, Tail::TwoSided).unwrap();
        // Both should find a highly significant difference.
        assert!(welch.p_value < 0.001);
        assert!(pooled.p_value < 0.001);
        // Welch dof ≤ pooled dof (n1+n2−2 = 10).
        assert!(welch.df <= 10.0 + 1e-9);
        // Identical groups → t = 0, p = 1.
        let same = t_test_two_sample(&a, &a, true, Tail::TwoSided).unwrap();
        assert!(close(same.statistic, 0.0, 1e-9));
        assert!(close(same.p_value, 1.0, 1e-9));
    }

    #[test]
    fn welch_degrees_of_freedom_are_scale_invariant() {
        // Regression test: the Welch–Satterthwaite dof squared `v/n`
        // directly. For a data spread ≲ 1e-77 (or ≳ 1e77) those squares
        // underflow (overflow), the dof became 0/0 = NaN (∞/∞) and
        // `StudentT::new(NaN)` panicked; just above the underflow threshold
        // the subnormal squares returned a wrong dof (5.5 instead of 4.824
        // at scale 2⁻²⁶⁸ ≈ 2.1e-81). The dof, t and p are
        // invariant under scaling both samples by the same factor.
        // Reference (unit scale): SciPy 1.18 `ttest_ind(a, b,
        // equal_var=False)`: t = −1.364096585169425,
        // df = 4.824272094820922 (mpmath: 4.82427209482092242),
        // p = 0.23272669004045568 (two-sided), 0.11636334502022784 (less),
        // 0.8836366549797722 (greater).
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 5.0, 7.0];
        // Powers of two keep the scaled data exact.
        for scale in [
            1.0,
            2f64.powi(-268),
            2f64.powi(-270),
            2f64.powi(-300),
            2f64.powi(300),
        ]
        {
            let sa: Vec<f64> = a.iter().map(|x| x * scale).collect();
            let sb: Vec<f64> = b.iter().map(|x| x * scale).collect();
            for (tail, want_p) in [
                (Tail::TwoSided, 0.232_726_690_040_455_68),
                (Tail::Less, 0.116_363_345_020_227_84),
                (Tail::Greater, 0.883_636_654_979_772_2),
            ]
            {
                let r = t_test_two_sample(&sa, &sb, false, tail).unwrap();
                assert!(
                    close(r.statistic, -1.364_096_585_169_425, 1e-12),
                    "scale {scale:e}: t = {}",
                    r.statistic
                );
                assert!(
                    close(r.df, 4.824_272_094_820_922, 1e-12),
                    "scale {scale:e}: df = {}",
                    r.df
                );
                assert!(
                    close(r.p_value, want_p, 1e-10),
                    "scale {scale:e}, {tail:?}: p = {}",
                    r.p_value
                );
            }
        }
    }

    #[test]
    fn welch_handles_constant_and_nan_samples_without_panicking() {
        // Regression test: two constant samples made the Welch dof 0/0 = NaN
        // and `StudentT::new(NaN)` panicked; so did any NaN in either
        // sample. Different constant means give t = ±∞, whose p-value is
        // 0 (or 1) for every dof (SciPy gives t = −inf, p = 0); the dof
        // itself is undefined and reported as NaN.
        let lo = [1.0, 1.0, 1.0];
        let hi = [2.0, 2.0, 2.0, 2.0];
        let r = t_test_two_sample(&lo, &hi, false, Tail::TwoSided).unwrap();
        assert_eq!(r.statistic, f64::NEG_INFINITY);
        assert!(r.df.is_nan());
        assert_eq!(r.p_value, 0.0);
        let less = t_test_two_sample(&lo, &hi, false, Tail::Less).unwrap();
        assert_eq!(less.p_value, 0.0);
        let greater = t_test_two_sample(&lo, &hi, false, Tail::Greater).unwrap();
        assert_eq!(greater.p_value, 1.0);
        let up = t_test_two_sample(&hi, &lo, false, Tail::Greater).unwrap();
        assert_eq!(up.statistic, f64::INFINITY);
        assert_eq!(up.p_value, 0.0);
        // Identical constant samples: t = 0/0, nothing to test.
        let same = t_test_two_sample(&lo, &lo, false, Tail::TwoSided).unwrap();
        assert!(same.statistic.is_nan() && same.p_value.is_nan());
        // NaN input propagates to a NaN p-value, as in the pooled test.
        let with_nan = [1.0, f64::NAN, 3.0];
        let r = t_test_two_sample(&with_nan, &hi, false, Tail::TwoSided).unwrap();
        assert!(r.p_value.is_nan());
        let pooled = t_test_two_sample(&with_nan, &hi, true, Tail::TwoSided).unwrap();
        assert!(pooled.p_value.is_nan());
        // One constant sample keeps a finite dof (n − 1 of the other).
        let b = [2.0, 3.0, 5.0, 7.0];
        let r = t_test_two_sample(&lo, &b, false, Tail::TwoSided).unwrap();
        assert!(close(r.df, 3.0, 1e-12), "df = {}", r.df);
        assert!(r.p_value > 0.0 && r.p_value < 1.0);
    }

    #[test]
    fn t_tests_do_not_depend_on_the_magnitude_of_the_data() {
        // Regression: the sample variances were formed from raw squared
        // deviations, which overflow for a spread above about 1e154 (t became
        // 0, p = 1, and the Welch dof NaN) and underflow below about 1e-154
        // (t = ±∞ with p = 0, or 0/0 = NaN; percent-level errors in the
        // subnormal range just above). t is invariant under a common positive
        // rescaling. Reference values at unit scale: SciPy 1.18
        // `ttest_ind` / `ttest_1samp`, cross-checked with mpmath at 60 digits.
        let a = [1.0, 1.1, 0.9];
        let b = [2.0, 2.1, 1.9];
        let c = [1.05, 0.95, 1.0];
        let (t_ab, p_ab) = (-12.247_448_713_915_883, 2.552_167_494_419_273e-4);
        let (t_one, p_one) = (17.320_508_075_688_767, 3.316_758_722_356_862e-3);
        let rel = |x: f64, y: f64| (x - y).abs() <= 1e-12 * y.abs();
        // Powers of two keep the rescaled samples exact.
        for s in [
            2f64.powi(-700),
            2f64.powi(-530),
            1.0,
            2f64.powi(530),
            2f64.powi(700),
        ]
        {
            let sa: Vec<f64> = a.iter().map(|v| v * s).collect();
            let sb: Vec<f64> = b.iter().map(|v| v * s).collect();
            let sc: Vec<f64> = c.iter().map(|v| v * s).collect();
            for equal_var in [true, false]
            {
                let r = t_test_two_sample(&sa, &sb, equal_var, Tail::TwoSided).unwrap();
                assert!(rel(r.statistic, t_ab), "s = {s:e}: t = {}", r.statistic);
                assert!(rel(r.df, 4.0), "s = {s:e}: df = {}", r.df);
                assert!(rel(r.p_value, p_ab), "s = {s:e}: p = {}", r.p_value);
                // Overlapping samples with equal means: t ≈ 0, p ≈ 1, not a
                // spurious p = 0 or NaN.
                let r = t_test_two_sample(&sa, &sc, equal_var, Tail::TwoSided).unwrap();
                assert!(r.statistic.abs() < 1e-12, "s = {s:e}: t = {}", r.statistic);
                assert!(r.p_value > 1.0 - 1e-12, "s = {s:e}: p = {}", r.p_value);
            }
            let r = t_test_one_sample(&sa, 0.0, Tail::TwoSided).unwrap();
            assert!(rel(r.statistic, t_one), "s = {s:e}: t = {}", r.statistic);
            assert!(rel(r.p_value, p_one), "s = {s:e}: p = {}", r.p_value);
            let r = t_test_one_sample(&sa, s, Tail::TwoSided).unwrap();
            assert!(r.statistic.abs() < 1e-12, "s = {s:e}: t = {}", r.statistic);
            assert!(r.p_value > 1.0 - 1e-12, "s = {s:e}: p = {}", r.p_value);
        }
    }

    #[test]
    fn anova_detects_and_rejects() {
        // Three clearly-separated groups → tiny p.
        let g1 = [1.0, 2.0, 1.5, 1.8, 2.2];
        let g2 = [5.0, 6.0, 5.5, 5.8, 6.2];
        let g3 = [9.0, 10.0, 9.5, 9.8, 10.2];
        let r = one_way_anova(&[&g1, &g2, &g3]).unwrap();
        assert!(r.p_value < 1e-6, "p = {}", r.p_value);
        assert!(r.df == 2.0);
        // Three copies of the same group → F ≈ 0, p ≈ 1.
        let r2 = one_way_anova(&[&g1, &g1, &g1]).unwrap();
        assert!(r2.statistic.abs() < 1e-9);
        assert!(close(r2.p_value, 1.0, 1e-9));
    }

    #[test]
    fn anova_ignores_empty_groups_when_computing_degrees_of_freedom() {
        let g1 = [1.0, 2.0];
        let g2 = [3.0, 4.0];
        let empty = [];
        let result = one_way_anova(&[&g1, &g2, &empty]).unwrap();
        assert!(close(result.statistic, 8.0, 1e-12));
        assert_eq!(result.df, 1.0);
    }

    #[test]
    fn anova_and_levene_do_not_depend_on_the_magnitude_of_the_data() {
        // Regression: the sums of squares were formed from raw squared
        // deviations, which overflow for a data spread above about 1e154 and
        // underflow below about 1e-154. `one_way_anova` then returned
        // F = ∞/∞ or 0/0 = NaN (and a NaN p-value), and `levene_test`
        // returned `None`; just above the underflow threshold F lost digits
        // (3.35620 instead of 3.35639 at scale 1e-160). Both statistics are
        // invariant under a common positive rescaling. Reference values at
        // unit scale: SciPy 1.18 `f_oneway` / `levene`, cross-checked with
        // mpmath at 50 digits.
        let a = [1.0, 2.0, 3.0, 4.5];
        let b = [2.0, 3.5, 5.0, 6.0, 4.0];
        let c = [0.5, 1.5, 2.5];
        let rel = |x: f64, y: f64, tol: f64| (x - y).abs() <= tol * y.abs();
        let lev_refs = [
            (
                LeveneCenter::Mean,
                0.402_611_916_469_761_8,
                0.680_036_913_823_352_2,
            ),
            (
                LeveneCenter::Median,
                0.361_215_303_825_956_5,
                0.706_487_460_510_983_8,
            ),
        ];
        // Powers of two keep the rescaled samples exact.
        for s in [
            2f64.powi(-1000),
            2f64.powi(-700),
            2f64.powi(-530),
            1.0,
            2f64.powi(530),
            2f64.powi(700),
            2f64.powi(1000),
        ]
        {
            let scale = |g: &[f64]| g.iter().map(|v| v * s).collect::<Vec<f64>>();
            let (sa, sb, sc) = (scale(&a), scale(&b), scale(&c));
            let groups: [&[f64]; 3] = [&sa, &sb, &sc];
            let r = one_way_anova(&groups).unwrap();
            assert!(
                rel(r.statistic, 3.356_394_129_979_035_6, 1e-12),
                "s = {s:e}: F = {}",
                r.statistic
            );
            assert_eq!(r.df, 2.0);
            assert!(
                rel(r.p_value, 0.081_461_355_208_934_58, 1e-10),
                "s = {s:e}: p = {}",
                r.p_value
            );
            for (center, w, p) in lev_refs
            {
                let r = levene_test(&groups, center)
                    .unwrap_or_else(|| panic!("s = {s:e}, {center:?}: None"));
                assert!(
                    rel(r.statistic, w, 1e-12),
                    "s = {s:e}, {center:?}: W = {}",
                    r.statistic
                );
                assert!(
                    rel(r.p_value, p, 1e-10),
                    "s = {s:e}, {center:?}: p = {}",
                    r.p_value
                );
            }
            // Constant groups with different means: F = ∞ and p = 0 at every
            // scale (an underflowing between-group sum gave 0/0 = NaN).
            let (lo, hi) = (scale(&[1.0, 1.0]), scale(&[2.0, 2.0]));
            let r = one_way_anova(&[&lo, &hi]).unwrap();
            assert_eq!(r.statistic, f64::INFINITY, "s = {s:e}");
            assert_eq!(r.p_value, 0.0, "s = {s:e}");
        }
    }

    #[test]
    fn chi_square_gof_rejects_non_finite_counts() {
        let obs = [10.0, 12.0, 8.0];
        let exp = [10.0, 10.0, 10.0];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        {
            let mut e = exp;
            e[1] = bad;
            assert!(chi_square_gof(&obs, &e, 0).is_none(), "expected {bad}");
            let mut o = obs;
            o[1] = bad;
            assert!(chi_square_gof(&o, &exp, 0).is_none(), "observed {bad}");
        }
    }

    #[test]
    fn chi_square_gof_large_counts_do_not_overflow() {
        // Scaling observed and expected by s scales the statistic by s.
        let obs = [12.0, 8.0, 10.0];
        let exp = [10.0, 10.0, 10.0];
        let base = chi_square_gof(&obs, &exp, 0).unwrap().statistic;
        let s = 1e300;
        let o: Vec<f64> = obs.iter().map(|v| v * s).collect();
        let e: Vec<f64> = exp.iter().map(|v| v * s).collect();
        let r = chi_square_gof(&o, &e, 0).unwrap();
        assert!(r.statistic.is_finite());
        assert!((r.statistic / (base * s) - 1.0).abs() < 1e-12);
        assert!(r.p_value.is_finite() && r.p_value >= 0.0);
    }

    #[test]
    fn chi_square_gof_fair_die() {
        // A near-fair die: expect ~1/6 each of 60 rolls.
        let observed = [9.0, 11.0, 10.0, 8.0, 12.0, 10.0];
        let expected = [10.0; 6];
        let r = chi_square_gof(&observed, &expected, 0).unwrap();
        assert!(r.df == 5.0);
        assert!(r.p_value > 0.5); // consistent with fair
        // A loaded die → significant.
        let loaded = [30.0, 6.0, 6.0, 6.0, 6.0, 6.0];
        let r2 = chi_square_gof(&loaded, &expected, 0).unwrap();
        assert!(r2.p_value < 1e-6);
    }

    #[test]
    fn ks_accepts_matching_and_rejects_mismatched() {
        // Deterministic sample from N(0,1) should NOT be rejected against N(0,1).
        let mut rng = crate::rng::SplitMix64::new(2024);
        let n = Normal::standard();
        let sample: Vec<f64> = (0..2000).map(|_| n.sample(&mut rng)).collect();
        let r = ks_test_one_sample(&sample, &n).unwrap();
        assert!(r.p_value > 0.05, "same dist p = {}", r.p_value);
        // The same sample vs a shifted normal IS rejected.
        let shifted = Normal::new(1.0, 1.0);
        let r2 = ks_test_one_sample(&sample, &shifted).unwrap();
        assert!(r2.p_value < 1e-6, "shifted p = {}", r2.p_value);
    }

    #[test]
    fn ks_rejects_nan_samples_instead_of_ignoring_them() {
        // Regression: NaN samples were counted in n but skipped by the
        // supremum (`f64::max` drops NaN), so an all-NaN sample reported
        // D = 0, p = 1, and 20 NaNs plus 8 well-fitting values reported
        // p ≈ 9e-11 against the very distribution they came from.
        let n = Normal::standard();
        let clean = [-1.2, -0.4, 0.1, 0.5, 0.9, 1.3, -0.7, 0.2];
        assert!(ks_test_one_sample(&clean, &n).is_some());
        assert!(ks_test_one_sample(&[f64::NAN; 5], &n).is_none());
        let mut padded = vec![f64::NAN; 20];
        padded.extend_from_slice(&clean);
        assert!(ks_test_one_sample(&padded, &n).is_none());
        let mut one = clean.to_vec();
        one.insert(3, f64::NAN);
        assert!(ks_test_one_sample(&one, &n).is_none());
    }

    #[test]
    fn ks_rejects_nan_cdf_values() {
        struct NanCdf;
        impl Distribution for NanCdf {
            fn pdf(&self, _x: f64) -> f64 {
                f64::NAN
            }
            fn cdf(&self, x: f64) -> f64 {
                if x > 0.0 { f64::NAN } else { 0.5 }
            }
            fn quantile(&self, _p: f64) -> f64 {
                f64::NAN
            }
            fn mean(&self) -> f64 {
                f64::NAN
            }
            fn variance(&self) -> f64 {
                f64::NAN
            }
        }
        assert!(ks_test_one_sample(&[-1.0, 2.0], &NanCdf).is_none());
        assert!(ks_test_one_sample(&[-1.0, -2.0], &NanCdf).is_some());
    }

    #[test]
    fn ks_accepts_infinite_samples_and_order_is_irrelevant() {
        let n = Normal::standard();
        let a = [f64::NEG_INFINITY, -0.3, 0.4, 1.1, f64::INFINITY];
        let b = [1.1, f64::INFINITY, -0.3, f64::NEG_INFINITY, 0.4];
        let (ra, rb) = (
            ks_test_one_sample(&a, &n).unwrap(),
            ks_test_one_sample(&b, &n).unwrap(),
        );
        // The CDF is transcendental: Miri perturbs its last bits on each call,
        // so the two orderings are compared with a tolerance, not bit for bit.
        assert_eq!(ra.df, rb.df);
        assert!(close(ra.statistic, rb.statistic, 1e-12));
        assert!(close(ra.p_value, rb.p_value, 1e-12));
        assert!(ra.statistic.is_finite() && (0.0..=1.0).contains(&ra.p_value));
    }

    #[test]
    fn kolmogorov_q_is_stable_for_small_lambda() {
        let q = kolmogorov_q(0.001);
        assert_eq!(q, 1.0);
        // Reference values from the Kolmogorov limiting distribution.
        assert!(close(kolmogorov_q(0.5), 0.963_945_243_664_875, 1e-13));
        assert!(close(kolmogorov_q(1.0), 0.269_999_671_677_354_56, 1e-13));
    }

    #[test]
    fn discrete_gof_poisson_matches_scipy() {
        use crate::discrete::Poisson;
        // Binned counts 0,1,2,3,4,≥5 (N = 100); a Poisson(1.98) fit.
        let observed = [10u64, 28, 32, 18, 8, 4];
        let dist = Poisson::new(1.98);
        // One parameter estimated ⇒ ddof = 1; every expected ≥ 5 ⇒ no pooling.
        let r = chi2_gof_discrete(&observed, &dist, 1, 5.0).unwrap();
        assert!(
            (r.statistic - 2.279_187_103).abs() < 1e-6,
            "chi2 = {}",
            r.statistic
        );
        assert!((r.df - 4.0).abs() < 1e-12);
        assert!(
            (r.p_value - 0.684_560_899).abs() < 1e-6,
            "p = {}",
            r.p_value
        );
    }

    #[test]
    fn discrete_gof_rejects_bad_fit_and_pools() {
        use crate::discrete::{Geometric, Poisson};
        // Data clearly not Poisson(1.0): far too much mass in the tail.
        let observed = [5u64, 5, 5, 5, 80];
        let bad = Poisson::new(1.0);
        let r = chi2_gof_discrete(&observed, &bad, 1, 5.0).unwrap();
        assert!(r.p_value < 1e-6, "bad fit not rejected, p = {}", r.p_value);
        // Geometric support starts at 1: bin 0 has zero probability and must
        // pool forward rather than break the test (expected > 0 everywhere).
        let obs_g = [0u64, 50, 25, 13, 12];
        let g = Geometric::new(0.5);
        let rg = chi2_gof_discrete(&obs_g, &g, 1, 5.0);
        assert!(rg.is_some());
        // Degenerate: a single bin cannot yield a test.
        assert!(chi2_gof_discrete(&[10], &bad, 0, 5.0).is_none());
        assert!(chi2_gof_discrete(&[0, 0, 0], &bad, 0, 5.0).is_none());
    }

    #[test]
    fn variance_tests_reject_bad_input() {
        let g = [1.0, 2.0, 4.0];
        for c in [LeveneCenter::Mean, LeveneCenter::Median]
        {
            assert!(levene_test(&[&g], c).is_none());
            assert!(levene_test(&[&g, &[]], c).is_none());
            assert!(levene_test(&[&[1.0], &[2.0]], c).is_none());
            assert!(levene_test(&[&g, &[1.0, f64::INFINITY]], c).is_none());
            // Every deviation equals its group mean: F is 0/0.
            assert!(levene_test(&[&[1.0, 3.0], &[5.0, 9.0]], c).is_none());
        }
        assert!(bartlett_test(&[&g]).is_none());
        assert!(bartlett_test(&[&g, &[3.0]]).is_none());
        assert!(bartlett_test(&[&g, &[2.0, 2.0]]).is_none());
        assert!(bartlett_test(&[&g, &[f64::NAN, 1.0]]).is_none());
    }

    #[test]
    fn variance_tests_invariants() {
        let a = [4.1, 5.3, 3.8, 6.0, 5.1, 4.4];
        let b = [2.0, 9.5, 5.5, 1.2, 8.8, 6.1, 3.3];
        // Shifting a group changes neither its spread nor the statistics.
        let shifted: Vec<f64> = b.iter().map(|v| v + 100.0).collect();
        for c in [LeveneCenter::Mean, LeveneCenter::Median]
        {
            let r = levene_test(&[&a, &b], c).unwrap();
            let s = levene_test(&[&a, &shifted], c).unwrap();
            assert!(close(r.statistic, s.statistic, 1e-9));
            assert!(r.p_value < 0.05);
            // Groups with identical spread give W = 0 and p = 1.
            let same = levene_test(&[&a, &a], c).unwrap();
            assert!(same.statistic.abs() < 1e-12);
            assert!(close(same.p_value, 1.0, 1e-9));
        }
        let r = bartlett_test(&[&a, &b]).unwrap();
        let s = bartlett_test(&[&a, &shifted]).unwrap();
        assert!(close(r.statistic, s.statistic, 1e-9));
        assert_eq!(r.df, 1.0);
        let same = bartlett_test(&[&b, &b, &b]).unwrap();
        assert!(same.statistic.abs() < 1e-12);
        assert!(close(same.p_value, 1.0, 1e-9));
    }

    #[test]
    fn bartlett_does_not_depend_on_the_magnitude_of_the_data() {
        // Regression: the group variances were formed from raw squared
        // deviations, which overflow for a data spread above about 1e154
        // (`None`) and underflow below about 1e-154 (`None`, or lost digits
        // in the subnormal range). Bartlett's statistic is invariant under a
        // common positive rescaling. Reference values at unit scale: SciPy
        // 1.18 `bartlett`, cross-checked with mpmath at 50 digits.
        let a = [1.0, 2.0, 3.0, 4.5];
        let b = [2.0, 3.5, 5.0, 6.0, 4.0];
        let c = [0.5, 8.5, 2.5, -3.0];
        let rel = |x: f64, y: f64, tol: f64| (x - y).abs() <= tol * y.abs();
        // Powers of two keep the rescaled samples exact.
        for s in [
            2f64.powi(-1000),
            2f64.powi(-700),
            2f64.powi(-530),
            1.0,
            2f64.powi(530),
            2f64.powi(700),
            2f64.powi(1000),
        ]
        {
            let scale = |g: &[f64]| g.iter().map(|v| v * s).collect::<Vec<f64>>();
            let (sa, sb, sc) = (scale(&a), scale(&b), scale(&c));
            let r = bartlett_test(&[&sa, &sb, &sc]).unwrap_or_else(|| panic!("s = {s:e}: None"));
            assert!(
                rel(r.statistic, 5.541_126_102_273_377, 1e-12),
                "s = {s:e}: T = {}",
                r.statistic
            );
            assert_eq!(r.df, 2.0);
            assert!(
                rel(r.p_value, 0.062_626_732_760_052_54, 1e-10),
                "s = {s:e}: p = {}",
                r.p_value
            );
        }
        // Groups whose spreads differ by a factor 2^1200 (mpmath:
        // T = 7304.36476658898459974, p ≈ 7.5e-1587, which underflows).
        let s = 2f64.powi(600);
        let wide: Vec<f64> = a.iter().map(|v| v * s).collect();
        let narrow: Vec<f64> = c.iter().map(|v| v / s).collect();
        let r = bartlett_test(&[&wide, &b, &narrow]).expect("mixed spreads: None");
        assert!(
            rel(r.statistic, 7_304.364_766_588_985, 1e-12),
            "T = {}",
            r.statistic
        );
        assert_eq!(r.p_value, 0.0);
    }

    /// Reference values from SciPy 1.18.1 (`levene(center=...)`, `bartlett`),
    /// on the NIST gear-diameter style data used in SciPy's own tests.
    #[test]
    fn variance_tests_match_scipy_oracle() {
        let a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
        let b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
        let c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
        let groups: [&[f64]; 3] = [&a, &b, &c];

        let r = levene_test(&groups, LeveneCenter::Mean).unwrap();
        assert_eq!(r.df, 2.0);
        assert!(close(r.statistic, 7.905194483442054, 1e-10));
        assert!(close(r.p_value, 0.001983795817472729, 1e-9));

        let r = levene_test(&groups, LeveneCenter::Median).unwrap();
        assert!(close(r.statistic, 7.584952754501659, 1e-10));
        assert!(close(r.p_value, 0.002431505967249677, 1e-9));

        let r = bartlett_test(&groups).unwrap();
        assert_eq!(r.df, 2.0);
        assert!(close(r.statistic, 22.789434813726768, 1e-10));
        assert!(close(r.p_value, 1.1254782518834628e-05, 1e-9));
    }

    /// Relative closeness for statistics that should agree up to rounding.
    fn rel_close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1e-300)
    }

    // Finite samples near `f64::MAX`: deviations from the mean used to
    // overflow to ±∞, so the statistics below were 0, NaN, or None.
    // Each must match the same data at unit scale (scale invariance).
    const M: f64 = f64::MAX;

    #[test]
    fn t_test_one_sample_near_f64_max() {
        let huge = t_test_one_sample(&[-M, M, M], 0.0, Tail::TwoSided).unwrap();
        let unit = t_test_one_sample(&[-1.0, 1.0, 1.0], 0.0, Tail::TwoSided).unwrap();
        assert!(rel_close(huge.statistic, unit.statistic), "{huge:?}");
        assert!(rel_close(huge.p_value, unit.p_value), "{huge:?}");
        // A huge `mu0` against moderate data: `x̄ − mu0` must not overflow.
        let r = t_test_one_sample(&[0.5 * M, 0.75 * M, M], -M, Tail::Greater).unwrap();
        let u = t_test_one_sample(&[0.5, 0.75, 1.0], -1.0, Tail::Greater).unwrap();
        assert!(rel_close(r.statistic, u.statistic), "{r:?} vs {u:?}");
    }

    #[test]
    fn t_test_two_sample_near_f64_max() {
        let a = [-M, M, M];
        let b = [M, 0.5 * M, 0.75 * M];
        let (ua, ub) = ([-1.0, 1.0, 1.0], [1.0, 0.5, 0.75]);
        for equal_var in [true, false]
        {
            let huge = t_test_two_sample(&a, &b, equal_var, Tail::TwoSided).unwrap();
            let unit = t_test_two_sample(&ua, &ub, equal_var, Tail::TwoSided).unwrap();
            assert!(rel_close(huge.statistic, unit.statistic), "{huge:?}");
            assert!(rel_close(huge.df, unit.df), "{huge:?}");
            assert!(rel_close(huge.p_value, unit.p_value), "{huge:?}");
        }
    }

    #[test]
    fn one_way_anova_near_f64_max() {
        // The plain sum of all values overflows, so the grand mean was ∞.
        let huge = one_way_anova(&[&[1.0e308, 1.5e308], &[1.6e308, 1.7e308]]).unwrap();
        let unit = one_way_anova(&[&[1.0, 1.5], &[1.6, 1.7]]).unwrap();
        assert!(rel_close(huge.statistic, unit.statistic), "{huge:?}");
        assert!(rel_close(huge.p_value, unit.p_value), "{huge:?}");
    }

    #[test]
    fn levene_and_bartlett_near_f64_max() {
        let (a, b) = ([-M, M, M, 0.5 * M], [0.1 * M, 0.2 * M, 0.4 * M]);
        let (ua, ub) = ([-1.0, 1.0, 1.0, 0.5], [0.1, 0.2, 0.4]);
        for center in [LeveneCenter::Mean, LeveneCenter::Median]
        {
            let huge = levene_test(&[&a, &b], center).unwrap();
            let unit = levene_test(&[&ua, &ub], center).unwrap();
            assert!(rel_close(huge.statistic, unit.statistic), "{huge:?}");
        }
        let huge = bartlett_test(&[&a, &b]).unwrap();
        let unit = bartlett_test(&[&ua, &ub]).unwrap();
        assert!(rel_close(huge.statistic, unit.statistic), "{huge:?}");
    }

    #[test]
    fn moderate_data_is_not_rescaled() {
        // Below `2^1022` the inputs are used unchanged, bit for bit.
        assert!(quartered_if_huge(&[&[1.0e307, -1.0e307]], false).is_none());
        assert!(quartered_if_huge(&[&[f64::INFINITY, 1.0]], false).is_none());
        assert!(quartered_if_huge(&[&[HUGE]], false).is_some());
    }
}
