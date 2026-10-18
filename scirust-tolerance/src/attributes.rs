//! Acceptance sampling **by attributes** — ISO 2859-1 / MIL-STD-105 single
//! sampling.
//!
//! The oldest and simplest lot-sentencing rule: draw `n` items, count the
//! defectives `d`, and **accept** the lot when `d ≤ c` (the acceptance number).
//! No measurement is needed — only a go/no-go verdict per item — so an attributes
//! plan applies where [`crate::variables`] (which needs a measured mean and a
//! normal assumption) cannot, at the cost of a larger sample for the same
//! protection.
//!
//! ## Operating characteristic
//!
//! For a large lot the defective count is binomial, so the probability of
//! acceptance at fraction defective `p` is the exact binomial tail
//!
//! ```text
//! Pa(p) = P(D ≤ c) = Σ_{d=0}^{c} C(n, d) pᵈ (1−p)ⁿ⁻ᵈ .
//! ```
//!
//! [`design_attributes_plan`] finds the smallest `(n, c)` whose OC clears the
//! producer point `(AQL, 1−α)` and the consumer point `(RQL, β)` by scanning the
//! acceptance number upward — the classical two-point construction.
//!
//! The **average outgoing quality** [`AttributesPlan::average_outgoing_quality`]
//! (rejected lots screened 100 %) peaks at the AOQL, the worst average quality a
//! rectifying scheme can let through.

use serde::{Deserialize, Serialize};

/// A single-sampling attributes plan: draw `sample_size`, accept when the
/// defective count is at most `acceptance_number`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributesPlan {
    /// Number of items to inspect, `n`.
    pub sample_size: usize,
    /// Acceptance number `c` — accept iff `d ≤ c`.
    pub acceptance_number: usize,
}

impl AttributesPlan {
    /// Build a plan from an explicit `(n, c)`.
    pub fn new(sample_size: usize, acceptance_number: usize) -> Self {
        AttributesPlan {
            sample_size,
            acceptance_number,
        }
    }

    /// Probability of accepting a lot of fraction defective `p`, the exact
    /// binomial tail `P(D ≤ c)`. Returns 1 for `p ≤ 0` and, for `p ≥ 1`, 1 only
    /// if `c ≥ n` (every item defective still passes an all-accepting plan).
    ///
    /// Evaluated as `I_{1−p}(n−c, c+1)` (regularised incomplete beta), so it
    /// stays accurate for large `n`, where `(1−p)ⁿ` alone underflows to `0`.
    pub fn probability_of_acceptance(&self, p: f64) -> f64 {
        // `binomial_cdf` handles the endpoints (`p ≤ 0` ⇒ 1; `p ≥ 1` ⇒ 1 iff
        // `c ≥ n`) and stays accurate where `(1−p)ⁿ` underflows.
        crate::special::binomial_cdf(self.sample_size, self.acceptance_number, p)
    }

    /// Average outgoing quality under rectifying inspection (rejected lots fully
    /// screened and cleaned), `AOQ(p) ≈ p·Pa(p)·(N−n)/N`. With `lot_size = None`
    /// the large-lot limit `p·Pa(p)` is used.
    pub fn average_outgoing_quality(&self, p: f64, lot_size: Option<usize>) -> f64 {
        let base = p * self.probability_of_acceptance(p);
        match lot_size
        {
            Some(nn) if nn > self.sample_size => base * (nn - self.sample_size) as f64 / nn as f64,
            _ => base,
        }
    }

    /// The operating-characteristic curve as `points` pairs `(p, Pa(p))` over
    /// `[0, max_p]`.
    pub fn oc_curve(&self, max_p: f64, points: usize) -> Vec<(f64, f64)> {
        if points == 0
        {
            return Vec::new();
        }
        (0..points)
            .map(|i| {
                let p = max_p * i as f64 / (points - 1).max(1) as f64;
                (p, self.probability_of_acceptance(p))
            })
            .collect()
    }
}

/// Design a single-sampling attributes plan whose OC clears the producer point
/// `(aql, 1−alpha)` and the consumer point `(rql, beta)`: fractions defective
/// `0 < aql < rql < 1`, risks in `(0, 1)`. Scans the acceptance number `c` upward
/// and, for each, takes the smallest `n` with `Pa(aql) ≥ 1−α`; the first `c` that
/// also delivers `Pa(rql) ≤ β` wins (the minimal-`c` plan). Returns `None` if no
/// plan up to `max_n` satisfies both points.
pub fn design_attributes_plan(
    aql: f64,
    rql: f64,
    alpha: f64,
    beta: f64,
    max_n: usize,
) -> Option<AttributesPlan> {
    if !(aql > 0.0 && aql < rql && rql < 1.0)
        || !(0.0..1.0).contains(&alpha)
        || !(0.0..1.0).contains(&beta)
        || alpha <= 0.0
        || beta <= 0.0
    {
        return None;
    }
    let want_good = 1.0 - alpha;
    for c in 0..max_n
    {
        // Smallest n (≥ c+1) meeting the producer point; Pa(aql) decreases in n.
        let mut plan = AttributesPlan::new(c + 1, c);
        let mut found = None;
        for n in (c + 1)..=max_n
        {
            plan.sample_size = n;
            if plan.probability_of_acceptance(aql) >= want_good
            {
                found = Some(n);
            }
            else
            {
                break; // once below, larger n only makes it worse
            }
        }
        if let Some(n) = found
        {
            let candidate = AttributesPlan::new(n, c);
            if candidate.probability_of_acceptance(rql) <= beta
            {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn binomial_oc_matches_hand_value() {
        // n=5, c=1, p=0.2: Pa = (0.8)^5 + 5·0.2·0.8^4.
        let plan = AttributesPlan::new(5, 1);
        let want = 0.8_f64.powi(5) + 5.0 * 0.2 * 0.8_f64.powi(4);
        assert_relative_eq!(plan.probability_of_acceptance(0.2), want, epsilon = 1e-12);
        // Endpoints.
        assert_relative_eq!(plan.probability_of_acceptance(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(plan.probability_of_acceptance(1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn accept_all_plan_always_accepts() {
        // c ≥ n ⇒ every lot passes, even all-defective.
        let plan = AttributesPlan::new(4, 4);
        assert_relative_eq!(plan.probability_of_acceptance(1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(plan.probability_of_acceptance(0.5), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn oc_is_monotone_decreasing() {
        let plan = AttributesPlan::new(50, 2);
        let curve = plan.oc_curve(0.2, 40);
        for w in curve.windows(2)
        {
            assert!(w[1].1 <= w[0].1 + 1e-12);
        }
    }

    #[test]
    fn design_meets_both_points() {
        let aql = 0.01;
        let rql = 0.10;
        let (alpha, beta) = (0.05, 0.10);
        let plan = design_attributes_plan(aql, rql, alpha, beta, 400).unwrap();
        assert!(plan.probability_of_acceptance(aql) >= 1.0 - alpha - 1e-9);
        assert!(plan.probability_of_acceptance(rql) <= beta + 1e-9);
        // Larger acceptance number needs a larger sample: c ≥ 0, n ≥ c+1.
        assert!(plan.sample_size > plan.acceptance_number);
    }

    #[test]
    fn aoq_peaks_between_zero_and_one() {
        let plan = AttributesPlan::new(50, 1);
        // AOQ is 0 at p=0, small near p=1 (Pa→0), and positive in between.
        assert_relative_eq!(
            plan.average_outgoing_quality(0.0, None),
            0.0,
            epsilon = 1e-12
        );
        let mid = plan.average_outgoing_quality(0.03, None);
        assert!(mid > plan.average_outgoing_quality(0.30, None));
        assert!(mid > 0.0);
    }

    #[test]
    fn oc_large_sample_matches_binomial_reference() {
        // `(1−p)ⁿ` alone underflows to 0 once n·ln(1−p) < −745 (0.46⁹⁶⁰,
        // 0.5²⁰⁰⁰, 0.95¹⁵⁰⁰⁰), which used to zero the whole OC. References:
        // exact binomial sums in mpmath at 50 digits, rounded to f64.
        let cases = [
            (960, 505, 0.54, 0.201_687_571_580_873_83),
            (960, 505, 0.50, 0.950_145_110_895_520_7),
            (2000, 1000, 0.50, 0.508_919_505_572_927_2),
            (15_000, 750, 0.05, 0.509_713_080_949_426_8),
            (50, 2, 0.04, 0.676_714_004_096_593_5),
        ];
        for (n, c, p, want) in cases
        {
            let got = AttributesPlan::new(n, c).probability_of_acceptance(p);
            assert_relative_eq!(got, want, max_relative = 1e-10);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "scans ~10⁵ candidate plans; too slow under Miri")]
    fn design_honours_consumer_risk_for_large_plans() {
        // AQL 0.50 / RQL 0.54 needs n ≈ 1000+. The underflowing OC reported
        // Pa(0.54) = 0 for (n, c) = (960, 505), so that plan was returned even
        // though its true consumer risk is 0.2017 > β = 0.10.
        let (aql, rql, alpha, beta) = (0.50, 0.54, 0.05, 0.10);
        let plan = design_attributes_plan(aql, rql, alpha, beta, 4000).unwrap();
        assert!(plan.probability_of_acceptance(aql) >= 1.0 - alpha);
        assert!(plan.probability_of_acceptance(rql) <= beta);
        // Same minimal-c plan as the identical search over scipy.stats.binom.cdf.
        assert_eq!((plan.sample_size, plan.acceptance_number), (1350, 705));
    }

    #[test]
    fn rejects_bad_design_inputs() {
        assert!(design_attributes_plan(0.10, 0.01, 0.05, 0.10, 200).is_none()); // aql≥rql
        assert!(design_attributes_plan(0.01, 0.10, 0.0, 0.10, 200).is_none());
    }
}
