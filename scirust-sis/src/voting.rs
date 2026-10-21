//! `MooN` voting architectures ("M out of N channels must vote trip") — the
//! logic layer that decides *whether the group trips* given per-channel
//! votes, on top of the quantitative `PFDavg`/`PFH` primitives already in
//! `scirust-reliability`.

use crate::error::{SisError, SisResult};
use scirust_reliability::{pfd_1oo1, pfd_1oo2, pfd_1oo3, pfd_2oo2, pfd_2oo3, pfd_moon};
use serde::{Deserialize, Serialize};

/// An `M`-out-of-`N` voting architecture: at least `m` of `n` channels must
/// vote "trip" for the group to trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Architecture {
    pub m: u8,
    pub n: u8,
}

impl Architecture {
    pub const OO1: Architecture = Architecture { m: 1, n: 1 };
    pub const OO2: Architecture = Architecture { m: 1, n: 2 };
    pub const TWO_OO2: Architecture = Architecture { m: 2, n: 2 };
    pub const TWO_OO3: Architecture = Architecture { m: 2, n: 3 };
    pub const OO3: Architecture = Architecture { m: 1, n: 3 };

    pub fn new(m: u8, n: u8) -> SisResult<Self> {
        if m == 0 || m > n
        {
            return Err(SisError::InvalidArchitecture { m, n });
        }
        Ok(Self { m, n })
    }

    pub fn label(&self) -> String {
        format!("{}oo{}", self.m, self.n)
    }

    /// `PFDavg` (low-demand mode) of this architecture. The five
    /// architectures IEC 61508-6 Annex B tabulates directly (1oo1, 1oo2,
    /// 2oo2, 2oo3, 1oo3) dispatch to those exact, literally-standard
    /// closed forms; any other valid `M`-out-of-`N` falls back to
    /// [`scirust_reliability::pfd_moon`]'s cross-validated generalization
    /// (see that function's documentation for its provenance). `beta`
    /// (common-cause fraction) is ignored by architectures that don't use
    /// it (1oo1, 2oo2, and generally any `M = N`).
    ///
    /// # Errors
    ///
    /// [`SisError::InvalidInput`] unless `lambda_du` and `t1` are finite and
    /// `≥ 0` and `beta` is a fraction in `[0, 1]` (checked for every
    /// architecture, including those that ignore `beta`). Out-of-domain
    /// inputs are rejected rather than evaluated because the closed forms can
    /// then return a negative "PFDavg" (e.g. a negative `lambda_du`, or
    /// `beta < 0` on 1oo2), which [`scirust_reliability::sil_from_pfd`]
    /// would otherwise have banded as SIL 4. Invalid shapes yield
    /// [`SisError::UnsupportedArchitecture`].
    pub fn pfd_avg(&self, lambda_du: f64, t1: f64, beta: f64) -> SisResult<f64> {
        validate_pfd_inputs(lambda_du, t1, beta)?;
        match (self.m, self.n)
        {
            (1, 1) => Ok(pfd_1oo1(lambda_du, t1)),
            (1, 2) => Ok(pfd_1oo2(lambda_du, t1, beta)),
            (2, 2) => Ok(pfd_2oo2(lambda_du, t1)),
            (2, 3) => Ok(pfd_2oo3(lambda_du, t1, beta)),
            (1, 3) => Ok(pfd_1oo3(lambda_du, t1, beta)),
            (m, n) => pfd_moon(m as u32, n as u32, lambda_du, t1, beta)
                .map_err(|reason| SisError::UnsupportedArchitecture { m, n, reason }),
        }
    }

    /// Whether the group trips given one vote per channel (`true` = "this
    /// channel demands trip"). `votes.len()` must equal `n`.
    pub fn evaluate_votes(&self, votes: &[bool]) -> SisResult<bool> {
        if votes.len() != self.n as usize
        {
            return Err(SisError::VoteCountMismatch {
                expected: self.n,
                got: votes.len(),
            });
        }
        let trip_votes = votes.iter().filter(|&&v| v).count() as u8;
        Ok(trip_votes >= self.m)
    }
}

/// Domain check shared by every `PFDavg` evaluation: a failure rate and a
/// proof-test interval are non-negative finite quantities and the
/// common-cause factor is a fraction. NaN fails every check.
fn validate_pfd_inputs(lambda_du: f64, t1: f64, beta: f64) -> SisResult<()> {
    if !lambda_du.is_finite() || lambda_du < 0.0
    {
        return Err(SisError::InvalidInput(format!(
            "lambda_du must be finite and >= 0, got {lambda_du}"
        )));
    }
    if !t1.is_finite() || t1 < 0.0
    {
        return Err(SisError::InvalidInput(format!(
            "t1 must be finite and >= 0, got {t1}"
        )));
    }
    if !(0.0..=1.0).contains(&beta)
    {
        return Err(SisError::InvalidInput(format!(
            "beta must be within [0, 1], got {beta}"
        )));
    }
    Ok(())
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rejects_invalid_architecture() {
        assert!(Architecture::new(0, 2).is_err());
        assert!(Architecture::new(3, 2).is_err());
        assert!(Architecture::new(1, 1).is_ok());
    }

    #[test]
    fn pfd_avg_dispatches_to_matching_formula() {
        let a = Architecture::OO2; // 1oo2
        let pfd = a.pfd_avg(1e-3, 1000.0, 0.1).unwrap();
        assert_relative_eq!(pfd, scirust_reliability::pfd_1oo2(1e-3, 1000.0, 0.1));
    }

    #[test]
    fn pfd_avg_falls_back_to_general_moon_formula_beyond_the_five_named_cases() {
        // 2oo4 isn't one of the five IEC-tabulated architectures but is a
        // valid MooN shape, so `pfd_avg` computes it via `pfd_moon` instead
        // of refusing it outright.
        let a = Architecture::new(2, 4).unwrap();
        let pfd = a.pfd_avg(1e-3, 1000.0, 0.1).unwrap();
        assert_relative_eq!(
            pfd,
            scirust_reliability::pfd_moon(2, 4, 1e-3, 1000.0, 0.1).unwrap()
        );
    }

    #[test]
    fn pfd_avg_rejects_out_of_domain_inputs_instead_of_returning_a_negative_pfd() {
        // Before the domain check these evaluated to negative "PFDavg"
        // values (e.g. 1oo1 with lambda_du = -1e-6 gave -4.38e-3, and 1oo2
        // with beta = -0.1 gave about -4.07e-4), which band as SIL 4.
        let cases: &[(Architecture, f64, f64, f64)] = &[
            (Architecture::OO1, -1e-6, 8760.0, 0.0),
            (Architecture::OO2, 1e-6, 8760.0, -0.1),
            (Architecture::TWO_OO3, 1e-6, -8760.0, 0.1),
            (Architecture::OO3, 1e-6, 8760.0, 1.5),
            (Architecture::OO1, f64::NAN, 8760.0, 0.0),
            (Architecture::OO1, 1e-6, f64::INFINITY, 0.0),
            (Architecture::TWO_OO2, 1e-6, 8760.0, f64::NAN),
            (Architecture { m: 2, n: 4 }, 1e-6, 8760.0, -0.5),
        ];
        for &(a, lambda_du, t1, beta) in cases
        {
            let r = a.pfd_avg(lambda_du, t1, beta);
            assert!(
                matches!(r, Err(SisError::InvalidInput(_))),
                "{a} with lambda_du={lambda_du}, t1={t1}, beta={beta} gave {r:?}"
            );
        }
    }

    #[test]
    fn pfd_avg_accepts_the_closed_domain_boundaries() {
        assert_eq!(Architecture::OO2.pfd_avg(0.0, 8760.0, 0.1).unwrap(), 0.0);
        assert_eq!(Architecture::OO2.pfd_avg(1e-6, 0.0, 0.1).unwrap(), 0.0);
        assert_relative_eq!(
            Architecture::OO2.pfd_avg(1e-6, 8760.0, 1.0).unwrap(),
            1e-6 * 8760.0 / 2.0
        );
        assert_relative_eq!(
            Architecture::OO2.pfd_avg(1e-6, 8760.0, 0.0).unwrap(),
            (1e-6f64 * 8760.0).powi(2) / 3.0
        );
    }

    #[test]
    fn pfd_avg_rejects_a_directly_constructed_invalid_shape() {
        let invalid = Architecture { m: 5, n: 2 };
        assert!(invalid.pfd_avg(1e-3, 1000.0, 0.1).is_err());
    }

    #[test]
    fn oo2_trips_on_either_channel() {
        let a = Architecture::OO2;
        assert!(a.evaluate_votes(&[true, false]).unwrap());
        assert!(a.evaluate_votes(&[false, true]).unwrap());
        assert!(a.evaluate_votes(&[true, true]).unwrap());
        assert!(!a.evaluate_votes(&[false, false]).unwrap());
    }

    #[test]
    fn two_oo2_needs_both_channels() {
        let a = Architecture::TWO_OO2;
        assert!(a.evaluate_votes(&[true, true]).unwrap());
        assert!(!a.evaluate_votes(&[true, false]).unwrap());
        assert!(!a.evaluate_votes(&[false, false]).unwrap());
    }

    #[test]
    fn two_oo3_needs_majority() {
        let a = Architecture::TWO_OO3;
        assert!(a.evaluate_votes(&[true, true, false]).unwrap());
        assert!(a.evaluate_votes(&[true, true, true]).unwrap());
        assert!(!a.evaluate_votes(&[true, false, false]).unwrap());
        assert!(!a.evaluate_votes(&[false, false, false]).unwrap());
    }

    #[test]
    fn evaluate_votes_rejects_wrong_channel_count() {
        let a = Architecture::TWO_OO3;
        assert!(a.evaluate_votes(&[true, false]).is_err());
    }

    #[test]
    fn label_matches_moon_notation() {
        assert_eq!(Architecture::OO1.label(), "1oo1");
        assert_eq!(Architecture::TWO_OO3.label(), "2oo3");
    }
}
