//! Boolean early-dispatch for Fourier modes.
//!
//! The router produces a canonical active-mode plan before the FNO spectral
//! channel-mixing stage. The corresponding low-level SciRust FNO path builds
//! DFT rows, trainable weight batches and inverse-DFT columns only for admitted
//! modes, rather than computing every configured mode and masking afterward.

use crate::{AnfPolynomial, NeuralOperatorError, Result};

/// Validated canonical subset of configured Fourier modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectralModePlan {
    total_modes: usize,
    active_modes: Vec<usize>,
}

impl SpectralModePlan {
    /// Validate a non-empty, strictly increasing subset of `0..total_modes`.
    pub fn new(total_modes: usize, active_modes: Vec<usize>) -> Result<Self> {
        if total_modes == 0
        {
            return Err(NeuralOperatorError::Empty {
                what: "configured spectral modes",
            });
        }
        if active_modes.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "active spectral modes",
            });
        }
        if let Some(&mode) = active_modes.iter().find(|&&mode| mode >= total_modes)
        {
            return Err(NeuralOperatorError::InvalidSpectralMode {
                mode,
                configured_modes: total_modes,
            });
        }
        if active_modes.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(NeuralOperatorError::NonCanonicalSpectralModes);
        }
        Ok(Self {
            total_modes,
            active_modes,
        })
    }

    /// Total number of modes available to the dense spectral branch.
    pub const fn total_modes(&self) -> usize {
        self.total_modes
    }

    /// Canonical active mode indices.
    pub fn active_modes(&self) -> &[usize] {
        &self.active_modes
    }

    /// Number of admitted modes.
    pub fn active_count(&self) -> usize {
        self.active_modes.len()
    }

    /// Number of rejected modes.
    pub fn skipped_count(&self) -> usize {
        self.total_modes - self.active_modes.len()
    }

    /// Fraction of configured spectral modes retained by the plan.
    pub fn retained_fraction(&self) -> f64 {
        self.active_modes.len() as f64 / self.total_modes as f64
    }
}

/// Structural complexity of a Boolean mode router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectralRouterComplexity {
    /// Number of Boolean predicates visible to every mode guard.
    pub input_arity: usize,
    /// Number of configured per-mode guards.
    pub mode_guards: usize,
    /// Total ANF monomials across all mode guards.
    pub monomials: usize,
    /// Total literal occurrences across all mode guards.
    pub literal_occurrences: usize,
    /// Largest degree among all guards.
    pub max_degree: usize,
}

/// One ANF guard per Fourier mode, evaluated before spectral mode mixing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanSpectralRouter {
    input_arity: usize,
    guards: Vec<AnfPolynomial>,
    fallback_mode: usize,
}

impl BooleanSpectralRouter {
    /// Construct a deterministic per-mode Boolean router.
    ///
    /// If every guard rejects, `fallback_mode` is retained so the spectral
    /// branch never receives an empty batch. The fallback is part of the router
    /// identity and is therefore explicit rather than silently choosing DC.
    pub fn new(
        input_arity: usize,
        guards: Vec<AnfPolynomial>,
        fallback_mode: usize,
    ) -> Result<Self> {
        if guards.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "spectral Boolean guards",
            });
        }
        if fallback_mode >= guards.len()
        {
            return Err(NeuralOperatorError::InvalidSpectralMode {
                mode: fallback_mode,
                configured_modes: guards.len(),
            });
        }
        if let Some(guard) = guards.iter().find(|guard| guard.variables() != input_arity)
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "spectral Boolean guard arity",
                expected: input_arity,
                got: guard.variables(),
            });
        }
        Ok(Self {
            input_arity,
            guards,
            fallback_mode,
        })
    }

    /// Evaluate all mode guards and return a canonical early-dispatch plan.
    pub fn route(&self, predicates: &[bool]) -> Result<SpectralModePlan> {
        if predicates.len() != self.input_arity
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "spectral router predicates",
                expected: self.input_arity,
                got: predicates.len(),
            });
        }
        let mut active = Vec::new();
        for (mode, guard) in self.guards.iter().enumerate()
        {
            if guard.evaluate(predicates)?
            {
                active.push(mode);
            }
        }
        if active.is_empty()
        {
            active.push(self.fallback_mode);
        }
        SpectralModePlan::new(self.guards.len(), active)
    }

    /// Number of available Fourier modes controlled by this router.
    pub fn modes(&self) -> usize {
        self.guards.len()
    }

    /// Explicit mode retained if all guards reject.
    pub const fn fallback_mode(&self) -> usize {
        self.fallback_mode
    }

    /// Deterministic structural complexity of the complete mode-control plane.
    pub fn complexity(&self) -> SpectralRouterComplexity {
        let complexities = self
            .guards
            .iter()
            .map(AnfPolynomial::complexity)
            .collect::<Vec<_>>();
        SpectralRouterComplexity {
            input_arity: self.input_arity,
            mode_guards: self.guards.len(),
            monomials: complexities.iter().map(|item| item.monomials).sum(),
            literal_occurrences: complexities
                .iter()
                .map(|item| item.literal_occurrences)
                .sum(),
            max_degree: complexities
                .iter()
                .map(|item| item.max_degree)
                .max()
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_router_builds_canonical_early_dispatch_plan() {
        // mode 0: p0; mode 1: p1; mode 2: p0*p1; mode 3: p0 XOR p1
        let guards = vec![
            AnfPolynomial::from_monomials(2, &[0b01]).unwrap(),
            AnfPolynomial::from_monomials(2, &[0b10]).unwrap(),
            AnfPolynomial::from_monomials(2, &[0b11]).unwrap(),
            AnfPolynomial::from_monomials(2, &[0b01, 0b10]).unwrap(),
        ];
        let router = BooleanSpectralRouter::new(2, guards, 0).unwrap();
        let plan = router.route(&[true, false]).unwrap();
        assert_eq!(plan.active_modes(), &[0, 3]);
        assert_eq!(plan.skipped_count(), 2);
        assert_eq!(plan.retained_fraction(), 0.5);
    }

    #[test]
    fn all_rejected_uses_declared_fallback_mode() {
        let guards = vec![
            AnfPolynomial::from_monomials(1, &[0b1]).unwrap(),
            AnfPolynomial::from_monomials(1, &[0b1]).unwrap(),
        ];
        let router = BooleanSpectralRouter::new(1, guards, 1).unwrap();
        assert_eq!(router.route(&[false]).unwrap().active_modes(), &[1]);
    }

    #[test]
    fn mode_plan_rejects_noncanonical_indices() {
        assert_eq!(
            SpectralModePlan::new(4, vec![2, 1]).unwrap_err(),
            NeuralOperatorError::NonCanonicalSpectralModes
        );
        assert_eq!(
            SpectralModePlan::new(4, vec![1, 1]).unwrap_err(),
            NeuralOperatorError::NonCanonicalSpectralModes
        );
    }
}
