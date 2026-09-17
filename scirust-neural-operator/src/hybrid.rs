//! Deterministic Boolean control plane for hybrid exact/surrogate execution.
//!
//! A router does not claim that Boolean control is preferable. It provides an
//! inspectable mechanism for choosing an execution action from declared Boolean
//! predicates, with ordered first-match semantics and explicit fallback.

use crate::boolean::{AnfPolynomial, BooleanComplexity};
use crate::error::{NeuralOperatorError, Result};

/// Execution action selected by a hybrid operator controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridAction {
    /// Use the learned surrogate without an immediate exact solve.
    UseSurrogate,
    /// Bypass the surrogate and run the authoritative solver.
    UseExact,
    /// Run or accept the surrogate only with an exact/reference verification.
    VerifySurrogate,
    /// Decline to produce a result at this control boundary.
    Abstain,
}

/// One ordered Boolean routing rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanRouteRule {
    guard: AnfPolynomial,
    action: HybridAction,
}

impl BooleanRouteRule {
    /// Construct a rule from an exact ANF guard and its action.
    pub const fn new(guard: AnfPolynomial, action: HybridAction) -> Self {
        Self { guard, action }
    }

    /// Guard expression used by this rule.
    pub const fn guard(&self) -> &AnfPolynomial {
        &self.guard
    }

    /// Action returned when this rule is the first matching rule.
    pub const fn action(&self) -> HybridAction {
        self.action
    }
}

/// Aggregate complexity of an ordered Boolean router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterComplexity {
    /// Number of input predicates visible to every rule.
    pub input_arity: usize,
    /// Number of ordered rules.
    pub rules: usize,
    /// Total number of ANF monomials over all rules.
    pub monomials: usize,
    /// Total literal occurrences over all rules.
    pub literal_occurrences: usize,
    /// Largest ANF degree among the guards.
    pub max_degree: usize,
}

/// Ordered, fail-closed Boolean controller for hybrid operator execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanRouter {
    input_arity: usize,
    rules: Vec<BooleanRouteRule>,
    fallback: HybridAction,
}

impl BooleanRouter {
    /// Create a router whose rules are evaluated in declaration order.
    pub fn new(
        input_arity: usize,
        rules: Vec<BooleanRouteRule>,
        fallback: HybridAction,
    ) -> Result<Self> {
        if rules.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "Boolean routing rules",
            });
        }
        if let Some(rule) = rules
            .iter()
            .find(|rule| rule.guard().variables() != input_arity)
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "Boolean router guard arity",
                expected: input_arity,
                got: rule.guard().variables(),
            });
        }
        Ok(Self {
            input_arity,
            rules,
            fallback,
        })
    }

    /// Select an action by ordered first-match evaluation.
    pub fn route(&self, predicates: &[bool]) -> Result<HybridAction> {
        if predicates.len() != self.input_arity
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "Boolean router predicates",
                expected: self.input_arity,
                got: predicates.len(),
            });
        }
        for rule in &self.rules
        {
            if rule.guard().evaluate(predicates)?
            {
                return Ok(rule.action());
            }
        }
        Ok(self.fallback)
    }

    /// Number of Boolean predicates exposed to the router.
    pub const fn input_arity(&self) -> usize {
        self.input_arity
    }

    /// Ordered rule set.
    pub fn rules(&self) -> &[BooleanRouteRule] {
        &self.rules
    }

    /// Fallback action when no guard matches.
    pub const fn fallback(&self) -> HybridAction {
        self.fallback
    }

    /// Deterministic structural complexity of the complete route set.
    pub fn complexity(&self) -> RouterComplexity {
        let guards = self
            .rules
            .iter()
            .map(|rule| rule.guard().complexity())
            .collect::<Vec<BooleanComplexity>>();
        RouterComplexity {
            input_arity: self.input_arity,
            rules: self.rules.len(),
            monomials: guards.iter().map(|item| item.monomials).sum(),
            literal_occurrences: guards.iter().map(|item| item.literal_occurrences).sum(),
            max_degree: guards.iter().map(|item| item.max_degree).max().unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_router_can_guard_surrogate_with_exact_fallback() {
        // p0 = in-domain, p1 = residual-small, p2 = verification-required.
        let use_surrogate = AnfPolynomial::from_monomials(3, &[0b011]).unwrap();
        let verify = AnfPolynomial::from_monomials(3, &[0b100]).unwrap();
        let router = BooleanRouter::new(
            3,
            vec![
                BooleanRouteRule::new(verify, HybridAction::VerifySurrogate),
                BooleanRouteRule::new(use_surrogate, HybridAction::UseSurrogate),
            ],
            HybridAction::UseExact,
        )
        .unwrap();

        assert_eq!(
            router.route(&[true, true, false]).unwrap(),
            HybridAction::UseSurrogate
        );
        assert_eq!(
            router.route(&[true, true, true]).unwrap(),
            HybridAction::VerifySurrogate
        );
        assert_eq!(
            router.route(&[true, false, false]).unwrap(),
            HybridAction::UseExact
        );
    }

    #[test]
    fn router_complexity_is_explicit() {
        let first = AnfPolynomial::from_monomials(2, &[0b01, 0b11]).unwrap();
        let second = AnfPolynomial::from_monomials(2, &[0b10]).unwrap();
        let router = BooleanRouter::new(
            2,
            vec![
                BooleanRouteRule::new(first, HybridAction::UseSurrogate),
                BooleanRouteRule::new(second, HybridAction::UseExact),
            ],
            HybridAction::Abstain,
        )
        .unwrap();
        assert_eq!(
            router.complexity(),
            RouterComplexity {
                input_arity: 2,
                rules: 2,
                monomials: 3,
                literal_occurrences: 4,
                max_degree: 2,
            }
        );
    }
}
