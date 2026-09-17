//! Deterministic Boolean control plane for hybrid exact/surrogate execution.
//!
//! A router does not claim that Boolean control is preferable. It provides an
//! inspectable mechanism for choosing an execution action from declared Boolean
//! predicates, with ordered first-match semantics and explicit fallback.

use crate::boolean::{AnfPolynomial, BooleanComplexity};
use crate::error::{NeuralOperatorError, Result};
use crate::{LearnedOperator, relative_l2};

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

/// Result of one routed hybrid operator execution.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridExecution {
    /// Action selected by the Boolean router.
    pub action: HybridAction,
    /// Returned field, or `None` for an abstention.
    pub output: Option<Vec<f32>>,
    /// Whether an authoritative solve was executed.
    pub exact_solve_executed: bool,
    /// Relative L2 discrepancy measured during verification, when applicable.
    pub verification_relative_l2: Option<f64>,
    /// Whether verification accepted the surrogate output.
    pub surrogate_accepted: Option<bool>,
}

/// Executes a learned operator under an inspectable Boolean control plane.
///
/// Verification always computes both paths. When the relative L2 discrepancy
/// exceeds the configured tolerance, the authoritative exact output is returned.
pub struct HybridExecutor<O> {
    router: BooleanRouter,
    surrogate: O,
    verification_tolerance: f64,
}

impl<O: LearnedOperator> HybridExecutor<O> {
    /// Construct a fail-closed hybrid executor.
    pub fn new(router: BooleanRouter, surrogate: O, verification_tolerance: f64) -> Result<Self> {
        if !verification_tolerance.is_finite() || verification_tolerance < 0.0
        {
            return Err(NeuralOperatorError::InvalidVerificationTolerance {
                tolerance: verification_tolerance,
            });
        }
        Ok(Self {
            router,
            surrogate,
            verification_tolerance,
        })
    }

    /// Execute one routed call. The supplied closure is the authoritative solver.
    pub fn execute<F>(
        &mut self,
        predicates: &[bool],
        input: &[f32],
        mut exact: F,
    ) -> Result<HybridExecution>
    where
        F: FnMut(&[f32]) -> Result<Vec<f32>>,
    {
        let action = self.router.route(predicates)?;
        match action
        {
            HybridAction::Abstain => Ok(HybridExecution {
                action,
                output: None,
                exact_solve_executed: false,
                verification_relative_l2: None,
                surrogate_accepted: None,
            }),
            HybridAction::UseSurrogate =>
            {
                let output = self.surrogate.predict(input)?;
                Ok(HybridExecution {
                    action,
                    output: Some(output),
                    exact_solve_executed: false,
                    verification_relative_l2: None,
                    surrogate_accepted: None,
                })
            },
            HybridAction::UseExact =>
            {
                let output = self.validate_exact_output(exact(input)?)?;
                Ok(HybridExecution {
                    action,
                    output: Some(output),
                    exact_solve_executed: true,
                    verification_relative_l2: None,
                    surrogate_accepted: None,
                })
            },
            HybridAction::VerifySurrogate =>
            {
                let surrogate = self.surrogate.predict(input)?;
                let exact_output = self.validate_exact_output(exact(input)?)?;
                let discrepancy = relative_l2(&surrogate, &exact_output)?;
                let accepted = discrepancy <= self.verification_tolerance;
                Ok(HybridExecution {
                    action,
                    output: Some(if accepted { surrogate } else { exact_output }),
                    exact_solve_executed: true,
                    verification_relative_l2: Some(discrepancy),
                    surrogate_accepted: Some(accepted),
                })
            },
        }
    }

    fn validate_exact_output(&self, output: Vec<f32>) -> Result<Vec<f32>> {
        if output.len() != self.surrogate.output_len()
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "exact operator output",
                expected: self.surrogate.output_len(),
                got: output.len(),
            });
        }
        if let Some((index, _)) = output
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NeuralOperatorError::NonFinite {
                what: "exact operator output",
                index,
            });
        }
        Ok(output)
    }

    /// Read the current router.
    pub const fn router(&self) -> &BooleanRouter {
        &self.router
    }

    /// Mutable access to the learned surrogate.
    pub fn surrogate_mut(&mut self) -> &mut O {
        &mut self.surrogate
    }

    /// Relative L2 tolerance used by the verification path.
    pub const fn verification_tolerance(&self) -> f64 {
        self.verification_tolerance
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

    struct ScaleSurrogate {
        factor: f32,
    }

    impl LearnedOperator for ScaleSurrogate {
        fn input_len(&self) -> usize {
            2
        }
        fn output_len(&self) -> usize {
            2
        }
        fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>> {
            if input.len() != 2
            {
                return Err(NeuralOperatorError::ShapeMismatch {
                    what: "scale surrogate input",
                    expected: 2,
                    got: input.len(),
                });
            }
            Ok(input.iter().map(|value| value * self.factor).collect())
        }
    }

    fn executor_for(
        action: HybridAction,
        factor: f32,
        tolerance: f64,
    ) -> HybridExecutor<ScaleSurrogate> {
        let guard = AnfPolynomial::from_monomials(1, &[0]).unwrap();
        let router = BooleanRouter::new(
            1,
            vec![BooleanRouteRule::new(guard, action)],
            HybridAction::Abstain,
        )
        .unwrap();
        HybridExecutor::new(router, ScaleSurrogate { factor }, tolerance).unwrap()
    }

    #[test]
    fn verification_accepts_surrogate_inside_tolerance() {
        let mut executor = executor_for(HybridAction::VerifySurrogate, 2.0, 1e-6);
        let result = executor
            .execute(&[true], &[1.0, 2.0], |input| {
                Ok(input.iter().map(|value| value * 2.0).collect())
            })
            .unwrap();
        assert_eq!(result.output, Some(vec![2.0, 4.0]));
        assert_eq!(result.surrogate_accepted, Some(true));
        assert_eq!(result.verification_relative_l2, Some(0.0));
        assert!(result.exact_solve_executed);
    }

    #[test]
    fn verification_falls_back_to_exact_outside_tolerance() {
        let mut executor = executor_for(HybridAction::VerifySurrogate, 1.0, 0.1);
        let result = executor
            .execute(&[true], &[1.0, 2.0], |input| {
                Ok(input.iter().map(|value| value * 2.0).collect())
            })
            .unwrap();
        assert_eq!(result.output, Some(vec![2.0, 4.0]));
        assert_eq!(result.surrogate_accepted, Some(false));
        assert!(result.verification_relative_l2.unwrap() > 0.1);
        assert!(result.exact_solve_executed);
    }

    #[test]
    fn abstain_executes_neither_path() {
        let guard = AnfPolynomial::from_monomials(1, &[1]).unwrap();
        let router = BooleanRouter::new(
            1,
            vec![BooleanRouteRule::new(guard, HybridAction::UseExact)],
            HybridAction::Abstain,
        )
        .unwrap();
        let mut executor =
            HybridExecutor::new(router, ScaleSurrogate { factor: 2.0 }, 0.1).unwrap();
        let mut exact_calls = 0;
        let result = executor
            .execute(&[false], &[1.0, 2.0], |_| {
                exact_calls += 1;
                Ok(vec![2.0, 4.0])
            })
            .unwrap();
        assert_eq!(result.output, None);
        assert_eq!(exact_calls, 0);
        assert!(!result.exact_solve_executed);
    }
}
