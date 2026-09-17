//! Bounded exhaustive sparse-ANF synthesis for Boolean control planes.
//!
//! This surface is deliberately an **exact-function** search: every development
//! assignment is unique and has one Boolean label. Empirical policy learning
//! with repeated/conflicting observations belongs on a separate weighted surface.
//! Candidate-space size is computed before enumeration and the search refuses to
//! run when the declared budget cannot cover the complete bounded grammar.

use std::collections::BTreeSet;

use crate::{AnfPolynomial, BooleanComplexity, NeuralOperatorError, Result};

/// One uniquely labelled Boolean assignment in a development set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanDevelopmentExample {
    /// Boolean predicate vector.
    pub input: Vec<bool>,
    /// Exact desired Boolean output.
    pub target: bool,
}

impl BooleanDevelopmentExample {
    /// Construct one labelled assignment.
    pub fn new(input: Vec<bool>, target: bool) -> Self {
        Self { input, target }
    }
}

/// Validated exact-function development set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanDevelopmentSet {
    variables: usize,
    examples: Vec<BooleanDevelopmentExample>,
}

impl BooleanDevelopmentSet {
    /// Validate one exact label per distinct assignment.
    pub fn new(variables: usize, examples: Vec<BooleanDevelopmentExample>) -> Result<Self> {
        if examples.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "Boolean development set",
            });
        }
        if variables > 20
        {
            return Err(NeuralOperatorError::TooManyBooleanVariables {
                variables,
                maximum: 20,
            });
        }
        let mut seen = BTreeSet::new();
        for (index, example) in examples.iter().enumerate()
        {
            if example.input.len() != variables
            {
                return Err(NeuralOperatorError::BooleanDatasetShapeMismatch {
                    example: index,
                    expected: variables,
                    got: example.input.len(),
                });
            }
            let assignment = pack_assignment(&example.input);
            if !seen.insert(assignment)
            {
                return Err(NeuralOperatorError::DuplicateBooleanAssignment { assignment });
            }
        }
        Ok(Self {
            variables,
            examples,
        })
    }

    /// Input arity of the exact-function surface.
    pub const fn variables(&self) -> usize {
        self.variables
    }

    /// Labelled development assignments in declaration order.
    pub fn examples(&self) -> &[BooleanDevelopmentExample] {
        &self.examples
    }
}

/// Complete bounded sparse-ANF grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseAnfSearchConfig {
    /// Maximum monomial degree admitted to the grammar.
    pub max_degree: usize,
    /// Maximum number of retained ANF monomials.
    pub max_terms: usize,
    /// Maximum complete candidate universe the caller authorizes evaluating.
    pub candidate_budget: usize,
}

/// Deterministic result of a complete bounded sparse-ANF search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseAnfSearchResult {
    /// Selected exact polynomial.
    pub polynomial: AnfPolynomial,
    /// Complete candidate count evaluated under the frozen grammar.
    pub candidates_evaluated: usize,
    /// Number of monomials available before term-count restriction.
    pub monomial_universe: usize,
    /// Structural complexity of the selected polynomial.
    pub complexity: BooleanComplexity,
}

/// Search the complete bounded ANF grammar for an exact fit.
///
/// The search order does not define the winner: all admitted candidates are
/// evaluated and exact fits are compared by `(terms, literal occurrences,
/// max degree, canonical monomial masks)` so the output is deterministic.
pub fn search_sparse_anf(
    development: &BooleanDevelopmentSet,
    config: SparseAnfSearchConfig,
) -> Result<SparseAnfSearchResult> {
    if config.max_degree > development.variables
    {
        return Err(NeuralOperatorError::InvalidAnfSearchDegree {
            degree: config.max_degree,
            variables: development.variables,
        });
    }
    if config.max_terms == 0
    {
        return Err(NeuralOperatorError::InvalidAnfSearchTerms { terms: 0 });
    }
    if config.candidate_budget == 0
    {
        return Err(NeuralOperatorError::CandidateBudgetExceeded {
            required: 1,
            budget: 0,
        });
    }

    let monomials = monomial_universe(development.variables, config.max_degree);
    let capped_terms = config.max_terms.min(monomials.len());
    let required = combination_prefix_count(monomials.len(), capped_terms);
    if required > config.candidate_budget
    {
        return Err(NeuralOperatorError::CandidateBudgetExceeded {
            required,
            budget: config.candidate_budget,
        });
    }

    let mut candidate = Vec::with_capacity(capped_terms);
    let mut best: Option<AnfPolynomial> = None;
    let mut evaluated = 0usize;
    for terms in 0..=capped_terms
    {
        enumerate_combinations(&monomials, terms, 0, &mut candidate, &mut |masks| {
            evaluated = evaluated.saturating_add(1);
            let polynomial = AnfPolynomial::from_monomials(development.variables, masks)
                .expect("enumerated monomials are valid for the declared arity");
            if development
                .examples
                .iter()
                .all(|row| polynomial.evaluate(&row.input).ok() == Some(row.target))
                && better_exact_candidate(&polynomial, best.as_ref())
            {
                best = Some(polynomial);
            }
        });
    }
    debug_assert_eq!(evaluated, required);

    let polynomial = best.ok_or(NeuralOperatorError::NoExactAnfCandidate)?;
    let complexity = polynomial.complexity();
    Ok(SparseAnfSearchResult {
        polynomial,
        candidates_evaluated: evaluated,
        monomial_universe: monomials.len(),
        complexity,
    })
}

fn pack_assignment(input: &[bool]) -> u64 {
    input.iter().copied().enumerate().fold(
        0u64,
        |mask, (index, bit)| {
            if bit { mask | (1u64 << index) } else { mask }
        },
    )
}

fn monomial_universe(variables: usize, max_degree: usize) -> Vec<u64> {
    let total = 1usize << variables;
    (0..total)
        .filter(|mask| mask.count_ones() as usize <= max_degree)
        .map(|mask| mask as u64)
        .collect()
}

fn combination_prefix_count(items: usize, max_terms: usize) -> usize {
    (0..=max_terms).fold(0usize, |sum, terms| {
        sum.saturating_add(binomial_saturating(items, terms))
    })
}

fn binomial_saturating(n: usize, k: usize) -> usize {
    if k > n
    {
        return 0;
    }
    let k = k.min(n - k);
    let mut value = 1u128;
    for step in 0..k
    {
        value = value
            .saturating_mul((n - step) as u128)
            .checked_div((step + 1) as u128)
            .unwrap_or(u128::MAX);
        if value > usize::MAX as u128
        {
            return usize::MAX;
        }
    }
    value as usize
}

fn enumerate_combinations<F>(
    universe: &[u64],
    remaining: usize,
    start: usize,
    candidate: &mut Vec<u64>,
    visit: &mut F,
) where
    F: FnMut(&[u64]),
{
    if remaining == 0
    {
        visit(candidate);
        return;
    }
    let last_start = universe.len() - remaining;
    for index in start..=last_start
    {
        candidate.push(universe[index]);
        enumerate_combinations(universe, remaining - 1, index + 1, candidate, visit);
        candidate.pop();
    }
}

fn better_exact_candidate(candidate: &AnfPolynomial, current: Option<&AnfPolynomial>) -> bool {
    let Some(current) = current
    else
    {
        return true;
    };
    let left = candidate.complexity();
    let right = current.complexity();
    (
        left.monomials,
        left.literal_occurrences,
        left.max_degree,
        candidate.monomials(),
    ) < (
        right.monomials,
        right.literal_occurrences,
        right.max_degree,
        current.monomials(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_three_variable_set<F>(f: F) -> BooleanDevelopmentSet
    where
        F: Fn(bool, bool, bool) -> bool,
    {
        let examples = (0usize..8)
            .map(|mask| {
                let input = vec![mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
                let target = f(input[0], input[1], input[2]);
                BooleanDevelopmentExample::new(input, target)
            })
            .collect();
        BooleanDevelopmentSet::new(3, examples).unwrap()
    }

    #[test]
    fn recovers_sparse_zhegalkin_function_exactly() {
        let development = complete_three_variable_set(|x0, x1, x2| x0 ^ (x1 & x2));
        let result = search_sparse_anf(
            &development,
            SparseAnfSearchConfig {
                max_degree: 2,
                max_terms: 2,
                candidate_budget: 64,
            },
        )
        .unwrap();
        assert_eq!(result.polynomial.monomials(), &[0b001, 0b110]);
        assert_eq!(result.complexity.monomials, 2);
        assert_eq!(result.complexity.max_degree, 2);
        assert_eq!(result.candidates_evaluated, 29);
    }

    #[test]
    fn refuses_incomplete_candidate_budget_instead_of_truncating() {
        let development = complete_three_variable_set(|x0, x1, x2| x0 ^ x1 ^ x2);
        let error = search_sparse_anf(
            &development,
            SparseAnfSearchConfig {
                max_degree: 3,
                max_terms: 3,
                candidate_budget: 10,
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            NeuralOperatorError::CandidateBudgetExceeded {
                required: 93,
                budget: 10,
            }
        );
    }

    #[test]
    fn duplicate_development_assignments_fail_closed() {
        let error = BooleanDevelopmentSet::new(
            2,
            vec![
                BooleanDevelopmentExample::new(vec![true, false], true),
                BooleanDevelopmentExample::new(vec![true, false], false),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            NeuralOperatorError::DuplicateBooleanAssignment { assignment: 1 }
        );
    }

    #[test]
    fn exact_search_reports_unrepresentable_function_under_degree_bound() {
        let development = complete_three_variable_set(|x0, x1, x2| x0 & x1 & x2);
        let error = search_sparse_anf(
            &development,
            SparseAnfSearchConfig {
                max_degree: 2,
                max_terms: 4,
                candidate_budget: 100,
            },
        )
        .unwrap_err();
        assert_eq!(error, NeuralOperatorError::NoExactAnfCandidate);
    }
}
