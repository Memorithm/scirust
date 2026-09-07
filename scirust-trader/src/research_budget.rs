//! Explicit work budgets for potentially combinatorial trading research.
//!
//! These helpers estimate work before allocation. They are computational
//! safety contracts, not statistical parameters and not profitability gates.

/// Default CSCV split budget used by agent-facing research calls.
pub const DEFAULT_CSCV_MAX_SPLITS: usize = 10_000;

/// Repository-level upper bound for one exact CSCV request.
///
/// Callers may choose a lower budget, but cannot raise a request above this
/// bound without changing and reviewing the code-level policy.
pub const HARD_CSCV_MAX_SPLITS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CscvBudgetError {
    InvalidSlices,
    InvalidBudget,
    CombinationCountOverflow,
    BudgetLimitTooHigh {
        requested: usize,
        hard_max: usize,
    },
    BudgetExceeded {
        required_splits: u128,
        max_splits: usize,
    },
}

/// Exact number of symmetric CSCV assignments: `C(slices, slices / 2)`.
///
/// Returns an error for an invalid (odd or smaller than two) slice count or if
/// the exact count cannot be represented in `u128`.
pub fn cscv_split_count(slices: usize) -> Result<u128, CscvBudgetError> {
    if slices < 2 || !slices.is_multiple_of(2) {
        return Err(CscvBudgetError::InvalidSlices);
    }
    let k = slices / 2;
    let mut result = 1_u128;
    for i in 1..=k {
        // Recurrence C(n-k+i, i) from C(n-k+i-1, i-1). Each division is exact.
        let numerator = (slices - k + i) as u128;
        result = result
            .checked_mul(numerator)
            .ok_or(CscvBudgetError::CombinationCountOverflow)?;
        result /= i as u128;
    }
    Ok(result)
}

/// Validate an exact CSCV request before the core enumerator allocates its
/// assignment vector.
///
/// On success, returns the exact split count. The caller can record this as the
/// declared work size in experiment provenance.
pub fn enforce_cscv_budget(
    slices: usize,
    max_splits: usize,
) -> Result<u128, CscvBudgetError> {
    if max_splits == 0 {
        return Err(CscvBudgetError::InvalidBudget);
    }
    if max_splits > HARD_CSCV_MAX_SPLITS {
        return Err(CscvBudgetError::BudgetLimitTooHigh {
            requested: max_splits,
            hard_max: HARD_CSCV_MAX_SPLITS,
        });
    }
    let required_splits = cscv_split_count(slices)?;
    if required_splits > max_splits as u128 {
        return Err(CscvBudgetError::BudgetExceeded {
            required_splits,
            max_splits,
        });
    }
    Ok(required_splits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_split_count_matches_known_values() {
        assert_eq!(cscv_split_count(4).unwrap(), 6);
        assert_eq!(cscv_split_count(10).unwrap(), 252);
        assert_eq!(cscv_split_count(40).unwrap(), 137_846_528_820);
    }

    #[test]
    fn forty_slices_are_rejected_before_enumeration() {
        assert_eq!(
            enforce_cscv_budget(40, DEFAULT_CSCV_MAX_SPLITS),
            Err(CscvBudgetError::BudgetExceeded {
                required_splits: 137_846_528_820,
                max_splits: DEFAULT_CSCV_MAX_SPLITS,
            })
        );
    }

    #[test]
    fn caller_cannot_raise_budget_past_repository_policy() {
        assert_eq!(
            enforce_cscv_budget(4, HARD_CSCV_MAX_SPLITS + 1),
            Err(CscvBudgetError::BudgetLimitTooHigh {
                requested: HARD_CSCV_MAX_SPLITS + 1,
                hard_max: HARD_CSCV_MAX_SPLITS,
            })
        );
    }
}
