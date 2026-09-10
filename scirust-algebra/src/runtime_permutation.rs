//! Runtime-sized permutation primitives and bounded lexicographic enumeration.
//!
//! This module complements [`crate::discrete::Permutation`], whose degree is a
//! const generic known at compile time. The runtime form is intended for finite
//! experiments whose carrier size is discovered only while executing.
//!
//! Exhaustive enumeration is explicitly budgeted. The exact factorial count is
//! checked before enumeration starts, and the enumerator keeps only `O(n)`
//! mutable state instead of materializing all `n!` permutations.

use core::fmt;

/// Failure while validating or enumerating runtime-sized permutations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimePermutationError {
    /// An image is not a permutation of `0..degree` because its length differs.
    WrongDegree {
        /// Expected permutation degree.
        expected: usize,
        /// Actual image length.
        actual: usize,
    },
    /// An image value lies outside the permutation domain.
    OutOfRange {
        /// Position containing the invalid value.
        position: usize,
        /// Invalid image value.
        value: usize,
        /// Permutation degree.
        degree: usize,
    },
    /// An image value occurs more than once.
    DuplicateImage {
        /// Repeated image value.
        value: usize,
    },
    /// Two runtime permutations with different degrees were composed.
    DegreeMismatch {
        /// Left-hand degree.
        left: usize,
        /// Right-hand degree.
        right: usize,
    },
    /// The exact factorial count cannot be represented by `usize`.
    FactorialOverflow {
        /// Degree whose factorial overflowed.
        degree: usize,
    },
    /// Exhaustive enumeration would exceed the caller-declared budget.
    EnumerationLimitExceeded {
        /// Permutation degree.
        degree: usize,
        /// Maximum number of permutations admitted by the caller.
        max_permutations: usize,
    },
}

impl fmt::Display for RuntimePermutationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDegree { expected, actual } => {
                write!(formatter, "permutation degree is {actual}, expected {expected}")
            }
            Self::OutOfRange {
                position,
                value,
                degree,
            } => write!(
                formatter,
                "permutation image at position {position} is {value}, outside 0..{degree}"
            ),
            Self::DuplicateImage { value } => {
                write!(formatter, "permutation image repeats value {value}")
            }
            Self::DegreeMismatch { left, right } => write!(
                formatter,
                "cannot compose runtime permutations of degree {left} and {right}"
            ),
            Self::FactorialOverflow { degree } => {
                write!(formatter, "factorial of permutation degree {degree} overflows usize")
            }
            Self::EnumerationLimitExceeded {
                degree,
                max_permutations,
            } => write!(
                formatter,
                "exhaustive enumeration of degree {degree} exceeds budget {max_permutations}"
            ),
        }
    }
}

impl std::error::Error for RuntimePermutationError {}

/// A validated permutation whose degree is known only at runtime.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RuntimePermutation {
    image: Vec<usize>,
}

impl RuntimePermutation {
    /// Construct and validate an owned runtime permutation.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimePermutationError::OutOfRange`] when an image leaves the
    /// domain and [`RuntimePermutationError::DuplicateImage`] when an image value
    /// occurs twice.
    pub fn new(image: Vec<usize>) -> Result<Self, RuntimePermutationError> {
        let degree = image.len();
        let mut seen = vec![false; degree];
        for (position, &value) in image.iter().enumerate() {
            if value >= degree {
                return Err(RuntimePermutationError::OutOfRange {
                    position,
                    value,
                    degree,
                });
            }
            if seen[value] {
                return Err(RuntimePermutationError::DuplicateImage { value });
            }
            seen[value] = true;
        }
        Ok(Self { image })
    }

    /// Construct the identity permutation of `degree`.
    #[must_use]
    pub fn identity(degree: usize) -> Self {
        Self {
            image: (0..degree).collect(),
        }
    }

    /// Return the permutation degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.image.len()
    }

    /// Return whether this is the unique degree-zero permutation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.image.is_empty()
    }

    /// Borrow the permutation image.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.image
    }

    /// Return the image of one domain point, or `None` when `point` is outside the degree.
    #[must_use]
    pub fn apply(&self, point: usize) -> Option<usize> {
        self.image.get(point).copied()
    }

    /// Compose `self ∘ rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimePermutationError::DegreeMismatch`] when the two degrees differ.
    pub fn compose(&self, rhs: &Self) -> Result<Self, RuntimePermutationError> {
        if self.degree() != rhs.degree() {
            return Err(RuntimePermutationError::DegreeMismatch {
                left: self.degree(),
                right: rhs.degree(),
            });
        }
        let image = rhs.image.iter().map(|&value| self.image[value]).collect();
        Ok(Self { image })
    }

    /// Return the inverse permutation.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let mut image = vec![0usize; self.degree()];
        for (position, &value) in self.image.iter().enumerate() {
            image[value] = position;
        }
        Self { image }
    }

    /// Return permutation parity: `1` for even and `-1` for odd.
    #[must_use]
    pub fn signature(&self) -> i8 {
        let mut parity = false;
        for left in 0..self.degree() {
            for right in (left + 1)..self.degree() {
                parity ^= self.image[left] > self.image[right];
            }
        }
        if parity { -1 } else { 1 }
    }
}

/// Return `degree!` when it is representable and does not exceed `max_permutations`.
///
/// The limit is checked before exhaustive enumeration is allowed to begin.
///
/// # Errors
///
/// Returns [`RuntimePermutationError::FactorialOverflow`] when the exact count
/// does not fit `usize`, or [`RuntimePermutationError::EnumerationLimitExceeded`]
/// when the exact count is larger than `max_permutations`.
pub fn checked_factorial(
    degree: usize,
    max_permutations: usize,
) -> Result<usize, RuntimePermutationError> {
    let mut count = 1usize;
    for factor in 2..=degree {
        count = count
            .checked_mul(factor)
            .ok_or(RuntimePermutationError::FactorialOverflow { degree })?;
        if count > max_permutations {
            return Err(RuntimePermutationError::EnumerationLimitExceeded {
                degree,
                max_permutations,
            });
        }
    }
    if count > max_permutations {
        return Err(RuntimePermutationError::EnumerationLimitExceeded {
            degree,
            max_permutations,
        });
    }
    Ok(count)
}

/// Streaming exhaustive permutation enumerator in lexicographic order.
///
/// Construction verifies `degree! <= max_permutations`. The enumerator stores
/// one mutable image of length `degree` plus counters, so exhaustive search does
/// not require materializing all permutations at once.
#[derive(Clone, Debug)]
pub struct LexicographicPermutations {
    current: Vec<usize>,
    exact_count: usize,
    yielded: usize,
}

impl LexicographicPermutations {
    /// Construct a bounded exhaustive enumerator.
    ///
    /// # Errors
    ///
    /// Returns the same factorial overflow or explicit-budget failures as
    /// [`checked_factorial`].
    pub fn new(
        degree: usize,
        max_permutations: usize,
    ) -> Result<Self, RuntimePermutationError> {
        let exact_count = checked_factorial(degree, max_permutations)?;
        Ok(Self {
            current: (0..degree).collect(),
            exact_count,
            yielded: 0,
        })
    }

    /// Return the runtime degree being enumerated.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.current.len()
    }

    /// Return the exact number `degree!` of permutations this enumerator will yield.
    #[must_use]
    pub fn exact_count(&self) -> usize {
        self.exact_count
    }

    /// Return how many permutations have already been yielded.
    #[must_use]
    pub fn yielded(&self) -> usize {
        self.yielded
    }

    /// Advance and borrow the next permutation image.
    ///
    /// The borrowed slice remains valid until the next mutable access to the
    /// enumerator. Degree zero yields one empty permutation, as required by `0! = 1`.
    pub fn next_slice(&mut self) -> Option<&[usize]> {
        if self.yielded == self.exact_count {
            return None;
        }
        if self.yielded != 0 {
            let advanced = next_lexicographic_permutation(&mut self.current);
            debug_assert!(advanced);
        }
        self.yielded += 1;
        Some(&self.current)
    }
}

fn next_lexicographic_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };

    let successor = ((pivot + 1)..values.len())
        .rev()
        .find(|&index| values[pivot] < values[index])
        .expect("a lexicographic pivot always has a larger suffix element");
    values.swap(pivot, successor);
    values[(pivot + 1)..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn runtime_permutation_validates_and_inverts() {
        let permutation = RuntimePermutation::new(vec![2, 0, 1]).unwrap();
        assert_eq!(permutation.degree(), 3);
        assert_eq!(permutation.apply(0), Some(2));
        assert_eq!(permutation.apply(3), None);
        assert_eq!(permutation.signature(), 1);

        let inverse = permutation.inverse();
        assert_eq!(permutation.compose(&inverse).unwrap(), RuntimePermutation::identity(3));
        assert_eq!(inverse.compose(&permutation).unwrap(), RuntimePermutation::identity(3));
    }

    #[test]
    fn malformed_runtime_images_fail_closed() {
        assert_eq!(
            RuntimePermutation::new(vec![0, 2]),
            Err(RuntimePermutationError::OutOfRange {
                position: 1,
                value: 2,
                degree: 2,
            })
        );
        assert_eq!(
            RuntimePermutation::new(vec![1, 1]),
            Err(RuntimePermutationError::DuplicateImage { value: 1 })
        );
    }

    #[test]
    fn composition_requires_matching_degrees() {
        let two = RuntimePermutation::identity(2);
        let three = RuntimePermutation::identity(3);
        assert_eq!(
            two.compose(&three),
            Err(RuntimePermutationError::DegreeMismatch { left: 2, right: 3 })
        );
    }

    #[test]
    fn factorial_budget_is_exact() {
        assert_eq!(checked_factorial(0, 1), Ok(1));
        assert_eq!(checked_factorial(1, 1), Ok(1));
        assert_eq!(checked_factorial(4, 24), Ok(24));
        assert_eq!(
            checked_factorial(4, 23),
            Err(RuntimePermutationError::EnumerationLimitExceeded {
                degree: 4,
                max_permutations: 23,
            })
        );
    }

    #[test]
    fn degree_three_enumeration_is_complete_and_lexicographic() {
        let mut enumerator = LexicographicPermutations::new(3, 6).unwrap();
        let mut observed = Vec::new();
        while let Some(image) = enumerator.next_slice() {
            observed.push(image.to_vec());
        }
        assert_eq!(
            observed,
            vec![
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0],
            ]
        );
        assert_eq!(enumerator.exact_count(), 6);
        assert_eq!(enumerator.yielded(), 6);
    }

    #[test]
    fn enumeration_has_no_duplicates() {
        let mut enumerator = LexicographicPermutations::new(5, 120).unwrap();
        let mut seen = BTreeSet::new();
        while let Some(image) = enumerator.next_slice() {
            assert!(seen.insert(image.to_vec()));
        }
        assert_eq!(seen.len(), 120);
    }

    #[test]
    fn empty_and_singleton_degrees_each_yield_once() {
        let mut empty = LexicographicPermutations::new(0, 1).unwrap();
        assert_eq!(empty.next_slice(), Some([].as_slice()));
        assert_eq!(empty.next_slice(), None);

        let mut singleton = LexicographicPermutations::new(1, 1).unwrap();
        assert_eq!(singleton.next_slice(), Some([0].as_slice()));
        assert_eq!(singleton.next_slice(), None);
    }

    #[test]
    fn replay_is_deterministic() {
        fn collect() -> Vec<Vec<usize>> {
            let mut enumerator = LexicographicPermutations::new(4, 24).unwrap();
            let mut result = Vec::new();
            while let Some(image) = enumerator.next_slice() {
                result.push(image.to_vec());
            }
            result
        }
        assert_eq!(collect(), collect());
    }
}
