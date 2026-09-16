//! Fallible public accessors for [`ParallelTape`].
//!
//! The historical `ParallelTape` compatibility methods intentionally keep their
//! panic behavior. This module adds typed index/shape validation without changing
//! `backward` semantics or hiding `RwLock` poisoning. Lock poisoning therefore
//! remains an explicit panic contract of the underlying proof/test tape.

use super::parallel::ParallelTape;
use super::reverse::Tensor;
use std::fmt;

/// Validation failures returned by the fallible [`ParallelTape`] accessors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelTapeAccessError {
    /// A node index is outside the allocated range.
    NodeOutOfBounds {
        /// Requested node index.
        index: usize,
        /// Number of nodes allocated when the access was validated.
        node_count: usize,
    },
    /// A replacement forward value has the wrong element count.
    ValueLengthMismatch {
        /// Target node index.
        index: usize,
        /// Number of elements required by the node's fixed shape.
        expected: usize,
        /// Number of elements supplied by the caller.
        actual: usize,
    },
}

impl fmt::Display for ParallelTapeAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NodeOutOfBounds { index, node_count } =>
            {
                write!(
                    f,
                    "ParallelTape node index {index} is out of bounds for {node_count} nodes"
                )
            },
            Self::ValueLengthMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "ParallelTape node {index} requires {expected} values, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ParallelTapeAccessError {}

impl ParallelTape {
    /// Returns a cloned forward value after validating `idx`.
    ///
    /// This is the fallible counterpart of [`ParallelTape::value`].
    ///
    /// # Errors
    ///
    /// Returns [`ParallelTapeAccessError::NodeOutOfBounds`] when `idx` has not
    /// been allocated.
    ///
    /// # Panics
    ///
    /// Panics if the internal node-count or value `RwLock` is poisoned. The
    /// proof/test tape deliberately keeps lock poisoning fail-loud rather than
    /// silently recovering shared state.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::parallel::ParallelTape;
    /// use scirust_core::autodiff::parallel_access::ParallelTapeAccessError;
    ///
    /// let tape = ParallelTape::new();
    /// assert!(matches!(
    ///     tape.try_value(0),
    ///     Err(ParallelTapeAccessError::NodeOutOfBounds {
    ///         index: 0,
    ///         node_count: 0,
    ///     })
    /// ));
    /// ```
    pub fn try_value(&self, idx: usize) -> Result<Tensor, ParallelTapeAccessError> {
        let node_count = self.num_nodes();
        if idx >= node_count
        {
            return Err(ParallelTapeAccessError::NodeOutOfBounds {
                index: idx,
                node_count,
            });
        }
        Ok(self.value(idx))
    }

    /// Returns the scalar gradient for `idx` after validating the node index.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelTapeAccessError::NodeOutOfBounds`] when `idx` has not
    /// been allocated.
    ///
    /// # Panics
    ///
    /// Panics if the internal node-count or gradient `RwLock` is poisoned.
    pub fn try_grad(&self, idx: usize) -> Result<f64, ParallelTapeAccessError> {
        let node_count = self.num_nodes();
        if idx >= node_count
        {
            return Err(ParallelTapeAccessError::NodeOutOfBounds {
                index: idx,
                node_count,
            });
        }
        Ok(self.grad(idx))
    }

    /// Replaces the forward value for `idx` after validating index and length.
    ///
    /// Node shapes are fixed at allocation time, so validating against the
    /// current cloned value is sufficient even if another clone allocates more
    /// nodes concurrently: existing slots are never removed or reshaped.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelTapeAccessError::NodeOutOfBounds`] for an unknown node,
    /// or [`ParallelTapeAccessError::ValueLengthMismatch`] when `data` does not
    /// match the node's fixed dense shape.
    ///
    /// # Panics
    ///
    /// Panics if an internal `RwLock` is poisoned. Lock poisoning remains a
    /// compatibility-level fail-loud contract; this method does not recover from
    /// potentially inconsistent shared state.
    pub fn try_set_value(&self, idx: usize, data: &[f32]) -> Result<(), ParallelTapeAccessError> {
        let current = self.try_value(idx)?;
        let expected = current.data.len();
        if data.len() != expected
        {
            return Err(ParallelTapeAccessError::ValueLengthMismatch {
                index: idx,
                expected,
                actual: data.len(),
            });
        }
        self.set_value(idx, data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Node, Op, SavedData};

    #[test]
    fn fallible_access_reports_index_and_shape_errors_without_mutation() {
        let tape = ParallelTape::new();
        let x = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 2),
            saved: SavedData::None,
        });

        assert_eq!(tape.num_nodes(), 1);
        assert!(matches!(
            tape.try_value(1),
            Err(ParallelTapeAccessError::NodeOutOfBounds {
                index: 1,
                node_count: 1,
            })
        ));
        assert_eq!(
            tape.try_set_value(x, &[1.0]),
            Err(ParallelTapeAccessError::ValueLengthMismatch {
                index: x,
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(tape.try_value(x).unwrap().data, vec![0.0, 0.0]);
    }

    #[test]
    fn clone_shares_values_gradients_and_node_count() {
        let tape = ParallelTape::new();
        let x = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let y = tape.alloc_node(Node {
            op: Op::Scale {
                input: x,
                scalar: 2.0,
            },
            shape: (1, 1),
            saved: SavedData::None,
        });
        let clone = tape.clone();

        clone.try_set_value(x, &[3.0]).unwrap();
        clone.try_set_value(y, &[6.0]).unwrap();
        assert_eq!(tape.try_value(x).unwrap().data, vec![3.0]);
        assert_eq!(clone.num_nodes(), tape.num_nodes());

        clone.backward(y);
        assert_eq!(tape.try_grad(x).unwrap(), 2.0);
        assert_eq!(clone.grads(), tape.grads());
    }
}
