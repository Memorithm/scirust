//! Public regressions for caller-controlled `ParallelTape::backward` boundary failures.
//!
//! `ParallelTape` remains a proof/test data-parallel tape. Its backward API is
//! intentionally fail-loud for an output index that does not identify an
//! allocated node; these tests keep that panic distinct from unsupported-op
//! refusal contracts.

use scirust_core::autodiff::parallel::ParallelTape;
use scirust_core::autodiff::reverse::{Node, Op, SavedData};

#[test]
#[should_panic(expected = "backward: idx 0 out of bounds (0 nodes)")]
fn backward_rejects_output_index_on_empty_tape() {
    ParallelTape::new().backward(0);
}

#[test]
#[should_panic(expected = "backward: idx 1 out of bounds (1 nodes)")]
fn backward_rejects_output_index_past_allocated_nodes() {
    let tape = ParallelTape::new();
    tape.alloc_node(Node {
        op: Op::Input,
        shape: (1, 1),
        saved: SavedData::None,
    });

    tape.backward(1);
}
