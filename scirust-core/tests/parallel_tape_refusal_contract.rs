//! Public regressions for the deliberately unsupported fused-op boundary on
//! `ParallelTape`.
//!
//! `ParallelTape` is a proof/test data-parallel tape with scalar gradient
//! reduction. Fused ops whose backward implementation belongs to the sequential
//! `Tape` must fail loudly here rather than silently producing zero gradients.

use scirust_core::autodiff::parallel::ParallelTape;
use scirust_core::autodiff::reverse::{Node, Op, SavedData};

fn input(tape: &ParallelTape) -> usize {
    let node = tape.alloc_node(Node {
        op: Op::Input,
        shape: (1, 1),
        saved: SavedData::None,
    });
    tape.set_value(node, &[1.0]);
    node
}

#[test]
#[should_panic(expected = "FlashAttention backward is not available on ParallelTape")]
fn flash_attention_backward_is_refused_not_silently_zeroed() {
    let tape = ParallelTape::new();
    let q = input(&tape);
    let k = input(&tape);
    let v = input(&tape);
    let output = tape.alloc_node(Node {
        op: Op::FlashAttention {
            q,
            k,
            v,
            mask: None,
            batch: 1,
            n_heads: 1,
            seq_len: 1,
            d_head: 1,
            scale: 1.0,
            block_size: 1,
        },
        shape: (1, 1),
        saved: SavedData::None,
    });
    tape.set_value(output, &[1.0]);

    tape.backward(output);
}

#[test]
#[should_panic(expected = "Conv2dTranspose backward is not available on ParallelTape")]
fn conv2d_transpose_backward_is_refused_not_silently_zeroed() {
    let tape = ParallelTape::new();
    let input_node = input(&tape);
    let weight = input(&tape);
    let output = tape.alloc_node(Node {
        op: Op::Conv2dTransposeForward {
            input: input_node,
            weight,
            bias: None,
            batch: 1,
            in_c: 1,
            h: 1,
            w: 1,
            out_c: 1,
            kernel: 1,
            stride: 1,
            pad: 0,
            output_padding: 0,
        },
        shape: (1, 1),
        saved: SavedData::None,
    });
    tape.set_value(output, &[1.0]);

    tape.backward(output);
}
