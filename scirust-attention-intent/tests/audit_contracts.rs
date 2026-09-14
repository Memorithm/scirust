//! Public-boundary regressions from the 2026-09-14 source audit.
//! Fixtures allocate graph metadata only, never the represented tensor payload.

use scirust_attention_intent::{
    AttentionExecutionIntent, IntentError, TensorRole, derive_attention_intent,
};
use scirust_compute::{DType, Shape};
use scirust_tensor_ir::{Graph, RepresentationPlan, TensorType};

fn derive(shapes: [[usize; 4]; 3]) -> Result<AttentionExecutionIntent, IntentError> {
    let mut graph = Graph::new();
    let q = graph
        .add_input("q", TensorType::new(DType::F32, Shape::new(shapes[0])))
        .expect("q metadata");
    let k = graph
        .add_input("k", TensorType::new(DType::F32, Shape::new(shapes[1])))
        .expect("k metadata");
    let v = graph
        .add_input("v", TensorType::new(DType::F32, Shape::new(shapes[2])))
        .expect("v metadata");
    let plan = RepresentationPlan::dense(&graph).expect("dense metadata plan");
    derive_attention_intent(&graph, &plan, q, k, v, false)
}

#[test]
fn audit_zero_k_heads_returns_error_not_remainder_by_zero_panic() {
    let error = derive([[1, 2, 4, 8], [1, 0, 4, 8], [1, 0, 4, 8]])
        .expect_err("zero K heads must be rejected before modulo");
    assert!(matches!(
        error,
        IntentError::InvalidDimension {
            role: TensorRole::Key,
            ..
        }
    ));
}

#[test]
fn audit_every_zero_dimension_reports_its_tensor_role() {
    let roles = [TensorRole::Query, TensorRole::Key, TensorRole::Value];
    for (role_index, &expected_role) in roles.iter().enumerate()
    {
        for axis in 0..4
        {
            let mut shapes = [[1, 2, 4, 8]; 3];
            shapes[role_index][axis] = 0;
            let error = derive(shapes).expect_err("zero dimension must be rejected");
            match error
            {
                IntentError::InvalidDimension { role, .. } => assert_eq!(role, expected_role),
                other => panic!("unexpected zero-dimension error: {other}"),
            }
        }
    }
}

#[test]
fn audit_k_and_v_head_counts_must_match() {
    let error = derive([[1, 4, 4, 8], [1, 2, 4, 8], [1, 3, 4, 8]])
        .expect_err("V cannot have a different number of heads from K");
    assert!(matches!(
        error,
        IntentError::InvalidDimension {
            role: TensorRole::Value,
            ..
        }
    ));
}

#[test]
fn audit_value_batch_mismatch_is_not_attributed_to_key() {
    let error = derive([[1, 2, 4, 8], [1, 2, 4, 8], [3, 2, 4, 8]])
        .expect_err("different V batch");
    assert!(matches!(
        error,
        IntentError::InvalidDimension {
            role: TensorRole::Value,
            ..
        }
    ));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn audit_large_dimensions_are_rejected_instead_of_truncated() {
    // Previously (2^32 + 1) as u32 became 1. Dense metadata fits in usize/u64;
    // these tests do not allocate multi-gigabyte tensor buffers.
    let too_large = u32::MAX as usize + 2;
    for axis in 0..4
    {
        let mut shape = [1usize; 4];
        shape[axis] = too_large;
        let error = derive([shape; 3]).expect_err("lossy narrowing must fail");
        assert!(matches!(
            error,
            IntentError::InvalidDimension {
                role: TensorRole::Query,
                ..
            }
        ));
    }
}

#[test]
fn audit_gqa_dimensions_and_exact_storage_are_preserved() {
    let intent = derive([[2, 4, 3, 8], [2, 2, 7, 8], [2, 2, 7, 8]])
        .expect("valid grouped-query metadata");
    assert_eq!(intent.batch, 2);
    assert_eq!(intent.q_heads, 4);
    assert_eq!(intent.kv_heads, 2);
    assert_eq!(intent.batch_q_heads, 8);
    assert_eq!(intent.q_len, 3);
    assert_eq!(intent.kv_len, 7);
    assert_eq!(intent.representation.total_storage_bytes, (192 + 224 + 224) * 4);
}

#[test]
fn audit_non_divisible_gqa_heads_remain_rejected() {
    assert!(matches!(
        derive([[1, 3, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]]),
        Err(IntentError::InvalidHeadGrouping {
            q_heads: 3,
            kv_heads: 2,
        })
    ));
}

#[test]
fn audit_executability_checks_the_documented_value_dimension() {
    let mut intent = derive([[1, 2, 4, 8]; 3]).expect("dense intent");
    assert!(intent.is_executable());
    // The fields are public; the predicate must honor its documented condition
    // even after a caller modifies the value dimension.
    intent.value_dim = 16;
    assert!(!intent.is_executable());
}
