//! Regression tests for the public per-tensor quantized representation contract.

use scirust_compute::{DType, Shape};
use scirust_tensor_ir::{Graph, RepresentationError, RepresentationPlan, TensorType};

fn tensor_type(dtype: DType, dims: &[usize]) -> TensorType {
    TensorType::new(dtype, Shape::new(dims.to_vec()))
}

#[test]
fn per_tensor_quantized_rejects_non_integer_codes_at_declaration() {
    let mut graph = Graph::new();
    let weight = graph
        .add_input("weight", tensor_type(DType::F32, &[2, 2]))
        .unwrap();
    graph.set_outputs(vec![weight]).unwrap();

    let mut plan = RepresentationPlan::dense(&graph).unwrap();
    let dense_f32 = plan.assignment(weight).unwrap();

    assert_eq!(
        plan.declare_quantized_per_tensor(
            tensor_type(DType::F32, &[2, 2]),
            dense_f32,
            TensorType::new(DType::F32, Shape::scalar()),
            dense_f32,
        ),
        Err(RepresentationError::QuantizedInvalidComponentDTypes {
            codes: DType::F32,
            scales: DType::F32,
        })
    );
}

#[test]
fn per_tensor_quantized_rejects_non_scalar_scale_when_bound() {
    let mut graph = Graph::new();
    let weight = graph
        .add_input("weight", tensor_type(DType::F32, &[2, 2]))
        .unwrap();
    graph.set_outputs(vec![weight]).unwrap();

    let mut plan = RepresentationPlan::dense(&graph).unwrap();
    let dense_f32 = plan.assignment(weight).unwrap();
    let dense_u8 = plan.declare_dense(DType::U8).unwrap();

    let quantized = plan
        .declare_quantized_per_tensor(
            tensor_type(DType::U8, &[2, 2]),
            dense_u8,
            tensor_type(DType::F32, &[1]),
            dense_f32,
        )
        .unwrap();

    assert_eq!(
        plan.assign(&graph, weight, quantized),
        Err(RepresentationError::QuantizedPerTensorIncompatibleShapes {
            codes: Shape::new(vec![2, 2]),
            scale: Shape::new(vec![1]),
            logical: Shape::new(vec![2, 2]),
        })
    );
    assert_eq!(plan.assignment(weight), Some(dense_f32));
}
