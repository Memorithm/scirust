use scirust_compute::{DType, Shape};
use scirust_gpu::CpuComputeAdapter;
use scirust_tensor_ir::{Graph, MorphoDiff, Operation, TensorType};
use scirust_tensor_runtime::{
    GraphConstants, GraphInputs, ReferenceGraphSession, ReferencePlanRuntime,
};

fn ty(dims: &[usize]) -> TensorType {
    TensorType::new(DType::F32, Shape::new(dims.to_vec()))
}

fn runtime() -> ReferencePlanRuntime<CpuComputeAdapter> {
    ReferencePlanRuntime::new(CpuComputeAdapter::new())
}

#[test]
fn prepared_reference_executes_rank2_matmul() {
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", ty(&[2, 3])).unwrap();
    let rhs = graph.add_input("rhs", ty(&[3, 2])).unwrap();
    let product = graph
        .add_node(Operation::MatMul, vec![lhs, rhs], ty(&[2, 2]))
        .unwrap();
    graph.set_outputs(vec![product]).unwrap();

    let session = ReferenceGraphSession::prepare(runtime(), &graph, &GraphConstants::new())
        .expect("rank-2 MatMul must prepare");
    let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(lhs, &left).bind(rhs, &right);

    let outputs = session.execute(&inputs).expect("rank-2 MatMul must execute");
    assert_eq!(outputs.into_values()[0].values, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn prepared_reference_executes_batch_matmul() {
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", ty(&[2, 1, 2])).unwrap();
    let rhs = graph.add_input("rhs", ty(&[2, 2, 2])).unwrap();
    let product = graph
        .add_node(Operation::BatchMatMul, vec![lhs, rhs], ty(&[2, 1, 2]))
        .unwrap();
    graph.set_outputs(vec![product]).unwrap();

    let session = ReferenceGraphSession::prepare(runtime(), &graph, &GraphConstants::new())
        .expect("BatchMatMul must prepare");
    let left = [1.0f32, 2.0, 3.0, 4.0];
    let right = [5.0f32, 6.0, 7.0, 8.0, 1.0, 0.0, 0.0, 1.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(lhs, &left).bind(rhs, &right);

    let outputs = session.execute(&inputs).expect("BatchMatMul must execute");
    assert_eq!(outputs.into_values()[0].values, vec![19.0, 22.0, 3.0, 4.0]);
}

#[test]
fn morphodiff_matmul_gradient_executes_through_prepared_cpu_pipeline() {
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", ty(&[2, 2])).unwrap();
    let rhs = graph.add_input("rhs", ty(&[2, 2])).unwrap();
    let product = graph
        .add_node(Operation::MatMul, vec![lhs, rhs], ty(&[2, 2]))
        .unwrap();
    graph.set_outputs(vec![product]).unwrap();

    // `grad` differentiates the sum of a non-scalar output by seeding an
    // all-one cotangent. The transformed graph therefore exercises OnesLike,
    // Transpose and MatMul through the prepared Reference path.
    let differentiated = MorphoDiff::grad(&graph, product, &[lhs]).unwrap();
    assert_eq!(differentiated.derivative_outputs.len(), 1);

    let session = ReferenceGraphSession::prepare(
        runtime(),
        &differentiated.graph,
        &GraphConstants::new(),
    )
    .expect("MorphoDiff MatMul gradient must prepare");

    let left = [2.0f32, -1.0, 0.5, 3.0];
    let right = [1.0f32, 2.0, 3.0, 4.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(lhs, &left).bind(rhs, &right);

    let outputs = session
        .execute(&inputs)
        .expect("MorphoDiff MatMul gradient must execute");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs.into_values()[0].values, vec![3.0, 7.0, 3.0, 7.0]);
}
