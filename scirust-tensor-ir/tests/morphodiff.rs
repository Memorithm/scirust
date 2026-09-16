use scirust_tensor_ir::{DType, Graph, MorphoDiff, Operation, Shape, TensorType};

fn matrix_type(rows: usize, cols: usize) -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![rows, cols]))
}

fn batched_matrix_type(batch: usize, rows: usize, cols: usize) -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![batch, rows, cols]))
}

#[test]
fn public_morphodiff_vjp_differentiates_matmul() {
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", matrix_type(2, 3)).unwrap();
    let rhs = graph.add_input("rhs", matrix_type(3, 4)).unwrap();
    let output = graph
        .add_node(
            Operation::MatMul,
            vec![lhs, rhs],
            matrix_type(2, 4),
        )
        .unwrap();
    graph.set_outputs(vec![output]).unwrap();

    let program = MorphoDiff::vjp(&graph, output, &[lhs, rhs]).unwrap();

    assert_eq!(program.seed_inputs.len(), 1);
    assert_eq!(program.derivative_outputs.len(), 2);
    assert_eq!(program.report.wrt_count, 2);
    assert_eq!(program.report.derivative_output_count, 2);
    assert_eq!(program.graph.validate(), Ok(()));
    assert!(program.report.generated_nodes > 0);
    assert!(
        program
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node.operation, Operation::Transpose { .. }))
    );
    assert!(
        program
            .graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.operation, Operation::MatMul))
            .count()
            >= 3
    );
}

#[test]
fn public_morphodiff_jvp_differentiates_batched_matmul() {
    let mut graph = Graph::new();
    let lhs = graph
        .add_input("lhs", batched_matrix_type(4, 2, 3))
        .unwrap();
    let rhs = graph
        .add_input("rhs", batched_matrix_type(4, 3, 5))
        .unwrap();
    let output = graph
        .add_node(
            Operation::BatchMatMul,
            vec![lhs, rhs],
            batched_matrix_type(4, 2, 5),
        )
        .unwrap();
    graph.set_outputs(vec![output]).unwrap();

    let program = MorphoDiff::jvp(&graph, output, &[lhs, rhs]).unwrap();

    assert_eq!(program.seed_inputs.len(), 2);
    assert_eq!(program.derivative_outputs.len(), 1);
    assert_eq!(program.report.wrt_count, 2);
    assert_eq!(program.graph.validate(), Ok(()));
    assert!(
        program
            .graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.operation, Operation::BatchMatMul))
            .count()
            >= 3
    );
}
