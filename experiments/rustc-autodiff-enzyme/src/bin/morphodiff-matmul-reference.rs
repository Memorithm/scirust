use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust_gpu::CpuComputeAdapter;
use scirust_tensor_ir::{DType, Graph, MorphoDiff, MorphoDiffReport, NodeId, Operation, Shape, TensorType};
use scirust_tensor_runtime::{
    GraphConstants, GraphInputs, ReferenceJitSession, ReferencePlanRuntime,
};

const DEFAULT_MATMUL_ITERS: usize = 2_000;
const DEFAULT_TRANSFORM_ITERS: usize = 1_000;
const MATMUL_DIM: usize = 16;
const MATMUL_ELEMENTS: usize = MATMUL_DIM * MATMUL_DIM;

struct MatrixFixture {
    session: ReferenceJitSession<CpuComputeAdapter>,
    a: NodeId,
    b: NodeId,
    report: MorphoDiffReport,
}

fn matrix_type() -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![MATMUL_DIM, MATMUL_DIM]))
}

fn matrix_graph() -> Result<(Graph, NodeId, NodeId, NodeId), Box<dyn Error>> {
    let ty = matrix_type();
    let mut graph = Graph::new();
    let a = graph.add_input("a", ty.clone())?;
    let b = graph.add_input("b", ty.clone())?;
    let product = graph.add_node(Operation::MatMul, vec![a, b], ty)?;
    graph.set_outputs(vec![product])?;
    Ok((graph, a, b, product))
}

fn matrix_inputs() -> (Vec<f32>, Vec<f32>) {
    let a = (0..MATMUL_ELEMENTS)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.0625)
        .collect();
    let b = (0..MATMUL_ELEMENTS)
        .map(|index| ((index % 13) as f32 - 6.0) * 0.03125)
        .collect();
    (a, b)
}

fn analytic_grad_a(b: &[f32]) -> Vec<f32> {
    let mut gradient = vec![0.0f32; MATMUL_ELEMENTS];
    for row in 0..MATMUL_DIM {
        for inner in 0..MATMUL_DIM {
            let mut sum = 0.0f32;
            for col in 0..MATMUL_DIM {
                sum += b[inner * MATMUL_DIM + col];
            }
            gradient[row * MATMUL_DIM + inner] = sum;
        }
    }
    gradient
}

fn build_fixture() -> Result<MatrixFixture, Box<dyn Error>> {
    let (graph, a, b, product) = matrix_graph()?;
    let differentiated = MorphoDiff::grad(&graph, product, &[a])?;
    let report = differentiated.report;
    let session = ReferenceJitSession::prepare(
        ReferencePlanRuntime::new(CpuComputeAdapter::new()),
        &differentiated.graph,
        &GraphConstants::new(),
    )?;
    Ok(MatrixFixture {
        session,
        a,
        b,
        report,
    })
}

fn execute_gradient(
    fixture: &MatrixFixture,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut inputs = GraphInputs::new();
    inputs.bind(fixture.a, a).bind(fixture.b, b);
    let outputs = fixture.session.execute(&inputs)?;
    let gradient = outputs
        .values()
        .first()
        .ok_or("MorphoDiff MatMul gradient output is missing")?;
    Ok(gradient.values.clone())
}

fn validate_correctness(fixture: &MatrixFixture) -> Result<(), Box<dyn Error>> {
    let (a, b) = matrix_inputs();
    let expected = analytic_grad_a(&b);
    let actual = execute_gradient(fixture, &a, &b)?;
    if actual.len() != expected.len() {
        return Err(format!(
            "MorphoDiff MatMul gradient length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 5.0e-4 * (1.0 + want.abs());
        if (got - want).abs() > tolerance {
            return Err(format!(
                "MorphoDiff MatMul gradient gate failed at A[{index}]: {got} vs {want}"
            )
            .into());
        }
    }
    Ok(())
}

fn bench_prepared(
    fixture: &MatrixFixture,
    iterations: usize,
) -> Result<Duration, Box<dyn Error>> {
    let (a, b) = matrix_inputs();
    let mut inputs = GraphInputs::new();
    inputs.bind(fixture.a, &a).bind(fixture.b, &b);

    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        let outputs = fixture.session.execute(black_box(&inputs))?;
        let gradient = outputs
            .values()
            .first()
            .ok_or("MorphoDiff MatMul gradient output is missing")?;
        sink = black_box(sink + gradient.values[0]);
    }
    black_box(sink);
    Ok(start.elapsed())
}

fn bench_transform(iterations: usize) -> Result<Duration, Box<dyn Error>> {
    let (graph, a, _, product) = matrix_graph()?;
    let start = Instant::now();
    let mut generated_nodes = 0usize;
    for _ in 0..iterations {
        let differentiated = MorphoDiff::grad(black_box(&graph), product, &[a])?;
        generated_nodes = black_box(generated_nodes + differentiated.report.generated_nodes);
    }
    black_box(generated_nodes);
    Ok(start.elapsed())
}

fn ns_per_iteration(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1.0e9 / iterations as f64
}

fn env_iterations(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let iterations = env_iterations("MORPHODIFF_MATMUL_ITERS", DEFAULT_MATMUL_ITERS);
    let transform_iterations = env_iterations(
        "MORPHODIFF_MATMUL_TRANSFORM_ITERS",
        DEFAULT_TRANSFORM_ITERS,
    );
    let fixture = build_fixture()?;
    validate_correctness(&fixture)?;

    let (a, b) = matrix_inputs();
    for _ in 0..10 {
        black_box(execute_gradient(&fixture, &a, &b)?);
    }

    let runtime = bench_prepared(&fixture, iterations)?;
    let transform = bench_transform(transform_iterations)?;

    println!("benchmark=morphodiff-matmul-reference-v1");
    println!("dtype=f32");
    println!("correctness_gate=passed");
    println!("matmul_dim={MATMUL_DIM}");
    println!("matmul_objective=sum(A@B)");
    println!("matmul_wrt=A");
    println!("source_nodes={}", fixture.report.source_nodes);
    println!("transformed_nodes={}", fixture.report.transformed_nodes);
    println!("generated_nodes={}", fixture.report.generated_nodes);
    println!(
        "prepared_kernel_count={}",
        fixture.session.compiled_kernel_count()
    );
    println!("prepared_dispatch_count={}", fixture.session.dispatch_count());
    println!("matmul_iterations={iterations}");
    println!("transform_iterations={transform_iterations}");
    println!(
        "morphodiff_prepared_matmul_grad_ns={:.3}",
        ns_per_iteration(runtime, iterations)
    );
    println!(
        "morphodiff_matmul_transform_ns_per_graph={:.3}",
        ns_per_iteration(transform, transform_iterations)
    );
    println!("morphodiff_runtime_lane=reference-jit-cpu-adapter");
    println!("native_speed_verdict=not-yet-comparable");
    Ok(())
}
