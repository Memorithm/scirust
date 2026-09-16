use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust_autodiff_enzyme_probe::{analytic_dx_f32, enzyme_dx_f32};
use scirust_tensor_core::Tensor;
use scirust_tensor_ir::{
    ConstantId, DType, Graph, MorphoDiff, MorphoDiffReport, NodeId, Operation, Scalar, Shape,
    TensorType,
};
use scirust_tensor_runtime::{Core2Constants, Core2Inputs, Core2ReferenceSession};

const DEFAULT_RUNTIME_ITERS: usize = 100_000;
const DEFAULT_TRANSFORM_ITERS: usize = 2_000;

struct MorphoDiffFixture {
    session: Core2ReferenceSession,
    x: NodeId,
    y: NodeId,
    gradient: NodeId,
    report: MorphoDiffReport,
}

fn scalar_type() -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![1]))
}

fn rosenbrock_graph() -> Result<(Graph, NodeId, NodeId, NodeId), Box<dyn Error>> {
    let ty = scalar_type();
    let mut graph = Graph::new();
    let x = graph.add_input("x", ty.clone())?;
    let y = graph.add_input("y", ty.clone())?;
    let one = graph.add_constant(ConstantId::new(0), ty.clone())?;

    let one_minus_x = graph.add_node(Operation::Sub, vec![one, x], ty.clone())?;
    let first = graph.add_node(
        Operation::Mul,
        vec![one_minus_x, one_minus_x],
        ty.clone(),
    )?;
    let x_squared = graph.add_node(Operation::Mul, vec![x, x], ty.clone())?;
    let residual = graph.add_node(Operation::Sub, vec![y, x_squared], ty.clone())?;
    let residual_squared = graph.add_node(Operation::Mul, vec![residual, residual], ty.clone())?;
    let weighted = graph.add_node(
        Operation::Scale {
            factor: Scalar::f32(100.0),
        },
        vec![residual_squared],
        ty.clone(),
    )?;
    let output = graph.add_node(Operation::Add, vec![first, weighted], ty)?;
    graph.set_outputs(vec![output])?;
    Ok((graph, x, y, output))
}

fn build_fixture() -> Result<MorphoDiffFixture, Box<dyn Error>> {
    let (graph, x, y, output) = rosenbrock_graph()?;
    let differentiated = MorphoDiff::grad(&graph, output, &[x])?;
    let gradient = differentiated.derivative_outputs[0];
    let report = differentiated.report;

    let mut constants = Core2Constants::new();
    constants.insert(
        ConstantId::new(0),
        Tensor::from_f32(vec![1.0], vec![1])?,
    )?;
    let session = Core2ReferenceSession::prepare(differentiated.graph, constants)?;

    Ok(MorphoDiffFixture {
        session,
        x,
        y,
        gradient,
        report,
    })
}

fn inputs(x_node: NodeId, y_node: NodeId, x: f32, y: f32) -> Result<Core2Inputs, Box<dyn Error>> {
    let mut inputs = Core2Inputs::new();
    inputs.insert(x_node, Tensor::from_f32(vec![x], vec![1])?)?;
    inputs.insert(y_node, Tensor::from_f32(vec![y], vec![1])?)?;
    Ok(inputs)
}

fn morphodiff_dx(
    fixture: &MorphoDiffFixture,
    x: f32,
    y: f32,
) -> Result<f32, Box<dyn Error>> {
    let inputs = inputs(fixture.x, fixture.y, x, y)?;
    let outputs = fixture.session.execute(&inputs)?;
    let gradient = outputs
        .get(fixture.gradient)
        .ok_or("MorphoDiff gradient output is missing")?
        .to_f32_vec()?;
    Ok(gradient[0])
}

fn validate_correctness(fixture: &MorphoDiffFixture) -> Result<(), Box<dyn Error>> {
    let points = [
        (1.0f32, 1.0f32),
        (3.0, 1.0),
        (-1.25, 0.75),
        (0.5, -0.25),
        (2.25, 4.0),
        (-0.75, 0.1),
    ];

    for (x, y) in points {
        let analytic = analytic_dx_f32(x, y);
        let enzyme = enzyme_dx_f32(x, y);
        let morphodiff = morphodiff_dx(fixture, x, y)?;
        let tolerance = 2.0e-4 * (1.0 + analytic.abs());

        if (enzyme - analytic).abs() > tolerance {
            return Err(format!(
                "Enzyme correctness gate failed at ({x}, {y}): {enzyme} vs {analytic}"
            )
            .into());
        }
        if (morphodiff - analytic).abs() > tolerance {
            return Err(format!(
                "MorphoDiff correctness gate failed at ({x}, {y}): {morphodiff} vs {analytic}"
            )
            .into());
        }
    }
    Ok(())
}

fn bench_enzyme(iterations: usize) -> Duration {
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        sink = black_box(sink + enzyme_dx_f32(black_box(3.0), black_box(1.0)));
    }
    black_box(sink);
    start.elapsed()
}

fn bench_morphodiff_reference(
    fixture: &MorphoDiffFixture,
    iterations: usize,
) -> Result<Duration, Box<dyn Error>> {
    let inputs = inputs(fixture.x, fixture.y, 3.0, 1.0)?;
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        let outputs = fixture.session.execute(black_box(&inputs))?;
        let gradient = outputs
            .get(fixture.gradient)
            .ok_or("MorphoDiff gradient output is missing")?
            .to_f32_vec()?;
        sink = black_box(sink + gradient[0]);
    }
    black_box(sink);
    Ok(start.elapsed())
}

fn bench_morphodiff_transform(iterations: usize) -> Result<Duration, Box<dyn Error>> {
    let (graph, x, _, output) = rosenbrock_graph()?;
    let start = Instant::now();
    let mut generated_nodes = 0usize;
    for _ in 0..iterations {
        let differentiated = MorphoDiff::grad(black_box(&graph), output, &[x])?;
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
    let runtime_iterations = env_iterations("MORPHODIFF_BENCH_ITERS", DEFAULT_RUNTIME_ITERS);
    let transform_iterations = env_iterations(
        "MORPHODIFF_TRANSFORM_ITERS",
        DEFAULT_TRANSFORM_ITERS,
    );
    let fixture = build_fixture()?;
    validate_correctness(&fixture)?;

    for _ in 0..1_000 {
        black_box(enzyme_dx_f32(3.0, 1.0));
    }
    for _ in 0..100 {
        black_box(morphodiff_dx(&fixture, 3.0, 1.0)?);
    }

    let enzyme = bench_enzyme(runtime_iterations);
    let morphodiff_reference = bench_morphodiff_reference(&fixture, runtime_iterations)?;
    let morphodiff_transform = bench_morphodiff_transform(transform_iterations)?;

    println!("benchmark=morphodiff-vs-enzyme-v1");
    println!("dtype=f32");
    println!("correctness_gate=passed");
    println!("source_nodes={}", fixture.report.source_nodes);
    println!("transformed_nodes={}", fixture.report.transformed_nodes);
    println!("generated_nodes={}", fixture.report.generated_nodes);
    println!("runtime_iterations={runtime_iterations}");
    println!("transform_iterations={transform_iterations}");
    println!(
        "enzyme_compiled_runtime_ns_per_derivative={:.3}",
        ns_per_iteration(enzyme, runtime_iterations)
    );
    println!(
        "morphodiff_core2_reference_ns_per_derivative={:.3}",
        ns_per_iteration(morphodiff_reference, runtime_iterations)
    );
    println!(
        "morphodiff_transform_ns_per_graph={:.3}",
        ns_per_iteration(morphodiff_transform, transform_iterations)
    );
    println!("morphodiff_runtime_lane=core2-reference-interpreter");
    println!("runtime_speed_verdict=not-comparable-until-compiled-backend");

    Ok(())
}
