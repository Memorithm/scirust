use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust_gpu::CpuComputeAdapter;
use scirust_tensor_core::Tensor;
use scirust_tensor_ir::{
    ConstantId, DType, Graph, MorphoDiff, MorphoDiffReport, NodeId, Operation, Scalar, Shape,
    TensorType,
};
use scirust_tensor_runtime::{
    Core2Constants, Core2Inputs, Core2ReferenceSession, GraphConstants, GraphInputs,
    ReferenceJitSession, ReferencePlanRuntime,
};

const DEFAULT_RUNTIME_ITERS: usize = 100_000;
const DEFAULT_TRANSFORM_ITERS: usize = 2_000;

struct MorphoDiffFixture {
    core_session: Core2ReferenceSession,
    prepared_session: ReferenceJitSession<CpuComputeAdapter>,
    x: NodeId,
    y: NodeId,
    cotangent: NodeId,
    core_gradient: NodeId,
    grad_report: MorphoDiffReport,
    vjp_report: MorphoDiffReport,
}

fn scalar_type() -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![1]))
}

fn analytic_dx_f32(x: f32, y: f32) -> f32 {
    -2.0 * (1.0 - x) - 400.0 * x * (y - x * x)
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

    let grad_program = MorphoDiff::grad(&graph, output, &[x])?;
    let core_gradient = grad_program.derivative_outputs[0];
    let grad_report = grad_program.report;
    let mut core_constants = Core2Constants::new();
    core_constants.insert(
        ConstantId::new(0),
        Tensor::from_f32(vec![1.0], vec![1])?,
    )?;
    let core_session = Core2ReferenceSession::prepare(grad_program.graph, core_constants)?;

    // The prepared lane uses an explicit scalar VJP seed. This avoids relying on
    // OnesLike lowering until that logical-kernel gap is implemented, while a
    // cotangent of one remains exactly equivalent to df/dx for scalar output.
    let vjp_program = MorphoDiff::vjp(&graph, output, &[x])?;
    let cotangent = vjp_program.seed_inputs[0];
    let vjp_report = vjp_program.report;
    let one = [1.0f32];
    let mut graph_constants = GraphConstants::new();
    graph_constants.bind(ConstantId::new(0), &one);
    let prepared_session = ReferenceJitSession::prepare(
        ReferencePlanRuntime::new(CpuComputeAdapter::new()),
        &vjp_program.graph,
        &graph_constants,
    )?;

    Ok(MorphoDiffFixture {
        core_session,
        prepared_session,
        x,
        y,
        cotangent,
        core_gradient,
        grad_report,
        vjp_report,
    })
}

fn core_inputs(
    x_node: NodeId,
    y_node: NodeId,
    x: f32,
    y: f32,
) -> Result<Core2Inputs, Box<dyn Error>> {
    let mut inputs = Core2Inputs::new();
    inputs.insert(x_node, Tensor::from_f32(vec![x], vec![1])?)?;
    inputs.insert(y_node, Tensor::from_f32(vec![y], vec![1])?)?;
    Ok(inputs)
}

fn morphodiff_core_dx(
    fixture: &MorphoDiffFixture,
    x: f32,
    y: f32,
) -> Result<f32, Box<dyn Error>> {
    let inputs = core_inputs(fixture.x, fixture.y, x, y)?;
    let outputs = fixture.core_session.execute(&inputs)?;
    let gradient = outputs
        .get(fixture.core_gradient)
        .ok_or("MorphoDiff Core2 gradient output is missing")?
        .to_f32_vec()?;
    Ok(gradient[0])
}

fn morphodiff_prepared_dx(
    fixture: &MorphoDiffFixture,
    x: f32,
    y: f32,
) -> Result<f32, Box<dyn Error>> {
    let x_value = [x];
    let y_value = [y];
    let cotangent = [1.0f32];
    let mut inputs = GraphInputs::new();
    inputs
        .bind(fixture.x, &x_value)
        .bind(fixture.y, &y_value)
        .bind(fixture.cotangent, &cotangent);
    let outputs = fixture.prepared_session.execute(&inputs)?;
    let gradient = outputs
        .values()
        .first()
        .ok_or("MorphoDiff prepared VJP output is missing")?;
    Ok(gradient.values[0])
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
        let morphodiff_core = morphodiff_core_dx(fixture, x, y)?;
        let morphodiff_prepared = morphodiff_prepared_dx(fixture, x, y)?;
        let tolerance = 2.0e-4 * (1.0 + analytic.abs());

        for (engine, value) in [
            ("MorphoDiff/Core2", morphodiff_core),
            ("MorphoDiff/prepared-VJP", morphodiff_prepared),
        ] {
            if (value - analytic).abs() > tolerance {
                return Err(format!(
                    "{engine} correctness gate failed at ({x}, {y}): {value} vs {analytic}"
                )
                .into());
            }
        }
    }
    Ok(())
}

fn bench_morphodiff_core(
    fixture: &MorphoDiffFixture,
    iterations: usize,
) -> Result<Duration, Box<dyn Error>> {
    let inputs = core_inputs(fixture.x, fixture.y, 3.0, 1.0)?;
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        let outputs = fixture.core_session.execute(black_box(&inputs))?;
        let gradient = outputs
            .get(fixture.core_gradient)
            .ok_or("MorphoDiff Core2 gradient output is missing")?
            .to_f32_vec()?;
        sink = black_box(sink + gradient[0]);
    }
    black_box(sink);
    Ok(start.elapsed())
}

fn bench_morphodiff_prepared(
    fixture: &MorphoDiffFixture,
    iterations: usize,
) -> Result<Duration, Box<dyn Error>> {
    let x = [3.0f32];
    let y = [1.0f32];
    let cotangent = [1.0f32];
    let mut inputs = GraphInputs::new();
    inputs
        .bind(fixture.x, &x)
        .bind(fixture.y, &y)
        .bind(fixture.cotangent, &cotangent);

    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        let outputs = fixture.prepared_session.execute(black_box(&inputs))?;
        let gradient = outputs
            .values()
            .first()
            .ok_or("MorphoDiff prepared VJP output is missing")?;
        sink = black_box(sink + gradient.values[0]);
    }
    black_box(sink);
    Ok(start.elapsed())
}

fn bench_morphodiff_transform(iterations: usize) -> Result<Duration, Box<dyn Error>> {
    let (graph, x, _, output) = rosenbrock_graph()?;
    let start = Instant::now();
    let mut generated_nodes = 0usize;
    for _ in 0..iterations {
        let differentiated = MorphoDiff::vjp(black_box(&graph), output, &[x])?;
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

    for _ in 0..100 {
        black_box(morphodiff_core_dx(&fixture, 3.0, 1.0)?);
        black_box(morphodiff_prepared_dx(&fixture, 3.0, 1.0)?);
    }

    let morphodiff_core = bench_morphodiff_core(&fixture, runtime_iterations)?;
    let morphodiff_prepared = bench_morphodiff_prepared(&fixture, runtime_iterations)?;
    let morphodiff_transform = bench_morphodiff_transform(transform_iterations)?;

    println!("benchmark=morphodiff-reference-v1");
    println!("dtype=f32");
    println!("correctness_gate=passed");
    println!("grad_source_nodes={}", fixture.grad_report.source_nodes);
    println!(
        "grad_transformed_nodes={}",
        fixture.grad_report.transformed_nodes
    );
    println!("grad_generated_nodes={}", fixture.grad_report.generated_nodes);
    println!("vjp_source_nodes={}", fixture.vjp_report.source_nodes);
    println!(
        "vjp_transformed_nodes={}",
        fixture.vjp_report.transformed_nodes
    );
    println!("vjp_generated_nodes={}", fixture.vjp_report.generated_nodes);
    println!(
        "prepared_kernel_count={}",
        fixture.prepared_session.compiled_kernel_count()
    );
    println!(
        "prepared_dispatch_count={}",
        fixture.prepared_session.dispatch_count()
    );
    println!("runtime_iterations={runtime_iterations}");
    println!("transform_iterations={transform_iterations}");
    println!(
        "morphodiff_core2_reference_ns_per_derivative={:.3}",
        ns_per_iteration(morphodiff_core, runtime_iterations)
    );
    println!(
        "morphodiff_prepared_reference_ns_per_derivative={:.3}",
        ns_per_iteration(morphodiff_prepared, runtime_iterations)
    );
    println!(
        "morphodiff_vjp_transform_ns_per_graph={:.3}",
        ns_per_iteration(morphodiff_transform, transform_iterations)
    );
    println!("morphodiff_prepared_lane=reference-jit-cpu-adapter");
    println!("morphodiff_native_codegen_lane=not-yet-implemented");
    println!("runtime_speed_verdict=not-comparable-until-native-codegen");

    Ok(())
}
