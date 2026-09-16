use std::collections::BTreeMap;
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use morphodiff_native_cpu::compile_scalar_f32_2_to_1;
use scirust_tensor_ir::{
    ConstantId, DType, Graph, MorphoDiff, Operation, Scalar, Shape, TensorType,
};

const DEFAULT_RUNTIME_ITERS: usize = 1_000_000;
const DEFAULT_TRANSFORM_ITERS: usize = 10_000;
const DEFAULT_COMPILE_ITERS: usize = 100;

fn scalar_type() -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![1]))
}

fn rosenbrock_graph() -> Result<(Graph, scirust_tensor_ir::NodeId, scirust_tensor_ir::NodeId), Box<dyn Error>> {
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
    Ok((graph, x, output))
}

fn analytic_dx(x: f32, y: f32) -> f32 {
    -2.0 * (1.0 - x) - 400.0 * x * (y - x * x)
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
    let runtime_iterations = env_iterations("MORPHODIFF_NATIVE_ITERS", DEFAULT_RUNTIME_ITERS);
    let transform_iterations = env_iterations("MORPHODIFF_NATIVE_TRANSFORM_ITERS", DEFAULT_TRANSFORM_ITERS);
    let compile_iterations = env_iterations("MORPHODIFF_NATIVE_COMPILE_ITERS", DEFAULT_COMPILE_ITERS);
    let (graph, x, output) = rosenbrock_graph()?;
    let mut constants = BTreeMap::new();
    constants.insert(ConstantId::new(0), 1.0);

    let transform_start = Instant::now();
    let mut generated_nodes = 0usize;
    for _ in 0..transform_iterations {
        let program = MorphoDiff::grad(black_box(&graph), output, &[x])?;
        generated_nodes = black_box(generated_nodes + program.report.generated_nodes);
    }
    let transform_elapsed = transform_start.elapsed();

    let program = MorphoDiff::grad(&graph, output, &[x])?;
    let report = program.report;

    let compile_start = Instant::now();
    for _ in 0..compile_iterations {
        black_box(compile_scalar_f32_2_to_1(&program.graph, &constants)?);
    }
    let compile_elapsed = compile_start.elapsed();

    let native = compile_scalar_f32_2_to_1(&program.graph, &constants)?;
    for (x_value, y_value) in [
        (1.0f32, 1.0f32),
        (3.0, 1.0),
        (-1.25, 0.75),
        (0.5, -0.25),
        (2.25, 4.0),
        (-0.75, 0.1),
    ] {
        let expected = analytic_dx(x_value, y_value);
        let actual = native.call(x_value, y_value);
        let tolerance = 2.0e-4 * (1.0 + expected.abs());
        if (actual - expected).abs() > tolerance {
            return Err(format!(
                "native correctness gate failed at ({x_value}, {y_value}): {actual} vs {expected}"
            )
            .into());
        }
    }

    let entry = black_box(native.entry());
    let mut sink = 0.0f32;
    for _ in 0..10_000 {
        sink = black_box(sink + entry(black_box(3.0), black_box(1.0)));
    }
    black_box(sink);

    let runtime_start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..runtime_iterations {
        sink = black_box(sink + entry(black_box(3.0), black_box(1.0)));
    }
    black_box(sink);
    let runtime_elapsed = runtime_start.elapsed();

    println!("benchmark=morphodiff-native-cpu-v1");
    println!("backend=cranelift-0.135.2");
    println!("dtype=f32");
    println!("correctness_gate=passed");
    println!("source_nodes={}", report.source_nodes);
    println!("transformed_nodes={}", report.transformed_nodes);
    println!("generated_nodes={}", report.generated_nodes);
    println!("runtime_iterations={runtime_iterations}");
    println!("transform_iterations={transform_iterations}");
    println!("compile_iterations={compile_iterations}");
    println!(
        "morphodiff_transform_ns_per_graph={:.3}",
        ns_per_iteration(transform_elapsed, transform_iterations)
    );
    println!(
        "morphodiff_cranelift_compile_ns_per_graph={:.3}",
        ns_per_iteration(compile_elapsed, compile_iterations)
    );
    println!(
        "morphodiff_native_runtime_ns_per_derivative={:.3}",
        ns_per_iteration(runtime_elapsed, runtime_iterations)
    );

    Ok(())
}
