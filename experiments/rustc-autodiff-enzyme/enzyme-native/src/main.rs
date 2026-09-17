#![feature(autodiff)]

use std::autodiff::{autodiff_forward, autodiff_reverse};
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_RUNTIME_ITERS: usize = 100_000;
const DEFAULT_MATMUL_ITERS: usize = 2_000;
const MATMUL_DIM: usize = 16;
const MATMUL_ELEMENTS: usize = MATMUL_DIM * MATMUL_DIM;

#[autodiff_forward(rosenbrock_forward_x_f32, Dual, Const, Dual)]
fn rosenbrock_f32(x: f32, y: f32) -> f32 {
    let one_minus_x = 1.0 - x;
    let residual = y - x * x;
    one_minus_x * one_minus_x + 100.0 * residual * residual
}

fn enzyme_dx_f32(x: f32, y: f32) -> f32 {
    let (_value, derivative) = rosenbrock_forward_x_f32(x, 1.0, y);
    derivative
}

fn analytic_dx_f32(x: f32, y: f32) -> f32 {
    -2.0 * (1.0 - x) - 400.0 * x * (y - x * x)
}

/// Scalar objective matching MorphoDiff's all-one cotangent for a matrix output:
/// `L(A, B) = sum(A @ B)`. Only `A` is differentiated; `B` is constant.
#[autodiff_reverse(matmul_sum_reverse_a, Duplicated, Const, Active)]
fn matmul_sum(a: &[f32], b: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for row in 0..MATMUL_DIM {
        for col in 0..MATMUL_DIM {
            let mut cell = 0.0f32;
            for inner in 0..MATMUL_DIM {
                cell += a[row * MATMUL_DIM + inner] * b[inner * MATMUL_DIM + col];
            }
            total += cell;
        }
    }
    total
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

fn analytic_matmul_grad_a(b: &[f32]) -> Vec<f32> {
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

fn enzyme_matmul_grad_a(a: &[f32], b: &[f32], gradient: &mut [f32]) -> f32 {
    gradient.fill(0.0);
    matmul_sum_reverse_a(a, gradient, b, 1.0)
}

fn validate_rosenbrock_correctness() -> Result<(), Box<dyn Error>> {
    let points = [
        (1.0f32, 1.0f32),
        (3.0, 1.0),
        (-1.25, 0.75),
        (0.5, -0.25),
        (2.25, 4.0),
        (-0.75, 0.1),
    ];

    for (x, y) in points {
        let expected = analytic_dx_f32(x, y);
        let actual = enzyme_dx_f32(x, y);
        let tolerance = 2.0e-4 * (1.0 + expected.abs());
        if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > tolerance {
            return Err(format!(
                "Enzyme correctness gate failed at ({x}, {y}): {actual} vs {expected}"
            )
            .into());
        }
    }
    Ok(())
}

fn validate_matmul_correctness() -> Result<(), Box<dyn Error>> {
    let (a, b) = matrix_inputs();
    let expected = analytic_matmul_grad_a(&b);
    let mut actual = vec![0.0f32; MATMUL_ELEMENTS];
    let primal = enzyme_matmul_grad_a(&a, &b, &mut actual);
    let expected_primal = matmul_sum(&a, &b);
    let primal_tolerance = 2.0e-4 * (1.0 + expected_primal.abs());
    if !primal.is_finite()
        || !expected_primal.is_finite()
        || (primal - expected_primal).abs() > primal_tolerance
    {
        return Err(format!(
            "Enzyme MatMul primal gate failed: {primal} vs {expected_primal}"
        )
        .into());
    }

    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 5.0e-4 * (1.0 + want.abs());
        if !got.is_finite() || !want.is_finite() || (got - want).abs() > tolerance {
            return Err(format!(
                "Enzyme MatMul gradient gate failed at A[{index}]: {got} vs {want}"
            )
            .into());
        }
    }
    Ok(())
}

fn bench_rosenbrock(iterations: usize) -> Duration {
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        sink = black_box(sink + enzyme_dx_f32(black_box(3.0), black_box(1.0)));
    }
    black_box(sink);
    start.elapsed()
}

fn bench_matmul(iterations: usize) -> Duration {
    let (a, b) = matrix_inputs();
    let mut gradient = vec![0.0f32; MATMUL_ELEMENTS];
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        gradient.fill(0.0);
        let primal = matmul_sum_reverse_a(
            black_box(a.as_slice()),
            black_box(gradient.as_mut_slice()),
            black_box(b.as_slice()),
            black_box(1.0),
        );
        sink = black_box(sink + primal + gradient[0]);
    }
    black_box(sink);
    start.elapsed()
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
    let iterations = env_iterations("MORPHODIFF_BENCH_ITERS", DEFAULT_RUNTIME_ITERS);
    let matmul_iterations = env_iterations("MORPHODIFF_MATMUL_ITERS", DEFAULT_MATMUL_ITERS);
    validate_rosenbrock_correctness()?;
    validate_matmul_correctness()?;

    for _ in 0..1_000 {
        black_box(enzyme_dx_f32(3.0, 1.0));
    }
    let (a, b) = matrix_inputs();
    let mut gradient = vec![0.0f32; MATMUL_ELEMENTS];
    for _ in 0..10 {
        black_box(enzyme_matmul_grad_a(&a, &b, &mut gradient));
    }

    let elapsed = bench_rosenbrock(iterations);
    let matmul_elapsed = bench_matmul(matmul_iterations);
    println!("benchmark=morphodiff-enzyme-native-v2");
    println!("dtype=f32");
    println!("correctness_gate=passed");
    println!("runtime_iterations={iterations}");
    println!(
        "enzyme_native_runtime_ns_per_derivative={:.3}",
        ns_per_iteration(elapsed, iterations)
    );
    println!("matmul_dim={MATMUL_DIM}");
    println!("matmul_objective=sum(A@B)");
    println!("matmul_wrt=A");
    println!("matmul_iterations={matmul_iterations}");
    println!(
        "enzyme_native_matmul_reverse_with_shadow_reset_ns={:.3}",
        ns_per_iteration(matmul_elapsed, matmul_iterations)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enzyme_matches_analytic_rosenbrock_gradient() {
        validate_rosenbrock_correctness().unwrap();
    }

    #[test]
    fn enzyme_matches_analytic_matmul_gradient() {
        validate_matmul_correctness().unwrap();
    }
}
