#![feature(autodiff)]

use std::autodiff::autodiff_forward;
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_RUNTIME_ITERS: usize = 100_000;

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

fn validate_correctness() -> Result<(), Box<dyn Error>> {
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
        if (actual - expected).abs() > tolerance {
            return Err(format!(
                "Enzyme correctness gate failed at ({x}, {y}): {actual} vs {expected}"
            )
            .into());
        }
    }
    Ok(())
}

fn bench(iterations: usize) -> Duration {
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        sink = black_box(sink + enzyme_dx_f32(black_box(3.0), black_box(1.0)));
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
    validate_correctness()?;

    for _ in 0..1_000 {
        black_box(enzyme_dx_f32(3.0, 1.0));
    }

    let elapsed = bench(iterations);
    println!("benchmark=morphodiff-enzyme-native-v1");
    println!("dtype=f32");
    println!("correctness_gate=passed");
    println!("runtime_iterations={iterations}");
    println!(
        "enzyme_native_runtime_ns_per_derivative={:.3}",
        ns_per_iteration(elapsed, iterations)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enzyme_matches_analytic_rosenbrock_gradient() {
        validate_correctness().unwrap();
    }
}
