//! End-to-end calibration: SciRust FEM oracle -> trained FNO surrogate -> measured benchmark.
//!
//! This example is intentionally a calibration, not a speed claim. The 1-D
//! tridiagonal FEM oracle is already O(N) and can legitimately be faster than a
//! small FNO. More expensive PDE solvers are the later target.

use std::time::Instant;

use scirust_neural_operator::{
    BenchmarkConfig, Fno1dConfig, Fno1dOperator, LearnedOperator, OperatorDataset1d,
    OperatorSample1d, benchmark_exact_vs_surrogate, relative_l2,
};
use scirust_solvers::scientific::FemSolver1D;

fn encoded_input(source: f32, points: usize, length: f32) -> Vec<f32> {
    let mut input = Vec::with_capacity(points * 2);
    for index in 0..points
    {
        let x = length * index as f32 / (points - 1) as f32;
        input.push(source);
        input.push(x);
    }
    input
}

fn fem_target(source: f32, points: usize, length: f64) -> Vec<f32> {
    FemSolver1D::new(points, length)
        .solve_steady_heat(source as f64)
        .expect("fixed benchmark FEM inputs are valid")
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn dataset(sources: &[f32], points: usize, length: f32) -> OperatorDataset1d {
    let samples = sources
        .iter()
        .map(|&source| {
            OperatorSample1d::new(
                encoded_input(source, points, length),
                fem_target(source, points, length as f64),
            )
            .unwrap()
        })
        .collect();
    OperatorDataset1d::new(samples, points, 2, 1).unwrap()
}

fn main() {
    let points = 32usize;
    let length = 1.0f32;
    let training = dataset(&[0.5, 1.0, 1.5, 2.0, 3.0, 4.0], points, length);
    let probe_source = 2.5f32;
    let probe_input = encoded_input(probe_source, points, length);
    let probe_target = fem_target(probe_source, points, length as f64);

    let mut operator = Fno1dOperator::new(Fno1dConfig {
        points,
        in_channels: 2,
        out_channels: 1,
        hidden_channels: 16,
        modes: 8,
        seed: 20260917,
    })
    .unwrap();

    let before = relative_l2(&operator.predict(&probe_input).unwrap(), &probe_target).unwrap();
    let training_start = Instant::now();
    let fit = operator.fit(&training, 250, 0.01).unwrap();
    let training_seconds = training_start.elapsed().as_secs_f64();
    let after = relative_l2(&operator.predict(&probe_input).unwrap(), &probe_target).unwrap();

    let report = benchmark_exact_vs_surrogate(
        &mut operator,
        &probe_input,
        BenchmarkConfig {
            warmup_calls: 10,
            repeats: 1000,
        },
        |input| {
            let source = input[0];
            Ok(fem_target(source, points, length as f64))
        },
    )
    .unwrap();
    let economics = report.economics(training_seconds).unwrap();

    println!("training_first_mse={:.9}", fit.first_mse);
    println!("training_last_mse={:.9}", fit.last_mse);
    println!("probe_relative_l2_before={before:.9}");
    println!("probe_relative_l2_after={after:.9}");
    println!("exact_mean_seconds={:.9}", report.exact_timing.mean_seconds);
    println!(
        "surrogate_mean_seconds={:.9}",
        report.surrogate_timing.mean_seconds
    );
    println!(
        "measured_online_speedup={:.6}",
        report.measured_online_speedup()
    );
    println!(
        "benchmark_mean_relative_l2={:.9}",
        report.accuracy.mean_relative_l2
    );
    println!("training_seconds={training_seconds:.6}");
    println!(
        "break_even_evaluations={:?}",
        economics.break_even_evaluations()
    );
}
