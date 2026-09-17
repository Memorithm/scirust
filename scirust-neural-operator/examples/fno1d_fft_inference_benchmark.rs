//! Measured dense-DFT vs radix-2 FFT inference on the exact same FNO snapshot.

use std::hint::black_box;
use std::time::Instant;

use scirust_neural_operator::{
    FftInferenceMode, Fno1dConfig, Fno1dFftInference, Fno1dOperator, LearnedOperator,
    SpectralModePlan, relative_l2,
};

fn mean_seconds<F>(warmups: usize, repeats: usize, mut call: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmups
    {
        call();
    }
    let start = Instant::now();
    for _ in 0..repeats
    {
        call();
    }
    start.elapsed().as_secs_f64() / repeats as f64
}

fn main() {
    let cfg = Fno1dConfig {
        points: 256,
        in_channels: 4,
        out_channels: 4,
        hidden_channels: 32,
        modes: 32,
        seed: 20260917,
    };
    let mut dense = Fno1dOperator::new(cfg).unwrap();
    let snapshot = dense.inference_snapshot();
    let mut fast = Fno1dFftInference::new(snapshot.clone(), FftInferenceMode::Fast).unwrap();
    let mut portable = Fno1dFftInference::new(snapshot, FftInferenceMode::Portable).unwrap();
    let input: Vec<f32> = (0..dense.input_len())
        .map(|index| ((index as f32) * 0.0137 - 1.2).sin() + 0.2 * ((index as f32) * 0.031).cos())
        .collect();

    let dense_output = dense.predict(&input).unwrap();
    let fast_output = fast.predict(&input).unwrap();
    let portable_output = portable.predict(&input).unwrap();
    println!(
        "fast_relative_l2={:.9}",
        relative_l2(&fast_output, &dense_output).unwrap()
    );
    println!(
        "portable_relative_l2={:.9}",
        relative_l2(&portable_output, &dense_output).unwrap()
    );

    let warmups = 20usize;
    let repeats = 1000usize;
    let dense_mean = mean_seconds(warmups, repeats, || {
        black_box(dense.predict(black_box(&input)).unwrap());
    });
    let fast_mean = mean_seconds(warmups, repeats, || {
        black_box(fast.predict(black_box(&input)).unwrap());
    });
    let portable_mean = mean_seconds(warmups, repeats, || {
        black_box(portable.predict(black_box(&input)).unwrap());
    });

    println!("dense_dft_mean_seconds={dense_mean:.9}");
    println!("fast_fft_mean_seconds={fast_mean:.9}");
    println!("portable_fft_mean_seconds={portable_mean:.9}");
    println!("fast_vs_dense_speedup={:.6}", dense_mean / fast_mean);
    println!(
        "portable_vs_dense_speedup={:.6}",
        dense_mean / portable_mean
    );

    let sparse_plan = SpectralModePlan::new(cfg.modes, (0..8).collect()).unwrap();
    let dense_sparse_output = dense.predict_with_mode_plan(&input, &sparse_plan).unwrap();
    let fft_sparse_output = fast.predict_with_mode_plan(&input, &sparse_plan).unwrap();
    println!(
        "sparse_fast_relative_l2={:.9}",
        relative_l2(&fft_sparse_output, &dense_sparse_output).unwrap()
    );
    let dense_sparse_mean = mean_seconds(warmups, repeats, || {
        black_box(
            dense
                .predict_with_mode_plan(black_box(&input), &sparse_plan)
                .unwrap(),
        );
    });
    let fft_sparse_mean = mean_seconds(warmups, repeats, || {
        black_box(
            fast.predict_with_mode_plan(black_box(&input), &sparse_plan)
                .unwrap(),
        );
    });
    println!("dense_8_of_32_mean_seconds={dense_sparse_mean:.9}");
    println!("fast_fft_8_of_32_mean_seconds={fft_sparse_mean:.9}");
    println!(
        "sparse_fft_vs_dense_speedup={:.6}",
        dense_sparse_mean / fft_sparse_mean
    );
    println!(
        "fft_sparse_vs_fft_full_speedup={:.6}",
        fast_mean / fft_sparse_mean
    );
}
