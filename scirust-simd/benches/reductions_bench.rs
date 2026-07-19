use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group};
use scirust_simd::reductions::{
    cosine_similarity_f32, dot_f32_deterministic, dot_f32_fast, l1_norm_f32, l2_norm_f32,
    sum_f32_deterministic, sum_f32_fast, sum_f32_kahan,
};

const SIZES: &[usize] = &[128, 1_024, 16_384, 262_144];

fn make_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn bench_sums(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions_sum");

    for &size in SIZES
    {
        let values = make_values(size, 0x51A5_0000 ^ size as u64);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("fast", size), &values, |b, values| {
            b.iter(|| black_box(sum_f32_fast(black_box(values))))
        });

        group.bench_with_input(
            BenchmarkId::new("deterministic", size),
            &values,
            |b, values| b.iter(|| black_box(sum_f32_deterministic(black_box(values)))),
        );

        group.bench_with_input(BenchmarkId::new("kahan", size), &values, |b, values| {
            b.iter(|| black_box(sum_f32_kahan(black_box(values))))
        });
    }

    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions_dot");

    for &size in SIZES
    {
        let lhs = make_values(size, 0xD07A_0000 ^ size as u64);
        let rhs = make_values(size, 0xD07B_0000 ^ size as u64);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("fast", size),
            &(&lhs, &rhs),
            |b, (lhs, rhs)| b.iter(|| black_box(dot_f32_fast(black_box(lhs), black_box(rhs)))),
        );

        group.bench_with_input(
            BenchmarkId::new("deterministic", size),
            &(&lhs, &rhs),
            |b, (lhs, rhs)| {
                b.iter(|| black_box(dot_f32_deterministic(black_box(lhs), black_box(rhs))))
            },
        );
    }

    group.finish();
}

fn bench_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions_norms");

    for &size in SIZES
    {
        let lhs = make_values(size, 0xA110_0000 ^ size as u64);
        let rhs = make_values(size, 0xA111_0000 ^ size as u64);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("l1", size), &lhs, |b, values| {
            b.iter(|| black_box(l1_norm_f32(black_box(values))))
        });

        group.bench_with_input(BenchmarkId::new("l2", size), &lhs, |b, values| {
            b.iter(|| black_box(l2_norm_f32(black_box(values))))
        });

        group.bench_with_input(
            BenchmarkId::new("cosine", size),
            &(&lhs, &rhs),
            |b, (lhs, rhs)| {
                b.iter(|| black_box(cosine_similarity_f32(black_box(lhs), black_box(rhs))))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sums, bench_dot, bench_norms);

fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
}
