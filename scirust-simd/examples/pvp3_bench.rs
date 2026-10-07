//! PVP3 native CPU SIMD benchmark harness.
//!
//! This example is an execution tool, not a claim of performance. It keeps
//! the scalar result as the independent correctness gate and emits one
//! machine-readable record per candidate. The caller is responsible for
//! pinning the exact binary, CPU identity, and process-isolation protocol.

use scirust_simd::pvp::{
    PvpBackendV1, PvpBitplanesV1, PvpLayoutV1, direct_subset_zeta_oracle_v1,
    pascal_subset_zeta_scalar_in_place, pascal_subset_zeta_with_backend_in_place,
};
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: pvp3_bench --k K --g G [--backend scalar|avx2|avx512|neon|sve] [--warmups N] [--reps N]"
    );
    std::process::exit(2);
}

fn value(flag: &str, args: &[String]) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn parse_usize(flag: &str, args: &[String]) -> usize {
    value(flag, args)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| usage())
}

fn backend(args: &[String]) -> PvpBackendV1 {
    match value("--backend", args).as_deref().unwrap_or("scalar")
    {
        "scalar" => PvpBackendV1::Scalar,
        "avx2" => PvpBackendV1::Avx2,
        "avx512" => PvpBackendV1::Avx512,
        "neon" => PvpBackendV1::Neon,
        #[cfg(feature = "nightly-simd")]
        "sve" => PvpBackendV1::Sve,
        _ => usage(),
    }
}

fn fixture(layout: PvpLayoutV1) -> Vec<u64> {
    let words = layout.gate_major_storage_words().expect("checked layout");
    let mut result = vec![0_u64; words];
    for gate in 0..layout.gates()
    {
        for address in 0..layout.addresses()
        {
            let bit = ((gate as u64)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add((address as u64).wrapping_mul(0x85eb_ca6b))
                .rotate_left((gate as u32) & 31)
                ^ (address as u64))
                .count_ones()
                .is_multiple_of(3);
            if bit
            {
                let index = gate * layout.address_words_per_gate() + address / 64;
                result[index] |= 1_u64 << (address % 64);
            }
        }
    }
    result
}

fn checksum(words: &[u64]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325, |hash, word| {
        let mixed = hash ^ word.wrapping_mul(0x1000_0000_01b3);
        mixed.rotate_left(13).wrapping_mul(0x1000_0000_01b3)
    })
}

fn percentile(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    let index = ((sorted.len() - 1) * numerator).div_ceil(denominator);
    sorted[index]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let k = parse_usize("--k", &args);
    let g = parse_usize("--g", &args);
    let warmups = value("--warmups", &args)
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let reps = value("--reps", &args)
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    if reps == 0
    {
        usage();
    }

    let selected = backend(&args);
    if !selected.available()
    {
        eprintln!(
            "backend {} is unavailable on this process",
            selected.label()
        );
        std::process::exit(3);
    }

    let layout = PvpLayoutV1::new(k, g).expect("invalid PVP geometry");
    let source = fixture(layout);
    let mut scalar = PvpBitplanesV1::from_gate_major_words(layout, &source).expect("fixture");
    let scalar_stats = pascal_subset_zeta_scalar_in_place(&mut scalar).expect("scalar oracle");

    // Keep the independent direct oracle on small geometries, so the benchmark
    // cannot silently time a shared implementation bug.
    if k <= 64 && g <= 257
    {
        let direct = direct_subset_zeta_oracle_v1(layout, &source).expect("direct oracle");
        assert_eq!(scalar, direct, "scalar/direct oracle mismatch");
    }

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..warmups
    {
        let mut candidate =
            PvpBitplanesV1::from_gate_major_words(layout, &source).expect("fixture");
        let _ = black_box(
            pascal_subset_zeta_with_backend_in_place(&mut candidate, selected).expect("candidate"),
        );
        assert_eq!(candidate, scalar, "candidate/scalar mismatch");
    }
    for _ in 0..reps
    {
        let mut candidate =
            PvpBitplanesV1::from_gate_major_words(layout, &source).expect("fixture");
        let start = Instant::now();
        let stats =
            pascal_subset_zeta_with_backend_in_place(&mut candidate, selected).expect("candidate");
        let elapsed = start.elapsed().as_nanos();
        assert_eq!(candidate, scalar, "candidate/scalar mismatch");
        assert_eq!(
            stats.logical_gate_xor_ops,
            scalar_stats.logical_gate_xor_ops
        );
        assert_eq!(stats.packed_word_updates, scalar_stats.packed_word_updates);
        black_box(candidate);
        samples.push(elapsed);
    }

    samples.sort_unstable();
    let median = percentile(&samples, 1, 2);
    let p95 = percentile(&samples, 95, 100);
    println!(
        "{{\"contract\":\"scirust-pvp3-native-bench/v1\",\"backend\":\"{}\",         \"k\":{},\"g\":{},\"warmups\":{},\"reps\":{},         \"median_ns\":{},\"p95_ns\":{},\"checksum\":\"{:016x}\",         \"storage_words\":{},\"storage_bits\":{},\"padding_bits\":{},         \"scratch_words\":0,\"oracle\":\"scalar-plus-direct-small\",         \"performance_claim\":false}}",
        selected.label(),
        k,
        g,
        warmups,
        reps,
        median,
        p95,
        checksum(scalar.words()),
        layout.storage_words(),
        layout.storage_bits(),
        layout.padding_bits(),
    );
}
