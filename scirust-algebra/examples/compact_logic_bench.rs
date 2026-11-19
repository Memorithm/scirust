//! Same-host scalar-vs-bitplane GF(4) microbenchmark, not an AI throughput claim.
//!
//! Run on the required physical machine after qualification:
//! cargo run -p scirust-algebra --locked --release --example compact_logic_bench
//!
//! Preparation and exhaustive equivalence checks are excluded from timings.
//! The scalar baseline stores one u8 per GF(4) symbol; the packed candidate
//! stores two bitplanes (2 bits/symbol). Identical GF(4) products are reduced
//! into the same digest. Compiler flags, CPU identity and memory hierarchy
//! affect timings; do not extrapolate to full-model performance.

use scirust_algebra::compact_logic::QuadBitplanes;
use std::hint::black_box;
use std::time::Instant;

const BLOCKS: usize = 2_048;
const REPEATS: usize = 32;
const GF4_MUL: [[u8; 4]; 4] = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]];

fn next_word(seed: &mut u64) -> u64 {
    *seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    (*seed).wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn time_packed(packed_inputs: &[(QuadBitplanes, QuadBitplanes)]) -> (u128, u64) {
    let start = Instant::now();
    let mut digest = 0u64;
    for _ in 0..REPEATS
    {
        for &(a, b) in packed_inputs
        {
            let product = black_box(a).gf4_mul(black_box(b));
            digest = digest
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(black_box(product.low ^ product.high.rotate_left(1)));
        }
    }
    (start.elapsed().as_nanos(), black_box(digest))
}

fn time_scalar(scalar_inputs: &[([u8; 64], [u8; 64])]) -> (u128, u64) {
    let start = Instant::now();
    let mut digest = 0u64;
    for _ in 0..REPEATS
    {
        for (left, right) in scalar_inputs
        {
            // Two optimization barriers per block, matching the packed path.
            let left = black_box(left);
            let right = black_box(right);
            let mut low = 0u64;
            let mut high = 0u64;
            for lane in 0..64usize
            {
                let value = GF4_MUL[left[lane] as usize][right[lane] as usize];
                low |= u64::from(value & 1) << lane;
                high |= u64::from(value >> 1) << lane;
            }
            digest = digest
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(black_box(low ^ high.rotate_left(1)));
        }
    }
    (start.elapsed().as_nanos(), black_box(digest))
}

fn main() {
    assert_eq!(
        std::mem::size_of::<QuadBitplanes>(),
        16,
        "64 two-bit lanes must occupy exactly 16 bytes"
    );
    let mut seed = 0x5eed_d15c_8a47_2026_u64;
    let mut packed_inputs = Vec::with_capacity(BLOCKS);
    let mut scalar_inputs = Vec::with_capacity(BLOCKS);

    for _ in 0..BLOCKS
    {
        let a = QuadBitplanes {
            low: next_word(&mut seed),
            high: next_word(&mut seed),
        };
        let b = QuadBitplanes {
            low: next_word(&mut seed),
            high: next_word(&mut seed),
        };
        let mut left = [0u8; 64];
        let mut right = [0u8; 64];
        let product = a.gf4_mul(b);
        for lane in 0..64u32
        {
            let index = lane as usize;
            left[index] = a.get(lane).expect("lane within block");
            right[index] = b.get(lane).expect("lane within block");
            assert_eq!(
                product.get(lane).expect("lane within block"),
                GF4_MUL[left[index] as usize][right[index] as usize]
            );
        }
        packed_inputs.push((a, b));
        scalar_inputs.push((left, right));
    }

    // Alternating call order limits systematic thermal/frequency/cache-order
    // bias; seven separate samples give a median rather than a single sample.
    // The two kernels perform the same logical computation, including a
    // packed output digest. Host noise and scalar repacking still matter.
    const TRIALS: usize = 7;
    let mut packed_samples = [0u128; TRIALS];
    let mut scalar_samples = [0u128; TRIALS];
    let mut expected_digest = None;

    for trial in 0..TRIALS
    {
        let (packed, scalar) = if trial % 2 == 0
        {
            let packed = time_packed(&packed_inputs);
            let scalar = time_scalar(&scalar_inputs);
            (packed, scalar)
        }
        else
        {
            let scalar = time_scalar(&scalar_inputs);
            let packed = time_packed(&packed_inputs);
            (packed, scalar)
        };
        assert_eq!(packed.1, scalar.1, "product digest differs");
        if let Some(expected) = expected_digest
        {
            assert_eq!(packed.1, expected, "digest changed across trials");
        }
        expected_digest = Some(packed.1);
        packed_samples[trial] = packed.0;
        scalar_samples[trial] = scalar.0;
    }

    packed_samples.sort_unstable();
    scalar_samples.sort_unstable();
    let packed_ns = packed_samples[TRIALS / 2];
    let scalar_ns = scalar_samples[TRIALS / 2];
    let packed_min = packed_samples[0];
    let packed_max = packed_samples[TRIALS - 1];
    let scalar_min = scalar_samples[0];
    let scalar_max = scalar_samples[TRIALS - 1];
    let packed_digest = expected_digest.expect("nonzero trial count");

    let symbols = BLOCKS * 64 * REPEATS;
    let packed_input_bytes = 2 * BLOCKS * 16;
    let scalar_input_bytes = 2 * BLOCKS * 64;
    println!(
        "arch={} os={} blocks={} repeats={} evaluated_symbols={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        BLOCKS,
        REPEATS,
        symbols
    );
    println!(
        "input_bytes_packed={} input_bytes_scalar={} (excludes Vec and allocator overhead)",
        packed_input_bytes, scalar_input_bytes
    );
    println!(
        "gf4_mul_packed_median_ns={packed_ns} gf4_mul_scalar_median_ns={scalar_ns} digest={packed_digest}"
    );
    println!(
        "gf4_mul_packed_range_ns={packed_min}..{packed_max} gf4_mul_scalar_range_ns={scalar_min}..{scalar_max} trials={TRIALS}"
    );
    println!(
        "relative_kernel_median_ratio_scalar_over_packed={:.4} (same-host cache-resident microbenchmark including scalar repacking, not a model speedup)",
        scalar_ns as f64 / packed_ns as f64
    );
}
