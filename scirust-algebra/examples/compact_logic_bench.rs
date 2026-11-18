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
const GF4_MUL: [[u8; 4]; 4] = [
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
];

fn next_word(seed: &mut u64) -> u64 {
    *seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    seed.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn main() {
    let mut seed = 0x5eed_d15c_8a47_2026_u64;
    let mut packed_inputs = Vec::with_capacity(BLOCKS);
    let mut scalar_inputs = Vec::with_capacity(BLOCKS);

    for _ in 0..BLOCKS {
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
        for lane in 0..64u32 {
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

    let packed_start = Instant::now();
    let mut packed_digest = 0u64;
    for _ in 0..REPEATS {
        for &(a, b) in &packed_inputs {
            let product = black_box(a).gf4_mul(black_box(b));
            packed_digest ^= black_box(product.low ^ product.high.rotate_left(1));
        }
    }
    let packed_elapsed = packed_start.elapsed();

    let scalar_start = Instant::now();
    let mut scalar_digest = 0u64;
    for _ in 0..REPEATS {
        for (left, right) in &scalar_inputs {
            let mut low = 0u64;
            let mut high = 0u64;
            for lane in 0..64usize {
                let value = GF4_MUL[black_box(left[lane]) as usize]
                    [black_box(right[lane]) as usize];
                low |= u64::from(value & 1) << lane;
                high |= u64::from(value >> 1) << lane;
            }
            scalar_digest ^= black_box(low ^ high.rotate_left(1));
        }
    }
    let scalar_elapsed = scalar_start.elapsed();
    assert_eq!(packed_digest, scalar_digest, "product parity differs");

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
        "gf4_mul_packed_ns={} gf4_mul_scalar_ns={} digest={packed_digest}",
        packed_elapsed.as_nanos(),
        scalar_elapsed.as_nanos()
    );
    println!(
        "relative_kernel_ratio_scalar_over_packed={:.4} (same-host microbenchmark, not a model speedup)",
        scalar_elapsed.as_secs_f64() / packed_elapsed.as_secs_f64()
    );
}
