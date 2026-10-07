# PVP3 native CPU benchmark protocol

Status: harness merged only after exact-head CI; no performance result is implied by
the source change.

The `scirust-simd/examples/pvp3_bench.rs` executable measures one selected
`PvpBackendV1` over the canonical address-major PVP transform. For every
candidate it:

1. constructs one deterministic gate-major input for the requested `(K, G)`;
2. computes the scalar transform once and, for small cases, checks it with the
   independent direct subset oracle;
3. runs five warmups by default;
4. clones the same input before each timed transform and records 30 samples by
   default;
5. rejects the candidate before emitting a record if its complete output differs
   from the scalar output;
6. emits one JSON object containing the contract version, backend label,
   geometry, median/p95 nanoseconds, checksum, storage accounting, and the
   explicit `performance_claim: false` boundary.

Example:

    cargo run --release -p scirust-simd --example pvp3_bench -- \
      --k 1024 --g 256 --backend avx2 --warmups 5 --reps 30

The measurement must be repeated on a fixed native host with the exact source
revision, Rust toolchain, target features, CPU identity, process isolation and
candidate order recorded by the caller. Hosted CI and QEMU are correctness
evidence only. A benchmark result cannot promote a default backend, an SML
architecture, a GPU claim, or a model-quality claim.

The deterministic generator is a harness fixture. A cross-repository campaign
using SML's frozen PVP6 corpus must replace it with the versioned fixture
adapter and retain the fixture SHA and the complete word-output oracle in its
evidence bundle.
