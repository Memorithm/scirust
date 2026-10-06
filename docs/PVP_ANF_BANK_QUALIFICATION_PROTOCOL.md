# SciRust PVP explicit ANF-bank qualification v1

Frozen before implementation on 2026-10-06. This is deterministic functional
qualification of reusable CPU primitives, with no model or timing promotion.

## Source and destination

Source corpus: Memorithm/FLAT-ATTENTION PR #341, merge
`50c0230cc645bd44f1520e2c7194878a4b13009c`, protocol commit
`701a6c440fa002cd68ab00898c08aa33ad4861a5`.
Destination: `scirust-simd::pvp`, versioned u64 address-major layout.
This ports the declared formulas and independent oracle to SciRust's existing
primitives; it adds no FLAT binary or runtime dependency.

## Frozen corpus

Use `(K,G)` = `(1,1)`, `(2,31)`, `(4,65)`, `(8,129)`, `(64,257)`,
`(256,513)`, `(2048,257)`, `(256,2048)`, each with three banks (24 cases):

- `boundary`: gate modulo six selects zero, constant one, lowest-variable
  monomial (constant at K=1), full-degree monomial, constant XOR lowest-variable
  monomial, or parity of all variables. Duplicate terms cancel over GF(2).
- `sparse4`: `min(4,K)` terms per gate.
- `dense32`: `min(32,K)` terms per gate.

For sparse4/dense32, term mask is `(257*g + 73*t) mod K` in declared term order.
Reject duplicate masks. Term mask zero means constant one.

Independently evaluate each gate at every assignment x by XORing
`(x & mask) == mask`. Expected truth words must be constructed directly in
`[K,ceil(G/64)] u64` without invoking any transform, submask enumeration or
layout transpose. Check complete words including zero gate padding, then
transform again and require exact recovery of all coefficient words.

## Execution gates

- Explicitly qualify Scalar, AVX2, AVX-512 and NEON when runtime-available;
  include SVE only with nightly-simd and runtime SVE support.
- Report unavailable candidates separately; unavailable execution must return
  BackendUnavailable before any coefficient mutation.
- Check reported backend, zero scratch, exact storage/padding and update counts.
- Qualify auto dispatch on every frozen case against the same direct oracle.
- Dedicated ARM/QEMU gate requires NEON, and dedicated SVE gate requires SVE;
  these tests must assert availability instead of silently skipping. Execute
  them under the existing SVE-enabled QEMU configuration.
- Preserve K=1 zero-stage and short-row paths as well as vector-body/tail cases.

QEMU evidence establishes functional execution, never native CPU speed. This
does not establish FLAT/SciRust binary compatibility, register spill behavior,
optimal dispatch, model quality or SML internalization. SML remains autonomous;
its future harness stays separate. PVP has no vendor GPU software dependency.
