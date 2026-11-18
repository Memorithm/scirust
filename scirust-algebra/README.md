# scirust-algebra

`scirust-algebra` is SciRust's deterministic, pure-Rust algebra and group-theory layer.
It is designed around fixed-size values, caller-owned workspaces and static dispatch so
hot paths do not require heap allocation or dynamic dispatch.

## Compatibility

The crate follows the root SciRust workspace baseline: Rust 1.89, edition 2021,
`#![forbid(unsafe_code)]`, and no external dependencies. The reference kernels are kept
portable across the stable x86_64, ARM64 and MSRV validation lanes used by the workspace.
The root lockfile includes the crate as a dependency-free workspace package so standard
`--locked` CI validation can build it without re-resolving the dependency graph.

## Architecture

- `core`: algebraic law traits (`Magma`, `Semigroup`, `Monoid`, `Group`,
  `AbelianGroup`, `Ring`, `Field`, `LieAlgebra`), quotient representatives, direct
  products, semidirect products and group actions.
- `discrete`: compact fixed-degree permutations, cycle/signature operations,
  deterministic finite-group enumeration/reference membership, a public automatic
  Schreier-Sims facade, and free-group word reduction/ordered rewriting.
- `schreier`: fixed-storage orbit/transversal construction, deterministic Schreier
  generator production, strong-generator completion, stabilizer-chain sifting,
  orbit-stabilizer order and exact membership.
- `presented`: fixed-capacity deterministic Todd-Coxeter coset enumeration with compact
  inverse-paired tables and union-find coincidence handling.
- `representation`: fixed matrices, character tables, finite-group irrep projectors,
  reference group Fourier transform and spherical-harmonic primitives.
- `lie`: `so(3)`, `SO(3)`, `se(3)`, `SE(3)`, the `SU(2)` quaternion model,
  second-order BCH and fixed-storage Clifford algebras `Cl(p,q)`.
- `equivariant`: statically typed representation actions, typed irrep labels,
  Clebsch-Gordan contractions and invariant bilinear contractions.

## Performance policy

The core value types use arrays and `const` generics. Algorithms whose state can grow
accept storage from the caller or use caller-selected compile-time capacities. The crate
has no dependencies and forbids `unsafe`. This makes the reference kernels deterministic
and suitable as correctness oracles for future SIMD/GPU specialisations.

## Algorithmic status

Permutation groups now have an automatic deterministic Schreier-Sims path. Starting from
ordinary generators, SciRust builds orbit transversals, produces Schreier generators,
sifts residues into deeper stabilizers and inserts missing strong generators until the
natural-base chain closes. The resulting BSGS provides exact order and membership without
full subgroup enumeration. Storage is bounded by the caller-selected strong-generator
capacity, and exhaustion is reported explicitly.

`PermutationGroup::enumerate_into` remains available intentionally as a small-group
reference oracle. `group_fourier` is still the direct finite-group transform rather than a
subgroup-factorised non-abelian FFT for `S_n`. Knuth-Bendix completion, general `SU(N)`,
Wigner-D matrices and algorithmic Clebsch-Gordan coefficient generation remain follow-up
implementation work.

## Example

```rust
use scirust_algebra::discrete::{Permutation, PermutationGroup, SchreierSims};

let swap = Permutation::new([1, 0, 2]).unwrap();
let cycle = Permutation::new([1, 2, 0]).unwrap();
let group = PermutationGroup::new([swap, cycle]);
let ss = SchreierSims::<3, 2, 8>::build(group).unwrap();
assert_eq!(ss.order(), Some(6));
assert!(ss.contains(&Permutation::new([0, 2, 1]).unwrap()));
```

## Packed binary/quaternary reference (research)

The `compact_logic` module adds a CPU-portable, allocation-free reference
for **64 independently addressed 2-bit lanes in 16 bytes**, represented as
two `u64` bitplanes. It provides exact lane get/set, mask-based equality,
conditional replacement and selection, lane-wise GF(4) XOR/addition and
multiplication (with `alpha² = alpha + 1`), and masked GF(4) parity.
Four symbolic labels do **not** intrinsically imply GF(4) arithmetic:
callers must opt in to the mathematical field interpretation.

A *separate two-bit GF(4) block parity* is also available: compute one
GF(4) sum over 64 packed symbols and retain the parity externally (130
total bits per protected 128-bit block: 1.5625% storage overhead when
amortized at this block size). It detects every single-symbol substitution.
If an **independent detector already supplies the exact erased lane**,
`recover_known_erasure` reconstructs that symbol without modifying any
others. It cannot locate unknown errors and cannot reliably detect multiple
faults (two equal substitutions can cancel). The parity must itself remain
intact; no authentication or adversarial resistance is implied.

A separately opt-in, extended Hamming **SECDED (8,4)** path encodes one
four-bit payload into one byte, correcting one bit error and detecting
all two-bit errors. Its batch admission helper takes an explicit
64-lane mask, never writes unselected or rejected output entries, and
reports accepted, corrected and rejected lane masks. This redundancy
cost is **8 bits per 4 payload bits**, and is not hidden inside the dense
2-bit state representation. Three or more bit flips may be miscorrected
or undetected. No adversarial integrity, authenticated storage,
biological fidelity, quantum capability or numerical speedup is claimed.

Example (a 64-lane block is a caller-owned value):

```rust
use scirust_algebra::compact_logic::{
    QuadBitplanes, DecodedNibble, secded_encode, secded_decode,
};

let mut state = QuadBitplanes::default();
state.set(7, 3).unwrap();
let active = state.matches(3).unwrap();
assert_eq!(active, 1 << 7);
assert_eq!(state.select(active).get(7), Ok(3));

let encoded = secded_encode(0b1011).unwrap();
assert_eq!(
    secded_decode(encoded ^ 1),
    Ok(DecodedNibble::Corrected {
        value: 0b1011,
        bit_position: 1,
    })
);
```

Source qualification: `cargo test -p scirust-algebra --locked compact_logic`.
A future SIMD implementation or integration into V888/SML/FLAT must be
tested against this scalar reference on the exact consumer workload,
including end-to-end memory traffic and runtime cost. No implicit model
behavior or feature is changed by adding this isolated module.
