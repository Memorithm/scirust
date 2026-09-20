# FLAT dependency pin

SciRust consumes FLAT through optional `flat-autotune` features.

Qualified bump target:

```
memorithm-elastic       git+https://github.com/Memorithm/ElasticXxx.git  rev = 354cfb372f568338b29a357b0671bf9315097b1d
flat-attention-planner  git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 7a5db9127bd9b76f6f4e58a47a371658f0c8f5e5
flat-elastic-kernel     git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 7a5db9127bd9b76f6f4e58a47a371658f0c8f5e5
```

The qualified FLAT revision is `7a5db9127bd9b76f6f4e58a47a371658f0c8f5e5`; its `flat-elastic-kernel` bridge is pinned to merged ElasticXxx `354cfb372f568338b29a357b0671bf9315097b1d`. SciRust consumes the same Elastic generation through the public `memorithm-elastic` facade.

## Rule

One SHA in flight across the org. The bump is accepted only when this revision passes:

```
cargo check -p scirust --features flat-autotune
```

WGPU must stay off on the newer FLAT revision in this crate (existing comment in `Cargo.toml`).

## Qualification

Issue #1424 is resolved by the dedicated pin-bump PR after `cargo check -p scirust --features flat-autotune` passes on the exact candidate head. `flat-autotune` continues to depend on FLAT with `default-features = false` for the planner dependency; this crate does not opt into FLAT WGPU features through that path.
