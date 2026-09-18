# FLAT dependency pin

SciRust consumes FLAT through optional `flat-autotune` features.

Qualified bump target:

```
elastic-core            git+https://github.com/Memorithm/ElasticXxx.git  rev = 9130a412857335cc5120b013b91552dd0808f9f1
flat-attention-planner  git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 4529a2079434965e13e90ddd2e98ecc88ee0cb3a
flat-elastic-kernel     git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 4529a2079434965e13e90ddd2e98ecc88ee0cb3a
```

The target revision is `4529a2079434965e13e90ddd2e98ecc88ee0cb3a`, previously identified by this tracker and already used by NoiseLab at the time the issue was opened.

## Rule

One SHA in flight across the org. The bump is accepted only when this revision passes:

```
cargo check -p scirust --features flat-autotune
```

WGPU must stay off on the newer FLAT revision in this crate (existing comment in `Cargo.toml`).

## Qualification

Issue #1424 is resolved by the dedicated pin-bump PR after `cargo check -p scirust --features flat-autotune` passes on the exact candidate head. `flat-autotune` continues to depend on FLAT with `default-features = false` for the planner dependency; this crate does not opt into FLAT WGPU features through that path.
