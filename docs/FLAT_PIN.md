# FLAT dependency pin

SciRust consumes FLAT through optional `flat-autotune` features.

Qualified bump target:

```
elastic                  git+https://github.com/Memorithm/ElasticXxx.git  rev = 36e1b73022d6bf3f0bc4cd6ae40ee1bc95e468b6
flat-attention-planner  git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 107854e0962979967c3425dd16de0c1c6efe2d8e
flat-elastic-kernel     git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 107854e0962979967c3425dd16de0c1c6efe2d8e
```

The target revision is `107854e0962979967c3425dd16de0c1c6efe2d8e`, previously identified by this tracker and already used by NoiseLab at the time the issue was opened.

## Rule

One SHA in flight across the org. The bump is accepted only when this revision passes:

```
cargo check -p scirust --features flat-autotune
```

WGPU must stay off on the newer FLAT revision in this crate (existing comment in `Cargo.toml`).

## Qualification

Issue #1424 is resolved by the dedicated pin-bump PR after `cargo check -p scirust --features flat-autotune` passes on the exact candidate head. `flat-autotune` continues to depend on FLAT with `default-features = false` for the planner dependency; this crate does not opt into FLAT WGPU features through that path.


## 2026-09-20 contract migration

The contextual rail now consumes the public Elastic facade package
`memorithm-elastic` at exact source `36e1b73022d6bf3f0bc4cd6ae40ee1bc95e468b6`. FLAT is pinned to
`107854e0962979967c3425dd16de0c1c6efe2d8e`, whose `flat-elastic-kernel` uses the same ElasticXxx
generation. This avoids two incompatible generations of Elastic freshness types
in one SciRust feature graph.

This is a host-only advisory dependency migration. It does not enable WGPU,
change attention semantics, or authorize execution of the contextual plan.
