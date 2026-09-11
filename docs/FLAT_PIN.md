# FLAT dependency pin

SciRust consumes FLAT through optional `flat-autotune` features.

Observed 2026-09-12 on `master` (`0e2eacc`):

```
elastic-core            git+https://github.com/Memorithm/ElasticXxx.git  rev = 9130a412857335cc5120b013b91552dd0808f9f1
flat-attention-planner  git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 75d3bd684643aedb98f55a892f93d727a8187cea
flat-elastic-kernel     git+https://github.com/Memorithm/FLAT-ATTENTION  rev = 75d3bd684643aedb98f55a892f93d727a8187cea
```

FLAT default tip at the same date: `4529a2079434965e13e90ddd2e98ecc88ee0cb3a` (already used by NoiseLab).

## Rule

One SHA in flight across the org. Do **not** edit the `rev` values in root `Cargo.toml` in this commit. The bump is a separate PR that must pass:

```
cargo check -p scirust --features flat-autotune
```

WGPU must stay off on the newer FLAT revision in this crate (existing comment in `Cargo.toml`).
