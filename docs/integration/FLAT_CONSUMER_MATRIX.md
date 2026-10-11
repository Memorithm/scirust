# FLAT consumer compatibility matrix

SciRust deliberately consumes two FLAT revisions for different contracts. They
are not interchangeable pins.

| Consumer | Feature | FLAT revision | WGPU contract | Qualification |
| --- | --- | --- | --- | --- |
| scirust-gpu runtime bridge | flat-attention | c74a3806f8cfe621671694692fc14adfe08116f0 | wgpu 0.20; optimized runtime results remain oracle-checked | sciagent-flat-decode.yml and the matrix workflow |
| root contextual advisory | flat-autotune | 4529a2079434965e13e90ddd2e98ecc88ee0cb3a | host-only planner; FLAT's wgpu 30 must not enter the active graph | flat-elastic-contextual.yml and the matrix workflow |

The contextual path also consumes elastic-core at
9130a412857335cc5120b013b91552dd0808f9f1. The machine-readable source of
truth is [flat-consumer-matrix-v1.json](flat-consumer-matrix-v1.json), checked
against both manifests, the committed lockfile and the feature-resolved Cargo
graphs.

## Candidate migration gate

A candidate revision is not adopted merely because it compiles. Record its
revision, packages, enabled features, WGPU version and MSRV; test every affected
consumer on that exact candidate; retain the independent SciRust reference
oracle; and keep runtime and contextual graphs separate until one type universe
is intentionally qualified. Lavapipe establishes portable WGPU correctness, not
physical-GPU performance.
