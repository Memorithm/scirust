# FLAT M33 — exact-SHA physical qualification record (unfilled)

Status: **NOT QUALIFIED**. This is a reproducibility and rejection template, not evidence of a successful physical run. Do not promote a FLAT release candidate or claim speedup from this file alone.

## Immutable identities (required)

- SciRust commit (full 40-hex SHA): **UNVERIFIED**
- SciRust Cargo.lock SHA-256: **UNVERIFIED**
- FLAT dependency commit from resolved Cargo.lock (full 40-hex SHA): **UNVERIFIED**
- Intended FLAT release-candidate commit (full 40-hex SHA): **UNVERIFIED**
- Equality between resolved dependency and intended release candidate: **NOT PROVEN**
- Workflow run ID, job ID, artifact URL and artifact SHA-256: **UNVERIFIED**
- Device name, WGPU backend, driver, runner identity: **UNVERIFIED**
- RemoteOps reservation/occupancy proof and contamination log: **UNVERIFIED**

## Mandatory evidence (all must pass on the exact identities above)

| Gate | Required observation | Status |
| --- | --- | --- |
| No adapter | `FLAT_REQUIRE_WGPU=1` fails instead of skipping | NOT VERIFIED |
| Physical adapter | WGPU adapter, physical device and driver match the run | NOT VERIFIED |
| Prefill | Model-level prefill against deterministic reference | NOT VERIFIED |
| Incremental decode | `query_len=1`, parity with reference and no hidden CPU fallback | NOT VERIFIED |
| Resident KV | Per-layer cache ownership; no host K/V roundtrip | NOT VERIFIED |
| Lifecycle | Reset/replay and EOS/empty-EOS parity | NOT VERIFIED |
| Performance | Same workload and device; old SciRust multi-dispatch versus resident FLAT, paired and ordered with warmups, medians/p95 and raw artifacts | NOT VERIFIED |
| Security | No unresolved applicable advisory or unaudited exception | NOT VERIFIED |
| CI | All required checks green for the exact final SciRust PR head | NOT VERIFIED |

## Negative and inconclusive evidence

Record every rejected run with exact run/job/SHA, failure phase, device occupancy, contamination, log/artifact digest, and whether numerical execution occurred. An unavailable Thor reservation, compilation-only result, software Vulkan run, or failed audit **must not** be reclassified as physical performance or model correctness evidence. Preserve unsuccessful timing pairs without clamping them into positive decode latencies.

## Promotion decision

**BLOCKED** until every required gate above is supported by immutable artifacts. This record is intentionally unfilled; do not substitute historical results from another SciRust or FLAT SHA.
