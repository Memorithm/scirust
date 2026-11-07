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

## Source-declared pin preflight (2026-11-07; not a lockfile attestation)

The opt-in runtime edge `scirust-sciagent/flat-attention -> scirust-gpu/flat-attention` declares FLAT Git revision `c74a3806f8cfe621671694692fc14adfe08116f0` in `scirust-gpu/Cargo.toml`. The separate host-only planning edge in the root `Cargo.toml` declares `4529a2079434965e13e90ddd2e98ecc88ee0cb3a`. These are **different dependencies and purposes**; neither may be silently substituted for the other.

The latest FLAT repository commit returned by the GitHub commit search at this preflight was `77c8abaa32d374814d9c36eeda25f1c10e0419a6`. It is **not equal** to either declared revision. This observation alone does not establish which commit is the intended immutable release candidate, and the resolved `Cargo.lock` entries have **not** been independently attested. Therefore the exact-release-candidate integration gate remains **BLOCKED**.

Before promoting M33 evidence, capture the intended candidate full SHA and verify the resolved `Cargo.lock` Git source for the **runtime** `flat-attention` package against that candidate; separately record the planner dependency and its feature scope. Preserve the prior revision as a valid historical test scope, never as evidence for the newer candidate. A physical benchmark on an old pin does not qualify a new FLAT SHA.

## CI rejection record — 2026-11-07 10:20 Europe/Paris

- Candidate branch: `fix/flat-m33-require-physical-wgpu-20261106`; previous reported commit `59afb6e274c37ed8fff9f8b9fdae162b672284d6` (current branch ref must be re-attested independently).
- CI run [`42354853312`](https://github.com/Memorithm/scirust/actions/runs/42354853312): **FAILURE**; job `128350574416` (`License & Security Audit`) fails specifically at `Audit MKL dependency graph`. Other audited feature graphs, format, Clippy and workspace tests succeeded in that run. The failure cannot be waived for M33 promotion.
- SciAgent Thor run [`42354853328`](https://github.com/Memorithm/scirust/actions/runs/42354853328): **FAILURE** at `Wait for CUDA runtime availability`, before hardware parity or benchmark; no numerical or physical FLAT result can be inferred from this run. This separate CUDA gate does not qualify portable WGPU.
- SciAgent FLAT decode contract run [`42354853146`](https://github.com/Memorithm/scirust/actions/runs/42354853146): **SUCCESS**, scoped to its CI contract only, not resident physical GPU proof.
- SciAgent FLAT M33 Thor run [`42354853200`](https://github.com/Memorithm/scirust/actions/runs/42354853200): **IN PROGRESS at observation**; no outcome or physical measurements asserted.
- Dependency-security blocker tracked independently in [SciRust #1734](https://github.com/Memorithm/scirust/issues/1734): `blas-mkl -> blas-src -> intel-mkl-src -> ocipkg 0.2.9`, including path traversal advisory `RUSTSEC-2026-0450`. Do not add an advisory ignore; require a compatible graph remediation and new exact-head audit.
- **Decision:** NOT QUALIFIED; no performance claim, merge or release. Next: resolve security chain on the existing PR, attest runtime Cargo.lock FLAT SHA and run physical paired evidence under RemoteOps reservation.

## Exact-head CI observation — 2026-11-07 11:16 Europe/Paris

- SciRust PR #1733 head at observation: `f2fbf97d7ea4737fa5c7458e8de8e8187e2bed60` (PR metadata, not an inferred merge commit).
- [CI run 42359346786](https://github.com/Memorithm/scirust/actions/runs/42359346786): **FAILURE**, job [128364186591](https://github.com/Memorithm/scirust/actions/runs/42359346786/job/128364186591), `License & Security Audit`, failing step `Audit MKL dependency graph`; the other enumerated graph audits passed. The step outcome does not independently prove a particular advisory or compatible remediation; see #1734 and retain its separate chain diagnosis.
- [SciAgent FLAT decode contract run 42359346762](https://github.com/Memorithm/scirust/actions/runs/42359346762): **SUCCESS**, contract scope only.
- [SciAgent Thor gate run 42359346761](https://github.com/Memorithm/scirust/actions/runs/42359346761): **IN PROGRESS at observation**, no device result asserted.
- [SciAgent FLAT M33 Thor run 42359346826](https://github.com/Memorithm/scirust/actions/runs/42359346826): **IN PROGRESS at observation**, no resident KV, numerical or physical timing proof asserted.
- [PR rustfmt diagnostic run 42359346718](https://github.com/Memorithm/scirust/actions/runs/42359346718): **SUCCESS** on the observed head.
- The newest FLAT commit returned by commit search is `77c8abaa32d374814d9c36eeda25f1c10e0419a6`; this search does **not** independently attest `refs/heads/main` or the resolved Cargo.lock pin.
- **Disposition: BLOCKED**. Do not merge or claim M33 physical qualification, release readiness or positive performance until security audit, pin equality and hardware evidence are independently demonstrated on the final exact head.
