# FLAT M33 — SciAgent resident decode/KV qualification

M33 qualifies SciAgent's already-wired FLAT decode path as an end-to-end resident KV-cache integration rather than introducing a second decode implementation.

## Runtime contract

With `scirust-sciagent/flat-attention` enabled:

- prompt prefill uses the M32 resident grouped-forward route;
- each layer stores RoPE-rotated K and raw V in fixed-capacity SciRust-owned `WgpuDenseKvCache` allocations;
- each subsequent token uses `query_len = 1` and the existing FLAT M15 pre-rotated-K path;
- Q, active K, active V and attention context remain WGPU-resident across the attention boundary;
- the existing KV prefix is not rebuilt on the host and no host K/V round-trip is introduced into the token loop;
- the final vocabulary logits retain the existing SciRust host-visible sampling boundary.

The feature-off path remains the explicit SciRust reference/fallback. FLAT itself does not silently select that path.

## Cache lifecycle and EOS

`generate_cached` allocates fresh per-layer caches for every generation request, so replay after completion starts from an empty logical cache without retaining hidden state from the previous request.

M33 adds `generate_cached_until_eos`, which uses the same M32 prefill + M15 decode path but terminates immediately after a generated token matches the caller-provided EOS set. An empty EOS set is required to be behaviorally identical to ordinary `generate_cached`.

`flat_m33_lifecycle` verifies deterministic replay, fresh-cache reset semantics, immediate EOS termination and the empty-EOS equivalence contract on a WGPU adapter.

## Real-device benchmark gate

`flat_m33_thor_bench` is the M33 model-level measurement harness. It runs through the public resident generation API and records:

- exact SciRust source revision;
- WGPU adapter name;
- model geometry and GQA head geometry;
- prompt length and generated-token count;
- warm-up and measured repeat counts;
- median prompt-prefill latency;
- median and p95 incremental decode latency per token;
- decode tokens/s.

The benchmark uses paired fresh-cache runs. `generate_cached(prompt, 1)` measures prompt prefill plus first-token selection; `generate_cached(prompt, N)` measures the same prefill followed by `N - 1` incremental decode forwards. The per-pair difference is divided by the actual decode-forward count. Execution order alternates to reduce systematic ordering bias.

The repository's self-hosted Jetson Thor workflow runs the lifecycle parity test and this benchmark on the real ARM64 Thor WGPU/Vulkan adapter. Results from software Vulkan remain correctness/qualification evidence only and must not be promoted as physical-GPU performance.

The harness emits `performance_claim=none`; measured values become a performance claim only when the exact real-device log/artifact and commit are cited.

## Sovereignty

M33 remains Rust-native host code plus WGPU/WGSL. It introduces no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK. The Thor workflow may inspect machine state, but FLAT's build and execution path does not depend on CUDA SDK components.

## Fail-closed adapter qualification (2026-11-06)

An ordinary developer run without a WGPU adapter may skip the two GPU-specific
integration tests. This **must not** count as a physical M33 qualification.
The physical Thor workflow sets `FLAT_REQUIRE_WGPU=1` for both the resident
cache reset/replay/EOS test and the prefill/incremental decode parity test.
When `ResidentModel::from_model` returns `None` under that setting, the test
fails instead of reporting a successful skip. Both tests also check `FLAT_EXPECTED_WGPU_ADAPTER` when present. The Thor workflow
requires the chosen runtime adapter name to contain `NVIDIA` after separately
verifying the physical Thor device and Vulkan adapter; a software adapter is not
accepted as evidence. The lifecycle test logs the actual adapter identity. The existing exclusive GPU
reservation, physical Vulkan identity check and contamination watchdog remain
mandatory and unchanged. This change does not by itself establish the exact
FLAT release-candidate pin or confer a speedup claim.

## Timing integrity gate (2026-11-06)

The paired prefill-versus-full-generation estimator must reject any measured
pair for which total elapsed time does not strictly exceed prefill elapsed time.
The previous `max(f64::MIN_POSITIVE)` clamp fabricated an extremely small
positive decode latency from an invalid/non-positive difference, producing an
artificially enormous tokens/s value. The M33 benchmark now terminates with
nonzero status on such a pair and reports the rejected pair index. Host-only
regression tests for positive, zero, negative and zero-decode-step inputs run
in the Thor workflow before the device reservation. This protects evidence
integrity; it does not make a noisy estimator a precise isolated decode timer.

## Exact-head qualification evidence register (2026-11-09)

This register is a **negative qualification record**, not a performance result.
It applies only to SciRust PR #1733 head
`78f130100bca1b1091312d3f2b1a456340bd531a` and must be
re-evaluated if that head changes. The intended immutable FLAT 1.0 candidate
pin is **not established** by these results.

| Gate | GitHub Actions run | Outcome and proof boundary |
| --- | --- | --- |
| SciAgent FLAT decode contract | [42538905722](https://github.com/Memorithm/scirust/actions/runs/42538905722) | Success; software/contract CI only, not physical M33 qualification. |
| FLAT consumer compatibility | [42538905689](https://github.com/Memorithm/scirust/actions/runs/42538905689) | Success; consumer compatibility, not an immutable release-candidate pin proof. |
| Native ARM64 | [42538905652](https://github.com/Memorithm/scirust/actions/runs/42538905652) | Failure; no physical portability acceptance. |
| SciAgent Thor gate | [42538905700](https://github.com/Memorithm/scirust/actions/runs/42538905700) | Failure; no physical resident decode/KV acceptance. |
| M33 Thor | [42538905769](https://github.com/Memorithm/scirust/actions/runs/42538905769) | Failure before accepted physical measurement; no M33 throughput or numerical qualification claim. |
| M53 Thor | [42538905712](https://github.com/Memorithm/scirust/actions/runs/42538905712) | Failure; no accepted physical candidate result. |
| M54 Thor | [42538905726](https://github.com/Memorithm/scirust/actions/runs/42538905726) | Failure; no accepted physical full-model prefill result. |

Earlier Thor logs identify a competing `checkupauto-character` process
preventing the required exclusive idle window. That historical observation
is **not** proof of current GPU occupancy, nor proof that every failed gate
has the same root cause. Never turn a reservation timeout into a passing
numerical test, and never terminate an unrelated workload to obtain a green
check. Physical scheduling and any rerun must be authorized through
Memorithm/RemoteOps.

Acceptance remains blocked pending a new exact-head physical run, actual
adapter/backend/driver identity, resident-KV and no-host-roundtrip evidence,
reset/replay/EOS parity, immutable SciRust lockfile plus FLAT candidate SHA,
and an M40-equivalent paired comparison against the old multi-dispatch path.
No latency, tokens/s, quality, or release 1.0 claim follows from this register.

## Follow-up exact-head CI register (2026-11-09, 16:19 Europe/Paris)

This follow-up records the GitHub Actions API snapshot for PR #1733 at
SciRust commit `da6d0fba78fbeaa92ae0424d3035892e59c3eefb` **before**
this documentation-only update. The earlier `78f1301` register remains a
separate negative record and is not retroactively transferred to this commit.
The follow-up does not establish the immutable FLAT release-candidate pin.

| Gate | Run | API state at observation | Scope |
| --- | --- | --- | --- |
| CI | [42615112229](https://github.com/Memorithm/scirust/actions/runs/42615112229) | completed / success | General CI only. |
| SciAgent FLAT decode contract | [42615112410](https://github.com/Memorithm/scirust/actions/runs/42615112410) | completed / success | Contract, not physical GPU acceptance. |
| FLAT consumer compatibility | [42615112771](https://github.com/Memorithm/scirust/actions/runs/42615112771) | completed / success | Consumer matrix, not exact FLAT candidate pin. |
| Native ARM64 | [42615112536](https://github.com/Memorithm/scirust/actions/runs/42615112536) | completed / failure | Physical ARM64 acceptance blocked. |
| SciAgent Thor | [42615111973](https://github.com/Memorithm/scirust/actions/runs/42615111973) | completed / failure | Physical resident-KV acceptance blocked. |
| M33 Thor | [42615112424](https://github.com/Memorithm/scirust/actions/runs/42615112424) | in_progress / no conclusion | No qualified result at observation. |
| M53 Thor | [42615112391](https://github.com/Memorithm/scirust/actions/runs/42615112391) | in_progress / no conclusion | No qualified result at observation. |
| M54 Thor | [42615112597](https://github.com/Memorithm/scirust/actions/runs/42615112597) | in_progress / no conclusion | No qualified result at observation. |

The historical Thor contention diagnosis identifies
`checkupauto-character.service` (PID 30957 in earlier logs); this is **not**
a fresh process-liveness check. No hardware reservation, benchmark, service
interruption or numerical test was performed in this documentation update.
The new documentation commit has its own exact SHA and requires fresh CI
checks; this snapshot cannot qualify that new commit for merge. RemoteOps
remains the sole machine-control plane. FLAT 1.0 is not authorized.
