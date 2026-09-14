# SciRust source audit — 2026-09-14

## Scope and evidence boundary

Baseline: `e1237827f73268bf43b2b971674a697cd186d95e` on `master`.

This is a targeted semantic audit within a broader repository review, not a claim that every workspace implementation has been verified. The review inspected the root manifest and README, the main CI entry points, API-lexicon CI, previous audit findings, repository-wide textual searches, and the statistics / attention-intent / graph boundaries described below. Previous reports are historical evidence, not a current defect inventory.

The corresponding PR adds executable positive controls and pinned-baseline negative controls. Their actual GitHub Actions results are authoritative for compilation and runtime validation. The editor environment did not contain a Rust toolchain or a network-capable Git checkout; no local Rust test execution is claimed.

`audit-contracts.yml` also emits a revision-bound inventory of Cargo workspace members and all tracked Rust files. Its textual marker counts include tests, comments, strings, examples and vendored code. They are **review candidates, not bug counts**, and are explicitly not counts of production-path panics or placeholders.

## Confirmed defects addressed by this patch

| ID | Priority | Baseline behavior and evidence | Change |
|---|---|---|---|
| ATT-01 | P1 | `derive_attention_intent` evaluates `q_heads % kv_heads` before rejecting zero dimensions. Graph construction accepts zero-sized shape metadata. K-head count zero can therefore panic instead of returning `IntentError`. | Validate every Q/K/V dimension in `rank4` before any head arithmetic. |
| ATT-02 | P1 | `rank4` uses unchecked `usize as u32` conversions. On 64-bit targets, dimension `2^32 + 1` becomes `1` even though the representation still accounts for the original large tensor. | Checked `u32::try_from`, with a role-specific `InvalidDimension`. No tensor payload is allocated by the regression fixtures. |
| ATT-03 | P1 | K and V sequence lengths are compared, but their head counts are not. A Q=4-head / K=2-head / V=3-head input can be described as a coherent 2-KV-head intent. | Require matching K/V head counts before producing an intent. |
| ATT-04 | P2 | A V batch mismatch is reported as a Key error. | Distinguish Key and Value batch validation. |
| ATT-05 | P2 | `is_executable` documents `value_dim == head_dim`, but checks only representation variants. Public fields can be modified after construction. | Check the documented value-dimension condition while preserving the `const` API. This is not a complete validator for arbitrarily mutated public fields. |
| ATT-06 | P2 | The test named `insertion_order_does_not_change_the_fingerprint` computes a second result and discards it. It cannot detect inequality. | Compare both fingerprints and rename the test to describe the equal-shape role permutation it actually exercises. |
| STAT-01 | P2 | `min([])` is positive infinity and `max([])` is negative infinity, contrary to their `NaN` contracts. All-NaN inputs have the same problem. | Fold from `NaN`, keeping the existing policy of ignoring individual NaN samples. |
| STAT-02 | P2 | `quantile([+infinity], p)` computes infinity minus infinity and returns NaN. A finite endpoint adjacent to infinity can also become NaN through zero times infinity. | Return exact order statistics before interpolation; specify the policy for infinite bounds. |
| STAT-03 | P2 | For `[-f64::MAX, f64::MAX]`, subtracting the bounds overflows although the median is exactly zero. Even the minimum endpoint can become NaN. | Use a weighted sum for opposite-sign finite bounds and preserve exact endpoints. Non-exact extreme interpolation tests use scale-aware rounding bounds. |
| STAT-04 | P2 | Treating every failed `partial_cmp` as equality does not provide a consistent ordering when NaN is present; the public quantile NaN policy is not documented. | Explicitly propagate NaN samples, reject NaN probabilities, and sort accepted samples with `total_cmp`. |

P1 denotes a high-priority public correctness/boundary failure, not a claim of exploitable memory corruption. P2 denotes an incorrect result, misleading contract, or missing regression assertion that should be fixed before relying on the affected path.

## Useful addition: batched empirical quantiles

`scirust_stats::describe::quantiles(data, probabilities)` computes multiple type-7 empirical quantiles with one sample sort. Probability order is preserved; scalar and batched APIs share the interpolation helper and non-finite-input contract. An invalid probability only contaminates its own output, while an invalid sample contaminates all results.

```rust
use scirust_stats::describe::quantiles;

let percentiles = quantiles(&[4.0, 1.0, 3.0, 2.0], &[0.5, 0.95, 0.99]);
assert_eq!(percentiles.len(), 3);
```

The structural cost is one `O(n log n)` sort plus `O(q)` interpolation for `q` requested probabilities, instead of `q` sorts. No wall-clock speedup is claimed without a benchmark. This is suitable for repeated percentile summaries in experiment and runtime reports; no downstream repository adoption is asserted by this patch.

## Validation protocol

Positive controls:

```bash
cargo +stable test -p scirust-stats -p scirust-attention-intent --locked
cargo +stable test -p scirust-stats --release --locked audit_
cargo +stable test -p scirust-attention-intent --release --locked --test audit_contracts
cargo +stable clippy -p scirust-stats -p scirust-attention-intent --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo +stable doc -p scirust-stats -p scirust-attention-intent --no-deps --locked
```

The main workspace CI remains unchanged and must also pass on the final PR head. Existing API-lexicon CI discovers the documented new callable and applies its documentation-debt baseline.

Negative controls in the dedicated workflow:

1. Fetch the exact baseline into a detached temporary worktree; do not modify `master` or the checked-out PR files.
2. Copy only the public attention regression tests to the baseline workspace. Require the six named runtime failures and the two passing controls. A Cargo compilation failure is explicitly not accepted as evidence.
3. Compile the baseline descriptive-statistics source with three added regression predicates. Require its three original tests to pass and all three added predicates to fail.
4. Retain the logs and revision-bound inventory as an artifact.

The expected failures belong exclusively to this pinned negative control. Patched tests, lint, documentation and normal CI failures are not suppressed. CPU metadata/statistics tests do not establish CUDA/Thor execution, numerical parity on devices, or measured performance.

## Remaining verified risks and implementation slices

### R1 — finite mean/variance overflow remains

`describe::mean` still computes a raw sum before dividing. For `[f64::MAX, f64::MAX]`, the representable mathematical mean is `f64::MAX`, but the intermediate sum overflows. The current variance then uses the non-finite mean, although two identical finite samples have zero mathematical variance.

This patch deliberately does not redesign reduction semantics incidentally. A separate numerical slice should add adversarial reference cases, choose a scaled/compensated reduction with an explicit non-finite policy, test cancellation and subnormal values, and assess callers' reproducibility requirements. Do not advertise the present two-pass variance as universally overflow-safe.

### R2 — legacy FLAT shape tuple is lossy

`flat_shape_tuple` returns `(batch_q_heads, kv_heads, max(q_len, kv_len), head_dim, causal, dtype)`, while its documentation labels the first field as batch. Separate Q/KV lengths cannot be recovered from that tuple. The repository search found its definition but no caller at the reviewed revision; no current downstream execution corruption is claimed.

Before wiring it to an executor, introduce a typed, fallible conversion to an explicitly versioned downstream shape contract. Preserve batch, Q heads, KV heads, Q length and KV length independently; reject unsupported rectangular/GQA configurations rather than silently flattening them.

### R3 — representation support is not execution support

Quantized-per-tensor intents are intentionally describable but non-executable. Legacy sparse and quantized skeletons lack reconstruction geometry and fail explicitly. This is an unfinished capability, not a hidden successful placeholder. Progress requires a verified reconstruction/layout contract, exact storage accounting, reference execution, and only then a kernel mapping. Boolean/sparse attention research must preserve that distinction.

### R4 — public mutable intent fields and identity scope

The builder validates invariants, but the intent is a public mutable record. `is_executable` is not a comprehensive validation method; callers can still change dimensions, dtypes, representation summaries or the stored fingerprint. A future compatible `validate` / checked adapter boundary should precede any persistence or kernel launch. A full canonical-record and cross-plan identity review remains necessary; no hash collision or exploitable cache confusion is asserted here.

### R5 — placeholder searches are not maturity assessment

Search hits include archived audits, research roadmaps, legitimate pattern placeholders and string fixtures. Conversely, incorrect arithmetic needs no TODO marker. Continue auditing public boundaries, actual call graphs, algorithmic contracts and effective tests instead of ranking crates solely by grep counts.

### R6 — hardware and excluded workspaces need separate evidence

The root manifest excludes independent workspaces including SOS, Studio, vendored CCOS-Core, fuzz and hypermemory. Root workspace success is not automatically their qualification. Opt-in WGPU/CUDA paths also need feature-specific and, when claimed, hardware-specific evidence. Do not replace those gates with these CPU-only tests.

## Cross-project relevance

The attention corrections strengthen SciRust's semantic boundary intended for FLAT-ATTENTION and elastic representation planning. They do not implement Boolean attention, enable sparse/quantized kernels, or prove a speedup. The batched-statistics API is a reusable candidate for TDI/KVLab/Forge reporting; adoption requires inspecting each consumer's current statistics convention and dependency revision first. No cross-repository integration is claimed without a corresponding patch and test result.

## Merge condition

Merge only after applicable checks are green on the exact final head. Neither the expected negative-control failures nor a successful hosted CPU job waive ordinary workspace or hardware-specific policies. This audit introduces no license change, dependency update, GPU workload termination, or branch-history rewrite.
