# SciRust optimization runtime roadmap — Optuna 5 / Rustuna

Date: 2026-09-19
Status: active implementation
Baseline audited: attached Optuna master snapshot reporting `5.1.0.dev`

## Objective

Build a native SciRust optimization runtime that can be compared reproducibly
against Optuna 5 and Rustuna on both optimizer overhead and optimization quality.
The target is not an API clone. SciRust should keep validated algorithmic ideas
while exploiting typed IR, contiguous Rust data structures, SIMD/GPU kernels,
incremental state, native scheduling, deterministic evidence and the existing
Memorithm research stack.

No performance claim is accepted without a pinned benchmark, seed, code revision
and confidence interval/evidence record.

## Competitive baseline

The benchmark matrix must include at least:

- Optuna 5 TPE (multivariate/default and univariate control).
- Optuna GP sampler.
- Optuna CMA-ES and NSGA-II/III where applicable.
- Rustuna current release when the tested feature exists.
- SciRust random/QMC controls.
- Objective-cost regimes: near-zero, about 1 ms, and expensive objectives.
- Single worker and rolling asynchronous multi-worker execution.
- Continuous, integer, categorical, mixed, conditional, constrained, noisy and
  multi-objective spaces.

Primary evidence families: COCO/BBOB, YAHPO Gym and repository-local synthetic
microbenchmarks with analytic optima. Repository-local results are not allowed
to replace external benchmark families.

## Phase 0 — measurement contract

Deliverables:

- Canonical benchmark schema through `scirust-bench-schema`.
- Mandatory seeds, code digests, environment/hardware identity and wall-clock
  confidence intervals.
- Metrics: ask/tell latency, trials/s, bytes/trial, RSS growth, regret,
  anytime-regret AUC, hypervolume, parallel efficiency, duplicate proposal rate,
  persistence/restart cost and determinism.
- Preregister comparison criteria before headline benchmark execution.

Definition of done: a third party can rerun a pinned comparison and classify the
result as exploratory or confirmatory.

## Phase 1 — unify Gaussian-process infrastructure

Status: **started in this branch**.

Deliverables:

- Remove AutoML's duplicated GP linear algebra from the Bayesian hot path.
- Reuse `scirust-gp` Matérn-5/2 + stored Cholesky factor.
- Preserve the existing `scirust-automl::GaussianProcess` facade for
  compatibility.
- Add checked fit and regression tests against the canonical GP.
- Next: make batch prediction genuinely matrix/batch based rather than a loop
  over scalar `predict`.

Definition of done: AutoML acquisition never rebuilds the training covariance
matrix for each candidate prediction.

## Phase 2 — `scirust-opt-core`

Status: **active implementation**.

Implemented foundation:
- typed SearchSpace/ParamId IR and column-oriented TrialStore;
- monotone reservation plus idempotent tell semantics;
- pending-aware `ask` / rolling `ask_batch`;
- deterministic random-access/sequential seed derivation;
- canonical append-only StudyEvent stream;
- deterministic replay, in-memory snapshots and watermark-based event deltas.

Remaining in this phase: persistence-backend abstraction and a process-safe atomic
reservation implementation; those contracts will build on the same event model.

Create a dedicated optimizer crate with:

- typed `ParamId`-based SearchSpace IR;
- continuous/log/integer/categorical distributions;
- conditional-space DAG;
- `Study`, `TrialId`, `ask`, `tell`, `ask_batch`, `reserve`;
- explicit running/completed/pruned/failed state;
- deterministic seed streams;
- append-only event log and snapshots;
- columnar/SoA TrialStore;
- idempotent tell and atomic reservation contracts.

Strings and hash maps remain API-boundary conveniences, not the hot-path storage
representation.

## Phase 3 — TPE parity baseline

Status: **active implementation**.

Implemented reference slice:
- independent/univariate single-objective TPE in `scirust-opt-tpe`;
- Optuna-v5 default startup/candidate/gamma/prior/magic-clip/endpoint constants;
- continuous, log-continuous, integer and categorical Parzen models;
- conditional search-space proposals;
- rolling-batch running-trial inclusion for constant-liar-style penalization;
- four numerical density oracles generated from the audited Optuna 5 snapshot;
- allocation-free objective lookup from `scirust-opt-core`.

Still required for full Phase 3 parity: pruned-trial intermediate ranking,
constraints, multivariate joint modeling, group decomposition and
multi-objective TPE.

Implement a reference TPE sufficiently aligned with Optuna 5 to support
differential testing:

- startup sampling;
- below/above split;
- continuous, discretized and categorical Parzen distributions;
- prior and endpoint/magic-clip behaviour;
- candidate acquisition;
- multivariate sampling;
- conditional grouping;
- constraints;
- pending/running-trial penalisation equivalent in purpose to constant liar;
- multi-objective TPE where the evidence suite requires it.

Definition of done: seeded differential/property tests explain any intentional
deviation from the Optuna reference.

## Phase 4 — native numerical kernels

Status: **active implementation**.

Measured progress:
- cached numerical/categorical observation ordering is in place;
- truncated-normal normalization is precomputed once per mixture component;
- direct probability-density accumulation has a stable log-domain fallback;
- the 24 EI candidates are scored in a component-major batch instead of
  reparsing the same component arrays candidate by candidate;
- scalar density paths remain as differential test oracles;
- controlled Thor benchmark at 1000 trials: SciRust is 13.6x–18.1x faster
  than Optuna 5.0.0 in the tested univariate regime;
- batched density scoring improves SciRust median proposal latency by about
  5.1% / 7.5% / 4.0% at 1/5/10 dimensions versus the pre-batch code, with all
  five paired seeds improving in every 1000-trial cell;
- the remaining SciRust/Rustuna latency ratio is about 1.26x / 1.31x / 1.34x
  at 1/5/10 dimensions;
- SciRust median process peak RSS remains about 2.5–3.25 MiB versus
  15–17 MiB Rustuna and about 49–51 MiB Optuna in those cells.

Next measured kernel step: precompute truncated-normal sampling CDF bounds that
currently repeat for every sampled EI candidate, then remove remaining temporary
mask/sigma allocations. Architecture-specific AArch64 SIMD/SVE follows only
after these scalar/layout costs are exhausted.

Move sampler hot paths into contiguous kernels:

- truncated normal sample/CDF/log-mass;
- erf/CDF/inverse-CDF;
- mixture log-pdf;
- categorical mixture scoring;
- deduplication of repeated distribution terms;
- stable log-domain reductions;
- persistent scratch buffers;
- architecture-specific SIMD behind verified scalar oracles.

Targets: x86_64 and AArch64 first; GPU dispatch only after CPU crossover
benchmarks demonstrate benefit.

## Phase 5 — Streaming-TPE

Status: **foundation started**.

Implemented foundations:
- TPE consumes StudyEvent deltas from a sampler watermark;
- objective-ranked completed trials and running trials are maintained
  incrementally;
- below/above groups are maintained incrementally and exposed to Parzen models
  in chronological TrialId order, matching Optuna's trial-number weighting;
- per-parameter observations are cached in chronological and numerical-value
  order.

The history/ranking cache alone was benchmarked as approximately neutral in
proposal latency; the dominant cost was mixture density evaluation. It is kept
because it removes repeated global reconstruction and is required for the later
fully incremental Parzen update.

Replace history-wide reconstruction with incremental state:

`S_(n+1) = update(S_n, trial_(n+1))`.

Maintain sufficient statistics/indexes required for proposal generation,
including conditional groups and pending proposals. Introduce explicit history
retention levels:

- RAW
- COMPACT
- STATISTICS
- DISCARDABLE

Definition of done: proposal cost and memory growth are measured versus trial
count and compared with both Optuna and Rustuna.

## Phase 6 — complete evolutionary and multi-objective engines

- Replace the current explicitly simplified `scirust-evo::CmaEs` with a
  complete, validated CMA-ES implementation.
- Retain NSGA-II and audit it against standard fronts.
- Add hypervolume engines and incremental updates.
- Add NSGA-III only with an explicit complexity/quality benchmark.
- Use SoA population layouts and SIMD-friendly objective/ranking primitives.

## Phase 7 — GP v2 and high-dimensional BO

Extend `scirust-gp` with:

- incremental Cholesky append/update;
- ARD kernels;
- hyperparameter warm starts;
- true batched cross-covariance/prediction;
- batched acquisition;
- mixed/categorical kernels;
- pending-point fantasies/conditioning;
- local/trust-region BO (TuRBO-style regime);
- sparse/high-dimensional regime only after benchmark evidence.

The optimizer portfolio must select regimes rather than force an exact global GP
onto every history size/dimension.

## Phase 8 — pruning and multi-fidelity

Implement validated equivalents for:

- median/percentile pruning;
- successive halving / ASHA;
- Hyperband;
- threshold/patient policies;
- statistically justified pruning for noisy objectives;
- resource/cost-aware fidelity.

Pruners consume the same event/trial substrate as samplers.

## Phase 9 — distributed runtime

- lease-based worker reservations;
- heartbeats and stale-trial recovery;
- atomic ID allocation without a study-wide hot lock;
- delta synchronization via watermark;
- worker-local caches;
- append-only WAL/event store;
- deterministic replay;
- rolling asynchronous scheduling;
- batch-aware proposals that avoid duplicate work.

## Phase 10 — PortfolioSampler / meta-controller

Characterize each optimization problem using observable features such as
dimension, parameter types, conditionality, objective count, noise, constraint
rate, history size and evaluation cost.

Allocate/search among TPE, Streaming-TPE, GP, trust-region BO, CMA-ES,
multi-objective engines and control samplers. Any adaptive selection policy must
be benchmarked on held-out problem families to prevent benchmark overfitting.

## Memorithm integration

- **SciRust**: numerical kernels, GP, SIMD/GPU, typed optimizer runtime.
- **ElasticXxx / elastic-autotuner**: hardware/problem-class aware execution and
  measured plan selection.
- **Forge**: search over optimizer policies/configurations under holdout.
- **TDI**: preregistered scientific confrontation and evidence publication.
- **scirust-bench-schema**: canonical result/provenance records.

## Merge discipline

Each phase is split into reviewable PRs. A PR may only merge when its relevant
tests pass and its benchmark claims, if any, are backed by committed evidence.
Large performance changes require a scalar/reference oracle before specialized
SIMD/GPU paths become authoritative.
