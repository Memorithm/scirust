# Tensor IR ecosystem reassessment — 2026-09-20

## Scope

This note reassesses `scirust-tensor-ir` against mechanisms that have matured in
other Memorithm repositories. It is deliberately conservative: an idea is moved
into the canonical tensor IR only when it strengthens a backend-neutral semantic
contract. Policy engines, research-only semantics and benchmark verdicts remain
in their owning projects.

Revision anchors used for this review:

- SciRust `9b34f52df617b71b233d00017fdc0728504b6342`;
- ElasticXxx `a232d252ce7db0a3c045ddf73acad6103f71588c`;
- TDI `12814e66937a2268b3d34bf4399bf4496c56c0db`;
- FLAT-ATTENTION `f1d17d704f6084d0d7b9e2dfd3b304951fe4d083`;
- Forge `da68e9703d7a523f5d3703ebedd3de85b39cb153`.

## Current Tensor IR strengths

`scirust-tensor-ir` already has several foundations that should be preserved:

- canonical acyclic graphs with typed outputs and explicit operation identity;
- deterministic structural optimization and semantic validation;
- backend-neutral sharding and `vmap` transforms;
- reverse/forward differentiation plus MorphoDiff integration;
- representation side tables that leave logical graph identity unchanged;
- exact integer `StorageBits` accounting;
- exact physical segment accounting with content/layout identity, shared ownership,
  reconstruction semantics and materialization-specific resident bits;
- atomic `RepresentationPlan::replan`: all requested rebindings are validated
  before assignments are changed.

Those contracts are stronger than replacing the IR with a general resource or
search graph. The canonical tensor graph should stay small, deterministic and
policy-free.

## Cross-project findings

### ElasticXxx — ADOPT the transaction boundary, not the policy engine

ElasticXxx now models resources through explicit invariants, admissible state
transitions, objectives and a control loop ending in
`PLAN -> VALIDATE -> ACT -> VERIFY -> COMMIT / ROLLBACK`. It also has typed
representational resources and Boolean eligibility gates.

Tensor IR already had atomic mutation, but callers could not validate and inspect
a projected representation state without cloning and mutating a plan themselves.
That made the useful `PLAN -> VALIDATE -> COMMIT` boundary implicit.

**Decision:** ADOPT a two-phase, policy-free representation transition primitive.
`PreparedReplan` validates a candidate on a cloned plan, exposes exact storage
before/after, and commits only if the target plan still equals the source state.
Dropping the token is the no-op rollback path. Elastic objectives, forecasts,
RAM/energy budgets and physical actuation remain in ElasticXxx.

### Forge — ADAPT evidence-driven search around the IR

Forge's strongest transferable property is not its evolutionary algorithm. It is
the separation of proposal, independent verification, measurement, environment
identity and Pareto selection. Incorrect candidates are not allowed to acquire
performance merit.

**Decision:** do not put Pareto selection, mutation or benchmark results in
`scirust-tensor-ir`. Instead, give search systems a stable candidate surface:
prepare an IR/representation transition, inspect exact structural costs, verify
it independently, measure it in a bound environment, then commit the winning
transition. `PreparedReplan` is the first such surface.

**Next investigation:** define a separate evidence/provenance sidecar keyed to
canonical graph/plan identity so Forge and compiler/autotune layers can bind
measurements without contaminating logical equality.

### TDI — keep Boolean/algebraic semantics experimental until operator contracts freeze

TDI now contains explicit Boolean-policy IR work and Boolean relational
architectures, with leakage-safe predicates, exact truth-table calibration and
bounded research protocols. This demonstrates that Boolean structure can be an
independent semantic object rather than an informal optimization hint.

**Decision:** INVESTIGATE an algebra/constraint sidecar for Tensor IR, but do not
add Boolean/F2/tropical/Zhegalkin variants to `TensorType` or `Operation` merely
because research benches use them. The canonical IR needs stable reconstruction
and operator semantics first. Candidate Boolean eligibility can already remain
external and feed validated rebindings.

### FLAT-ATTENTION — graduate only qualified execution semantics

FLAT-ATTENTION now has qualified streaming softmax paths plus a research-only
multi-algebra attention programme combining Boolean admission, F2 predicates,
Zhegalkin polynomials and max-plus readiness constraints. Its own documentation
explicitly states that this research is not yet a softmax replacement or GPU
performance result.

**Decision:** preserve the current boundary. Stable FLAT execution should lower
through compiler/attention intent layers; multi-algebra research should not add
canonical Tensor IR operations until its semantics graduate. The IR should,
however, remain able to represent alternative physical representations and
preflight their exact costs.

## Implemented in this slice

### `PreparedReplan`

The new `scirust-tensor-ir::PreparedReplan` provides:

1. non-mutating preparation using the same validation as
   `RepresentationPlan::replan`;
2. exact `before_storage_bits` and `after_storage_bits`;
3. access to projected canonical node assignments;
4. commit only when the destination `RepresentationPlan` is byte-for-byte
   semantically equal to the source plan captured at preparation;
5. fail-closed stale-plan rejection;
6. no policy, device, benchmark, random identity or external runtime dependency.

A regression test exercises a real `[2,2] F32` tensor replan to a per-tensor U8
code representation with one F32 scale. The exact representation cost changes
from 128 bits to 64 bits while the source plan remains unchanged until commit.

## IR-E2 implementation follow-up

The second slice adds three explicitly versioned canonical byte domains:

1. complete graph structural identity, including input display names;
2. representation-plan graph-anchor identity, deliberately ignoring only input
   display names to match existing plan-compatibility semantics;
3. complete representation-plan identity, binding that anchor to the ordered
   declaration table and canonical node assignments.

The encoder uses explicit variant tags, fixed-width little-endian integers and
length-prefixed sequences. It adds no cryptographic dependency. Canonical bytes
are the authoritative comparison surface; downstream SHA-256 or other digests
remain indexing/provenance values and cannot replace structural validation.

This distinction is required by the Forge integration model: candidate identity
may be hashed in an external envelope, while Tensor IR remains responsible for
the deterministic semantic payload being hashed.

## Next IR candidates

Priority order after this slice:

1. **Canonical plan/evidence identity.** Evaluate a deterministic content digest
   for graph + representation plan + physical layout identity, suitable for
   Forge/FLAT qualification records. The digest must not replace structural
   validation.
2. **Exact cost-vector surface.** Generalize storage-only comparison into a
   backend-neutral vector of exact/declared costs where the quantity is actually
   defined: serialized bits, resident bits for a named materialization, transfer
   bytes and recomputation markers. Unknown costs must remain unknown, not zero.
3. **Constraint/eligibility sidecar.** Study a typed Boolean eligibility layer
   inspired by ElasticXxx/TDI without changing logical tensor identity.
4. **Algebra semantics graduation gate.** Define explicit acceptance criteria
   before Boolean/F2/tropical/Zhegalkin operations can enter canonical Tensor IR:
   frozen operator semantics, shape/type rules, reference interpreter,
   differential oracles where applicable, and independent evidence.
5. **Search bridge, not search in IR.** Let Forge/scirust-opt-* propose plans
   through a bridge crate; keep candidate mutation, Pareto objectives and
   experiment state outside canonical IR.

## Rejected directions

- embedding ElasticXxx's runtime loop directly in Tensor IR;
- making performance measurements part of graph equality;
- treating `compiled` or representable as evidence of executable hardware;
- introducing research-only multi-algebra operations before semantic graduation;
- replacing exact `StorageBits` with average/estimated bits-per-value;
- accepting stale prepared transitions after the underlying plan changes.

These boundaries allow the Memorithm projects to cooperate without collapsing
their responsibilities into one oversized IR.
