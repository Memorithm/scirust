# BANC V888 connectome bootstrap for SciRust

Status: research bootstrap, not a performance or model-quality claim.

## Scope and source boundary

This track uses only the FlyWire BANC v888 connectome. FAFB v783 and other connectomes are excluded from the experimental source set for this programme.

The authoritative public metadata is the Codex BANC v888 dataset page and FAQ. At bootstrap time Codex identifies BANC v888 as Female Adult Fly Brain and Nerve Cord, snapshot v888 dated 2026-05-20, with 158,262 neurons and 3,037,361 aggregated connections. These connection counts are not to be relabelled as raw synapse counts.

External data must remain outside the SciRust source tree. SciRust may contain schemas, import adapters, deterministic fixtures derived from tiny synthetic graphs, source identifiers, checksums and reproducibility manifests. It must not vendor the BANC dataset or assume redistribution rights. Before any external artifact is published, the exact license/citation metadata shipped by the selected Codex download must be captured in the evidence bundle.

Sources:
- https://codex.flywire.ai/?dataset=banc
- https://codex.flywire.ai/faq

## Mission

Make SciRust the reusable pure-Rust substrate for sparse recurrent, graph-dynamical and event-driven model research. The V888 programme is a forcing function for capabilities that are broadly useful beyond connectomics: directed weighted sparse graphs, typed edges, deterministic graph statistics, sparse matrix kernels, delayed events, spiking-state dynamics, surrogate gradients, and portable accelerator execution.

SciRust owns reusable mechanisms, not the SML-GENIUS model hypothesis and not the TDI scientific verdict.

## Existing foundations to reuse

Do not create duplicate abstractions when these are sufficient:

- scirust-sparse: COO/CSR/CSC and deterministic sparse linear algebra;
- scirust-graph: graph algorithms, motifs, modularity and related structural analysis;
- scirust-sim: deterministic simulation and existing discrete-event machinery;
- scirust-core: tensors, reverse-mode autodiff and neural-network primitives;
- scirust-gpu: portable WGPU execution;
- scirust-simd: CPU vector kernels.

A dedicated connectomics or spiking crate is authorized only after a concrete API boundary cannot be expressed cleanly through these foundations.

## Bootstrap sequence

### SR-V888-0 — data contract and provenance

Deliver a Rust data contract for externally supplied V888 tables.

Required fields:
- stable node identifier;
- directed pre/post relation;
- integer or explicitly typed edge weight;
- optional anatomical/type metadata represented as optional data, never inferred;
- source dataset = BANC;
- source snapshot = v888;
- source date and artifact checksum;
- import filter parameters.

Acceptance:
- fail closed on duplicate or unknown node references;
- deterministic ordering independent of input row order;
- no network access in library code;
- tiny checked-in synthetic fixture only;
- round-trip manifest tests.

### SR-V888-1 — directed sparse graph substrate

Extend existing graph/sparse primitives instead of creating a second graph stack.

Required capabilities:
- directed weighted graph;
- inbound and outbound compressed adjacency;
- stable node/edge indexing;
- explicit self-loop and parallel-edge policy;
- deterministic SCCs, degree distributions, reciprocity and reachability;
- typed node/edge annotations without making them mandatory for generic algorithms.

Acceptance:
- O(V+E) traversal;
- property tests against simple independent oracles;
- deterministic serialization fingerprint.

### SR-V888-2 — structural descriptor panel

Implement reusable descriptors required for matched topology controls:
- in/out degree distributions;
- weighted degree;
- reciprocity;
- strongly connected components;
- clustering/motif counts where computationally bounded;
- modularity/community summaries;
- shortest-path and reachability summaries;
- hub/rich-club candidate diagnostics;
- region/type mixing matrices only when source annotations explicitly support them.

Every descriptor must document complexity and approximation policy. Large-graph approximate algorithms require explicit seeds and error reporting.

### SR-V888-3 — sparse numerical kernels

Generalize the sparse numerical path for model workloads.

Targets:
- f32 CSR/CSC in addition to current scientific f64 surfaces where justified;
- SpMV and batched sparse-dense multiplication;
- row/segment gather-scatter;
- deterministic CPU reference;
- portable WGPU candidate path;
- exact memory accounting for indices, values, padding and temporary buffers.

Do not add CUDA as a requirement. Vendor-specific backends are outside this V888 track.

### SR-V888-4 — deterministic temporal event engine

Add a reusable event scheduler suitable for sparse recurrent dynamics:
- monotonic simulation clock;
- delayed events;
- deterministic tie-breaking;
- bounded queues;
- event cancellation/invalidations where needed;
- exact counters for scheduled, delivered, dropped and coalesced events;
- replayable traces.

Compare fixed-step and event-driven execution on the same reference dynamics.

### SR-V888-5 — spiking reference semantics

Add a correctness-first neuronal dynamics layer after the graph/event substrate is stable:
- LIF reference;
- alpha/exponential synaptic state;
- refractory state;
- signed/typed edge handling only when input data supplies such information;
- externally driven Poisson or deterministic spike sources;
- activation and silencing interventions;
- fixed-step oracle and event-driven candidate.

Do not claim biological fidelity beyond the declared equations.

### SR-V888-6 — differentiable spike boundary

Provide an explicitly experimental surrogate-gradient boundary:
- hard forward spike;
- declared surrogate backward function;
- finite-difference and known-answer tests on tiny systems;
- no hidden continuous replacement during inference;
- deterministic training traces where the optimizer permits it.

This is a general SciRust primitive, not evidence that V888 topology improves learning.

### SR-V888-7 — portable acceleration

Promote only after CPU correctness:
- WGPU sparse propagation;
- WGPU event-window execution where beneficial;
- CPU SIMD fallback;
- capability detection;
- no implicit CPU fallback when a caller requires a GPU path;
- benchmark reports tied to exact hardware/backend/commit.

### SR-V888-8 — ecosystem APIs

Publish narrow versioned interfaces needed by:
- SML-GENIUS for connectome-derived structural priors and sparse recurrent execution;
- FLAT-ATTENTION for graph-derived sparse admission masks;
- NNIS for runtime execution contracts;
- TDI for deterministic experimental traces;
- Forge for topology candidate representation;
- NoiseLab for perturbation injection;
- ElasticXxx for active-edge/resource telemetry.

No consumer may import internal modules to bypass these contracts.

## Required control generators

SciRust must provide deterministic synthetic topology generators used by downstream scientific controls:
- Erdos-Renyi with matched edge count;
- directed configuration model with matched in/out degree sequence when feasible;
- degree-distribution matched rewiring;
- reciprocity-matched rewiring;
- modular/block controls;
- motif-matched controls only after the matching algorithm is independently validated.

V888-derived topology must never be compared only against a dense network or an unconstrained random graph.

## Exit gate

This bootstrap graduates only when a fresh checkout can:

1. import an external V888-shaped fixture through the versioned contract;
2. build deterministic directed sparse adjacency;
3. compute the frozen structural descriptor panel;
4. execute the same small recurrent system by fixed-step and event-driven references;
5. report exact active-edge, memory and event counters;
6. pass CPU oracle tests;
7. optionally execute qualified WGPU kernels without any NVIDIA dependency.

Promotion to model-level claims belongs to SML-GENIUS/TDI, not SciRust.
