# V888-GROWTH — Brain growth research programme

Status: active exploratory programme.  
Primary execution host: Thor.  
Primary adult structural dataset: BANC v888.  
Scientific boundary: adult structural correlations are not causal developmental mechanisms.

## Question

Which measurable developmental-lineage, morphology, connectivity and energetic
variables are associated with the expansion and organization of the adult
Drosophila nervous system, and which of those relationships survive matched
structural controls and independent developmental comparison?

## Stage VG-0 — provenance and schema

Freeze exact file identities, SHA-256 hashes, byte sizes, schemas, row counts,
join keys and source snapshot. Raw data remain outside the SciRust source tree.

Required adult V888 artefacts include metadata, morphology metrics, v3
neuron-neuron edgelist, compartment-resolved edgelist and the full v3
individual-synapse parquet.

## Stage VG-1 — lineage-to-morphology scaling

Aggregate by annotated hemilineage:

- neuron count;
- L2 reconstruction-node count;
- skeletal cable length;
- segmentation volume;
- incoming and outgoing synapse counts;
- primary dendrite width;
- axon/dendrite segregation index.

Report totals, per-neuron distributions, robust summaries, log-log scaling,
residuals and region-stratified results. Raw size rankings alone are not an
explanation of growth.

## Stage VG-2 — energetic/metabolic scaling

Test whether mitochondrial count and volume scale with cable length, neuronal
volume and synaptic load, both globally and after lineage/region stratification.

This stage asks whether larger or more highly connected neuronal structures
carry a consistent energetic signature. It does not infer metabolic causality.

## Stage VG-3 — topology-to-growth proxies

Join morphology to directed graph structure:

- in/out degree and weighted degree;
- reciprocity;
- strongly connected components;
- reachability;
- centrality/hub diagnostics;
- modular/network membership;
- lineage and region mixing.

Compare V888 against deterministic matched controls:

- edge-count matched random sparse;
- in/out-degree matched;
- reciprocity matched;
- block/modularity matched.

The purpose is to determine which adult structural regularities exceed trivial
consequences of size and degree sequence.

## Stage VG-4 — synapse-allocation geometry

Use the full individual-synapse v3 table to quantify:

- synapses per unit cable and volume;
- regional allocation;
- pre/post asymmetry;
- concentration versus distributed allocation;
- within-lineage versus cross-lineage allocation;
- spatial concentration where coordinates permit it.

This stage must retain exact upstream row/filter provenance.

## Stage VG-5 — developmental comparison

BANC v888 is an adult snapshot. Claims about mechanisms that *cause* brain
growth require independent temporal/developmental evidence.

Candidate comparison rails include:

- complete first-instar larval connectome;
- published first- versus third-instar circuit reconstructions;
- lineage timing / neuroblast temporal-factor datasets;
- perturbation evidence for nutrient, insulin/TOR, ecdysone and glial control.

The comparison must distinguish neuron production, neurite growth, synapse
addition, pruning/reorganization and cell death.

## Stage VG-6 — artificial growth-rule translation

Only after VG-1 through VG-5 evidence, translate qualified motifs into bounded
SML-GENIUS/TDI experiments. Candidate process vocabulary:

PROLIFERATE -> SPECIALIZE -> GROW -> COMPETE -> PRUNE -> REGROW -> STABILIZE

This is a hypothesis space, not a claim that artificial networks should mimic
biological development.

## Ownership

SciRust owns reusable deterministic statistics, graph, sparse and event
primitives. SML-GENIUS owns model hypotheses. TDI owns comparative scientific
protocols and intervention/recovery interpretation. NoiseLab owns controlled
perturbation machinery. Forge may search bounded candidate rules after the
protocol is frozen.

## Current execution

VG-0, VG-1 and VG-2 have retained Thor evidence. The executable V888 pair graph
is separately qualified by the SciRust V8CSR001 contract.

VG-3A established the first Rust-owned topology-to-growth input surface: exact
per-neuron in/out degree, incoming/outgoing contact multiplicity,
reciprocal-neighbour count and canonical SCC membership/size.

VG-3B now adds the required matched-control ladder as reusable Rust helpers:
edge-count, degree, reciprocity and anatomical-block unit-adjacency controls
over deterministic induced subsets of the same frozen V8CSR001 graph. The
dedicated Thor workflow verifies the graph hash, runs the control exporter on a
bounded subset, and records a compact summary. Block labels remain caller-supplied
annotations, not inferred communities.

VG-3C now adds Rust-owned deterministic reachability, directed harmonic
closeness, directed Brandes betweenness and rich-club descriptors on the same
unit-adjacency reference and VG-3B control arms, with an exact integer delta
table for arm comparison. The pilot exporter stays at VG-3B subset scale
(≤512) because exact betweenness is cubic.

VG-3C deliberately stops before inferred modularity, morphology/hemilineage
residual joins, mixing/ensemble qualification and any topology-advantage claim.
Those remain VG-3D / later VG-3 work. No developmental-causality claim follows
from this slice.

VG-3D (first slice) adds Rust-owned deterministic directed modularity, a
deterministic directed Louvain community inference with exact integer move
gains, exact label-mixing tables with nominal assortativity for caller
annotations such as hemilineage, adjusted-Rand partition agreement, and a
per-arm comparison against the VG-3B controls
(`docs/research/V888_GROWTH_VG3D.md`). Inferred communities are descriptive
local optima, not biological modules. The Thor exporter, ensemble dispersion
and any qualification remain later VG-3D work.

The earlier Thor workflow `V888 Thor dataset and growth phase 0` downloads the
external adult dataset into persistent host storage and emits only compact
manifests and analysis outputs as GitHub evidence. Raw BANC files and the
qualified graph binary are never uploaded as repository artefacts.

Tracking: SciRust issue #1500.
