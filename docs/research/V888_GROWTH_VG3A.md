# V888-GROWTH VG-3A — exact per-neuron topology profile

Status: implementation/qualification slice for SciRust issue #1500.

This slice converts the already-qualified BANC v888 executable pair graph into a
deterministic per-neuron table that can later be joined to morphology,
hemilineage and region evidence. It is an adult structural descriptor stage, not
a developmental-causality result.

## Frozen graph input

VG-3A consumes only the qualified V8CSR001 graph retained by the executable
graph contract:

- source qualification run: 35717145115;
- metadata nodes: 188,508;
- directed pairs: 13,620,865;
- induced contacts: 42,309,621;
- graph bytes: 166,466,540;
- graph SHA-256: 385111a69cc8a1d748c0bdfd9b0b738c51fe83cde98f45762553435a2d15a2a4.

The raw BANC dataset and graph binary remain external to Git.

## Rust-owned profile

The reusable function
scirust_graph::v888_growth::v888_growth_topology_profile emits one row per
canonical BANC root ID with:

- in-degree;
- out-degree;
- exact incoming contact multiplicity;
- exact outgoing contact multiplicity;
- reciprocal-neighbour count;
- canonical strongly-connected-component label;
- strongly-connected-component size.

The complete profile also records graph-wide directed pairs, contacts,
reciprocal pairs, SCC count and largest SCC size.

Contact multiplicity remains an exact source count. It is not conductance,
event rate, physiological strength or a developmental-growth variable.

## Deterministic export

The example v888_growth_vg3a reads the qualified V8CSR001 graph, rejects a
SHA-256 mismatch, and writes a deterministic CSV ordered by canonical root ID.

The Thor workflow for this slice verifies the retained graph byte size and
SHA-256 before export, runs the Rust tests/doctest, emits the CSV and records its
SHA-256 plus row count as compact evidence.

## What VG-3A does not claim

VG-3A does not yet implement or qualify:

- all-pairs reachability;
- centrality or rich-club diagnostics;
- modular/community membership;
- lineage/region joins;
- random, degree-matched, reciprocity-matched or block-matched controls;
- any topology advantage;
- any causal brain-growth mechanism.

Those remain VG-3B/VG-3C work. The purpose of this slice is to make the exact
per-neuron graph facts a reusable Rust contract before higher-level statistics
or controls are applied.
