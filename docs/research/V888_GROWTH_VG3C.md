# V888-GROWTH VG-3C — reachability, centrality and rich-club descriptors

Status: implementation/qualification slice for SciRust issue #1500.

This slice adds Rust-owned deterministic reachability, directed harmonic
closeness, directed Brandes betweenness and rich-club descriptors that can be
computed on VG-3A/VG-3B unit-adjacency digraphs. Reference and matched-control
arms therefore share one metric contract. It is an adult structural descriptor
stage, not a developmental-causality or topology-advantage result.

## Why VG-3C now

VG-3A established exact per-neuron degree, contact multiplicity, reciprocity and
SCC facts. VG-3B added the matched-control ladder as reusable unit-edge
generators. The programme document deferred:

- all-pairs reachability;
- centrality / rich-club diagnostics;

to VG-3C. Inferred modularity and morphology / hemilineage residual joins remain
explicitly deferred to VG-3D.

## Frozen graph input

VG-3C consumes only the qualified V8CSR001 graph retained by the executable
graph contract:

- source qualification run: 35717145115;
- metadata nodes: 188,508;
- directed pairs: 13,620,865;
- induced contacts: 42,309,621;
- graph bytes: 166,466,540;
- graph SHA-256: 385111a69cc8a1d748c0bdfd9b0b738c51fe83cde98f45762553435a2d15a2a4.

Raw BANC files and the graph binary remain external to Git.

## Rust-owned metric surface

Reusable helpers re-exported from `scirust_graph::v888_growth`:

- `v888_growth_unit_digraph` — temporary `DirectedGraph<()>` from unit edges;
- `v888_growth_reachability_profile` / `v888_growth_reachability_from_edges` —
  exact out/in reach counts (self included) via `DirectedGraph::reachable_from`
  and its transpose;
- `v888_growth_bfs_distances` — exact directed BFS distances;
- `v888_growth_harmonic_closeness` — unnormalized out/in harmonic closeness
  `Σ 1/d` over reachable others (IEEE-754 sums of exact reciprocal hop counts);
- `v888_growth_directed_betweenness` — exact directed Brandes betweenness with
  `(n-1)(n-2)` normalization for `n > 2`;
- `v888_growth_rich_club_curve` — directed rich-club points over unique total-
  or out-degree thresholds, with exact `club_edges` / `max_edges` counts and
  deterministic `phi`;
- `v888_growth_metric_bundle` / `v888_growth_compare_metric_arms` — per-arm
  bundle plus exact integer deltas (and nano-scaled f64 deltas with documented
  half-away-from-zero rounding).

Mean out-reach is retained as the exact rational
`(reachable_ordered_pairs / node_count)`. Contact multiplicity stays outside
these unit-adjacency metrics and is not conductance.

## Pilot scale

Exact Brandes is cubic in the worst case. The example exporter
`v888_growth_vg3c` therefore rejects subset sizes above 512 — the same pilot
scale used by VG-3B — so reviewable Thor evidence stays bounded. The reusable
API itself does not hard-cap `n`; callers must choose a scale they can afford.

## Deterministic export

The example rejects a SHA-256 mismatch against the frozen qualification
envelope, selects a subset, builds VG-3B matched controls, computes the VG-3C
metric bundle per arm, and writes:

- `nodes.tsv`
- `metrics_per_arm.tsv`
- `deltas.tsv`
- `rich_club_reference_total.tsv`
- `summary.json`

No raw BANC rows are emitted. The Thor workflow recipe is retained as
`docs/research/V888_GROWTH_VG3C_THOR_WORKFLOW.yml` until a credential with the
GitHub `workflow` scope can install it at
`.github/workflows/v888-growth-vg3c-thor.yml`.

## What VG-3C does not claim

VG-3C does not yet implement or qualify:

- inferred modular / community membership;
- morphology / hemilineage residual joins to control deltas;
- a uniform null ensemble or mixing-time proof;
- any topology advantage;
- any causal brain-growth mechanism.

Those remain VG-3D / later VG-3 work. Adult structural deltas between reference
and matched controls are descriptive only.
