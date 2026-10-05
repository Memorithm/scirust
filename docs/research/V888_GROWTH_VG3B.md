# V888-GROWTH VG-3B — deterministic matched topology controls

Status: implementation/qualification slice for SciRust issue #1500.

This slice promotes the topology-matched control ladder required by stage VG-3
into a reusable Rust contract under `scirust_graph::v888_growth`. It consumes the
same qualified V8CSR001 executable graph as VG-3A and generates unit-adjacency
control arms against deterministic induced subsets. It is an adult structural
null-model staging step, not a developmental-causality or topology-advantage
result.

## Why VG-3B now

VG-3A (`docs/research/V888_GROWTH_VG3A.md`, master commits through `f8e66ed`)
established exact per-neuron degree, contact multiplicity, reciprocity and SCC
facts. The programme document and VG-3A evidence boundary both defer the
required matched controls to VG-3B/VG-3C:

- edge-count matched random sparse;
- in/out-degree matched;
- reciprocity matched;
- block/modularity matched.

BOOL-0.2a previously staged the same ladder under `scripts/v888_bool02` for the
Boolean programme. VG-3B lifts the reusable generators into `scirust-graph` so
growth-proxy work can call them without depending on research-staging scripts.

## Frozen graph input

VG-3B consumes only the qualified V8CSR001 graph retained by the executable
graph contract:

- source qualification run: 35717145115;
- metadata nodes: 188,508;
- directed pairs: 13,620,865;
- induced contacts: 42,309,621;
- graph bytes: 166,466,540;
- graph SHA-256: 385111a69cc8a1d748c0bdfd9b0b738c51fe83cde98f45762553435a2d15a2a4.

Raw BANC files and the graph binary remain external to Git.

## Rust-owned control surface

Reusable helpers re-exported from `scirust_graph::v888_growth`:

- `v888_growth_select_subset` — deterministic ranked or weak-BFS dense subsets;
- `v888_growth_unit_reference` — induced unit adjacency (multiplicities dropped);
- `v888_growth_edge_count_control` — Floyd sampling of exact non-self digraphs;
- `v888_growth_rewire_control` — bounded directed double-edge switches;
- `v888_growth_matched_controls` — fail-closed four-arm ladder with audits.

Every comparison arm has identical node identities and the same directed unit
edge count `E` as the reference:

1. `edge_count`
2. `degree`
3. `degree_reciprocal`
4. `degree_reciprocal_block`

Contact multiplicity stays an exact measured source count on the reference pair
table only. Control arms are unit adjacency and are not conductance, event rate,
physiological strength or developmental-growth variables.

Block labels are caller-supplied anatomical annotations (for example region).
They are not inferred communities. A single distinct label makes the block
constraint vacuous; the exporter records that fact explicitly.

## Deterministic export

The example `v888_growth_vg3b` rejects a SHA-256 mismatch against the frozen
qualification envelope, selects a subset, writes:

- `nodes.tsv`
- `reference.edges.tsv`
- one `*.edges.tsv` per arm
- `summary.json`

and prints a compact stderr summary. The Thor workflow verifies the retained
graph byte size and SHA-256 before generation, runs the Rust tests/doctest, and
records the summary SHA-256 as compact evidence.

Pilot defaults intentionally stay within BOOL-0.2a scale (for example 512-node
weak-BFS subsets) so control generation remains reviewable before any full-graph
rewiring campaign.

The Thor workflow recipe is retained as
`docs/research/V888_GROWTH_VG3B_THOR_WORKFLOW.yml` until a credential with the
GitHub `workflow` scope can install it at `.github/workflows/v888-growth-vg3b-thor.yml`.

## What VG-3B does not claim

VG-3B does not yet implement or qualify:

- all-pairs reachability;
- centrality or rich-club diagnostics;
- inferred modular/community membership;
- morphology / hemilineage joins to control residuals;
- a uniform null ensemble or mixing-time proof;
- any topology advantage;
- any causal brain-growth mechanism.

Those remain VG-3C / later VG-3 work. Bounded switch chains that preserve
declared invariants are not automatically qualified random null models.
