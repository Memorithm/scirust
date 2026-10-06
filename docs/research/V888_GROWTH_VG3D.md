# V888-GROWTH VG-3D — inferred modularity, label mixing and partition agreement

Status: implementation slices for SciRust issue #1500 (Rust-owned primitives,
the annotation join, pilot exporter and Thor workflow recipe, then multi-seed
control ensembles wired into that exporter; the Thor evidence run itself has
not been executed yet).

VG-3C stopped before inferred modularity, morphology / hemilineage joins and
mixing descriptors. This slice adds the reusable deterministic primitives those
joins need, on the same unit-adjacency contract as VG-3A/3B/3C, so reference and
matched-control arms are scored identically. It is an adult structural
descriptor stage, not a developmental-causality or topology-advantage result.

## Rust-owned surface

Re-exported from `scirust_graph::v888_growth` (module
`scirust_graph::v888_growth_modularity`):

- `v888_growth_directed_modularity` — Leicht–Newman directed modularity of a
  caller-supplied partition, retained as the exact rational
  `(m * intra - Σ_c out_c * in_c) / m²` (`None` for an edgeless graph);
- `v888_growth_infer_communities` — deterministic directed Louvain (local
  moving in ascending node order, then aggregation into weighted super-nodes).
  Move decisions compare exact `i128` gains
  `m * (w_to(C) + w_from(C)) - (k_out * in_C + k_in * out_C)`; a node moves only
  on a strict improvement and ties go to the smallest community label. Every
  accepted move strictly increases exact modularity, so the search terminates;
  `V888GrowthLouvainOptions` additionally bounds levels and passes and the
  result reports `hit_bound`. Final labels are canonical (numbered by smallest
  member), so identical inputs give identical partitions on every platform;
- `v888_growth_label_mixing` — exact source-label × target-label edge counts
  and Newman nominal assortativity `(m * same - S) / (m² - S)` for an opaque
  caller annotation (hemilineage, anatomical block, or an inferred community);
- `v888_growth_partition_agreement` — exact pair counts and the adjusted Rand
  index between two partitions (for example inferred modules versus a
  hemilineage annotation);
- `v888_growth_compare_modularity_arms` — per-arm inferred partition, optional
  annotation modularity / mixing / agreement, and nano-scaled integer deltas
  against the reference arm (half-away-from-zero rounding shared with VG-3C).

Contact multiplicity stays outside these unit-adjacency metrics and is not
conductance. Annotations are caller-supplied labels; no BANC rows or raw
annotation tables are added to Git.

## Interpretation boundary

- Louvain returns a local optimum of directed modularity under a fixed sweep
  order. It is a reproducible descriptor, not the globally optimal partition and
  not a biological module claim.
- Annotation joins (hemilineage, block) measure how strongly unit edges follow
  the labels on each arm. A reference-versus-control delta is descriptive only.
- No uniform null ensemble or mixing-time property of the VG-3B rewiring chains
  is claimed. The multi-seed ensemble below reports dispersion across
  realisations; its below / equal / above-reference counts are descriptive,
  not p-values, and ensemble qualification remains open.
- No topology advantage and no developmental-causality claim follows.

## Annotation join and pilot exporter (second VG-3D slice)

Re-exported from `scirust_graph::v888_growth` (module
`scirust_graph::v888_growth_annotation`):

- `v888_growth_parse_annotation_tsv` — fail-closed reader for
  `node_id<TAB>label` rows (comments and blank lines skipped; malformed ids,
  empty labels, extra fields, duplicate ids and the reserved bucket name are
  rejected with the offending line);
- `v888_growth_join_annotation` — maps the selected subset onto dense label
  indices numbered in lexicographic label order, with every unannotated node
  placed in one explicit trailing `<unlabelled>` bucket. The dense labels do
  not depend on table row order, so the join is reproducible across platforms.

The example `scirust-graph/examples/v888_growth_vg3d.rs`

```text
v888_growth_vg3d <graph.csr> <out_dir> <size<=512> <seed> <ranked|weak_bfs> \
    [--blocks blocks.tsv] [--labels labels.tsv]
```

verifies the qualified graph SHA-256, selects the same pilot subset and VG-3B
matched controls as VG-3B/3C, runs `v888_growth_compare_modularity_arms` with
the joined annotation, and writes only compact summaries:

| file | content |
|---|---|
| `nodes.tsv` | local index, node id, block, annotation index, reference community |
| `annotation_labels.tsv` | dense index, label, selected-node count (with `--labels`) |
| `mixing_reference.tsv` | reference-arm source-label × target-label unit-edge counts |
| `modularity_per_arm.tsv` | inferred communities, exact `Q` rational, Louvain work, annotation `Q` / assortativity / ARI per arm |
| `deltas.tsv` | nano-scaled integer deltas of each control arm against the reference |
| `summary.json` | provenance, Louvain bounds, annotation coverage and the explicit `false` claim flags |

The Thor workflow recipe `docs/research/V888_GROWTH_VG3D_THOR_WORKFLOW.yml`
(intended path `.github/workflows/v888-growth-vg3d-thor.yml`; kept under
`docs/` because the available token lacks `workflow` scope) derives the
hemilineage table from the external `banc_888_meta.feather` into a private
runner directory that is never uploaded, records only its row count and
SHA-256, and uploads the compact summaries above.

## Multi-seed control ensembles (third and fourth VG-3D slices)

A single VG-3B draw cannot show how far a descriptor moves from one control
realisation to the next. Re-exported from `scirust_graph::v888_growth`
(module `scirust_graph::v888_growth_ensemble`):

- `v888_growth_control_ensemble` — regenerates the four-arm ladder for a
  duplicate-free list of 2 to 1024 seeds, evaluates the VG-3C reachability /
  harmonic closeness / betweenness descriptors and the VG-3D inferred
  modularity / annotation descriptors on every arm, and returns raw per-seed
  samples plus one dispersion summary per `(arm, descriptor)`. Each arm also
  reports how many distinct control graphs its seeds produced, so a stuck
  rewiring chain shows up instead of reading as zero dispersion;
- `v888_growth_dispersion` — deterministic minimum, maximum, input-order mean,
  two-pass sample standard deviation, median, and nano-rounded counts of draws
  below / equal to / above the reference value;
- `v888_growth_ensemble_seeds` — the exporter's seed rule,
  `seed, seed + 1, …, seed + count - 1` (wrapping at `u64::MAX`), so the first
  ensemble draw is the same realisation as the single-draw rows.

The pilot exporter takes `--ensemble <count>` (2 to 64 for the pilot):

```text
v888_growth_vg3d <graph.csr> <out_dir> <size<=512> <seed> <ranked|weak_bfs> \
    [--blocks blocks.tsv] [--labels labels.tsv] [--ensemble count]
```

and then also writes

| file | content |
|---|---|
| `ensemble_dispersion.tsv` | per arm and descriptor: draws, distinct graphs, reference, defined / undefined, min, max, mean, SD, median, below / equal / above-reference counts |
| `ensemble_samples.tsv` | one row per `(seed, arm)` with accepted swaps and every descriptor value |

plus an `ensemble` object in `summary.json` (seed count and range, sample
count, per-arm distinct graphs and nano-scaled inferred-`Q` mean / SD, and an
explicit `comparison_counts_are_p_values: false`). Annotation-only descriptors
are omitted without `--labels`. The exporter fails closed if the ensemble's
reference modularity disagrees with the single-draw reference arm. The Thor
recipe runs `--ensemble 16` (seeds 29 to 44) and checks these fields.

## Next VG-3D work

- install the workflow recipe with a `workflow`-scoped credential and record
  the first Thor evidence run, including the 16-seed ensemble summaries;
- only after that run, decide whether any descriptor's dispersion is narrow
  enough to support a later qualification step.
