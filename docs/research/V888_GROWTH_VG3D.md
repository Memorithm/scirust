# V888-GROWTH VG-3D — inferred modularity, label mixing and partition agreement

Status: implementation slice for SciRust issue #1500 (Rust-owned primitives;
Thor exporter and evidence run follow in a later VG-3D slice).

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
  is claimed; ensemble qualification remains open.
- No topology advantage and no developmental-causality claim follows.

## Next VG-3D work

- deterministic exporter example and Thor workflow recipe (kept under `docs/`
  until a credential with `workflow` scope can install it) that joins the
  pilot subset with hemilineage labels and records compact TSV / JSON summaries;
- multi-seed control ensembles with per-arm dispersion of the VG-3C/VG-3D
  descriptors, as a precondition for any later qualification.
