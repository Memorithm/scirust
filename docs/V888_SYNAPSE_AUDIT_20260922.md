# BANC v888 — full synapse and induced-graph audit

Author: Memorithm research pipeline. Observed: 2026-09-22.
Scope: external data integrity and exploratory adult-connectome structure, not developmental causality.

## Completed on Thor

The complete `banc_888_synapses_v3_enriched.parquet` is stored in persistent Thor storage, outside Git, under the dataset directory `/mnt/nvme/github-runners/home/datasets/banc_v888`.

The selected file is **19,733,123,829 bytes** (19.73 decimal GB), not the older approximately 5.6 GB estimate in the dataset tutorial. The file contains **198,816,365 rows**. No raw files were removed, filtered in place, committed to Git, or uploaded as workflow artifacts.

| Verified principal input | Bytes | Observed rows |
| --- | ---: | ---: |
| `banc_888_meta.feather` | 57,503,026 | 188,508 |
| `banc_888_metrics.feather` | 12,285,378 | 188,508 |
| `banc_888_edgelist_simple_v3.feather` | 359,161,658 | 13,620,865 |
| `banc_888_synapses_v3_enriched.parquet` | 19,733,123,829 | 198,816,365 |

The four inputs were fully read to recompute SHA-256 and MD5. Their SHA-256 values agree with the retained Phase-0 manifest. Their byte sizes and MD5 values agree with the Google Cloud Storage object headers observed during both full audits. Object-generation identifiers are retained in `source_identity.json`; these are not biological development dates.

Raw synapse SHA-256:

`0dfb5cf89ba156d076beab2da38d87eaa63dcbe45d76f86b108570fb5b961dd0`

Raw synapse object generation: `1786578708197684`.

## Full raw-file inspection

A std-only Rust executable consumes little-endian integer endpoint records from bounded Arrow-decoding batches. It checks integer overflow, truncated streams, duplicate node IDs, and zero edge weights. Five Rust tests and three independent Python/process/Arrow tests passed on Thor before the data scan. The induced-graph pass also checks Arrow membership against an independent Python-set fixture, including IDs above the exact floating-point integer range.

Run **35709233870**, source **e7a61ed220d994076adb6504025cd72fdd1f73e8**, read all **198,816,365** raw rows for pre/post endpoints, size and region. It observed zero self-endpoint records, minimum size 10, and zero records with size below 10. This is not an audit of every value in all 31 columns, nor a uniqueness proof for synapse IDs.

The first pass intentionally compared the full raw endpoint totals with the published v3 edgelist. It found different source scopes: the raw table is not interchangeable with the edgelist without an explicit node-set restriction. These differences are retained as diagnostics rather than discarded as a failed scientific result.

## Exact reconciliation on a declared node set

Run **35710150870**, source **6c74a9dc42a7b9ae82e6bd0c8499446041ae9bcd**, again scanned every raw endpoint pair. The node set was the unique `banc_888_id` values in the frozen metadata table. A raw record was included in the induced comparison if and only if **both** endpoints belonged to that set.

| Membership in the metadata node set | Raw records |
| --- | ---: |
| Both endpoints present | 42,309,621 |
| Only presynaptic endpoint present | 141,812,486 |
| Only postsynaptic endpoint present | 14,694,258 |
| Neither endpoint present | 0 |
| Total | 198,816,365 |

The 42,309,621 retained contacts equal the sum of the integer weights in the 13,620,865-row v3 edgelist. Incoming and outgoing integer totals were compared separately for every one of the **188,508** metadata nodes: **zero mismatching nodes**. No edgelist endpoint fell outside this metadata node set.

This establishes exact **node-total reconciliation**, not pair-by-pair edge equality or synapse-ID uniqueness. The biological identity or reconstruction status of endpoints absent from metadata was not determined. Absence must not be relabelled as a particular cell type or segmentation defect without independent evidence.

## Growth-analysis safeguards

Nonempty hemilineage labels occur for **50,501** metadata neurons, not the entire nervous system. The 259 distinct label strings in Phase 0 are not 259 independently certified developmental lineages. Current descriptive grouping uses `(region, side, hemilineage, neuromere)`, producing **1,252 literal annotation groups**, with ambiguous labels retained rather than silently resolved.

The upstream tutorial documents metadata `input_connections` and `output_connections` as detector-v2 counters. This audit recomputes detector-v3 counts directly instead of mixing those counters with v3 observations. Morphology aggregates carry per-metric valid counts and preserve missingness. A zero measurement is not automatically interpreted as biological absence.

One adult snapshot does not supply growth rates, a developmental time course, or evidence that a particular variable causes growth. Future work must distinguish descriptive adult allometry from developmental causality. The immediate research use is a qualified source set for morphology/connectivity comparisons and explicit missing-data controls. No biological growth law, AI-model advantage, runtime policy, or GPU performance claim is promoted here.

## Reproducible evidence

- Phase-0 run: https://github.com/Memorithm/scirust/actions/runs/35706848881
- Full raw scan: https://github.com/Memorithm/scirust/actions/runs/35709233870
- Induced-graph reconciliation: https://github.com/Memorithm/scirust/actions/runs/35710150870
- Full-scan artifact **10685938370**: SHA-256 `a54e364ca84eef5c58ba10ff7b035a80663fa5073878c4b0c1b461bf6c4edd2f`
- Reconciliation artifact **10687100052**: SHA-256 `0d781e7f21206b9259e92ca1f1ba5a2ea4d4e5b9bc70ef32f4e2bc3d143baf51`
- Both ZIP digests were independently recomputed after retrieving the artifacts and matched the GitHub artifact metadata.
- Each bundle retains the executed commit and compiled Rust worker hash. These identities are reproducibility metadata, not a signed build attestation.
- Upstream schema documentation: https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md

Tracking: issue #1500; research pipeline PR #1502; reusable statistics primitive PR #1501. Repository-wide CI and public-library promotion are separate from the successful scoped Thor data audits.
