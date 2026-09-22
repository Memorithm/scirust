# V888-BOOL-0.1 — executable graph identity

Programme revision: 2026-09-22.1. Tracking: Memorithm/itd-simulator#58 and SciRust #1500.
This is a research-staging format and source qualification, not a public graph API or a Boolean task result.

## Source and hypothesis

The inputs are the exact metadata, v3 edgelist and raw-v3 Parquet previously audited in `docs/V888_SYNAPSE_AUDIT_20260922.md`. Source bytes are rehashed before reading the full endpoint stream. Original data is never filtered in place. No further download is necessary.

The stronger question is whether each directed pair in the metadata-induced raw graph has exactly the v3 edgelist contact multiplicity. Per-node equality alone cannot answer it: swapping two destinations can preserve every node total while changing the graph.

The criterion is zero unexpected induced pairs and zero differing reference pair weights, with complete accounting of both-known, pre-only, post-only and neither-known raw endpoints. IDs remain exact unsigned integers even beyond floating-point integer precision. Synapse-ID uniqueness is outside this audit.

The Rust executable in `scripts/v888_bool01/pair_graph.rs` owns exact counting and CSR writing. The Python layer performs Arrow decoding, provenance and independent binary readback; it is not the future simulation engine.

## Binary contract: V8CSR001

All integers are little-endian. No host-sized integer is serialized.

| Order | Type | Meaning |
| --- | --- | --- |
| 1 | eight bytes | ASCII magic `V8CSR001` |
| 2 | u64 | number of nodes N |
| 3 | u64 | number of directed pairs E |
| 4 | N × u64 | original BANC root IDs in strictly increasing order |
| 5 | (N+1) × u64 | outgoing row offsets, beginning at 0 and ending at E |
| 6 | E × u32 | target indices, strictly increasing within each source row |
| 7 | E × u64 | positive integer contact multiplicities in the same order |

Exact file length is `32 + 16*N + 12*E` bytes. Every source node is retained, including isolated nodes. Rows are presynaptic and targets are postsynaptic. There is no conductance, sign, threshold, delay or stimulation policy encoded here.

The graph SHA-256 is bound to source hashes, filtering policy, index convention and the exact node-map SHA-256 in a separate graph-identity record. Execution commit, compiler, binary hash and protocol identity are retained separately.

Publication is conditional on pairwise agreement. The output directory and graph file are exclusive-create; failed runs retain diagnostics but do not publish an accepted graph. Consumers must require the successful qualification envelope and matching graph hash, not merely a file with a plausible name.

## Controls and scope

Seven Rust tests exercise multiplicities, boundaries, changed pairs with unchanged node totals, large IDs, invalid references, truncation and overflow. Eight independent Python/process test methods include twelve random multiset seeds, exact Counter references, input-order replay and independent CSR parsing.

The Arrow batch is bounded at 131,072 rows. The qualification job requests an 8 GiB address-space ceiling and 45-minute wall-clock timeout. These are configured limits, not measured resource usage or a performance comparison.

The CPU run uses the existing Thor-labelled ARM64 runner. Checkout is isolated in a per-run path so an old non-writable artifact in another worktree does not require deletion, privilege escalation or a global permission change. Existing raw data and unrelated workspace artifacts remain untouched.

## Integration and continuation

This branch is stacked on `research/v888-growth-thor-20260922` / PR #1502. Qualification on a research branch does not mean availability in master. Integration remains subject to exact-head checks and applicable merge policies.

A successful source qualification allows V888-BOOL-0.2 to begin deterministic subset extraction and topology-matched controls. Subsets must preserve source identity, sampling seed and cut-edge accounting. Recurrent dynamics and delayed-XOR evaluation remain separate downstream milestones; no model or biological advantage is inferred from this graph audit.
