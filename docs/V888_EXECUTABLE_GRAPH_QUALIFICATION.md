# V888 executable graph qualification for downstream consumers

Status: reusable SciRust graph contract for issue #1520. This document records a software/source qualification boundary, not a model-quality or topology-advantage result.

## Qualified source evidence

The public consumer contract reuses the already observed V888-BOOL-0.1 source qualification rather than inventing another graph representation.

Pinned source/software evidence:

- programme revision: `2026-09-22.1`;
- successful source run: `35717145115`;
- producer commit: `d61ffb6a79ebe975c1d867ad1a728fd01126fde0`;
- Rust worker SHA-256: `dbf349dafcde9b4967bb514e8875f375801a2994315126c5f09458e2f799ed02`;
- protocol SHA-256: `497cdcd57ac8c1d9052b8df6a0b1fb5b2839cdb61079ac4ad719d4f7bb080ad7`;
- metadata SHA-256: `86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098`;
- v3 edgelist SHA-256: `8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b`;
- raw-v3 synapse SHA-256: `0dfb5cf89ba156d076beab2da38d87eaa63dcbe45d76f86b108570fb5b961dd0`.

The selected metadata-induced graph has:

| scope | exact count |
| --- | ---: |
| metadata nodes | 188,508 |
| directed pairs | 13,620,865 |
| contacts on those pairs | 42,309,621 |
| raw-v3 records scanned | 198,816,365 |
| raw records with only presynaptic endpoint in metadata | 141,812,486 |
| raw records with only postsynaptic endpoint in metadata | 14,694,258 |
| raw records with neither endpoint in metadata | 0 |

The scopes are intentionally different. A directed pair is not a contact and neither quantity is the full raw-record count.

The canonical executable graph is the retained `V8CSR001` artifact:

- graph bytes: 166,466,540;
- graph SHA-256: `385111a69cc8a1d748c0bdfd9b0b738c51fe83cde98f45762553435a2d15a2a4`;
- node-map SHA-256: `eaa481b3aef42f63fb9e2e8f9c27d1405c073c54d11cbcd656ca9432904f425a`;
- graph/source/filter identity: `3cfbed6392a097865147ac08fae65f36a986b2ba6d386b1f5d61b9cfbefd1dd9`;
- retained logical host path: `$HOME/datasets/banc_v888/analysis/bool01-35717145115-1/graph.csr`.

The original detailed evidence remains in `V888_BOOL_0_1_RESULT_20260922.md` and `V888_SYNAPSE_AUDIT_20260922.md`.

## Public Rust contract

`scirust_graph::v888_executable` promotes the previously research-only graph format into a reusable, fail-closed SciRust interface.

It validates and exposes:

- exact non-zero BANC root IDs;
- deterministic dense node-index mapping;
- exact directed pairs;
- positive integer contact multiplicities;
- canonical byte replay;
- raw SHA-256 graph integrity identity;
- subset cut-edge accounting with pairs and contacts kept separate;
- active-node, directed-pair, contact-multiplicity and caller-observed event accounting.

The static graph does **not** invent or infer:

- neurotransmitter sign;
- conductance;
- threshold;
- physiological delay;
- event rate;
- stimulation policy;
- biological time.

Contact multiplicity is never silently converted into one of those quantities.

## Reproduction gate

The exact-head qualification workflow for this slice performs two distinct checks on the Thor-labelled ARM64 runner:

1. rerun the retained V888-BOOL-0.1 source qualification from the frozen external BANC v888 inputs, producing a fresh canonical `V8CSR001` graph;
2. consume that freshly reproduced graph through the public `scirust-graph` validator and require exact byte size, node count, directed-pair count, contact count, SHA-256 and byte-for-byte canonical replay.

The graph binary and raw source data stay outside Git and are not uploaded as repository artifacts. Compact test/provenance logs may be retained.

## Downstream boundary

This gate supplies a qualified executable graph interface to SML-GENIUS CSP / V888-BOOL-4.0 and the ITD V888-BOOL programme. It does not establish that V888 topology is superior to random, degree-matched, reciprocity-matched or module-matched controls.

SML model promotion remains controlled by its own independent quality/resource gates. ITD scientific conclusions remain controlled by ITD. SciRust owns the reusable graph/source/accounting substrate.

Closing SciRust #1520 therefore means the executable graph prerequisite is satisfied at this declared source/software scope. It does not close #1500, ITD #58, SML CSP research, delayed-XOR qualification, protected final evaluation or any runtime-default decision.
