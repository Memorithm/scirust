# V888-BOOL-0.2a — reproducible subsets and topology-control invariants

Author: Memorithm research programme. Protocol written 2026-09-22 before pilot execution.
Parent: ITD V888-BOOL-0.2, issue Memorithm/itd-simulator#58; source research issue #1500.
This is a research-staging CPU programme, not a public graph API or neural-task result.

## Source and executable separation

Consume the separately qualified V8CSR001 graph from run 35717145115, graph SHA-256
`385111a69cc8a1d748c0bdfd9b0b738c51fe83cde98f45762553435a2d15a2a4`.
Require its successful, hash-pinned qualification envelope and the exact metadata SHA.
No network download is needed. Raw synapses and the full graph remain external on Thor.

The existing SciRust bootstrap assigns general graph/sparse/event APIs to reusable
crates. This slice does not introduce another public graph library or new crate.
It adds a std-only staging reader/generator under `scripts/v888_bool02`, with independent
Python/NumPy oracles. Generic promotion into the shared graph layer remains separate.
No changes to existing product semantics, ITD V29.18 or protected evaluation surfaces.

## Preregistered pilot

`protocol.json` is the executable source of the grid, boundaries and exact source hashes.
Use weak-neighbor BFS subsets of 128, 512 and 2048 nodes for seeds 17, 29 and 43;
also retain rank-only 2048-node subsets for the same seeds. Run every case twice.
No case is dropped for low edge count, disconnected ports or unsuccessful rewiring.

Root IDs are ranked with the specified 64-bit SplitMix transformation and seed.
BFS visits both incoming and outgoing neighbors, ordered by the same rank; disconnected
components are refilled with the next unvisited ranked root. Original IDs are sorted
in the saved subset and remain unsigned integers. This selection is topology-biased,
not a representative sample of all neural tissue. Rank-only selection is retained as
an explicitly different sampling policy, not as a replacement chosen after results.

## Exact boundary and weight semantics

Save every crossing edge of the qualified induced graph, with original source/target
IDs, measured integer contact multiplicity and incoming/outgoing category. Independently
reconstruct the complete cut-edge list and the inside/outside partition. This boundary
is relative to the qualified metadata node set; earlier raw unknown endpoints remain
outside that source scope and are not recategorized as anatomical tissue.

Preserve measured internal multiplicities in `source_pairs.tsv`. All five comparison
arms instead use **unit adjacency**, including the reference arm. Thus the generator
does not pretend that shuffled contact multiplicities are measured conductances, nor
that weighted strengths are matched. Physiological signs/delays/thresholds are absent.

## Control ladder

Every arm has identical node identities and E directed non-self unit edges:

1. reference induced topology;
2. edge-count control: Floyd sampling over ordered non-self pairs;
3. degree control: bounded directed double-edge switches, preserving each node's in/out degree;
4. reciprocal control: also preserving each node's number of reciprocal neighbors;
5. anatomical-block control: also preserving the complete directed block-pair edge-count matrix.

Each rewired arm receives `min(32*E, 1000000)` proposals with an explicit per-arm seed.
Reject loops, duplicate pairs and switches without four distinct endpoints. The
reciprocal constraint is evaluated as an exact local degree change, not an aggregate
reciprocity approximation. Rust recounts the global invariants after construction;
Python independently recounts them from saved files.

Blocks come from the frozen metadata `region` field. Null/empty values are explicit,
not guessed. These are anatomical labels, **not inferred functional communities**.
A block with only one label makes the block constraint vacuous; expose that fact.
No preserved modularity score, SCC structure, path length or higher motif is claimed.

## Ports, replay and rejection tests

Freeze the same two input indices and one readout index for every arm from a separate
seeded root-ID ranking. This is only port placement: no encoder, readout fitting or
neural dynamics is introduced. Record reachability from each input to the readout;
never move an inconvenient port after observing a task result.

Synthetic tests use exact IDs above 2^53, independent selection/Counter-style graph
references, malformed byte streams, constant/empty/dense examples and deliberate
corruptions that preserve degree while changing reciprocity or block mixing. Repeated
real generation must reproduce every file byte-for-byte, not only the graph counts.

## Crucial qualification limit and next slice

An invariant-preserving control is **not automatically a qualified random null model**.
Record accepted proposals, final changed-edge count, disconnected ports and unchanged
outputs. No uniform stationary distribution, mixing time or independent ensemble is
established by the bounded switch chain. Degenerate graphs are retained as such.

V888-BOOL-0.2 remains active after a successful 0.2a pilot. Continuation 0.2b must
characterize diversity/mixing, partition choice and port-placement sensitivity before
using these controls to attribute neural-task effects. A finite, bounded recurrence
smoke can be prepared separately, but cannot be advertised as V888 topology superiority.

No delayed XOR, learning, growth law, GPU acceleration, model promotion or runtime
actuation is delivered by this slice. The first functional target stays delayed XOR
under the downstream dynamics and causal-I/O gates.
