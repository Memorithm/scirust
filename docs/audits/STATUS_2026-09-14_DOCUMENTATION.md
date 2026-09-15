# SciRust documentation audit continuation — 2026-09-14

## Baseline evidence

This continuation follows PR #1430 at head
`e5fee8165ccc6ad9f4b1e6f2a50845ab99c34c18`. Its earlier source audit was pinned to
`e1237827f73268bf43b2b971674a697cd186d95e`; it is not an exhaustive semantic
verification of the repository.

The downloaded artifacts from Actions runs `34896662474` and `34896662430`
identify checkout `e86c4b9885dbf2bad0aafbf6030c8394c6beaa39`, a synthetic PR
merge commit rather than evidence of an actual master merge. SHA-256 digests
were recomputed and matched the artifact API digests:

- Lexicon: `b5bd0924328f2fe6c6ce777b20f495d56158517c8b9b54c75046b28bf5f0a298`.
- Contracts: `195ccdca4acf1209724242bcc34726981f9c80ef0856a9a3b2694814c845617f`.

| Observation at that checkout | Count |
|---|---:|
| Workspace members | 160 |
| Tracked Rust files, including archives/tests/vendor | 3,181 |
| Tracked Rust lines in the same broad scope | 1,089,280 |
| Library packages represented in the lexicon | 147 |
| Directly declared public callables indexed | 12,924 |
| Entries with adjacent rustdoc summaries | 10,543 |
| Entries without adjacent summaries | 2,381 |

The last two counts measure a lexical observation, not complete semantic
documentation. The prior index does not measure examples or their execution.
The negative-control logs were inspected: six attention regressions fail with
two passing controls on the original implementation; three new statistics
predicates fail while three original tests pass. Compilation failures were
not substituted for expected runtime failures.

## Work in this slice

Add an example-observation tool consuming the existing lexicon index, an
explicit reviewed source policy, CPU-only parser/policy regression tests,
JSON/Markdown artifacts and a separate actual doctest CI step. Expand all nine
`describe` functions to two ordinary examples apiece (18 in total, including
the pre-existing quantiles example). The new gate fails for each of those nine
functions when applied to their original documentation. The gate also covers
the six `comb` functions and their twelve examples added by the separate
corrective commit `f5a1040f454650c29cb7341513ecae3a440d94dd`. Thus the initial
policy covers 15 functions and 30 ordinary candidates, not the whole workspace.
See [the combinatorics follow-up](COMBINATORICS_2026-09-14.md) for the observed
Miri failure, source correction, boundary defects and exact-head validation
requirements. No green outcome for the new commits is assumed in this report.

The original `describe.rs` blob was reconstructed and verified as
`499585a6bb4538081c753c517ce977bd15e73c51`. A local comparison excluding only
`///` and `//!` documentation lines established that this slice does not change its implementations or
existing unit tests. The editing environment has no Rust toolchain; no local
Rust execution is claimed. Exact-revision GitHub Actions doctest results are
required before claiming execution or merging.

Known mean/variance overflow is now visible in public rustdoc, including its
impact on standard deviation and standard error. It remains an open numerical
fix, not an accepted permanent behavior or a completed correctness claim.

## Open work remains explicit

1. Overflow-/cancellation-aware mean and dispersion reductions with adversarial
   reference tests, non-finite policies and reproducibility review.
2. A typed, checked, lossless attention adapter; validation of mutable intent
   records before persistence or execution.
3. Verified reconstruction and kernel contracts for sparse/quantized variants.
4. Documentation and examples for the remaining API, prioritizing core,
   SciAgent, tensor compilation/runtime and other high-debt packages.
5. Separate feature/hardware and excluded-workspace qualification.

A grep hit is not a confirmed production defect; a successful HTML build is not
an executed example; a supported representation is not an executable kernel.
The new source gate does not waive any existing CI, hardware, license or
security policy and does not establish downstream adoption in TDI/KVLab/Forge.

## Reproduction and review

See [API example maintenance](../API_EXAMPLES.md) for commands, classifications,
source limitations and extension policy. Keep this report revision-bound;
regenerate the artifacts rather than treating these counts as current forever.
