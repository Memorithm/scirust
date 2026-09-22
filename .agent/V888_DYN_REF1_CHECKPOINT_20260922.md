# DYN-REF1 checkpoint — 2026-09-22

Reusable delayed recurrent Rust reference for the V888 forcing programme. This is research staging, not a public scirust-sim API or ML-maturity promotion.

Implementation: PR #1509, research/v888-dyn-ref1, source 75f43ec35b1118c98b23bbde8fc8e1d6bfd42036. Successful Thor CPU run: https://github.com/Memorithm/scirust/actions/runs/35761411608 . Sixteen Rust unit tests and eight independent process methods passed. The 152 synthetic cases span 9,576 network ticks / 58,832 neuron updates per panel pass, with zero oracle mismatches and byte-identical replay.

Artifact 10709639798, ZIP SHA-256 dea5b888055feaace9f847345891703bdbd9dd8089250dab0bee0a1de9a05fe9, was retrieved and rechecked. Exact sources, protocol, fixtures and all 152 traces were independently revalidated after retrieval; this is not a second Thor run or signed attestation.

Canonical detailed report: https://github.com/Memorithm/itd-simulator/blob/4a4549dbcf5bc82b00bcd0a28a48c9aafce2822c/.agent/RESEARCH_STATUS_20260922_DYN_REF1.md . ITD bootstrap continuation: commit 60ac40598af704f0d6e810122bb51ea195325a88.

The core exposes separate discrete LIF, signed bounded integer and Boolean (affine/nonlinear) recurrence. Delays, synchronous commit, refractory/reset, cancellation-before-clipping, payload and work budgets are explicit. Immutable configuration and transient state are separate. No weights, readout or learned checkpoints are trained. The designed pulse cycle is an oracle, not learned memory.

No BANC graph was read, game candidate connected or training executed. Fixed-step edge scans and byte histories are not event-driven or bit-packing performance evidence. BOOL-0.2b graph/control/partition/port diagnostics and source-bound integration remain open. Reuse this qualified module instead of rebuilding its equations; integrate into scirust-sim only through a separately reviewed general API and independent consumer tests.

PR #1509 remains separate from master integration and required exact-head CI/review. No merge, public API promotion, maturity-score change, raw-data mutation, biological claim, final-holdout authorization or trading work is recorded by this checkpoint.
