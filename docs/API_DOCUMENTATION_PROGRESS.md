# SciRust API documentation progress

Tracking issue: #1431.

This file records reviewed documentation slices. Counts from source scanners are
navigation evidence, not claims of effective public reachability.

## Completed before this branch

- `scirust-stats::describe` and `scirust-stats::comb`: reviewed two-example
  source policy merged through #1432.
- The stats slice executes its ordinary examples as doctests in the API lexicon
  workflow.
- The original attention/statistics correctness audit was merged through #1430.

## Current slice: `scirust-core::compute_backend`

The module is being normalized to the #1431 standard:

- module-level execution and numerical-safety contract;
- documented backend error variants;
- trait semantics for availability and centered 1-D convolution;
- explicit error behavior for non-finite input, empty kernels and output
  overflow;
- two ordinary runnable examples on the public trait and each trait method;
- two ordinary runnable examples on `CpuFallback` and `get_backend`;
- dedicated CI executing `scirust-core` doctests and rustdoc with warnings
  denied.

No production convolution implementation is intentionally changed by this
slice. Exact-head CI is the evidence for example compilation and execution.

## Next slices

Continue through `scirust-core` by coherent module rather than by raw function
count. Prioritize modules whose public contracts control memory, numerical
correctness, backend selection, serialization/checkpointing and autodiff before
pure convenience helpers. After `scirust-core`, continue with the remaining
large gaps recorded in #1431.
