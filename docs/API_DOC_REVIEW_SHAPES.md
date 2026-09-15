# Tensor shape contract review prompts

For shape-bearing APIs, inspect:

- rank validation before indexing dimensions;
- zero dimensions;
- checked conversion between `usize`, `u32`, `u64` and byte counts;
- checked products/sums for storage sizes;
- batch/head/channel agreement across operands;
- broadcast versus exact-shape semantics;
- rectangular sequence/matrix cases;
- output shape derivation;
- representation/storage accounting consistency.

Prefer returning a typed validation error before arithmetic that can panic or
truncate. Document unsupported shape families explicitly rather than coercing
them into a lossy tuple/configuration.
