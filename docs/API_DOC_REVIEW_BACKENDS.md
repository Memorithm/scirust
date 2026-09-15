# Backend contract review prompts

For backend-related APIs, inspect/document:

- capability detection;
- explicit versus silent fallback;
- operation/dtype/shape support matrix;
- compile-time feature and runtime availability distinction;
- allocation/transfer/synchronization behavior;
- error mapping;
- numerical/reference parity expectations;
- deterministic/reproducibility scope;
- actual qualification evidence by target/device.

A public factory should not claim to return an interchangeable backend when
implementations expose materially different operation contracts without an
adapter enforcing equivalence.
