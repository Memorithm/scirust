# Typed error examples

Prefer matching stable error variants over exact error strings:

```rust
# use scirust_core::compute_backend::{BackendError, ComputeBackend, CpuFallback};
let result = CpuFallback.execute_kernel(&[], &[1.0]);
assert!(matches!(result, Err(BackendError::Internal(_))));
```

When the error type does not implement `Debug`, avoid `.unwrap_err()` because
`Result::unwrap_err` requires the `Ok` type to implement `Debug`. A `match` can
extract the error without adding an unrelated trait requirement.

Exact message assertions are appropriate only when the textual message itself is
a documented stable interface.
