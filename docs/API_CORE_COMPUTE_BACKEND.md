# `scirust-core` convolution backend

Module: `scirust_core::compute_backend`.

This is the small CPU convolution abstraction implemented directly in
`scirust-core`. It should not be confused with the matrix/tensor GPU adapters in
`scirust-gpu`.

## Select the available backend

```rust
use scirust_core::compute_backend::get_backend;

let backend = get_backend()?;
assert!(backend.is_available());
# Ok::<(), scirust_core::compute_backend::BackendError>(())
```

The current factory returns the CPU fallback. It does not silently select a GPU.

## Execute a centered kernel

```rust
use scirust_core::compute_backend::{ComputeBackend, CpuFallback};

let backend = CpuFallback;
let output = backend.execute_kernel(&[1.0, 0.0, -1.0], &[1.0, 2.0, 3.0])?;
assert_eq!(output.len(), 3);
# Ok::<(), scirust_core::compute_backend::BackendError>(())
```

The operation keeps the input length and treats samples outside the input range
as zero. Products are accumulated in `f64` and checked before conversion to
`f32`.

## Handle invalid numerical input

```rust
use scirust_core::compute_backend::{BackendError, ComputeBackend, CpuFallback};

let result = CpuFallback.execute_kernel(&[f32::INFINITY], &[1.0]);
assert!(matches!(result, Err(BackendError::NanDetected { .. })));
```

An empty kernel is also rejected. A non-finite or out-of-range accumulated
output is returned as `BackendError::Overflow`; the backend does not silently
saturate it.

## Related references

- `scirust_core::compute_backend` rustdoc: item-level semantic contracts and
  executable examples.
- [API documentation standard](API_DOCUMENTATION_STANDARD.md): repository-wide
  requirements for public API documentation.
- [API lexicon](API_LEXICON.md): source-level callable navigation.
- [GPU guide](GPU.md): separately qualified GPU execution paths.
