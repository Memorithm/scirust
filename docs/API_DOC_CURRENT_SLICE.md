# Current API documentation slice

Package: `scirust-core`

Module: `compute_backend`

## Why this module

It is a small public numerical/backend boundary with caller-visible non-finite,
empty-kernel and overflow behavior. It is suitable for establishing the
`scirust-core` documentation/doctest pattern before moving into larger tensor,
autodiff and persistence modules.

## Changes

- English module contract aligned with implementation.
- Error variants have semantic descriptions.
- `ComputeBackend`, both trait methods, `CpuFallback`, and `get_backend` have
  executable examples and explicit error/backend semantics where applicable.
- Dedicated core doctest/rustdoc workflow added.
- Task guide added without changing the convolution implementation.

## Acceptance

Do not merge this slice until exact-head CI proves the doctests compile/execute
and rustdoc builds with warnings denied. Normal repository checks remain
applicable.
