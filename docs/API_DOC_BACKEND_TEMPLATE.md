# Backend API documentation template

Backend-facing APIs require additional qualification beyond ordinary function
syntax.

Document:

- operation semantics and data layout;
- supported dtypes/shapes/alignments;
- device/driver/runtime prerequisites;
- whether availability is compile-time, runtime or both;
- fallback behavior, especially whether fallback is explicit or silent;
- synchronization/ownership requirements;
- numerical differences from reference execution;
- errors for unavailable devices, unsupported operations and allocation/launch
  failures;
- safety invariants for unsafe FFI/device interfaces;
- evidence level: compile-only, software adapter, or physical hardware run.

Provide one small portable/contract example when possible and a second example
covering capability detection or an unsupported/error path. Do not count a CUDA
or WGPU feature build as proof that a physical device executed the operation.
