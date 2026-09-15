# Resource limits for documentation examples

Ordinary doctests should remain fast and bounded so repository-wide example
coverage is sustainable.

Prefer tiny tensors/matrices, metadata-only shape fixtures, fixed short signals,
small deterministic solver problems and in-memory serialization. Do not download
models/datasets, require GPUs, sleep/retry network services or allocate huge
buffers in ordinary examples.

Expensive end-to-end demonstrations belong in examples/bench/evidence workflows
and can be linked from Rustdoc. The item itself still needs a small executable
contract example unless an explicit exception is approved.
