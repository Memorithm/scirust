# Parallel and reproducibility API documentation template

For public parallel/SIMD/distributed APIs, document:

- partitioning and reduction semantics;
- ordering guarantees;
- thread/process safety and mutable shared state;
- deterministic versus schedule-dependent behavior;
- floating-point accumulation differences;
- required feature flags/targets;
- resource/concurrency limits and failure behavior.

Examples should demonstrate a small deterministic property and a second
serial/parallel equivalence or validation case when the API promises it. Do not
claim bit-identical reproducibility from approximate numerical agreement.
