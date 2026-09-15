# Performance documentation policy

API documentation may describe algorithmic complexity or measured performance,
but these are different claims.

Complexity claims should name the variables and operation being counted.
Measured claims require a reproducible benchmark with workload, environment,
revision and comparison baseline. Avoid evergreen statements such as "fast" or
"faster than X" when the evidence is a historical run.

Examples in Rustdoc are correctness/usage fixtures, not benchmarks. Keep them
small and deterministic. Link performance-sensitive APIs to revision-bound
benchmark evidence instead of turning doctests into timing tests.
