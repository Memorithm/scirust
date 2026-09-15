# API documentation evidence levels

Documentation claims should identify the strongest evidence actually available.

## E0 — source observation

A scanner found a declaration, summary, code fence or marker. This supports an
inventory statement only.

## E1 — compile evidence

The documented API and example compile for a stated target/toolchain. This does
not establish runtime behavior.

## E2 — execution evidence

A deterministic doctest/unit/integration test executed and asserted the stated
contract on the exact revision.

## E3 — cross-path qualification

The contract was exercised on additional relevant targets/backends such as
Windows, macOS, aarch64, WGPU or CUDA. Only name paths that actually ran.

## E4 — external/reference validation

A numerical/scientific result was compared against an exact derivation,
independent implementation, published reference or high-precision oracle under
a documented tolerance.

## E5 — measured performance evidence

A performance claim is supported by a reproducible benchmark with environment,
workload and comparison baseline recorded.

Higher levels do not automatically apply to every claim in a module. For
example, a CUDA compile check is E1 for CUDA compilation, not E3 execution on a
physical CUDA device. The lexicon may eventually surface these qualifications
without collapsing them into a single "verified" flag.
