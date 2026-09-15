# Compiler/lowering API review prompts

For public compiler/lowering APIs, document/review:

- accepted source/IR version and invariants;
- transformation semantics and preserved properties;
- target/backend capability requirements;
- deterministic ordering/canonicalization;
- unsupported operation behavior;
- diagnostics/errors with source/node context;
- caching/fingerprint inputs;
- output artifact ownership/version compatibility.

Examples should lower/compile a tiny supported graph and a distinct unsupported
or invalid graph, asserting typed diagnostics rather than only successful object
construction.
