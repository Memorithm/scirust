# SciRust public API documentation standard

This document defines the completion criterion for issue #1431. It supplements rustdoc; it does not replace rustdoc's effective-visibility model.

## Per-callable contract

Every externally reachable public function or method must document the behavior a caller needs to use it correctly. At minimum:

1. **Purpose and semantics** — what the operation computes or changes; name mathematical conventions when ambiguous.
2. **Inputs** — domains, tensor shapes, units, ordering, ownership and relevant finite/non-finite policies.
3. **Output** — meaning, shape, units and invariants.
4. **Failure behavior** — `# Errors` for fallible APIs, `# Panics` for caller-triggerable panics, and `# Safety` for public unsafe APIs.
5. **Execution qualifications** — determinism, backend/device limitations, numerical limitations or resource behavior when these affect correct use.
6. **Examples** — two ordinary runnable Rust examples for non-trivial public callables. A trivial accessor may use one example only under an explicit reviewed policy.

Examples are executable evidence only when the crate's doctest job actually runs them successfully. `no_run`, `ignore`, unknown-language fences and prose snippets do not count as executed examples.

## Example design

Prefer two complementary examples: a minimal nominal use with a semantic assertion, then a boundary/error/alternative configuration or deterministic property. Examples must not require credentials, mutable Internet services or unbounded workloads.

## Lexicon quality

The generated lexicon should eventually expose capability domain/package, module/source path, callable/signature, semantic summary, documentation status, ordinary example count/execution status, applicable Errors/Panics/Safety sections, and source provenance. Source-level `pub fn` discovery is not proof of effective external reachability; compiler/rustdoc evidence remains authoritative.

## Monotonic gates

A reviewed slice must never increase undocumented-callable debt or reduce reviewed example coverage; new reachable APIs require documentation/examples; relevant rustdoc/doctests and ordinary workspace CI must remain green. Baselines are updated only after reviewing the corresponding improvement.

Empty boilerplate, copied descriptions, artificial examples and blanket exception lists are documentation defects even when a counter passes.
