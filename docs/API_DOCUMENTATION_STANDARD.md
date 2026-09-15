# SciRust public API documentation standard

This document defines the completion criterion for issue #1431. It supplements
rustdoc; it does not replace rustdoc's effective-visibility model.

## Per-callable contract

Every externally reachable public function or method must document the behavior
a caller needs to use it correctly. At minimum:

1. **Purpose and semantics** — what the operation computes or changes. For a
   mathematical API, name the convention or formula when ambiguity is possible.
2. **Inputs** — domains, tensor shapes, units, ordering, ownership and relevant
   finite/non-finite policies.
3. **Output** — meaning, shape, units and invariants.
4. **Failure behavior** — `# Errors` for fallible APIs, `# Panics` for
   caller-triggerable panics, and `# Safety` for public unsafe APIs.
5. **Execution qualifications** — determinism, backend/device limitations,
   numerical limitations or resource behavior when these affect correct use.
6. **Examples** — two ordinary runnable Rust examples for non-trivial public
   callables. A trivial accessor may use one example only when a reviewed policy
   scope explicitly records that exception.

Examples are executable evidence only when the crate's doctest job actually
runs them successfully. `no_run`, `ignore`, unknown-language fences and prose
snippets may be useful documentation but do not count as executed examples.

## Example design

Prefer two complementary examples rather than duplicated happy paths:

- example A: minimal nominal use with an assertion on the result;
- example B: boundary case, error path, alternative configuration, shape or
  deterministic property.

Examples must not require external credentials, mutable Internet services or
unbounded workloads. Hardware-specific examples should be paired with a
portable contract example and an explicit qualification instead of being
silently marked `ignore`.

## Lexicon quality

The generated lexicon is a navigation and audit surface. For each discovered
callable it should eventually expose:

- capability domain and package;
- module/source path and symbol;
- signature or an authoritative rustdoc link;
- first semantic summary;
- documentation status;
- ordinary example count and execution qualification;
- `Errors` / `Panics` / `Safety` section status when applicable;
- source revision or content fingerprint.

Source-level discovery must continue to state its limits. Syntactic `pub fn`
does not prove external reachability, and macro/trait/re-exported APIs can evade
a source regex. Rustdoc remains authoritative for the public surface.

## Monotonic gates

Documentation work is accepted in reviewable slices. A slice must:

- never increase the existing undocumented-callable debt;
- never reduce reviewed example coverage;
- add documentation and examples for newly introduced reachable APIs;
- run rustdoc and the relevant doctests;
- keep ordinary workspace CI green;
- update a baseline only after the corresponding source delta has been reviewed.

A counter is not the objective. Empty boilerplate, copied descriptions,
artificial examples, and blanket exception lists are documentation defects even
when they satisfy a mechanical scanner.

## Current execution order

The initial source inventory identified the largest adjacent-rustdoc gaps in
`scirust-core`, `scirust-sciagent`, `scirust-trader`, `scirust-variational`,
`scirust-gpu`, and `scirust-tensor-compile`. Work proceeds in focused module
slices so that semantics and doctests can be reviewed rather than hidden in a
multi-thousand-line generated patch.
