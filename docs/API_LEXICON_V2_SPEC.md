# API lexicon v2 — qualitative metadata specification

The existing `scripts/api-lexicon.py` is the canonical source-level discovery
mechanism. The v2 work extends that mechanism; it must not create a second list
of workspace APIs.

## Entry model

A qualitative lexicon entry should expose the following fields when they can be
established without inventing semantics:

```text
package
capability_domains[]
source
line
symbol
signature
summary
documentation_status
ordinary_example_count
example_execution_status
has_errors_section
has_panics_section
has_safety_section
source_sha256
rustdoc_url_or_path
```

`has_errors_section`, `has_panics_section` and `has_safety_section` are
observations, not automatic correctness judgments. Their requiredness depends on
the actual signature and implementation contract.

## Reachability

The regex source inventory currently discovers directly declared `pub fn`
items. It must continue to label this as syntactic discovery. Effective public
reachability requires rustdoc metadata or another compiler-derived view because
module visibility, cfg expansion, re-exports, macros and trait methods alter the
real API surface.

The v2 implementation should therefore support a three-state reachability field
when compiler-derived evidence is added:

- `reachable` — confirmed in the generated public rustdoc surface;
- `not_reachable` — syntactically public but hidden behind a non-public path;
- `unknown` — source observation has not yet been reconciled with rustdoc.

No `unknown` item may be silently counted as externally reachable.

## Example evidence

The existing `scripts/api-example-audit.py` distinguishes ordinary Rust fences
from `no_run`, `compile_fail`, `ignore`, `should_panic`, unknown-language,
empty and malformed fences. The lexicon should consume those observations.

An ordinary example becomes `executed` only when the relevant doctest command
succeeds on the same source revision. Merely finding a code fence is
`observed`, not execution evidence.

## Generated outputs

Maintain two complementary outputs:

- Markdown: human navigation grouped by capability domain/package/module;
- JSON: complete machine-readable observations for CI, audits and downstream
  documentation tooling.

Generated files must carry the source revision or source hashes necessary to
prevent stale documentation metrics from being presented as current evidence.

## CI policy

The migration should be monotonic. Existing documentation-debt and reviewed
example baselines remain valid until their replacements have equivalent or
stronger semantics. Do not lower a baseline to make a PR green. A newly added
public callable should fail CI when it introduces new documentation/example
debt in a policy-covered scope.
