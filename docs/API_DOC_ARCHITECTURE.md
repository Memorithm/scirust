# API documentation evidence architecture

The #1431 program separates four concerns that are easy to conflate.

## 1. Discovery

`scripts/api-lexicon.py` scans workspace library sources for directly declared
public callables. This gives deterministic repository-wide discovery and
capability-domain navigation. It is not a compiler visibility proof.

## 2. Semantic documentation

Rustdoc beside the implementation is the primary callable contract. It explains
what the function means, its valid inputs, output, failures, numerical/backend
qualifications and examples. Hand-written guides may compose several APIs into a
task, but they do not replace item-level contracts.

## 3. Example observation

`scripts/api-example-audit.py` observes documentation fences and distinguishes
ordinary Rust examples from `no_run`, `compile_fail`, `ignore`, `should_panic`,
unknown-language and malformed fences. Observation answers "is an example
present?", not "did it execute successfully?".

## 4. Execution evidence

`cargo test --doc` on the exact source revision establishes that ordinary
examples compile and execute. `cargo doc` with warnings denied checks the
rendered documentation surface. Normal workspace CI remains necessary because a
documentation review can expose and then change production contracts.

## Data flow

```text
Cargo metadata + source
        |
        v
api-lexicon.py ---------> API_LEXICON.md / JSON index
        |                         |
        |                         v
        +-----------------> api-example-audit.py
                                  |
                                  v
                         example observations
                                  |
                                  v
                          cargo test --doc
                                  |
                                  v
                     revision-bound evidence
```

A future compiler-derived reconciliation layer will add effective reachability
to this flow. Until then, source-level items must remain explicitly labelled as
syntactic discoveries.
