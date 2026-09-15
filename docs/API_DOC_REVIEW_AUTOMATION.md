# Automation boundaries for API documentation review

Automation is appropriate for:

- Cargo/source inventory;
- locating missing summaries/sections;
- classifying documentation code fences;
- source hashes/staleness;
- compiler/rustdoc reachability extraction;
- executing doctests/tests/lints;
- generating indexes and metric reports.

Automation is not accepted as sole authority for:

- mathematical convention;
- caller-visible semantics;
- whether an error/panic is intended;
- safety invariants;
- performance/security/hardware claims;
- whether two examples are semantically distinct.

Agents can draft documentation, but completion requires evidence-grounded review
against implementation/tests and exact-head CI.
