# Performance review prompts during documentation

When performance is part of a public contract, inspect/document:

- asymptotic work and memory;
- avoidable repeated sorting/allocation/copies;
- whether a batched API can share preprocessing;
- backend/data-transfer overhead;
- cache/layout assumptions;
- benchmark evidence for any measured claim.

A documentation review can motivate a useful API such as a batched operation,
but do not claim measured speedup from complexity reasoning alone. Add a
benchmark before publishing a numeric performance comparison.
