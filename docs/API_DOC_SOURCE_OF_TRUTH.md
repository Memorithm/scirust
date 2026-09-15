# Documentation sources of truth

When documentation surfaces disagree, resolve them against the following
sources rather than choosing the most convenient prose:

1. implementation + validated tests for current runtime behavior;
2. compiler/rustdoc for effective public API shape and visibility;
3. item Rustdoc for the intended caller contract;
4. generated lexicon/example inventories for repository-wide navigation/metrics;
5. task guides for composed workflows;
6. historical audits/reports for the exact revisions they name.

A mismatch between implementation/tests and item Rustdoc is a defect to resolve,
not a reason to silently prefer one. A historical report never overrides current
source behavior.
