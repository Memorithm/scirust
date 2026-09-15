# Documentation release gate

Before a release or documentation-completion milestone:

- regenerate the API lexicon/example inventories from the release revision;
- reconcile compiler-confirmed public reachability for covered packages;
- run rustdoc and doctests for all policy-covered scopes;
- verify no documentation/example baseline regressed;
- review active exceptions and remove obsolete ones;
- check README/docs navigation and glossary links;
- ensure hardware/performance/security claims still point to applicable evidence;
- record unresolved qualification discrepancies rather than hiding them.

A release can contain experimental APIs, but their current contract and
limitations must still be documented precisely.
