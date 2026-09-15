# Stale documentation detection

Documentation becomes stale when code changes without the corresponding contract
or generated observations changing.

Use several signals:

- source fingerprints in generated example/lexicon observations;
- CI regeneration/diff checks for checked-in generated indexes;
- doctest execution after relevant source changes;
- code review requiring contract updates for behavior/signature changes;
- revision labels on historical reports.

A matching source hash proves freshness of the observed source, not semantic
correctness. Semantic staleness still requires review when behavior changes
without a syntactic signature change.
