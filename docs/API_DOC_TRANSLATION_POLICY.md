# API documentation translation policy

The source Rustdoc and canonical API contract should have one authoritative
version per code revision. Translations may improve accessibility but must not
become independent semantic specifications that drift from implementation.

When translated guides exist:

- link back to the canonical source/API documentation;
- record revision/freshness where practical;
- update changed technical conventions consistently;
- do not translate generated inventories by hand;
- preserve equations, units, error variant names and code identifiers exactly
  where translation would alter their technical meaning.

The #1431 completion metric applies to the canonical public API documentation;
translation coverage is tracked separately.
