# Contributing to the API documentation program

Before editing public Rustdoc:

1. read `API_DOCUMENTATION_STANDARD.md`, `API_DOC_CONVENTIONS.md` and the
   relevant domain review aid;
2. inspect implementation/tests/call sites for the selected module;
3. use `API_DOC_REVIEW_CHECKLIST.md` while editing;
4. keep the PR scope coherent and identify any runtime change separately;
5. execute package doctests/rustdoc and applicable tests;
6. update progress/evidence only from the final PR head.

Do not bulk-fill missing summaries with generated prose. If the implementation's
behavior is unclear, add/repair tests or open a focused finding rather than
inventing a contract.
