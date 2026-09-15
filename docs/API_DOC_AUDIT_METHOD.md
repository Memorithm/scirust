# Semantic audit method for public APIs

For each callable under review:

1. locate implementation and all directly relevant tests;
2. identify input/output invariants from code, not from the old comment alone;
3. test edge cases implied by arithmetic/indexing/conversions;
4. inspect error and panic paths;
5. inspect backend/feature/cfg differences;
6. compare numerical formulas against a known definition/reference when the
   convention is ambiguous or precision-sensitive;
7. write Rustdoc from the verified contract;
8. write examples that exercise distinct parts of that contract;
9. run doctests/package tests and record exact-head evidence.

Searches for TODO/unwrap/unsafe are candidate generators only. A function can be
wrong with no marker, and a marker can occur harmlessly in tests/prose.
