# Concise public API documentation requirements

For each externally reachable public callable:

- explain what it does;
- define inputs, output and important conventions;
- document errors, caller-triggerable panics and safety invariants where
  applicable;
- state numerical/backend/determinism limitations that affect correct use;
- provide two complementary executable examples by default for non-trivial APIs;
- keep examples small, deterministic and assertion-based;
- qualify exceptions explicitly;
- keep the contract synchronized with implementation/tests.

Repository-wide completion additionally requires compiler-derived reachability,
generated lexicon navigation, example execution evidence and non-regression CI.
