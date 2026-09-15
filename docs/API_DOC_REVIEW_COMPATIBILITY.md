# Compatibility review during documentation

When review suggests changing a public API, inspect:

- existing call sites and re-export paths;
- serialized/wire/checkpoint compatibility;
- feature/backend behavior;
- error variant matching by downstream callers;
- deterministic/fingerprint/cache identity;
- deprecation/migration path.

Prefer compatible validation/fix additions when they correct accidental behavior
without changing intended semantics. Intentional breaking changes need an API
classification and migration documentation, not a hidden docs commit.
