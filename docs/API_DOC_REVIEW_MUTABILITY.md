# Public mutability review prompts

For public structs with mutable fields/setters or mutable records used as
validated intents/plans, document/review:

- which invariants construction establishes;
- whether callers can later invalidate them;
- whether validation/fingerprint/cache values become stale after mutation;
- whether execution adapters revalidate before use;
- compatibility implications of making fields private/builders in future.

A predicate such as `is_executable` should not be described as full validation
unless it actually checks every mutable invariant required for execution.
