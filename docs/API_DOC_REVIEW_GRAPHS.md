# Graph and IR API review prompts

For public graph/IR APIs, document/review:

- node/edge identity and ownership;
- insertion/order guarantees;
- cycle/topology requirements;
- shape/type inference and validation;
- foreign/stale handle behavior;
- mutation invalidation rules;
- canonicalization/fingerprint ordering;
- serialization/versioning;
- errors for unknown nodes, invalid edges or incompatible plans.

Examples should construct a tiny graph and assert topology/type/output behavior,
plus a distinct invalid/foreign-handle or transformation case.
