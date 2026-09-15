# Memory/allocation API review prompts

For public memory/allocation APIs, document/review:

- size/alignment units and valid ranges;
- zero-size behavior;
- ownership/deallocation;
- pointer provenance/alignment;
- initialization guarantees;
- growth/reallocation behavior;
- overflow checks;
- thread/device locality;
- `# Safety` invariants for unsafe access.

Examples should prefer safe wrappers and small allocations. Empty/sentinel paths
should be exercised under Miri/Strict Provenance where applicable rather than
assuming non-null alignment tricks are harmless.
