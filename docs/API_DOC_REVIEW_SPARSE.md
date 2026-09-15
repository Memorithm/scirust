# Sparse API review prompts

For public sparse APIs, document/review:

- format (CSR/CSC/COO/etc.) and index ordering;
- dimensions and index dtype;
- duplicate-entry policy;
- sortedness/canonicalization requirements;
- explicit-zero behavior;
- storage accounting;
- validation of offsets/indices;
- conversion/execution support versus skeleton representation;
- errors for malformed structures.

Examples should use a tiny sparse structure with exact dense interpretation and
a distinct malformed/duplicate/unsupported case.
