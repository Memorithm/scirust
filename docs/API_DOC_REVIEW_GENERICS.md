# Generic API review prompts

For public generic functions/methods, document/review:

- semantic requirements represented by trait bounds;
- supported numeric/data types beyond what bounds syntactically permit;
- ownership/lifetime relationships;
- specialization/backend behavior if any;
- error/panic differences by type;
- type inference/turbofish needs only when they affect usability.

Examples should use representative concrete types and a second type/configuration
only when it demonstrates a distinct contract, not merely generic syntax.
