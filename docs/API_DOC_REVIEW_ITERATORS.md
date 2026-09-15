# Iterator and stream API review prompts

For public iterators/streams, document/review:

- item order and determinism;
- ownership/borrowing/lifetime;
- exact/upper size hints when promised;
- laziness and side effects;
- error termination/retry behavior;
- fused/double-ended/exact-size guarantees only when implemented;
- concurrency/backpressure for async streams.

Examples should collect a small exact sequence and cover a distinct empty/error
or ordering property.
