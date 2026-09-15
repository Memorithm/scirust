# Concurrency API review prompts

For public concurrent/parallel APIs, document/review:

- thread safety and Send/Sync expectations;
- ownership of shared state;
- ordering and completion guarantees;
- cancellation/retry behavior;
- deterministic reduction semantics;
- resource/concurrency limits;
- deadlock/reentrancy restrictions where relevant;
- error propagation across workers.

Examples should be small/deterministic and avoid timing-based assertions when a
synchronization/property assertion is available.
