# Async API review prompts

For public async APIs, document/review:

- when work begins (call versus poll/await);
- cancellation/drop behavior;
- timeout/retry/idempotency;
- Send/Sync/runtime assumptions;
- resource ownership while pending;
- ordering/concurrency limits;
- error propagation;
- external side effects.

Examples should avoid real network/time dependencies when a deterministic local
fixture can exercise the future. Do not imply a specific async runtime unless
the API requires one.
