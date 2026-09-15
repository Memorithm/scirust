# Opaque handle API review prompts

For public opaque handles/IDs, document/review:

- originating owner/session/program;
- copy/equality/order/hash semantics;
- lifetime/invalidation;
- foreign/stale handle detection limitations;
- serialization/persistence stability;
- whether numeric/internal identity is intentionally private;
- errors for unknown/foreign handles.

Examples should show normal same-owner use and a distinct invalid/foreign case
when the API can detect it. If cross-owner collisions cannot be detected, state
that limitation rather than implying stronger branding.
