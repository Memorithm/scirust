# Agent/tooling API review prompts

For APIs intended for automated agents/tooling, document/review:

- input/output schema and versioning;
- deterministic/canonical serialization;
- capability/permission boundaries;
- timeout/cancellation/retry behavior;
- validation and typed errors;
- idempotency/side effects;
- provenance/evidence returned;
- unsafe/external action boundaries.

Examples should use local deterministic requests/responses rather than external
services where possible. A tool being callable by an agent does not imply that
all side effects are safe or automatically authorized.
