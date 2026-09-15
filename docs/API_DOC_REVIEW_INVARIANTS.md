# Invariant documentation prompts

Document invariants callers can rely on, for example:

- output length/shape equals a stated function of inputs;
- values remain normalized/bounded/sorted;
- graph remains acyclic/typed;
- representation assignment remains compatible;
- state update is transactional;
- serialization round trips preserve identity.

Only state invariants supported by implementation/tests. Examples should assert
an invariant when it communicates more than a specific incidental output.
