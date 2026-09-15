# Mutable-state API review prompts

For stateful public APIs, document/review:

- initial state and constructor defaults;
- ownership/borrowing and whether methods mutate in place;
- state transitions after success and after error;
- reset/clear semantics;
- thread safety / Send+Sync assumptions where relevant;
- serialization/checkpoint behavior;
- determinism with stateful RNG/cache/optimizer/tape;
- whether handles are valid only for the originating session/program.

Examples should assert a state transition or invariant, not merely that a method
can be called.
