# Test-quality review during API documentation

Documentation review should also detect tests that do not assert their stated
purpose.

Check for:

- computed values that are discarded;
- assertions that only check nonzero/nonempty when equality/property is intended;
- tests named for insertion/order invariance that never compare variants;
- broad `is_err()` when a typed error matters;
- tolerances without scale/reference justification;
- tests that pass because the relevant branch is never reached;
- examples/tests that duplicate setup without testing a second semantic case.

Correct vacuous tests before relying on them as documentation evidence.
