# Builder/configuration API review prompts

For public builders/configuration APIs, document/review:

- defaults;
- required fields and validation time;
- units/ranges;
- repeated setter/override semantics;
- interaction/incompatibility between options;
- deterministic canonicalization;
- build-time versus execution-time errors;
- ownership/reuse after `build`.

Examples should include a minimal valid configuration and a distinct explicit
option/validation failure. Avoid examples that only call every setter without
asserting resulting behavior.
