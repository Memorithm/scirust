# SciRust public API documentation program

Tracking issue: #1431.

Goal: provide a precise, navigable and executable contract for every externally
reachable public callable, with one or two examples as appropriate and two by
default for non-trivial APIs.

The program combines:

- item/module Rustdoc for canonical semantics;
- ordinary doctests for executable usage evidence;
- generated API lexicon for repository-wide discovery;
- example observer for code-fence classification;
- compiler/rustdoc reconciliation for eventual effective reachability;
- glossary/task guides for conceptual and workflow navigation;
- monotonic CI baselines to prevent regression;
- focused semantic review that can expose and fix real implementation defects.

Current reviewed/merged example scope includes `scirust-stats::describe` and
`comb`. The current `scirust-core` slice begins with `compute_backend` and awaits
exact-head CI before integration.
