# Scope of complete SciRust API documentation

"Document every function" in issue #1431 is interpreted as every externally
reachable public callable that forms part of the SciRust API, plus documentation
needed to make those callables understandable.

## Included

- public free functions;
- public inherent methods;
- public trait methods exposed as part of SciRust's API;
- public unsafe callables, with explicit safety invariants;
- re-exported and macro-generated public callables once compiler-derived
  reachability inventory can observe them;
- task-oriented guides and glossary terms needed to navigate major capabilities.

## Not counted as missing public API documentation

- private helpers;
- test-only functions that are not part of an external API;
- binaries' private command handlers unless separately exposed as a library API;
- vendored external project internals that are not part of the SciRust public
  API;
- historical source snapshots retained only as evidence.

Private functions can still merit internal comments when algorithms are subtle,
but they are not part of the public-callable completion metric.

## Examples

The target is one or two executable examples per public callable: two for
non-trivial APIs by default, with a reviewed one-example exception only when the
second example would add no semantic information. Hardware/network-only cases
follow the explicit exception registry rather than being silently ignored.
