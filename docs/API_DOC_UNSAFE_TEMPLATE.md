# Unsafe public API documentation template

Every public `unsafe fn` or public API exposing an unsafe caller contract must
have a `# Safety` section that states concrete invariants.

Review at least:

- pointer validity and provenance;
- alignment requirements;
- initialized/readable/writable byte ranges;
- aliasing and exclusivity;
- object lifetime and ownership transfer;
- thread/device synchronization;
- FFI ABI and representation assumptions;
- valid enum/tag/value ranges;
- what the implementation guarantees after the caller satisfies the contract.

A useful example should demonstrate construction through a safe precursor when
possible. A second example can demonstrate the invariant without intentionally
executing undefined behavior. Never provide a doctest whose pedagogical purpose
requires actually triggering UB.

Miri is valuable execution evidence for applicable CPU paths, but a passing Miri
run does not qualify GPU/FFI behavior it cannot execute. Provenance warnings
must be investigated rather than reclassified as documentation-only concerns.
