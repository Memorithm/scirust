# Public panic policy

A caller-triggerable panic in a public API must be intentional and documented in
`# Panics`, or preferably replaced by a typed error when the API contract can
support it without an inappropriate breaking change.

During review, inspect indexing, division/modulo, unchecked conversions,
`unwrap`/`expect` and assertions reachable from caller input. Textual matches are
only candidates; establish the actual path before classifying a panic.

Examples should not intentionally crash ordinary doctest execution. Use typed
error examples after fixing accidental panics; `should_panic` examples are
tracked separately and do not count as ordinary runnable examples.
