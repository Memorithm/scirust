# API documentation review ordering

Within a package, documentation is reviewed by semantic importance rather than
alphabetical order or raw missing-comment count.

Prioritize persistence/parsing boundaries, numerical routines with important
non-finite/tolerance behavior, tensor shape/representation/backend contracts,
autodiff/optimization state transitions, and then convenience transforms and
accessors.

The ordering is pragmatic: documentation review reads implementation and tests,
so interfaces with richer invariants are more likely to reveal mismatches that
need regression tests. Each slice remains small enough for its semantics and
examples to be reviewed before merge.
