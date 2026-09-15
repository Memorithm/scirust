# Public supporting types

The callable-focused #1431 metric does not eliminate the need to document public
supporting types. A function cannot be understood if its public enum/struct/trait
parameters or error variants are opaque.

During each callable slice, document public supporting types necessary to
understand the reviewed functions: semantic field/variant meanings, invariants,
units and construction constraints. Examples can live on the type or callable
whichever produces the clearest public usage.

A future repository-wide type inventory can extend the lexicon, but callable
completion should not wait to fix an undocumented error/type that blocks the
current module's usability.
