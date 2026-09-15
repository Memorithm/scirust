# SciRust glossary

This glossary defines recurring terms used across SciRust documentation. It is
not an exhaustive mathematical dictionary; module-specific terminology belongs
next to the API that implements it.

## API and evidence

**Callable** — a function or method that can be invoked. The source lexicon
currently inventories directly declared `pub fn` callables; this is broader
than the set proven externally reachable.

**Doctest** — a Rust code example embedded in documentation and compiled/tested
by `cargo test --doc`. A normal Rust code fence is execution evidence only after
the corresponding doctest job succeeds on the same revision.

**Effective public reachability** — whether an API item is actually accessible
to an external crate after module visibility, re-exports, cfg expansion, macros
and traits are resolved. Rustdoc/compiler-derived metadata is authoritative;
textual `pub` discovery alone is insufficient.

**Lexicon** — SciRust's generated repository-wide navigation index of public
callable declarations. It complements rustdoc by supporting capability-domain
search and documentation audits.

**Revision-bound evidence** — a test result, benchmark, inventory or audit tied
to an exact Git commit/source fingerprint. It must not automatically be treated
as evidence for a later revision.

## Tensor and execution architecture

**Logical tensor** — a tensor described by mathematical shape and logical dtype,
independent of how its values are physically represented.

**Physical representation** — the storage/layout encoding used for a logical
tensor, such as dense or a supported quantized representation. Representability
does not imply that an execution kernel exists.

**Representation plan** — the side-table assigning physical representations to
nodes in Tensor IR while preserving the logical graph semantics.

**Tensor IR** — SciRust's intermediate representation for tensor computations,
including graph nodes, logical tensor types and representation planning.

**Execution intent** — a validated semantic description of a computation to be
mapped onto an execution backend. An intent can describe a representation that
is not yet executable; this distinction prevents silent fallback.

**Backend** — an implementation responsible for executing a supported operation
on a concrete compute path, for example a CPU implementation or an explicitly
qualified GPU path.

**Canonical façade** — the user-facing tensor-runtime layer that hides internal
IR identifiers and backend buffers behind stable handles and `TensorND` values.

## Numerical terminology

**Finite value** — a floating-point value that is neither `NaN` nor positive or
negative infinity.

**NaN** — IEEE-754 "not a number" value. Each SciRust numerical API must state
whether NaNs are rejected, propagated or deliberately ignored.

**Overflow** — an operation whose result cannot be represented in the target
numeric format. Intermediate overflow can occur even when the mathematical
final result is representable, so evaluation order is part of numerical
correctness.

**Quantile** — a value associated with a cumulative probability in an empirical
sample or probability distribution. `scirust-stats::describe` uses a documented
linear-interpolation convention for empirical quantiles.

**Determinism** — repeated execution under the stated conditions produces the
same result. This is weaker than bit-identical reproducibility across every
backend unless that stronger guarantee is explicitly stated.

**Bit-identical reproducibility** — equality of floating-point bit patterns, not
merely numerical closeness. Backend changes can alter accumulation order and
therefore invalidate this property.

## Documentation contract

**Ordinary runnable example** — a normal Rust documentation code fence intended
to compile and execute as a doctest. `no_run`, `ignore`, `compile_fail` and
unknown-language fences do not count as executed ordinary examples.

**Documentation debt** — a measured absence of required public API
 documentation. Scanner counts are indicators; they are not a substitute for
semantic review.

**Policy-covered scope** — a reviewed set of APIs for which CI enforces a
specific documentation/example requirement. Coverage is expanded monotonically
until the complete externally reachable API is governed.
