# SciRust glossary

This glossary defines recurring terms used across SciRust documentation. Module-specific terminology belongs next to the API that implements it.

## API and evidence

**Callable** — a function or method that can be invoked. Source-level `pub fn` discovery can be broader than the externally reachable API.

**Doctest** — a Rust code example embedded in documentation and compiled/tested by `cargo test --doc`.

**Effective public reachability** — whether an API item is actually accessible to an external crate after module visibility, re-exports, cfg expansion, macros and traits are resolved. Compiler/rustdoc evidence is authoritative.

**Lexicon** — SciRust's generated repository-wide navigation index of callable declarations, complementing rustdoc with capability-domain search and audit information.

**Revision-bound evidence** — a test result, benchmark, inventory or audit tied to an exact Git revision/source fingerprint.

## Tensor and execution architecture

**Logical tensor** — a tensor described by mathematical shape and logical dtype independently of physical storage.

**Physical representation** — storage/layout encoding of a logical tensor. Representability does not imply an execution kernel exists.

**Representation plan** — the side-table assigning physical representations to Tensor IR nodes while preserving logical graph semantics.

**Tensor IR** — SciRust's intermediate representation for tensor computations, including graph nodes, logical tensor types and representation planning.

**Execution intent** — a validated semantic description of a computation to be mapped onto an execution backend. An intent can be describable but non-executable.

**Backend** — an implementation executing a supported operation on a concrete compute path, such as CPU or an explicitly qualified GPU path.

## Numerical terminology

**Finite value** — a floating-point value that is neither `NaN` nor positive/negative infinity.

**NaN** — IEEE-754 "not a number". Numerical APIs should state whether NaNs are rejected, propagated or deliberately ignored.

**Overflow** — a result/intermediate value that cannot be represented in the target numeric format.

**Quantile** — a value associated with a cumulative probability in a sample/distribution; SciRust statistics documents its interpolation convention at the callable.

**Determinism** — repeated execution under stated conditions produces the same result; this is weaker than cross-backend bit-identical reproducibility unless explicitly established.

## Documentation contract

**Ordinary runnable example** — a normal Rust documentation fence intended to compile and execute as a doctest. `no_run`, `ignore`, `compile_fail` and unknown-language fences do not count as executed ordinary examples.

**Documentation debt** — measured absence of required public API documentation. Scanner counts are indicators, not substitutes for semantic review.

**Policy-covered scope** — a reviewed API subset for which CI enforces documentation/example requirements; coverage expands monotonically.
