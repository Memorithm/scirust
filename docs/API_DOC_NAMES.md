# Naming and terminology in public documentation

Use the public Rust identifier exactly when referring to an API. Expand acronyms
on first conceptual use unless they are universally established in the immediate
domain. Keep the glossary for recurring cross-module terms.

When legacy names no longer match semantics, documentation should state the
actual behavior rather than rationalize the name. If the mismatch materially
misleads callers, track an API rename/deprecation separately rather than hiding
it with prose.

Avoid synonyms that blur distinct SciRust concepts such as logical dtype versus
storage dtype, representation versus execution, compile support versus runtime
qualification, or deterministic versus bit-identical reproducibility.
