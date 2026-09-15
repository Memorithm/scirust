# API documentation readability

Precision does not require opaque prose.

- Define specialized terms in the glossary or locally before relying on them.
- Keep identifiers/equations exact; explain their role in prose.
- State units and shape order explicitly.
- Prefer small examples with assertions over long demonstrations.
- Separate normal behavior from errors/panics/safety qualifications with standard
  Rustdoc sections.
- Avoid unexplained acronyms in first use.
- Use tables for comparable options/backends, not for long narrative contracts.

The documentation should remain useful to researchers who know the domain but
are not specialists in SciRust's internal crate architecture.
