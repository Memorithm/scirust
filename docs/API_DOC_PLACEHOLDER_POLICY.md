# Placeholder and skeleton API documentation policy

A public skeleton/placeholder must never look like a completed capability.

If retained intentionally for research/compatibility, document:

- what data/contract is representable today;
- what execution/validation is missing;
- what calls fail explicitly;
- whether the API is unstable;
- the condition required before it can be called implemented/executable.

Prefer typed unsupported errors to silent fallback. TODO-marker searches are not
the authority: a function can be incomplete without a TODO, and TODO text can be
a harmless fixture. Review actual behavior/tests.
