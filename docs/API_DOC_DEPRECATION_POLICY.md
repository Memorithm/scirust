# Deprecated API documentation policy

Deprecated public APIs remain part of the externally reachable surface until
removed. They therefore still require a usable contract.

Document:

- why the API is deprecated when technically relevant;
- the supported replacement and migration difference;
- whether behavior/security/correctness limitations motivated deprecation;
- removal timeline only when one is actually established.

Examples may prioritize migration to the replacement, but at least one example
should make the deprecated contract understandable while it remains callable.
Generated lexicon entries should preserve deprecation status rather than simply
hiding the item and understating the public surface.
