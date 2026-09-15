# SciRust lexicon

SciRust uses two complementary lexicons:

1. **[API lexicon](API_LEXICON.md)** — generated index of directly declared
   public callables, organized for source/API discovery.
2. **[Glossary](GLOSSARY.md)** — human definitions of recurring architectural,
   numerical and evidence terminology.

For documentation requirements and example policy, see the
[public API documentation standard](API_DOCUMENTATION_STANDARD.md). For the
current completion program, see [API documentation progress](API_DOCUMENTATION_PROGRESS.md).

The distinction is intentional: API entries should remain generated from code,
while conceptual definitions require semantic editorial review. Combining both
into one generated table would either duplicate rustdoc or make the terminology
section dependent on incidental source layout.
