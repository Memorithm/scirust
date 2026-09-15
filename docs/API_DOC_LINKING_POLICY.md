# API documentation linking policy

Use links to make documentation navigable without duplicating contracts.

- Item Rustdoc should link to related types/traits/errors using intra-doc links.
- Task guides should link to the relevant module/item Rustdoc conceptually and
  avoid copying complete API contracts that can become stale.
- The generated API lexicon links discovery metadata to source/rustdoc where
  available.
- The glossary links recurring concepts to canonical architecture/API guides.
- Historical audit/evidence documents keep revision-specific links and should
  not be rewritten to look current.

Broken links are documentation defects. Rustdoc is built with warnings denied in
covered scopes so invalid intra-doc links fail CI rather than silently degrade
navigation.
