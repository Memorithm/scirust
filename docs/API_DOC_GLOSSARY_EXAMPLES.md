# Glossary versus API examples

Glossary entries explain concepts; they do not satisfy callable example policy.
A glossary can show a conceptual equation or tiny illustration, but each public
callable still needs examples beside its Rustdoc so signature/behavior changes
are checked by doctests.

Conversely, do not repeat long conceptual definitions in every function example.
Link recurring terms to module docs/glossary and keep callable examples focused
on actual invocation and assertions.
