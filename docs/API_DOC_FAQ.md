# API documentation FAQ

## Why not generate descriptions for all missing functions automatically?

Because a syntactically plausible description can be scientifically wrong.
Automation inventories APIs and executes examples; semantic contracts are
reviewed against implementation and tests.

## Why two examples?

One nominal example often omits the most important boundary. The default pair is
nominal use plus a complementary error/boundary/configuration/state case.

## Why is the source lexicon not the final public API count?

Module visibility, re-exports, cfgs, macros and trait methods change effective
reachability. Compiler/rustdoc reconciliation is required.

## Does a code fence count as tested documentation?

No. It counts as observed. It becomes execution evidence only after the relevant
doctest succeeds on the same revision.

## Are private functions ignored?

They are outside the public-callable completion metric, though subtle private
algorithms can and should have internal explanatory comments.

## Can hardware examples be ignored?

Only through an explicit reviewed exception with alternative evidence. Compile
support and physical-device execution remain distinct claims.
