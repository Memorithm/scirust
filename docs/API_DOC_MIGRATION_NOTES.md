# API documentation migration notes

The existing API lexicon introduced a useful baseline: directly declared public
callables and adjacent `///` summaries. The #1431 program deliberately preserves
that evidence while moving to a stronger model.

Migration principles:

- do not delete the old metric before the new one can reproduce/explain it;
- distinguish source discovery from effective public reachability;
- distinguish example presence from execution;
- expand two-example policy in reviewed scopes rather than declaring repository
  coverage from a global code-fence count;
- retain package/domain navigation already provided by the lexicon;
- add semantic contract observations only when the parser can measure them
  without inventing requiredness;
- keep source/runtime defects discovered during documentation review visible.

This staged migration prevents a more sophisticated-looking dashboard from
quietly weakening the guarantees of the simpler baseline it replaces.
