# Stop condition for a documentation batch

Stop expanding a PR and submit the current slice when its coherent module
contract is complete and validated. Do not keep adding unrelated modules merely
to increase the function count.

A batch is complete when its reviewed public surface has accurate docs/examples,
its applicable tests/rustdoc/doctests are green, and discovered defects are
fixed or explicitly tracked. The next batch starts from the newly integrated
`master`, preventing long-lived branches from accumulating unrelated conflicts
and stale generated metrics.
