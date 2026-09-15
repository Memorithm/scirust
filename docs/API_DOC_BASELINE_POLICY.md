# Documentation baseline policy

Baselines protect against silent regression while a large legacy debt is being
removed incrementally.

- The existing adjacent-summary baseline remains exact until deliberately
  refreshed after a reviewed improvement.
- Example-policy baselines cover only reviewed scopes and expand monotonically.
- A baseline refresh must explain package/scope deltas; it is not a generic CI
  repair step.
- New debt is not offset by unrelated improvements elsewhere unless the metric
  explicitly defines such aggregation and the review accepts it.
- When compiler-derived reachability replaces a source metric, preserve a
  reconciliation report explaining denominator changes.

The long-term target is zero missing semantic contracts/examples on the reachable
surface, but incremental baselines keep the repository usable while reaching it.
