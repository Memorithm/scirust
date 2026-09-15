# Scalable completion strategy

The inventory is large enough that quality depends on batching discipline.

## Batch size

Prefer one coherent module or a small family of related modules per PR. The
correct size is the amount whose implementation, tests, docs and examples can be
semantically reviewed, not a fixed number of functions.

## Parallelism

Independent packages can be documented in parallel only when each branch starts
from a known base and generated baselines are reconciled after merges. Avoid
multiple branches rewriting the same global generated artifact without a clear
integration order.

## Automation

Automate discovery, metrics, stale-source checks, code-fence classification,
doctest execution and generated navigation. Do not automate semantic claims.

## Progress

Expand reviewed policy coverage monotonically. The end state is compiler-derived
reachable API coverage, not merely zero source-regex missing summaries.
