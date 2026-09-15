# Scheduler/planner API review prompts

For public schedulers/planners, document/review:

- input resource/workload model;
- feasibility constraints;
- objective/tie-breaking ordering;
- deterministic behavior;
- overflow/capacity validation;
- fallback/rollback semantics;
- partial plan behavior on failure;
- evidence/metrics produced;
- concurrency/device assumptions.

Examples should use a tiny feasible plan and a distinct infeasible/capacity or
tie-breaking case with exact assertions.
