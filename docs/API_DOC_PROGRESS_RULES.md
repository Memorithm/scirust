# Documentation progress accounting rules

Count progress only after a reviewed slice is integrated or has exact-head green
evidence clearly identified as pending merge.

Keep separate counts for:

- source-discovered callables;
- compiler-confirmed reachable callables;
- semantic summaries/contracts reviewed;
- policy-covered callables;
- ordinary examples observed;
- ordinary examples executed;
- active exceptions/open findings.

Do not add these into one "completion score". Report the definition and exact
revision beside every aggregate.
