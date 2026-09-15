# Time-dependent API review prompts

For APIs involving time, document/review:

- units and monotonic/wall-clock distinction;
- timezone/UTC/local assumptions;
- interval endpoint convention;
- clock source and reproducibility;
- timeout/deadline behavior;
- overflow/wraparound;
- simulation time versus real time.

Examples should use fixed/deterministic timestamps or simulated clocks rather
than depending on the current wall clock whenever possible.
