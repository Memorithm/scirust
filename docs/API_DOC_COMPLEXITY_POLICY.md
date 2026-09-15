# Algorithmic complexity documentation policy

Document complexity when it materially affects correct use or resource planning.
Name the variables and distinguish time from memory/storage complexity.

Examples:

- "sorts `n` samples once in `O(n log n)`, then computes `q` quantiles in
  `O(q)` interpolation work";
- "allocates an output vector of length `n`";
- "does not allocate represented tensor payload; operates on graph metadata".

Do not infer wall-clock speedups from asymptotic complexity alone. Measured
performance belongs under the benchmark policy with revision/environment data.
