# API documentation metrics

Metrics guide the #1431 program but do not replace review.

## Tracked metrics

- syntactically discovered directly declared public callables;
- callables with an adjacent semantic summary;
- callables with ordinary example fences;
- reviewed policy-covered callables with at least two ordinary examples;
- ordinary examples executed successfully as doctests on the same revision;
- public unsafe callables with `# Safety` sections;
- fallible/panicking APIs with the applicable contract sections once the parser
  can determine requiredness reliably;
- compiler-confirmed externally reachable callables, once reachability
  reconciliation is implemented.

## Anti-gaming rules

The following do not count as completion:

- adding empty or tautological summaries;
- duplicating one example to reach a count of two;
- converting a failing example to `ignore`/`no_run` without a reviewed reason;
- lowering a baseline to match new debt;
- counting syntactic `pub` as externally reachable without compiler evidence;
- counting compilation as hardware execution;
- claiming semantic correctness solely from code-fence counts.

## Reporting

Report absolute numerator/denominator values with the metric definition and
source revision. Avoid a single composite percentage because coverage,
executability, semantic completeness and evidence strength are distinct.
