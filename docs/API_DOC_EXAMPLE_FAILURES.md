# Doctest failure triage

When a new/changed ordinary example fails:

1. identify whether failure is compile, link, runtime assertion or environment;
2. verify the public import path and required feature/cfg;
3. compare the asserted semantics with implementation/tests;
4. fix the example when it is wrong;
5. fix implementation + add regression when the example exposes a real bug;
6. use an explicit exception only for a genuine environment-only constraint.

Do not convert a failing ordinary example to `ignore` or `no_run` as the default
repair. That changes the evidence level and must be reviewed as such.
