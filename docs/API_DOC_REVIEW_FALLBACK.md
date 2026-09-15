# Fallback behavior review prompts

For APIs with fallback, document/review:

- trigger condition;
- whether fallback is explicit to caller or silent;
- semantic/numerical differences;
- performance/resource differences only with evidence;
- whether unsupported representations/operations are rejected instead;
- provenance/evidence indicating which path actually ran.

Silent fallback is inappropriate when it changes the promised semantics or would
make execution evidence misleading. Prefer explicit unsupported errors in those
cases.
