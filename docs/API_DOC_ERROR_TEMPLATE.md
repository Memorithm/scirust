# Fallible API documentation template

For a public function returning `Result`, the `# Errors` section should describe
conditions, not merely repeat error variant names.

Good structure:

```text
# Errors

Returns `ShapeMismatch` when ... . Returns `NonFiniteInput` when ... . Returns
`NotConverged` when the stopping criterion is not reached within ... .
```

Also document whether failure is transactional: does the function leave mutable
state unchanged, partially updated, or unspecified? For I/O/network APIs,
separate validation errors from transport/protocol failures. For iterative
numerics, distinguish invalid input, numerical breakdown and ordinary
non-convergence.

Examples should include a nominal successful call and, when practical, a typed
error assertion. Do not require string matching when a stable error variant is
available.
