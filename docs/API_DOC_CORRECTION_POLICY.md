# Correction policy for documentation-discovered defects

If reviewing a public contract exposes a reproducible implementation defect:

1. capture the failing case in a focused regression test;
2. establish intended semantics from implementation context/reference evidence;
3. fix the smallest responsible implementation layer;
4. update Rustdoc/examples to the corrected contract;
5. run package and applicable workspace checks;
6. record the fix separately from documentation-only changes in the PR.

Do not document an accidental panic/wrong numerical result as intended behavior
merely to avoid changing code. Conversely, do not change behavior when the
intended contract cannot be established from evidence.
