# Handling code/documentation contract mismatches

When existing Rustdoc and implementation disagree:

1. do not assume the prose or code is automatically correct;
2. inspect tests/call sites/history and, for scientific semantics, the relevant
   mathematical/reference definition;
3. write a regression for the intended behavior when it can be established;
4. fix code or documentation accordingly;
5. record compatibility impact if external callers may depend on old behavior.

If intent cannot be established from evidence, mark the finding open rather than
inventing a contract. Documentation completion must reduce ambiguity, not encode
guesses permanently.
