# Public API documentation exceptions

Issue #1431 requires executable examples for externally reachable public
callables. Exceptions must be explicit and reviewed; this file is the registry.

## Rules

An exception must identify the exact callable, source revision/scope, reason,
alternative executable evidence, and removal condition. Acceptable reasons can
include a hardware-only launch, external credentials, OS-only behavior, or an
API whose only correct example is intentionally compile-failing. Cost alone is
not sufficient when a small deterministic fixture can be constructed.

`ignore` and `no_run` are not automatic exceptions. They are code-fence
properties observed by the example auditor and do not count as successful
execution.

## Active exceptions

None have been approved by the #1431 documentation program yet.

As policy coverage expands, any necessary exception must be added in the same
PR as the affected scope and must not be used to conceal missing documentation
or a broken doctest.
