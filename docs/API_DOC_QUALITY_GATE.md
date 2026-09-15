# Per-slice API documentation quality gate

A #1431 slice is mergeable only when:

- its public contract has been reviewed against implementation/tests;
- required examples are ordinary runnable fences or justified exceptions;
- relevant doctests pass on the exact final head;
- rustdoc builds with warnings denied;
- package tests/Clippy and applicable repository CI are green;
- documentation/example debt does not regress;
- any runtime behavior change has a regression test and is identified separately;
- progress/evidence records refer to the final head, not an earlier commit.

A green documentation-specific job does not waive a red normal CI job.
