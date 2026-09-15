# Public Rustdoc versus internal comments

Issue #1431 focuses on externally reachable public callables. Private/internal
code still needs comments when implementation reasoning is not obvious.

Use public Rustdoc for caller contracts: semantics, domains, outputs, errors,
panics, safety and examples. Use internal comments for why a numerical
rearrangement avoids cancellation, why an invariant makes indexing safe, why a
specific backend workaround exists, or why an algorithmic branch is selected.

Do not expose implementation trivia in public docs unless callers need it for
correctness/reproducibility. Conversely, do not hide caller-visible limitations
only in internal comments.
