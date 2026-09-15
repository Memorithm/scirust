# Documentation lint adoption policy

Rustdoc/Clippy lints can enforce useful structure but should be introduced with
measured baselines rather than globally enabled in a way that makes thousands of
unrelated legacy items fail at once.

For reviewed scopes, prefer warnings denied for broken intra-doc links and
package doctest/rustdoc jobs. Add missing-errors/panics/safety style checks only
when the tool can determine applicability with acceptable false-positive rates.

A lint waiver must identify why the rule does not apply; blanket crate-level
allowances are not the completion mechanism for #1431.
