# Exact-head evidence requirement

A CI result qualifies the source revision it actually checked out. If a branch
changes afterward, earlier green results remain historical evidence but are not
merge evidence for the new head.

Documentation examples are especially sensitive to this: a one-line Rustdoc
change can alter a doctest without changing production code. Therefore progress
reports and PR comments should name the final head/run when claiming examples
executed successfully.

Synthetic pull-request merge commits are acceptable hosted evidence when their
parents are the stated base and exact PR head; record that relationship when it
matters to an audit.
