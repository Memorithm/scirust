# Source API discovery limitations

The current lexicon scanner is intentionally conservative and transparent.

It detects directly declared `pub fn` forms in workspace library source trees.
It does not by itself establish:

- external module reachability;
- re-export paths;
- macro-generated callables;
- trait-provided methods that do not use the same source form;
- cfg/feature-expanded surfaces;
- every exact multiline signature relationship.

These are reasons to add compiler/rustdoc reconciliation, not reasons to discard
the source inventory. Source discovery remains valuable for locating debt and
linking entries to exact files/lines.
