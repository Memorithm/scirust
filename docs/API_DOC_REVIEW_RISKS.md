# Documentation program risks

**Mechanical prose at scale** — mitigated by coherent semantic slices and
implementation/test review.

**Stale generated indexes** — mitigated by revision/source fingerprints and CI
regeneration.

**False public-surface counts** — mitigated by compiler/rustdoc reachability
reconciliation while retaining source inventory limits.

**Example-count gaming** — mitigated by ordinary-fence classification,
semantic-pair review and doctest execution.

**CI bypass via ignored examples/tolerance changes** — mitigated by explicit
exception/tolerance policies.

**Long-lived branch conflicts** — mitigated by small PR stop conditions and
starting each next slice from integrated `master`.

**Documentation hiding code bugs** — mitigated by a separate findings/fix path
with regression tests.
