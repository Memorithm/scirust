# Classification of documentation-driven changes

During #1431, classify each finding/change so review evidence stays clear.

- **DOC** — prose/examples/navigation only; no runtime behavior intended.
- **TEST** — adds/corrects a regression or qualification test without changing
  runtime behavior.
- **FIX** — changes runtime behavior to correct a verified contract defect;
  requires regression evidence.
- **API** — intentionally changes public signature/semantics; requires migration
  documentation and compatibility review.
- **TOOL** — changes lexicon/example/reachability generation or CI policy.

A PR may contain more than one class when the documentation review exposes a
bug, but the description should separate them. This prevents a large docs patch
from obscuring a production behavior change.
