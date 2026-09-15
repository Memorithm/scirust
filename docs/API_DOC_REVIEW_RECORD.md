# Per-module API documentation review record

For each reviewed module, record:

```text
package:
module/source:
base revision:
final revision:
public surface source:
callables reviewed:
ordinary examples observed:
ordinary examples executed:
errors/panics/safety contracts reviewed:
runtime defects found:
regression tests added:
applicable feature/backend qualifications:
open findings:
CI evidence:
```

This record can live in the PR description or a revision-bound audit file. The
important property is traceability: later reports must be able to distinguish
what was actually reviewed/executed from what remains inferred or planned.
