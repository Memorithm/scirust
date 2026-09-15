# Documentation surfaces by user need

Use the narrowest authoritative surface for the question.

| Need | Primary surface |
|---|---|
| "What can SciRust do?" | README + capability index/API lexicon |
| "What does this term mean?" | Glossary |
| "How do I perform a task?" | Quickstart/task guide |
| "What exactly does this function accept/return?" | Item rustdoc |
| "Which functions exist in this domain?" | Generated API lexicon |
| "Are examples present and executed?" | API example inventory + doctest evidence |
| "Was this backend/device actually qualified?" | Test protocol + revision-bound evidence |
| "How should I document a new API?" | Documentation standard/conventions/checklist |
| "What remains undocumented?" | Progress/status + generated metrics |

This separation reduces duplication: rustdoc owns callable semantics, generated
indexes own discovery, guides own workflows, and evidence documents own
qualification claims.
