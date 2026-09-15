# Evidence-emitting API review prompts

For APIs whose output is intended as benchmark/proof/experiment evidence,
document/review:

- measurement definition and units;
- environment/workload identifiers;
- warmup/sample aggregation;
- deterministic versus statistical components;
- missing/failed measurement representation;
- schema/version/provenance;
- whether values are measured, modeled or estimated;
- comparison/reference baseline.

Examples should create/parse a small deterministic evidence record and a
distinct missing/invalid/version case. Never label modeled estimates as measured
runtime data.
