# SciAgent specialization: corpus admission v1

SciAgent is the target model for the finance/Rust specialization program.
The first real-checkpoint diagnostic in Replikans PR #52 produced no valid
A/B/C continuations on twelve synthetic routing cases. This is a scoped baseline,
not a judgement of all SciAgent tasks or architectures. No financial checkpoint
has been trained by this change.

## Implemented first stage

`train::corpus_manifest::validate` checks exact UTF-8 content identities and
caller-declared source, revision, license, rights-review reference and group
identity **before tokenization**. It rejects missing fields, tampering, repeated
IDs, exact duplicate contents (including within a split), noncausal timestamps,
incorrect declared temporal splits and groups crossing retained splits.

`group` must identify the canonical upstream project for Rust (forks share an
origin) or the event family for financial records. These identities must be
curated; changing a group string can defeat metadata-based grouping. The module
does not discover project ancestry, near duplicates, paraphrases or hidden
training contamination. It does not retrieve sources or verify licenses. A
nonempty review reference records a caller assertion, not legal approval.

Financial times are unsigned UTC Unix milliseconds. Require
`available_at <= decision_at <= label_end`. Training labels must end strictly
before `train_end`. Validation decisions start at `train_end + embargo_ms`,
and their labels end strictly before `validation_end`. Test decisions start at
`validation_end + embargo_ms`. Rows overlapping a boundary or embargo are
explicitly returned with `split: null` and must be excluded by the downstream
packer. Rust rows use an explicitly assigned project-level split. The validator
never silently moves rows into a different split.

Any malformed row rejects the entire manifest without an accepted report.
Purged records are retained in the report; a report with no retained rows is not
a usable training corpus. Neither balanced classes nor nonempty train/validation/
test sets are guaranteed. Those remain task-level acceptance checks.

## Command

```bash
cargo +stable run --locked -p scirust-sciagent --bin sciagent-corpus-check -- /external/corpus/manifest.json
cargo +stable test --locked -p scirust-sciagent --test corpus_manifest
```

The CLI is read-only and caps input at 32 MiB. The library accepts at most 100,000
records, each text at most 1 MiB. Unknown schema fields and duplicate JSON struct
fields fail parsing. Example manifest shape (replace the SHA with the actual
SHA-256 of the exact text; the placeholder deliberately fails validation):

```json
{
  "schema_version": 1,
  "train_end": 100,
  "validation_end": 200,
  "embargo_ms": 10,
  "records": [{
    "id": "synthetic-example",
    "source_uri": "synthetic://example-not-training-data",
    "revision": "v1",
    "license": "test-only",
    "rights_review": "synthetic demonstration, not a rights approval",
    "group": "synthetic-project",
    "text": "example",
    "sha256": "REPLACE_WITH_EXACT_SHA256",
    "split": "train",
    "domain": {"kind": "rust"}
  }]
}
```

For finance use `{"kind":"finance","available_at":40,"decision_at":50,
"label_end":60}` as the domain. Output binds the report to the exact input
manifest SHA-256 and explicitly records `rights_verified: false` and
`training_performed: false`. Report generation neither authorizes training nor
changes the existing trainers: the manifest gate is not yet connected to shard
packing/training. Keep corpora external under the existing corpus storage rules.

## Following stages

1. Inventory actual candidate sources with exact revisions and reviewed rights.
2. Curate canonical project/event identities, timestamps, labels and task-specific
   train/validation/test boundaries; freeze held-out evaluation before training.
3. Connect accepted/purged IDs and content hashes to the external shard packer;
   prevent the existing random window splitter from remixing approved splits.
4. Compare unchanged, finance-only, Rust-only and mixed training at documented
   budgets; track forgetting, semantic decisions and Rust compilation/tests.
5. Evaluate typed outputs, calibration, latency and memory on unseen data before
   Replikans paper-trading qualification. Rust financial policy stays authoritative.

The admission contract is reusable for other SciAgent specializations. Financial
labels and policy remain owned by Replikans; generic corpus checks and training
remain in SciRust. Remote execution uses Memorithm/RemoteOps. No performance,
financial profitability or ML-maturity promotion is made by this stage.
