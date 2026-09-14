# API examples: inventory, reviewed gates and executed evidence

The [API lexicon](API_LEXICON.md) observes adjacent rustdoc summaries. It does
not establish whether a callable has an executable example. Example coverage is
a separate quality dimension, measured by `scripts/api-example-audit.py` using
the lexicon's existing machine-readable index rather than a second workspace
or capability catalogue.

## Reproduce the inventory

From the repository root, with Python 3.10+ and the repository Rust toolchain:

```bash
python3 scripts/api-lexicon.py --index-json /tmp/scirust-api-index.json \
  --output /tmp/scirust-api-functions.md
python3 scripts/api-example-audit.py \
  --index /tmp/scirust-api-index.json \
  --policy docs/api-example-policy.json \
  --output /tmp/scirust-api-examples.json \
  --markdown /tmp/scirust-api-examples.md
```

The JSON contains every indexed callable, its source location, source-content
SHA-256, fence classifications, aggregate observations and policy violations.
The Markdown includes the per-symbol backlog with fewer than two ordinary
runnable candidates. Both reports are deterministic for fixed inputs.

A full, unfiltered schema-1 lexicon index is required. Filtered indexes,
duplicate entries, invalid fields, stale symbol/line pairs and paths escaping
the repository fail explicitly. In each policy source, current declarations
are also checked against index locations so omitting a function cannot make
the policy pass vacuously.

## What the observations mean

| Classification | Observation | Satisfies the ordinary-example gate? |
|---|---|---|
| `runnable_candidates` | Closed nonempty Rust/default fence without a disabling tag | Yes, as presence evidence only |
| `compile_only` | `no_run` | No |
| `compile_fail` | Compilation is expected to fail | No |
| `ignored` | `ignore` or target-specific `ignore-*` | No |
| `should_panic` | Execution is expected to panic | No |
| `other_fences` | Non-Rust language or unrecognized attributes | No |
| `empty_fences` | A closed fence has no non-whitespace body | No |
| `unclosed_fences` | A fence lacks its closing delimiter | No |

A candidate is **not** proof of compilation, execution, assertions, API
reachability or meaningful example content. The inventory does not execute any
source code. A `compile_fail` example can be valuable, but cannot replace an
ordinary successful-use example in this policy. Hardware/unsafe APIs may need
a different reviewed policy; do not relabel their samples merely to pass a
counter.

The observer reads adjacent `///` comments and skips blanks and single-line
attributes. It does not infer multi-line attributes, block doc comments,
`#[doc = ...]`, indented Markdown code blocks, macro expansion, cfg visibility
or trait-generated methods. It inherits the lexicon's source-declaration
limits. Missing observations in these cases are **not proof that an API lacks
examples**. Full Rustdoc semantics remain authoritative.

## Reviewed source gate

`docs/api-example-policy.json` currently requires **two ordinary runnable
candidates for every indexed public callable in `scirust-stats/src/describe.rs`
and `scirust-stats/src/comb.rs`**.
Its required-symbol list prevents deletion/empty-scope accidents from making
the check pass. Any newly declared public callable in that source is also
subject to the minimum. This initial gate does not pretend that all workspace
functions already meet the same requirement.

The nine descriptive-statistics functions are `mean`, `variance`, `std_dev`, `std_error`,
`quantile`, `quantiles`, `median`, `min` and `max`. Their documentation specifies
input mutation, boundary behavior, complexity and known limitations. In
particular, the mean/variance reduction still has documented intermediate
overflow risks; this documentation change does not fix or hide them.

The six combinatorics functions are `factorial`, `ln_factorial`, `binomial`,
`ln_binomial`, `permutations` and `multichoose`. Together the two reviewed
sources contain 15 functions and 30 ordinary fenced examples. The combinatorics
fixes and their numerical limits are recorded in the
[combinatorics audit](audits/COMBINATORICS_2026-09-14.md).

To extend coverage, review another complete source, add meaningful examples
and its explicit policy entry, then include its owning package in an actual
CI doctest command. Do not lower a minimum, accept an empty scope, increase the
existing summary-debt baseline, or count `ignore` as execution to make CI green.
The summary-debt baseline remains a separate unchanged control.

## Execute the examples

Presence and execution are separate steps:

```bash
python3 scripts/test_api_example_audit.py -v
RUSTDOCFLAGS='-D warnings' cargo +stable test -p scirust-stats --doc --locked
```

The existing API-lexicon workflow now performs both steps, emits the full
example inventory and uploads the reports even when a preceding policy check
fails, unless the run was cancelled. Failure remains failure; the upload step
does not relax any gate. The main workspace, lint and hardware workflows keep
their own independent acceptance conditions.

Rust's documentation-test behavior is defined in the [Rustdoc book](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html).
Only actual command results for a recorded revision establish executed
examples. Generating HTML with `cargo doc` alone is not that evidence.
