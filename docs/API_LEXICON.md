# SciRust API lexicon and capability index

SciRust is large enough that a hand-maintained function catalogue would drift
quickly. The documentation stack therefore separates orientation, operational
reference, source-level discovery, capability navigation, documentation-debt
control, and authoritative API documentation.

## Documentation layers

1. `README.md` gives the supported entry points and repository map.
2. `docs/REFERENCE.md` documents commands, quality gates, features, and API
   entry points.
3. `scripts/api-lexicon.py` generates and searches a compact lexicon of directly
   declared public functions and methods across Cargo library targets.
4. `docs/api-domains.json` is the reviewed navigation taxonomy that maps
   workspace packages to functional domains.
5. `docs/api-doc-baseline.json` is the reviewed adjacent-rustdoc debt baseline
   used as a CI ratchet.
6. The generated capability index summarizes API volume and adjacent-rustdoc
   coverage by domain and package.
7. Rustdoc remains the authoritative API reference for effective visibility,
   signatures, re-exports, cfg expansion, trait-provided methods, generated
   items, and intra-doc links.

This distinction is intentional: the lexicon and capability taxonomy are
optimized for discovering where a capability lives; the debt baseline controls
measurable drift; rustdoc is optimized for exact API semantics.

## Generate the current function lexicon

From the repository root:

```bash
python3 scripts/api-lexicon.py \
  --output /tmp/scirust-api-functions.md \
  --stats-json /tmp/scirust-api-functions.json \
  --index-json /tmp/scirust-api-index.json \
  --domain-index /tmp/scirust-capabilities.md
```

The outputs have separate purposes:

- the Markdown lexicon is a human-readable function inventory;
- the statistics JSON aggregates public-callable and adjacent-rustdoc counts;
- the index JSON contains one machine-readable record per indexed callable,
  including its domain membership;
- the capability index summarizes the same inventory by functional domain and
  package.

The generated data is deterministic for a fixed checkout. Each function entry
contains the workspace package, declared symbol, first adjacent `///` sentence
when present, source file and line. The machine-readable index also includes
its domain ids.

## Search by capability

Search is case-insensitive and checks the function symbol, adjacent rustdoc
summary, package name, and source path. Every whitespace-separated search term
must match somewhere in that combined record.

```bash
python3 scripts/api-lexicon.py --query matrix
python3 scripts/api-lexicon.py --query "kalman filter"
python3 scripts/api-lexicon.py --query tensor --output /tmp/tensor-api.md
```

Search can be restricted to one or more reviewed capability domains:

```bash
python3 scripts/api-lexicon.py --list-domains
python3 scripts/api-lexicon.py --domain tensor-compute --query tensor
python3 scripts/api-lexicon.py --domain algebra-symbolic --query simplify
python3 scripts/api-lexicon.py --domain control-robotics
python3 scripts/api-lexicon.py --domain trading-finance
```

Multiple `--domain` arguments form a union of the selected domains. Multiple
`--query` arguments contribute additional terms that all have to match.

Package filtering remains available when a crate boundary is already known:

```bash
python3 scripts/api-lexicon.py --package scirust-core
python3 scripts/api-lexicon.py \
  --package scirust-solvers \
  --package scirust-symbolic
```

## Find documentation debt

The lexicon observes whether a directly declared public callable has an
adjacent `///` summary. It can therefore produce a focused work list:

```bash
python3 scripts/api-lexicon.py --missing-docs
python3 scripts/api-lexicon.py \
  --domain tensor-compute \
  --missing-docs \
  --output /tmp/tensor-doc-gaps.md
```

A missing adjacent `///` summary is an observation, not proof that the API is
undocumented everywhere. The symbol may be explained by module-level docs,
traits, generated documentation, or a longer guide. The metric is useful for
prioritization because it is mechanical and reproducible, but it must not be
presented as semantic documentation coverage.

## Documentation-debt ratchet

`docs/api-doc-baseline.json` records, per represented package, the number of
directly declared public callables for which the scanner did not find an
adjacent `///` summary. CI requires an exact match with this baseline:

```bash
python3 scripts/api-lexicon.py \
  --check-doc-baseline docs/api-doc-baseline.json \
  --output /tmp/scirust-api-functions.md
```

This produces two useful failure modes:

- if the count increases, CI exposes a documentation regression instead of
  allowing new undocumented public callables to accumulate silently;
- if the count decreases, CI also asks for an explicit baseline refresh so the
  improvement becomes the new lower ceiling rather than being lost later.

After reviewing an intentional documentation improvement, refresh the baseline
with the same source-level scanner:

```bash
python3 scripts/api-lexicon.py \
  --write-doc-baseline docs/api-doc-baseline.json \
  --output /tmp/scirust-api-functions.md

git diff -- docs/api-doc-baseline.json
```

A baseline increase should normally be treated as a regression to fix, not a
number to accept automatically. The file exists to make that decision visible
in code review. Baseline check/write operations deliberately reject package,
domain, query, and `--missing-docs` filters so a partial inventory cannot be
mistaken for the workspace baseline.

## Capability taxonomy

`docs/api-domains.json` is a reviewed navigation layer. Domain assignments are
explicit rather than inferred from crate names at generation time. The
generator validates every package named by the taxonomy against current Cargo
metadata and rejects stale references.

The current taxonomy covers these broad surfaces:

- foundation and tooling;
- tensor and compute;
- learning and AI;
- optimization and algorithm search;
- algebra and symbolic mathematics;
- numerics, statistics and estimation;
- simulation and physics;
- control, robotics and navigation;
- industrial and engineering domains;
- security, provenance and safety evidence;
- data transport, messaging and events;
- trading and financial tooling;
- agent systems and code transformation;
- Studio application services;
- deployment and operations;
- research methods;
- research benchmarking and evaluation infrastructure.

A package may be moved or assigned to multiple domains when the architecture
actually warrants it. Such changes are reviewable documentation changes; the
generator does not silently invent classifications.

CI invokes `--require-domain-coverage`, so a library package represented by at
least one indexed public callable cannot appear without a domain assignment.
This converts taxonomy drift into an explicit review failure rather than a
stale documentation problem.

## What the lexicon indexes

The scanner first uses `cargo metadata` to select workspace packages that expose
library-like Cargo targets. It then indexes directly declared public Rust
callables matching `pub fn` and the corresponding `async`, `const`, `unsafe`,
and `extern` forms. Conventional binary sources (`main.rs` and `src/bin/`) are
excluded so binary helper functions are not presented as library API.

The scanner deliberately does **not** claim that this source-level inventory is
the exact externally reachable API. In particular, Rust visibility can be
changed by module boundaries and re-exports, traits expose methods without a
`pub` token on each method, macros can generate functions, and `cfg` controls
which items exist for a selected build. Use rustdoc whenever exact reachability
or signatures matter:

```bash
RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --no-deps --locked --open
```

## Root `scirust` facade

The root crate is intentionally small and re-exports the principal framework
surfaces:

- `scirust::core`
- `scirust::learning`
- `scirust::rsi`
- `scirust::simd`
- `scirust::solvers`
- `scirust::symbolic`
- `scirust::prelude`

Optional root surfaces are exposed behind features:

- `scirust::autotune` with `autotune`
- `scirust::flat_autotune` with `flat-autotune`
- `scirust::tensor_canonical` with `tensor-canonical` and backend-specific
  variants

For a user deciding where to start, this facade is the first API boundary to
inspect before dropping into specialist crates.

## CI and review artifacts

The `API function lexicon` GitHub Actions workflow runs when Rust sources,
Cargo manifests, the generator, taxonomy, debt baseline, guide, or workflow
itself change. It:

1. validates Python, taxonomy JSON, and baseline JSON syntax;
2. regenerates the full lexicon;
3. requires every represented package to be classified;
4. requires the adjacent-rustdoc debt counts to match the reviewed baseline;
5. generates aggregate statistics, the machine-readable callable index, and the
   capability index;
6. smoke-tests domain search and the documentation-debt filter;
7. publishes the largest adjacent-rustdoc gaps in the Actions summary;
8. uploads all generated discovery artifacts for review.

The workflow is intentionally non-destructive: it does not commit generated
output or rewrite source documentation from CI.

## Documentation maintenance policy

Function documentation should live next to the Rust item as `///` rustdoc.
Long-form guides should link to APIs rather than duplicate signatures or
behavioral contracts. The generated lexicon may summarize the first rustdoc
sentence, but it must never become a second manually maintained API truth.

The domain taxonomy should describe discoverability, not redefine architecture.
When a crate's responsibility changes, update its code/docs first and then
update the taxonomy to match the reviewed reality. The debt baseline should
ratchet downward as documentation improves and should not be raised merely to
make CI green.
