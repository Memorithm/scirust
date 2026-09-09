# SciRust API lexicon

SciRust is large enough that a hand-maintained function catalogue would drift
quickly. The documentation stack therefore separates orientation, operational
reference, source-level discovery, and authoritative API documentation.

## Documentation layers

1. `README.md` gives the supported entry points and repository map.
2. `docs/REFERENCE.md` documents commands, quality gates, features, and API
   entry points.
3. `scripts/api-lexicon.py` generates a compact, searchable lexicon of directly
   declared public functions and methods across workspace crates.
4. Rustdoc remains the authoritative API reference for effective visibility,
   signatures, re-exports, cfg expansion, trait-provided methods, generated
   items, and intra-doc links.

This distinction is intentional: the lexicon is optimized for finding a
capability quickly; rustdoc is optimized for exact API semantics.

## Generate the current function lexicon

From the repository root:

```bash
python3 scripts/api-lexicon.py \
  --output /tmp/scirust-api-functions.md \
  --stats-json /tmp/scirust-api-functions.json
```

To inspect one or more crates:

```bash
python3 scripts/api-lexicon.py --package scirust-core
python3 scripts/api-lexicon.py \
  --package scirust-solvers \
  --package scirust-symbolic
```

The generated Markdown is deterministic for a fixed checkout. Each entry
contains the workspace package, declared symbol, first adjacent `///` sentence
when present, and source file/line.

## What the lexicon indexes

The scanner includes directly declared public Rust callables matching `pub fn`
and the corresponding `async`, `const`, `unsafe`, and `extern` forms. This
covers free functions and inherent public methods declared in workspace `src/`
trees.

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

## Documentation maintenance policy

Function documentation should live next to the Rust item as `///` rustdoc.
Long-form guides should link to APIs rather than duplicate signatures or
behavioral contracts. The generated lexicon may summarize the first rustdoc
sentence, but it must never become a second manually maintained API truth.

The `API function lexicon` GitHub Actions workflow regenerates the inventory on
changes to Rust sources, Cargo manifests, the generator, or this guide. It
uploads both the Markdown lexicon and JSON statistics as review artifacts. The
workflow is intentionally non-destructive: it does not commit generated output
or rewrite documentation from CI.

The JSON statistics make documentation debt measurable without inventing a
coverage claim. A missing adjacent `///` summary is reported as an observation,
not as proof that an API is undocumented elsewhere. This metric can later be
used to prioritize high-value documentation work crate by crate.
