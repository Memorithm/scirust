# SciRust

SciRust is a Rust scientific-computing and machine-learning workspace. It combines numerical methods, tensor/autodiff infrastructure, optimization, simulation, signal processing, symbolic tools, domain crates, and optional GPU execution paths.

The repository is research-oriented. Capabilities differ in maturity, and a compiled feature is not automatically a validated hardware path. Use the linked documentation and revision-bound evidence when deciding whether a component is suitable for a particular experiment or application.

## Quick start

Prerequisites:

- Rust toolchain from [`rust-toolchain.toml`](rust-toolchain.toml)
- Git
- platform build tools required by the crates/features you enable

Clone and run the default smoke path:

```bash
git clone https://github.com/Memorithm/scirust.git
cd scirust
cargo run -p quickstart_v2
```

The quickstart trains the repository's small deterministic classifier example.
It is a functional smoke test, not a performance benchmark.

## Library example

The CLI is a thin entry point over workspace crates. Applications can depend on
individual path crates while developing inside the workspace:

```toml
[dependencies]
scirust-core = { path = "scirust-core" }
scirust-solvers = { path = "scirust-solvers" }
```

For an end-to-end classifier, start with
[`examples/quickstart_v2`](examples/quickstart_v2) or run:

```bash
cargo run -p quickstart_v2
```

## Repository map

| Path | Purpose |
|---|---|
| [`scirust-core/`](scirust-core/) | Core tensor, autodiff, neural-network, and quantization code |
| [`scirust-cli/`](scirust-cli/) | Unified `scirust` command-line interface |
| [`scirust-tensor-core/`](scirust-tensor-core/) | Canonical tensor representation |
| [`scirust-tensor-runtime/`](scirust-tensor-runtime/) | Canonical execution runtime |
| [`scirust-gpu/`](scirust-gpu/) | CPU reference, WGPU, and CUDA adapters behind features |
| [`scirust-learning/`](scirust-learning/) | Learning algorithms, NLP utilities, and reinforcement learning |
| [`scirust-solvers/`](scirust-solvers/) | Numerical solvers and linear algebra |
| [`scirust-symbolic/`](scirust-symbolic/) | Symbolic expressions and differentiation |
| [`scirust-sim/`](scirust-sim/) | Deterministic simulation environments |
| [`scirust-signal/`](scirust-signal/) | Signal processing and radar-oriented reference algorithms |
| [`scirust-industrial/`](scirust-industrial/) | Industrial demonstration CLI |
| [`scirust-mcp/`](scirust-mcp/) | Model Context Protocol server |
| [`scirust-sciagent/`](scirust-sciagent/) | Experimental local language-model tooling |
| [`examples/`](examples/) | Runnable examples |
| [`docs/`](docs/) | Guides, design notes, evidence, and archived research notes |
| [`paper/`](paper/) | Technical reports and paper material; measurements are historical unless dated otherwise |
| [`external/`](external/) | Vendored external project snapshots; not part of the SciRust public API |

This map highlights entry points rather than listing every crate. Use
`cargo metadata --no-deps` for the authoritative workspace inventory.

## Features and hardware backends

The root crate has no default features. The canonical tensor facade is exposed
through explicit features:

| Feature | Effect |
|---|---|
| `tensor-canonical` | Canonical tensor facade with a caller-supplied backend |
| `tensor-canonical-cpu` | Facade plus the deterministic CPU reference adapter |
| `tensor-canonical-wgpu` | Facade plus the WGPU adapter |
| `tensor-canonical-cuda` | Facade plus the CUDA adapter |

Examples:

```bash
cargo run --example canonical_tensor_cpu --features tensor-canonical-cpu
cargo test -p scirust-gpu --features wgpu
cargo test -p scirust-gpu --features cuda
```

Enabling a feature proves only that the selected code is built and tested in
the current environment. It does not imply that compatible GPU hardware or
drivers are present.

## Verification

The main workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
Run the standard local checks with:

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +nightly-2026-07-02 clippy --workspace --all-targets --locked -- -D warnings
cargo +nightly-2026-07-02 build --workspace --all-targets --locked
cargo +nightly-2026-07-02 test --workspace --locked
```

Additional workflows cover native ARM64, release, SOS, and Studio-specific
paths. Some hardware tests require self-hosted runners and therefore provide a
narrower guarantee than portable hosted CI. See
[`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md) for the repository's acceptance
protocol and [`docs/evidence/`](docs/evidence/) for retained evidence.

Test totals and benchmark numbers are intentionally omitted here because they
become stale as the workspace changes. Reproduce a result from its dated report
and exact command instead.

## Documentation

- [Documentation hub](docs/README.md)
- [Quickstart](docs/QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Command and API reference](docs/REFERENCE.md)
- [Public function lexicon](docs/API_LEXICON.md)
- [API example inventory](docs/API_EXAMPLES.md)
- [Human terminology lexicon](docs/LEXICON.md)
- [Glossary](docs/GLOSSARY.md)
- [Public API documentation standard](docs/API_DOCUMENTATION_STANDARD.md)
- [API documentation progress](docs/API_DOCUMENTATION_PROGRESS.md)
- [GPU status and usage](docs/GPU.md)
- [Test protocol](docs/TEST_PROTOCOL.md)
- [Release process](docs/RELEASING.md)
- [Security policy](SECURITY.md)
- [Licensing guide](LICENSING.md)
- [Changelog](CHANGELOG.md)
- [Research notes](docs/research-notes/)
- [Audit archive](docs/audits/)
- [Documentation overview](docs/translations/Documentation_EN.md)

Historical reports record the state and measurements of a particular revision.
They should not be read as guarantees for the current `master` branch.

## Contributing and security

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a change. Report
security issues through the private process in [`SECURITY.md`](SECURITY.md), not
through a public issue.

## License

SciRust is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md). Commercial use is not
granted by that license. See [`LICENSING.md`](LICENSING.md) for the separate
commercial licensing path.

Copyright © 2026 Tarek Zekriti.
