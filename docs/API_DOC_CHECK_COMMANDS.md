# API documentation verification commands

Run from the repository root. Replace `<package>` with the package under review.

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +stable test -p <package> --doc --locked
RUSTDOCFLAGS='-D warnings' cargo +stable doc -p <package> --no-deps --locked
cargo +stable test -p <package> --locked
cargo +stable clippy -p <package> --all-targets --locked -- -D warnings
```

Generate/search the source-level lexicon:

```bash
python3 scripts/api-lexicon.py --help
python3 scripts/api-lexicon.py --package <package> --missing-docs
```

Inspect example-audit options:

```bash
python3 scripts/api-example-audit.py --help
```

The repository's hosted workflows remain the merge evidence. Local commands are
useful preflight checks but must not be reported as exact-head hosted results.
