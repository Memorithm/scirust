# SML PVP6 fixture adapter

This adapter validates the frozen SML-owned CSV interchange boundary before a
SciRust backend consumes the same coefficients, query schedules and expected
outputs.

The checker is deliberately structural. It verifies:

- the v1 header and exact K=2048/G=16 geometry;
- all 2048 unique BANK records and 6144 ordered QUERY records;
- all 24576 ordered OUTPUT records;
- lowercase 64-bit words and zero output padding bits 16..63;
- the exact END trailer and absence of trailing records.

It does not replace SML-GENIUS's independent monomial oracle, and it does not
claim that a fixture is a trained-model artifact.

## Reproduce

Obtain the exact fixture from SML-GENIUS at the frozen contract SHA:

`feb1a144e23d9a29439b7cad6704f56637a7f7ead1a9b9abb39ad9c0ac5325fd`

Then run:

```bash
sha256sum pvp6_fixture.csv
cargo run -p scirust-simd --example pvp6_fixture_check -- --fixture pvp6_fixture.csv
```

The SHA-256 comparison is intentionally kept outside this dependency-free
Rust checker; the caller must compare it with the frozen value before using the
fixture for cross-repository timing.

A successful structural check prints a versioned JSON record with
`"status":"structural-ok"`. Native timing remains a separate campaign and
must use the identical fixture after the SHA check.
