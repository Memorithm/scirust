# SciRust BLAS MKL provider — local installation only

Status: **system-provider contract**, not an MKL performance qualification.

The optional `scirust-core/blas-mkl` backend no longer selects
`blas-src/intel-mkl`, whose `intel-mkl-src 0.8.1` build dependency pulls
`ocipkg 0.2.9`. The latter is affected by the OCI archive path-traversal
advisory **RUSTSEC-2026-0450** (arbitrary filesystem modification / possible
code execution). Silencing cargo-deny is **not** a remediation.

Instead, `blas-mkl` requires a **separately installed, trusted oneMKL**
runtime under an absolute `MKLROOT` path. The build script checks for
`libmkl_rt.so` on Linux, `libmkl_rt.dylib` on macOS or `mkl_rt.lib`
on Windows inside `MKLROOT/lib`, `lib/intel64` or `lib/intel64_lin`.
If missing, compilation refuses the feature rather than downloading or
falling back to another provider. No GPU vendor SDK is introduced.

Example for an already-installed oneMKL on Linux:

```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
cargo check --locked -p scirust-core --no-default-features --features blas-mkl
```

An installation path must be supplied by the operator; the example does
not assert that this host has oneMKL installed. Deployment is responsible
for configuring the dynamic loader to locate the runtime. Do not enable
`blas-openblas` and `blas-mkl` simultaneously.

`blas-openblas` remains the available auto-provisioned alternative.
The CI `Audit MKL dependency graph` must pass without exemptions for
RUSTSEC-2026-0450. System MKL linking and numerical parity still require
a separate physical validation on a host with licensed system libraries.
