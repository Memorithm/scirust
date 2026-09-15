# Rust toolchain qualification for documentation

SciRust declares its toolchain/MSRV in repository configuration/CI. Public
examples should avoid relying accidentally on newer language/library features
unless the relevant package/toolchain contract permits them.

Documentation-specific stable doctest jobs complement, not replace, the declared
MSRV and nightly feature checks in normal CI. If an example needs a nightly-only
feature because the public API itself is nightly-only, state the feature/toolchain
condition and execute it in an applicable job rather than silently excluding it.
