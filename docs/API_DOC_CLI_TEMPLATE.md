# CLI and library-boundary documentation template

For APIs exposed through both library and command-line surfaces, document the
library contract independently from CLI presentation.

Library Rustdoc should state typed inputs/outputs/errors. CLI documentation
should additionally state flags, defaults, environment variables, files/stdin,
stdout/stderr format, exit status and destructive/network side effects.

Examples for library callables belong in Rustdoc and should execute without
shelling out when a direct API exists. CLI examples should be reproducible
commands using small local fixtures. Do not present a CLI wrapper as the only
way to use an underlying public library capability.
