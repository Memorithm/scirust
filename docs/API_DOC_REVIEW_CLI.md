# CLI review prompts

For command-line surfaces, document/review:

- subcommand/flag syntax and defaults;
- input files/stdin/environment variables;
- stdout/stderr format and exit codes;
- destructive/network side effects;
- reproducibility/seed/output path behavior;
- validation/error messages;
- relationship to the underlying library API.

CLI examples are command examples, not substitutes for Rustdoc examples on
public library callables.
