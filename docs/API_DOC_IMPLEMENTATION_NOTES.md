# Documentation infrastructure implementation notes

Existing infrastructure already provides two useful foundations:

- `scripts/api-lexicon.py`: deterministic source-level public callable discovery,
  package/domain taxonomy, Markdown/JSON output and exact adjacent-summary debt
  baseline checking;
- `scripts/api-example-audit.py`: source-bound documentation code-fence
  observation and reviewed example-policy enforcement introduced through #1432.

The v2 work should compose these rather than create parallel inventories. The
main missing architectural layer is compiler/rustdoc effective reachability;
additional semantic observations can be added incrementally after parser tests
establish their reliability.
