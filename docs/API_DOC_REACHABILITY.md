# Public reachability reconciliation

The final documentation metric must answer which callables an external crate can
actually use.

## Why source scanning is insufficient

`pub fn` can occur inside a private module and remain unreachable. Conversely,
public re-exports, trait methods and macro-generated items can expose callables
that a simple declaration regex does not inventory. `cfg`/feature selection also
changes the surface.

## Target reconciliation

Generate compiler/rustdoc metadata for selected feature/target configurations,
extract externally visible callable identities/paths, and reconcile them with
source observations. Preserve unmatched items rather than dropping them:

- source-only → `reachability: unknown/not_reachable` after evidence;
- compiler-only → generated/re-export/trait item requiring semantic mapping;
- matched → `reachable` with canonical public path and source provenance.

## Acceptance

Do not replace the current 12k-scale source inventory with a smaller number until
the reconciliation can explain every material difference. The goal is accuracy,
not a cosmetically lower documentation-debt denominator.
