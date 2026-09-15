# Const API review prompts

For public `const fn`, document the same runtime semantics as ordinary functions
plus any meaningful const-evaluation limitations. Examples can demonstrate both
const-context construction and ordinary runtime use when those are semantically
distinct.

Do not imply that every operation reachable at runtime is available during const
evaluation; document feature/compiler constraints when they matter.
