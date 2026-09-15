# Exact versus approximate result review prompts

Public numerical/symbolic APIs should state whether results are exact,
floating-point approximations, heuristic estimates or bounds/certificates.

For approximate results, document tolerance/error semantics and reference when
relevant. For bounds/certificates, state what property is actually certified and
under which assumptions. For symbolic APIs, distinguish syntactic rewrite from
proved equivalence.

Examples should assert the corresponding kind of guarantee rather than treating
all outputs as exact equality by default.
