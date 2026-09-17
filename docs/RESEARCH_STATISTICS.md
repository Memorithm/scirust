# Bounded research statistics

`scirust-stats` supplies general statistical computation. TDI owns experimental
units, pairing, exclusions, protected domains, preregistration and scientific
decisions. SciRust does not acquire a scheduler or a product-specific evaluator.

The additions build on the existing `describe::quantiles` type-7 implementation
and `SplitMix64`. Inventory at base `aa13f62ea829c5c42e370a32d51143f016a0dc44`
found no equivalent bootstrap, Holm, Morris or Sobol entry points in
`scirust-stats`; `scirust-tdi` retains its existing finite-state dynamics role.

## Library and process interfaces

| API | Input and method | Failure policy |
| --- | --- | --- |
| `resampling::paired_mean_percentile` | One paired contrast per independent unit; equally weighted mean, independent SplitMix64 resampling, two-sided type-7 percentile interval | 2..10000 units, 100..100000 resamples, maximum 10 million sampled contributions; reject non-finite data/arithmetic |
| `resampling::holm_adjust` | Complete caller-declared family of p-values; Holm step-down adjustment in original order | 1..10000 finite values in [0,1] |
| `sensitivity::morris_effects` | Complete normalized one-factor-at-a-time trajectories; signed/absolute mean and sample SD of actual finite differences | 2..10000 trajectories, 1..32 factors, 1 million input scalars; reject invalid/duplicate/multifactor steps |
| `sensitivity::sobol_first_total` | A, B and A-with-B-factor-j outputs; centered Saltelli 2010 first/total-order estimates, combined A/B population variance | 2..100000 base rows, 1..32 factors, 1 million output scalars; explicit zero-variance rejection |

The library has no new production dependency. The `research_stats` example uses
serde/serde_json only as development dependencies and exposes
`scirust-research-stats-json/v1`: one JSON request on stdin, one response on
stdout, at most 1 MiB input. Success exits 0 with `schema:1,status:computed` and a
`result`. Errors exit 21 with `status:rejected` and no numeric result. Unknown or
duplicate fields, unsupported schemas, overflow and oversized input are rejected.
The caller provides OS containment and a time budget for this trusted executable.

```bash
cargo build --locked -p scirust-stats --example research_stats
echo '{"schema":1,"operation":"holm","p_values":[0.01,0.04,0.03]}' | target/debug/examples/research_stats
echo '{"schema":1,"operation":"paired_mean_percentile","contrasts":[0,2],"resamples":1000,"confidence":0.95,"seed":"731"}' | target/debug/examples/research_stats
```

Other operations are `morris` with `inputs` (rows) and `outputs`, and `sobol`
with `a`, `b`, `ab` (one output vector per factor). Seeds are canonical decimal
u64 strings, preserving the full range through JavaScript-compatible JSON.

## Qualification and interpretation

```bash
cargo test --locked -p scirust-stats
cargo clippy --locked -p scirust-stats --all-targets -- -D warnings
python3 -m venv .research-reference
.research-reference/bin/pip install -r scripts/requirements-research-stats.txt
.research-reference/bin/python scripts/check-research-stats-reference.py --worker target/debug/examples/research_stats
```

The bounded reference script was executed with Rust 1.89.0 on Linux x86_64,
Python 3.12, SciPy 1.18.1, NumPy 2.5.3 and SALib 1.5.2. It checks the percentile
interval against complete enumeration of a four-value empirical population and
an independent SciPy bootstrap; Morris summaries against SALib on a nonlinear
polynomial; and Sobol estimates against SALib on additive and Ishigami functions.
The additive example is also checked against analytical variance shares with a
declared absolute tolerance of 0.015 for 1024 scrambled base samples. Direct
SALib numerical agreement uses an absolute/relative tolerance of 5e-12. Tests
also cover explicit rejections and repeat evaluation in separate processes.

This is empirical numerical qualification of these fixtures, not a coverage
theorem or hardware result. A common seed does not establish independence.
Percentile intervals have approximate finite-sample coverage; cluster weighting,
missing-data rules, multiple-comparison families and exploratory/confirmatory
roles must be declared by the caller. Holm does not replace a prescribed family
or manufacture p-values. Sensitivity assumes independent input factors and the
appropriate design; correlated/constrained inputs require other methods. Morris
standard deviations are not confidence intervals. Sobol estimates are not
clipped to [0,1], and this API supplies no second-order or uncertainty estimate.
Constant-output Sobol data is explicitly rejected. Sampling/layout contracts
remain the consumer's responsibility; an output-only API cannot verify them.

The TDI consumer is a separate bounded process adapter and paired report path.
It first pairs repeats and averages within each declared independent unit,
then calls this binary. Source commit and binary digest are recorded separately;
declaring a source commit is not a reproducible-build attestation. Existing TDI
series-specific frozen statistics remain unchanged. Cross-platform bitwise
equivalence has not been established; determinism claims refer to the tested
algorithm, ordered inputs, seed, build and target.

References used to specify and independently qualify the methods:
[SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
[SALib analysis API](https://salib.readthedocs.io/en/latest/api/SALib.analyze.html),
[SALib Sobol estimator documentation](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html).
