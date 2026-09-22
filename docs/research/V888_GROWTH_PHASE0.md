# V888-GROWTH Phase 0 — Thor adult-snapshot profile

Status: **exploratory evidence; not a developmental-causality result**.

Execution:
- workflow: `V888 Thor dataset and growth phase 0`
- successful run: `35706848881`
- candidate source: `e803a9265ecf04cc31bf4f399f965be014ada006`
- runner class: `jetson-thor`
- dataset snapshot: BANC v888
- persistent raw-data location: external to the repository

## Data actually materialized

The current public objects are larger/richer than some earlier documentation implied.

| Object | Bytes | Rows |
|---|---:|---:|
| metadata | 57,503,026 | 188,508 |
| morphology/connection metrics | 12,285,378 | 188,508 |
| v3 neuron-neuron edgelist | 359,161,658 | 13,620,865 |
| compartment-resolved edgelist v2 | 847,546,546 | 14,332,296 |
| individual synapses v3 enriched | **19,733,123,829** | **198,816,365** |
| neuron skeleton ZIP | 215,596,097 | n/a |

The complete selected raw-file set is about 21.27 GB in decimal units before local analysis-environment overhead. The individual-synapse Parquet alone is about 19.73 GB (18.38 GiB), not 5.6 GB in the current object served to Thor.

The Parquet contains 1,989 row groups and 31 columns, including pre/post root identities, pre/post coordinates, neuropil/region/side labels, neurotransmitter probabilities and pre/post labels.

Individual-synapse SHA-256:

`0dfb5cf89ba156d076beab2da38d87eaa63dcbe45d76f86b108570fb5b961dd0`

The complete file manifest is retained in the compact workflow evidence bundle.

## Adult hemilineage coverage

- metadata neurons: 188,508
- neurons with a non-empty hemilineage annotation used in Phase 0: 50,501
- distinct hemilineage labels: 259

The metric table exposes more growth-relevant fields than the initial public summary used by the bootstrap, including:
- branchpoints;
- endpoints;
- axon length;
- dendrite length;
- projection score.

These fields are now included in the profiler.

## Exploratory lineage-size scaling

Phase 0 fits an unadjusted OLS relation in log-log space:

`total_metric = a * n_neurons^b`

This is a descriptive adult-snapshot model. It does not establish that lineage size causes the metric or that the exponent is universal.

| Lineage total | b | R² | groups |
|---|---:|---:|---:|
| L2 reconstruction nodes | 0.778 | 0.781 | 259 |
| cable length | 0.776 | 0.770 | 259 |
| branchpoints | 0.895 | 0.721 | 259 |
| endpoints | 0.853 | 0.715 | 259 |
| dendrite length | 0.774 | 0.688 | 258 |
| input connections | 0.778 | 0.679 | 259 |
| segmentation volume | 0.727 | 0.662 | 259 |
| axon length | 0.759 | 0.620 | 253 |
| output connections | 0.837 | 0.604 | 259 |

All listed point estimates are below one. Under this descriptive model, lineage-wide morphology/connectivity increases with neuron count sublinearly rather than as a simple constant-per-neuron multiplication.

This pattern is a hypothesis generator, not yet a biological conclusion. It can arise from cell-type composition, lineage annotation conventions, region composition, differential reconstruction/proofreading, heterogeneous neuron sizes, or true developmental allocation rules.

## First structural observation

The strongest simple size relation in this panel is not synapse count but reconstruction/morphology:
- L2 nodes and cable length have R² around 0.78 and 0.77;
- branchpoints/endpoints are also strongly related to lineage neuron count;
- connection counts are more dispersed;
- volume is more dispersed than cable length.

A useful next question is therefore not merely "which lineage has the most neurons?", but:

> Given the same lineage neuron count, which lineages allocate unusually high or low cable, arbor complexity, volume and synaptic connectivity?

That is the VG-1 residual-analysis target.

## Scientific boundary

BANC v888 is one adult nervous system snapshot. Phase 0 can characterize final architecture and lineage-associated scaling, but it cannot identify developmental causality by itself.

VG-1/VG-3 must add region and matched structural controls. VG-5 must use independent temporal/developmental evidence before statements such as "factor X makes the brain grow" are accepted.

Tracking: SciRust issue #1500.
