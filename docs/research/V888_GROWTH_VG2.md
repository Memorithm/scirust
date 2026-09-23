# V888-GROWTH VG-2 — mitochondrial coverage and association audit

Status: **coverage-limited exploratory evidence; not a global energetic-growth result**.

Execution:
- workflow: `V888 VG-2 mitochondrial audit on Thor`
- successful run: `35708072025`
- candidate head: `c9b71b1a7d4ebb57328f70827d84e7086bcc89a9`
- complete BANC v888 raw-file manifest verified before analysis

## Coverage result

Across 188,508 metric rows:

| State | Mitochondria count | Mitochondrial volume |
|---|---:|---:|
| positive | 13,950 | 13,950 |
| zero | 160,866 | 160,866 |
| null/non-finite | 13,692 | 13,692 |
| negative | 0 | 0 |

The positive count and positive-volume sets coincide exactly.

Only **7.4002 %** of all rows contain positive values for both mitochondrial fields.

The released documentation identifies these values as mitochondrion count inside the neuron segmentation and summed mitochondrial volume from the CAVE mitochondria table. The current public documentation does not justify treating a recorded zero as proof of biological absence.

## Strong selection signature

The positive-mitochondria subset is not morphologically representative of the zero-valued subset.

Selected medians:

| Adult per-neuron metric | positive mitochondria | zero mitochondria |
|---|---:|---:|
| cable length (µm) | 89.79 | 400.145 |
| volume (nm³) | 27.16e9 | 51.77e9 |
| input connections | 2 | 114 |
| output connections | 2 | 388 |
| branchpoints | 23 | 29 |
| endpoints | 64 | 73 |
| axon length | 104.7 | 129.3 |
| dendrite length | 185.2 | 197.7 |

This asymmetry is too large to treat the positive subset as a random sample of the CNS.

## Associations inside the positive subset

Within the 13,950 positive rows, Spearman associations are moderate for morphology:

| Pair with mitochondria count | ρ |
|---|---:|
| segmentation volume | 0.653 |
| cable length | 0.614 |
| dendrite length | 0.546 |
| endpoints | 0.538 |
| branchpoints | 0.482 |
| input connections | 0.452 |
| axon length | 0.442 |
| output connections | 0.362 |

Mitochondrial count versus mitochondrial volume has ρ = 0.757.

Mitochondrial volume shows similar associations with volume (ρ=0.660) and cable length (ρ=0.643).

These associations are descriptive **within the selected positive subset** and cannot currently be generalized to the whole connectome.

## Regional variation inside the selected subset

Per-neuron log-log fits remain region dependent. For mitochondria count versus neuron volume:

- central brain: slope 0.682, R² 0.418, n=3,985;
- optic lobe: slope 0.694, R² 0.321, n=6,553;
- VNC: slope 0.781, R² 0.487, n=3,408.

For mitochondria count versus cable length:

- central brain: slope 0.389, R² 0.373;
- optic lobe: slope 0.214, R² 0.194;
- VNC: slope 0.381, R² 0.413.

## Decision

VG-2 does **not** support promoting mitochondrial abundance as a global brain-growth driver from the current compiled metrics.

The signal remains scientifically interesting because:
- the positive subset shows coherent morphology/mitochondria associations;
- the original BANC project automatically detected tens of millions of mitochondria in the underlying EM volume;
- therefore the gap between global detection and the sparse positive per-neuron compiled metric requires provenance/coverage clarification before inference.

The next energetic step is not a larger regression. It is a provenance audit of how CAVE mitochondrial objects were assigned to neuron segmentations and why the released per-neuron table contains positive values for only 7.4 % of rows.

## Boundary

Zero is not interpreted as biological absence. No claim about metabolism causing neural growth is accepted from VG-2.

Tracking: SciRust issue #1500.
