# PORTS-DYN1: source-bound port diagnostics and Boolean impulse responses

This research slice advances V888-BOOL-0.2b port sensitivity and prepares source-bound dynamics for 0.3. It does not complete the parent null-ensemble/partition qualification and is not a game learner.

The existing DYN-REF1 recurrent library is reused without modification. The build requires its exact SHA-256, as qualified at source 75f43ec35b1118c98b23bbde8fc8e1d6bfd42036. The new driver only binds qualified TSV graph arms to that library, provides bounded administrative impulses and emits evidence. No graph generator, game or new numerical crate is introduced.

## Frozen scope

Consume all 12 cases and all 5 arms from BOOL-0.2a run 35726122553. Accepted receipt hashes and each consumed node/edge/port hash are checked before and after. The old raw data, graph, cut-edge files, original ports and failed/sparse cases remain untouched. The protocol retains detector-v3 scope through the accepted parent evidence. No second connectome or network download is performed.

Seventeen port configurations are reported per graph: the original plus sixteen deterministic ID-ranked placements. Every arm receives the same placements within each source case. Alternative panels measure sensitivity, do not replace the original ports, and are not selected for success. Duplicate panels remain in the count, explicitly not independent samples.

Rust emits directed shortest hop distances (-1 means unreachable), reachability and route-intersection counts. Independently implemented set-frontier breadth layers verify every report. A directed path is a structural possibility, not proof of transmission under any particular equation.

## Source-bound dynamics

Only the original placement is stimulated dynamically. Both Boolean equations from DYN-REF1 are used: affine XOR and XOR plus the AND of the first two canonically ordered incoming bits. Every edge is unit-weight and has delay one. These are declared model assumptions, not inferred biology; synapse multiplicity is not conductance.

Each model receives four trials: zero drive, an A impulse, a B impulse, and simultaneous A+B impulses at tick zero, followed by zero drive through tick 128. Activity resets between trials. The output includes every node state in little-bit-order hexadecimal, simulation counters, direct vector payload, and a separate zero-drive reset probe. No weights or gates are trained.

The NumPy oracle gathers predecessor bits and counts them before modulo-two reduction; its sums are exact because bounded in-degrees are <=4095. It also propagates Boolean OR support, independently of model activity. This distinguishes missing paths from even-path cancellation. Affine superposition is checked exactly. Nonlinear superposition failure is a diagnostic of the declared gate, not learned deduction.

Every full Rust trace is replayed byte-for-byte. Counters report one pass; replay doubles Rust simulation/reset work. They are not wall-clock energy or process-memory measurements. Full CI and scoped Thor execution remain separate; exact head and result identities must be recorded after execution.

## Boundaries

There is no readout training, episode learning, checkpoint retention, protected-final study, model promotion, trading or biological-growth conclusion. Unit adjacency, fixed synchronous scans and numeric scratch remain those of DYN-REF1: no bit-packed implementation or event-driven speedup is asserted.

Remaining work includes independent null ensembles/partition sensitivity, input/readout design for the game, persistent learned parameters and actual unseen-episode learning comparisons. ITD owns those protocols and their interpretation. The graph-binding and influence diagnostics may support SML-GENIUS and FLAT sparse routing only after their own qualification.
