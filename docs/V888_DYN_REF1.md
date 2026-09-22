# DYN-REF1 — three explicit recurrent CPU references

Research preparation for V888-BOOL-0.3, dated 2026-09-22. No task or learned-model result is claimed by this document.

## Scope and reuse

The previously inspected scirust-sim provides ODE/environment machinery but no qualified delayed LIF/integer/Boolean network path. This small std-only Rust reference is staged in scripts/v888_dyn01, with a separate library module and administrative differential driver. It is a candidate for later integration into scirust-sim; no new crate or public ecosystem API is created. Generic topology, rules, delays and counters contain no two-door game logic, brain-specific labels or market code.

This independent synthetic qualification can precede consumption of V888. It does not complete V888-BOOL-0.2, bypass its port/partition/control diagnostics, or justify a topology comparison. The game and GAME-IO1 are retained unchanged. No BANC dataset is read by this slice.

## Exact update semantics

Each call advances one synchronous integer tick. A directed edge of delay d reads its source spike from t-d; before the first update all past spikes are zero. Delay is in [1,64]. The output of tick t is not visible to another node until at least t+1. Original input edge order is canonicalized by (source,target); duplicate directed pairs are rejected, self-connections are allowed. There is one model weight/delay per directed pair, not a raw individual-synapse model.

Three model families are separate:

* Discrete LIF: v_next = leak*v + sum(weight*delayed_spike) + drive, with leak in [0,1], positive threshold, reset zero and an explicit refractory count. This is a discrete equation, not a claimed continuous-time integration accuracy or physiological calibration.
* Integer: truncate v/divisor toward zero, add signed impulses and current drive with checked i64 arithmetic, clip ONCE to [-cap,cap], then apply threshold/reset. Per-edge clipping is deliberately forbidden because it changes cancellation semantics. Clipping is counted.
* Boolean: XOR the current external bit with parity of all incoming delayed bits. The nonlinear variant also XORs the conjunction of the first two incoming delayed bits in canonical source order. With fewer than two incoming edges that conjunction is zero. Both variants require unit weights and 0/1 drives; numeric weights are not silently discarded.

After a numeric neuron fires, the next K updates are refractory: state remains zero, incoming drive for those updates is discarded, and the counter decrements. All edge visits and arriving active-edge signals are still accounted. Boolean models have no refractory state.

Weights, signs, delays and thresholds in the synthetic fixtures are declared model parameters. Contact multiplicity is never relabelled as measured conductance or excitation/inhibition.

## State, memory and work

Connectivity and parameters are immutable in this slice. reset_activity clears current state, the pending history ring and episode work counters, without changing the parameters. This separation prepares later learning/checkpoint work but does not implement any training, learned checkpoint, readout or reward rule.

The reference scans every edge on every tick. It is not an optimized sparse event scheduler. Histories and spikes are byte arrays, not packed bits. Numeric scratch remains reserved even in the Boolean reference, and is included in direct vector-payload accounting. The bound excludes caller-owned input arrays, Vec headers, allocator overhead, stack, tracing and OS memory. It must not be presented as process RSS or a performance gain.

Max ticks, complete edge visits and direct vector bytes are caller-specified. Size/configuration/shape checks precede construction or state commit. Scratch computation is transactional with respect to observable state; on an invalid drive or exceeded work budget no new tick/history/state is committed. User-supplied work budgets are not an OS sandbox or a wall-clock guarantee.

## Qualification plan

Rust unit tests cover quiescence, exact delays including 64 updates, synchronous ordering, recurrent pulse persistence, reset, a geometric discrete LIF response, negative integer leak, signed cancellation, saturation, refractory length, Boolean nonlinearity, ordering, bad input/topology and resource budgets.

The independent Python reference is target-gather over complete time history, rather than Rust source-scatter with a ring buffer. Signed sums use Python arbitrary precision and reverse incoming order. Every spike, integer potential, refractory value and work counter must agree; LIF state tolerances are 1e-12 absolute and relative. A complete trace replay must be byte-identical on the same compiler/host/source.

The frozen synthetic panel contains 32 seeds across four variants (LIF, integer, affine Boolean, nonlinear Boolean), explicit delays, quiescence and programmed recurrent pulse cycles. A pulse persisting in a deliberately constructed loop is a dynamical oracle, not learned memory or V888 intelligence. All fixture inputs are an administrative test interface, never the game candidate protocol.

## Remaining work

Read actual retained run results before claiming this reference qualified. Graph-control diversity, structural versus anatomical partitions, port placement, real V888 integration, causal game adapter, explicit learned-state persistence and learning curves remain separate tasks. Event-driven equivalence and acceleration require their own references and measurements. Trading remains deferred; ITD V29.18, source data and protected evaluation boundaries remain unchanged.
