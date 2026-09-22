//! DYN-REF1: deterministic delayed recurrent CPU references.
//!
//! Research staging for scirust-sim, not a public API promotion or fitted animal model.
//! Edges are supplied model parameters, NOT inferred synaptic conductances/signs.
//! At update t an edge with delay d reads spike[t-d]; all negative times are zero.
//! All outputs commit simultaneously. Failed steps leave observable state unchanged.
//! Numerical LIF, bounded integer integration and Boolean parity/nonlinear recurrence
//! are distinct equations. This is fixed-step execution, not an event-driven speedup.
#![forbid(unsafe_code)]

use std::mem::size_of;

/// One directed model connection; source and target are caller-owned index identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub weight: i32,
    pub delay: u16,
}

/// Explicit discrete-time model assumptions. Zero reset and rest are fixed in v1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dynamics {
    Lif { leak: f64, threshold: f64, refractory: u16 },
    Integer { divisor: u32, threshold: i64, cap: i64, refractory: u16 },
    // Boolean mode requires unit edges and 0/1 external drives.
    // nonlinear adds AND(first incoming delayed bit, second incoming delayed bit)
    // to the parity. Incoming ordering is by canonical source index, never by label.
    Boolean { nonlinear: bool },
}

/// Per-episode work and direct vector-payload bounds, checked before allocations/steps.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_ticks: u64,
    pub max_edge_visits: u64,
    pub max_payload_bytes: usize,
}

/// Integer counters; no inference about wall time, energy, allocator or OS memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub ticks: u64,
    pub edge_visits: u64,
    pub active_edge_events: u64,
    pub spikes: u64,
    pub clipped_nodes: u64,
    pub refractory_nodes: u64,
}

/// Immutable connectivity/parameters plus resettable transient state and work counters.
/// The reference reserves numeric scratch even in Boolean mode: packing is future work.
pub struct Network {
    n: usize,
    edges: Vec<Edge>,
    dynamics: Dynamics,
    limits: Limits,
    slots: usize,
    payload_bytes: usize,
    counters: Counters,
    history: Vec<u8>,
    spikes: Vec<u8>,
    integer: Vec<i64>,
    voltage: Vec<f64>,
    refractory: Vec<u16>,
    accumulator: Vec<i64>,
    parity: Vec<u8>,
    first: Vec<u8>,
    pair: Vec<u8>,
    seen: Vec<u8>,
    next_spikes: Vec<u8>,
    next_integer: Vec<i64>,
    next_voltage: Vec<f64>,
    next_refractory: Vec<u16>,
}

impl Network {
    pub fn new(n: usize, edges: &[Edge], dynamics: Dynamics, limits: Limits) -> Result<Self, &'static str> {
        if n == 0 || n > 1_000_000 { return Err("invalid node count"); }
        if limits.max_ticks == 0 { return Err("zero tick budget"); }
        match dynamics {
            Dynamics::Lif { leak, threshold, .. } => {
                if !leak.is_finite() || !(0.0..=1.0).contains(&leak) || !threshold.is_finite() || threshold <= 0.0 {
                    return Err("invalid LIF parameters");
                }
            }
            Dynamics::Integer { divisor, threshold, cap, .. } => {
                if divisor == 0 || threshold <= 0 || cap < threshold || cap > i64::from(i32::MAX) {
                    return Err("invalid integer parameters");
                }
            }
            Dynamics::Boolean { .. } => {}
        }
        let mut max_delay = 1usize;
        for e in edges {
            if e.source >= n || e.target >= n || e.delay == 0 || e.delay > 64 || e.weight == 0 {
                return Err("invalid edge");
            }
            if matches!(dynamics, Dynamics::Boolean { .. }) && e.weight != 1 {
                return Err("Boolean reference requires unit edges");
            }
            max_delay = max_delay.max(usize::from(e.delay));
        }
        let slots = max_delay + 1;
        let history_len = n.checked_mul(slots).ok_or("size overflow")?;
        // 2 spike arrays, 2 integer arrays, 2 voltage arrays, 2 refractory arrays,
        // one i64 accumulator, 4 byte arrays, and the delayed spike ring.
        let per_node = 2 * size_of::<u8>() + 2 * size_of::<i64>() + 2 * size_of::<f64>()
            + 2 * size_of::<u16>() + size_of::<i64>() + 4 * size_of::<u8>();
        let payload_bytes = edges.len().checked_mul(size_of::<Edge>())
            .and_then(|x| n.checked_mul(per_node).and_then(|y| x.checked_add(y)))
            .and_then(|x| x.checked_add(history_len)).ok_or("size overflow")?;
        if payload_bytes > limits.max_payload_bytes { return Err("payload budget exceeded"); }
        let mut ordered = edges.to_vec();
        ordered.sort_unstable_by_key(|e| (e.source, e.target));
        if ordered.windows(2).any(|p| (p[0].source, p[0].target) == (p[1].source, p[1].target)) {
            return Err("duplicate directed pair");
        }
        Ok(Self {
            n, edges: ordered, dynamics, limits, slots, payload_bytes,
            counters: Counters::default(), history: vec![0; history_len],
            spikes: vec![0; n], integer: vec![0; n], voltage: vec![0.0; n], refractory: vec![0; n],
            accumulator: vec![0; n], parity: vec![0; n], first: vec![0; n], pair: vec![0; n], seen: vec![0; n],
            next_spikes: vec![0; n], next_integer: vec![0; n], next_voltage: vec![0.0; n], next_refractory: vec![0; n],
        })
    }

    pub fn counters(&self) -> Counters { self.counters }
    pub fn payload_bytes(&self) -> usize { self.payload_bytes }
    pub fn spikes(&self) -> &[u8] { &self.spikes }
    pub fn integer_state(&self) -> &[i64] { &self.integer }
    pub fn voltage_state(&self) -> &[f64] { &self.voltage }
    pub fn refractory_state(&self) -> &[u16] { &self.refractory }
    pub fn parameters(&self) -> Dynamics { self.dynamics }
    pub fn edges(&self) -> &[Edge] { &self.edges }

    /// Reset activity and episode counters only. No rule/edge/parameter is trained here.
    pub fn reset_activity(&mut self) {
        self.history.fill(0); self.spikes.fill(0); self.integer.fill(0);
        self.voltage.fill(0.0); self.refractory.fill(0); self.counters = Counters::default();
    }

    /// One synchronous tick; refractory nodes discard this tick's input and stay at zero.
    /// Integer clipping is applied ONCE after a checked signed sum (never per-edge).
    /// The input is current drive, not a target or an externally retained cue history.
    pub fn step(&mut self, drive: &[i32]) -> Result<&[u8], &'static str> {
        if drive.len() != self.n { return Err("drive dimension mismatch"); }
        if matches!(self.dynamics, Dynamics::Boolean { .. }) && drive.iter().any(|&x| x != 0 && x != 1) {
            return Err("Boolean drive must be 0 or 1");
        }
        let mut next = self.counters;
        next.ticks = next.ticks.checked_add(1).ok_or("tick overflow")?;
        next.edge_visits = next.edge_visits.checked_add(self.edges.len() as u64).ok_or("counter overflow")?;
        if next.ticks > self.limits.max_ticks || next.edge_visits > self.limits.max_edge_visits {
            return Err("work budget exceeded");
        }
        self.accumulator.fill(0); self.parity.fill(0); self.first.fill(0); self.pair.fill(0); self.seen.fill(0);
        self.next_spikes.fill(0); self.next_integer.fill(0); self.next_voltage.fill(0.0); self.next_refractory.fill(0);
        let tick = self.counters.ticks;
        let boolean = matches!(self.dynamics, Dynamics::Boolean { .. });
        let mut active = 0u64;
        for e in &self.edges {
            let d = u64::from(e.delay);
            let bit = if tick < d { 0 } else {
                self.history[((tick - d) % self.slots as u64) as usize * self.n + e.source]
            };
            active = active.checked_add(u64::from(bit)).ok_or("counter overflow")?;
            if boolean {
                self.parity[e.target] ^= bit;
                if self.seen[e.target] == 0 { self.first[e.target] = bit; }
                if self.seen[e.target] == 1 { self.pair[e.target] = self.first[e.target] & bit; }
                self.seen[e.target] = (self.seen[e.target] + 1).min(2);
            } else if bit != 0 {
                self.accumulator[e.target] = self.accumulator[e.target].checked_add(i64::from(e.weight)).ok_or("signed accumulation overflow")?;
            }
        }
        let mut clipped = 0u64;
        let mut blocked = 0u64;
        for (i, &input) in drive.iter().enumerate() {
            match self.dynamics {
                Dynamics::Boolean { nonlinear } => {
                    self.next_spikes[i] = (input as u8) ^ self.parity[i] ^ if nonlinear { self.pair[i] } else { 0 };
                }
                Dynamics::Integer { divisor, threshold, cap, refractory } => {
                    if self.refractory[i] > 0 {
                        self.next_refractory[i] = self.refractory[i] - 1; blocked += 1; continue;
                    }
                    // Rust signed integer division truncates toward zero; specified in DYN-REF1.
                    let raw = (self.integer[i] / i64::from(divisor))
                        .checked_add(self.accumulator[i]).and_then(|v| v.checked_add(i64::from(input)))
                        .ok_or("signed integration overflow")?;
                    let potential = raw.clamp(-cap, cap);
                    if potential != raw { clipped += 1; }
                    if potential >= threshold {
                        self.next_spikes[i] = 1; self.next_refractory[i] = refractory;
                    } else { self.next_integer[i] = potential; }
                }
                Dynamics::Lif { leak, threshold, refractory } => {
                    if self.refractory[i] > 0 {
                        self.next_refractory[i] = self.refractory[i] - 1; blocked += 1; continue;
                    }
                    let potential = (leak * self.voltage[i] + self.accumulator[i] as f64) + f64::from(input);
                    if !potential.is_finite() { return Err("nonfinite LIF state"); }
                    if potential >= threshold {
                        self.next_spikes[i] = 1; self.next_refractory[i] = refractory;
                    } else { self.next_voltage[i] = potential; }
                }
            }
        }
        next.active_edge_events = next.active_edge_events.checked_add(active).ok_or("counter overflow")?;
        next.spikes = next.spikes.checked_add(self.next_spikes.iter().map(|&x| u64::from(x)).sum()).ok_or("counter overflow")?;
        next.clipped_nodes = next.clipped_nodes.checked_add(clipped).ok_or("counter overflow")?;
        next.refractory_nodes = next.refractory_nodes.checked_add(blocked).ok_or("counter overflow")?;
        // No fallible operations after this point. Commit all node states together.
        let slot = (tick % self.slots as u64) as usize * self.n;
        self.history[slot..slot + self.n].copy_from_slice(&self.next_spikes);
        std::mem::swap(&mut self.spikes, &mut self.next_spikes);
        std::mem::swap(&mut self.integer, &mut self.next_integer);
        std::mem::swap(&mut self.voltage, &mut self.next_voltage);
        std::mem::swap(&mut self.refractory, &mut self.next_refractory);
        self.counters = next;
        Ok(&self.spikes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lim() -> Limits { Limits { max_ticks: 512, max_edge_visits: 1_000_000, max_payload_bytes: 1_000_000 } }
    fn modes() -> [Dynamics; 3] {
        [Dynamics::Lif { leak: 0.5, threshold: 1.0, refractory: 0 },
         Dynamics::Integer { divisor: 2, threshold: 1, cap: 100, refractory: 0 },
         Dynamics::Boolean { nonlinear: true }]
    }
    fn edge(s: usize, t: usize, d: u16) -> Edge { Edge { source:s, target:t, delay:d, weight:1 } }
    #[test]
    fn zero_drive_is_quiescent_in_all_models() {
        for m in modes() { let mut n = Network::new(2, &[edge(0,1,1),edge(1,0,2)], m, lim()).unwrap();
            for _ in 0..100 { assert_eq!(n.step(&[0,0]).unwrap(), &[0,0]); }
            assert_eq!(n.counters().active_edge_events, 0);
        }
    }
    #[test]
    fn propagation_has_exact_positive_delay() {
        for m in modes() { for delay in [1,2,7,64] {
            let mut n = Network::new(2, &[edge(0,1,delay)], m, lim()).unwrap();
            assert_eq!(n.step(&[1,0]).unwrap(), &[1,0]);
            for t in 1..=usize::from(delay)+2 { assert_eq!(n.step(&[0,0]).unwrap(), &[0,u8::from(t==usize::from(delay))]); }
        } }
    }
    #[test]
    fn no_within_tick_cascade() {
        for m in modes() { let mut n=Network::new(3,&[edge(0,1,1),edge(1,2,1)],m,lim()).unwrap();
            assert_eq!(n.step(&[1,0,0]).unwrap(), &[1,0,0]);
            assert_eq!(n.step(&[0,0,0]).unwrap(), &[0,1,0]);
            assert_eq!(n.step(&[0,0,0]).unwrap(), &[0,0,1]);
        }
    }
    #[test]
    fn recurrent_cycle_retains_a_pulse_without_external_history() {
        for m in modes() { let mut n=Network::new(2,&[edge(0,1,1),edge(1,0,1)],m,lim()).unwrap();
            for t in 0..128 { let d=if t==0 { [1,0] } else { [0,0] };
                assert_eq!(n.step(&d).unwrap(), if t%2==0 { &[1,0] } else { &[0,1] }); }
        }
    }
    #[test]
    fn reset_removes_pending_delayed_activity_and_retains_parameters() {
        for m in modes() { let e=[edge(0,1,8)];let mut n=Network::new(2,&e,m,lim()).unwrap();
            n.step(&[1,0]).unwrap(); n.reset_activity(); assert_eq!(n.parameters(),m); assert_eq!(n.edges(),&e);
            assert_eq!(n.counters(),Counters::default());
            for _ in 0..16 { assert_eq!(n.step(&[0,0]).unwrap(),&[0,0]); }
        }
    }
    #[test]
    fn lif_matches_discrete_geometric_response() {
        let mut n=Network::new(1,&[],Dynamics::Lif{leak:0.5,threshold:10.0,refractory:0},lim()).unwrap();
        for t in 1..=32 { n.step(&[1]).unwrap();assert_eq!(n.voltage_state()[0],2.0*(1.0-0.5_f64.powi(t))); }
    }
    #[test]
    fn integer_signed_leak_truncates_toward_zero() {
        let mut n=Network::new(1,&[],Dynamics::Integer{divisor:2,threshold:8,cap:20,refractory:0},lim()).unwrap();
        n.step(&[-3]).unwrap(); n.step(&[0]).unwrap(); assert_eq!(n.integer_state(),&[-1]);
        n.step(&[0]).unwrap();assert_eq!(n.integer_state(),&[0]);
    }
    #[test]
    fn integer_clips_only_after_signed_accumulation() {
        let edges=[Edge{weight:100,..edge(0,2,1)},Edge{weight:-99,..edge(1,2,1)}];
        let mut n=Network::new(3,&edges,Dynamics::Integer{divisor:1,threshold:2,cap:10,refractory:0},lim()).unwrap();
        n.step(&[2,2,0]).unwrap(); n.step(&[0,0,0]).unwrap();
        assert_eq!(n.integer_state()[2],1);assert_eq!(n.counters().clipped_nodes,0);
    }
    #[test]
    fn saturation_is_counted() {
        let mut n=Network::new(1,&[],Dynamics::Integer{divisor:1,threshold:3,cap:5,refractory:0},lim()).unwrap();
        n.step(&[-100]).unwrap();assert_eq!(n.integer_state(),&[-5]);
        n.step(&[100]).unwrap();assert_eq!(n.spikes(),&[1]);assert_eq!(n.integer_state(),&[0]);
        assert_eq!(n.counters().clipped_nodes,2);
    }
    #[test]
    fn refractory_blocks_exactly_k_following_updates() {
        for mode in [Dynamics::Integer{divisor:1,threshold:1,cap:5,refractory:2},Dynamics::Lif{leak:1.0,threshold:1.0,refractory:2}] {
            let mut n=Network::new(1,&[],mode,lim()).unwrap();
            for t in 0..9 { assert_eq!(n.step(&[1]).unwrap(),&[u8::from(t%3==0)]); }
            assert_eq!(n.counters().refractory_nodes,6);
        }
    }
    #[test]
    fn nonlinear_boolean_is_not_affine_parity() {
        for nonlinear in [false,true] { for a in 0..=1 {for b in 0..=1 {
            let mut n=Network::new(3,&[edge(0,2,1),edge(1,2,1)],Dynamics::Boolean{nonlinear},lim()).unwrap();
            n.step(&[a,b,0]).unwrap();n.step(&[0,0,0]).unwrap();
            assert_eq!(n.spikes()[2],((a^b)^if nonlinear {a&b}else{0}) as u8);
        } } }
    }
    #[test]
    fn edge_order_does_not_change_results() {
        let e=[edge(0,2,1),edge(1,2,2),edge(2,0,1)];let mut rev=e;rev.reverse();
        for m in modes(){let mut a=Network::new(3,&e,m,lim()).unwrap();let mut b=Network::new(3,&rev,m,lim()).unwrap();
            for t in 0..50{let d=[i32::from(t%3==0),i32::from(t%5==0),0];assert_eq!(a.step(&d),b.step(&d));assert_eq!(a.counters(),b.counters());}}
    }
    #[test]
    fn invalid_input_does_not_advance_state() {
        let mut n=Network::new(1,&[],Dynamics::Boolean{nonlinear:true},lim()).unwrap();n.step(&[1]).unwrap();
        let before=n.counters();assert!(n.step(&[2]).is_err());assert!(n.step(&[]).is_err());
        assert_eq!(n.counters(),before);assert_eq!(n.spikes(),&[1]);
    }
    #[test]
    fn work_budget_fails_before_commit() {
        let l=Limits{max_ticks:1,max_edge_visits:1,..lim()};
        let mut n=Network::new(1,&[edge(0,0,1)],modes()[0],l).unwrap();n.step(&[1]).unwrap();
        let before=n.counters();assert!(n.step(&[0]).is_err());assert_eq!(n.counters(),before);assert_eq!(n.spikes(),&[1]);
    }
    #[test]
    fn rejects_invalid_topology_and_parameters() {
        for e in [[edge(0,1,0)],[edge(0,1,65)],[edge(2,1,1)]] {assert!(Network::new(2,&e,modes()[0],lim()).is_err());}
        assert!(Network::new(2,&[edge(0,1,1),edge(0,1,2)],modes()[0],lim()).is_err());
        assert!(Network::new(0,&[],modes()[0],lim()).is_err());
        assert!(Network::new(1,&[],Dynamics::Lif{leak:f64::NAN,threshold:1.0,refractory:0},lim()).is_err());
        assert!(Network::new(1,&[],Dynamics::Integer{divisor:0,threshold:1,cap:2,refractory:0},lim()).is_err());
        assert!(Network::new(2,&[Edge{weight:-1,..edge(0,1,1)}],modes()[2],lim()).is_err());
    }
    #[test]
    fn direct_payload_bound_matches_reserved_lengths() {
        let e=[edge(0,1,3)];let n=Network::new(2,&e,modes()[2],lim()).unwrap();
        assert_eq!(n.payload_bytes(),size_of::<Edge>()+2*(50+4));
        assert!(Network::new(2,&e,modes()[2],Limits{max_payload_bytes:n.payload_bytes()-1,..lim()}).is_err());
    }
}
