//! Deterministic matched topology controls for V888-GROWTH VG-3B.
//!
//! These generators promote the BOOL-0.2a control ladder into a reusable
//! SciRust graph contract for growth-proxy work. Every arm uses unit adjacency.
//! Measured contact multiplicities remain outside the control graphs and are
//! not treated as conductance, event rates, or developmental weights.
//!
//! The slice does not claim topology advantage, mixing-time uniformity, or any
//! causal brain-growth mechanism.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::v888_executable::BancV888ExecutableGraph;

/// Directed unit edge `(source, target)` in dense canonical subset indices.
pub type V888GrowthUnitEdge = (usize, usize);

/// Unit-adjacency edge set used by VG-3B controls.
pub type V888GrowthUnitEdges = BTreeSet<V888GrowthUnitEdge>;

/// Subset selection policy for VG-3B pilot graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V888GrowthSubsetPolicy {
    /// Rank root IDs by SplitMix of `root_id ^ seed`, then take the first `count`.
    Ranked,
    /// Weak-neighbor BFS seeded by the same ranking, refill from unused ranks.
    WeakBfs,
}

/// Declared matching strength for double-edge switch proposals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V888GrowthControlMatch {
    /// Preserve each node's exact in-degree and out-degree.
    Degree,
    /// Also preserve each node's reciprocal-neighbour count.
    Reciprocal,
    /// Also preserve the directed anatomical-block pair edge-count matrix.
    Block,
}

/// Named control arm in the VG-3B ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V888GrowthControlArm {
    /// Exact edge-count matched random sparse digraph (Floyd sampling).
    EdgeCount,
    /// Degree-preserving rewiring of the unit reference.
    Degree,
    /// Degree- and reciprocity-preserving rewiring.
    DegreeReciprocal,
    /// Degree-, reciprocity-, and block-matrix-preserving rewiring.
    DegreeReciprocalBlock,
}

impl V888GrowthControlArm {
    /// Stable machine name used in exporters and Thor summaries.
    pub const fn name(self) -> &'static str {
        match self
        {
            Self::EdgeCount => "edge_count",
            Self::Degree => "degree",
            Self::DegreeReciprocal => "degree_reciprocal",
            Self::DegreeReciprocalBlock => "degree_reciprocal_block",
        }
    }
}

/// Exact degree / reciprocity / block statistics for a unit digraph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthControlStats {
    /// Incoming unit-edge counts per dense node.
    pub incoming: Vec<usize>,
    /// Outgoing unit-edge counts per dense node.
    pub outgoing: Vec<usize>,
    /// Reciprocal-neighbour counts per dense node.
    pub reciprocal: Vec<usize>,
    /// Directed block-pair edge counts keyed by `(source_block, target_block)`.
    pub blocks: BTreeMap<(u32, u32), usize>,
}

/// One generated control arm with invariant audit counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthControlArmResult {
    /// Ladder arm identity.
    pub arm: V888GrowthControlArm,
    /// Exact per-arm seed used for sampling or rewiring.
    pub seed: u64,
    /// Generated unit-adjacency edges.
    pub edges: V888GrowthUnitEdges,
    /// Accepted double-edge switches (zero for the edge-count arm).
    pub accepted_swaps: usize,
    /// Proposed switch attempts (zero for the edge-count arm).
    pub attempts: usize,
    /// Number of reference edges absent from the generated arm.
    pub replaced_edges: usize,
    /// Nodes whose in/out degree disagrees with the reference.
    pub degree_mismatch_nodes: usize,
    /// Nodes whose reciprocal count disagrees with the reference.
    pub reciprocal_mismatch_nodes: usize,
    /// Directed block-pair cells that disagree with the reference.
    pub block_mismatch_cells: usize,
}

/// Complete four-arm VG-3B control bundle for one unit reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthControlBundle {
    /// Dense node count of the subset / reference.
    pub node_count: usize,
    /// Reference unit-adjacency induced from the qualified graph.
    pub reference_edges: V888GrowthUnitEdges,
    /// Number of distinct block labels present in the supplied groups.
    pub distinct_block_labels: usize,
    /// Whether the block constraint is vacuous (single label).
    pub block_constraint_vacuous: bool,
    /// Generated arms in ladder order.
    pub arms: Vec<V888GrowthControlArmResult>,
}

/// Fail-closed errors from VG-3B control generation.
#[derive(Debug)]
pub enum V888GrowthControlError {
    /// Subset size is outside the declared pilot bounds.
    SubsetSizeOutOfBounds {
        /// Requested subset size.
        count: usize,
        /// Available source nodes.
        available: usize,
    },
    /// Unknown or unsupported selection policy string.
    UnknownSubsetPolicy,
    /// Selected dense indices are empty, duplicated, or out of range.
    InvalidSelection,
    /// Block-group vector length disagrees with the node count.
    BlockGroupLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Observed length.
        observed: usize,
    },
    /// Requested edge count cannot exist on a simple digraph without loops.
    ImpossibleEdgeCount {
        /// Node count.
        nodes: usize,
        /// Requested edges.
        edges: usize,
    },
    /// Integer overflow while counting the non-self pair universe.
    PairUniverseOverflow,
    /// A generated arm violated its declared matching invariant.
    InvariantViolation {
        /// Arm that failed verification.
        arm: &'static str,
    },
}

impl fmt::Display for V888GrowthControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::SubsetSizeOutOfBounds { count, available } => write!(
                formatter,
                "subset size {count} is outside bounds for {available} available nodes"
            ),
            Self::UnknownSubsetPolicy => formatter.write_str("unknown subset selection policy"),
            Self::InvalidSelection => formatter.write_str("invalid dense-node selection"),
            Self::BlockGroupLengthMismatch { expected, observed } => write!(
                formatter,
                "block-group length {observed} does not match node count {expected}"
            ),
            Self::ImpossibleEdgeCount { nodes, edges } => write!(
                formatter,
                "edge count {edges} is impossible for {nodes} nodes without loops"
            ),
            Self::PairUniverseOverflow => formatter.write_str("non-self pair universe overflowed"),
            Self::InvariantViolation { arm } =>
            {
                write!(
                    formatter,
                    "generated control arm `{arm}` violated its declared invariant"
                )
            },
        }
    }
}

impl std::error::Error for V888GrowthControlError {}

#[derive(Clone, Copy)]
struct ControlRng(u64);

impl ControlRng {
    fn next(&mut self) -> u64 {
        let value = splitmix64(self.0);
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        value
    }

    fn below(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bound = upper as u64;
        let limit = u64::MAX - u64::MAX % bound;
        loop
        {
            let sample = self.next();
            if sample < limit
            {
                return (sample % bound) as usize;
            }
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Deterministic SplitMix64 mix used by VG-3B subset ranking and RNG streams.
pub fn v888_growth_control_mix(value: u64) -> u64 {
    splitmix64(value)
}

/// Select a deterministic dense-index subset from a qualified executable graph.
///
/// `# Errors`
///
/// Returns an error when `count` is outside `[3, min(4096, node_count)]` or the
/// policy is unsupported.
pub fn v888_growth_select_subset(
    graph: &BancV888ExecutableGraph,
    count: usize,
    seed: u64,
    policy: V888GrowthSubsetPolicy,
) -> Result<(Vec<usize>, usize), V888GrowthControlError> {
    let available = graph.node_count();
    if count < 3 || count > available || count > 4096
    {
        return Err(V888GrowthControlError::SubsetSizeOutOfBounds { count, available });
    }

    let ids = graph.node_ids();
    let mut ranked: Vec<usize> = (0..available).collect();
    ranked.sort_unstable_by_key(|&index| (splitmix64(ids[index].get() ^ seed), ids[index].get()));

    match policy
    {
        V888GrowthSubsetPolicy::Ranked =>
        {
            ranked.truncate(count);
            ranked.sort_unstable();
            Ok((ranked, 0))
        },
        V888GrowthSubsetPolicy::WeakBfs =>
        {
            let directed = graph.graph();
            let mut reverse = vec![Vec::new(); available];
            for edge in directed.edges()
            {
                reverse[edge.target].push(edge.source);
            }

            let mut seen = vec![false; available];
            let mut queue = VecDeque::new();
            let mut selected = Vec::with_capacity(count);
            let mut anchor = 0_usize;
            let mut starts = 0_usize;

            while selected.len() < count
            {
                if queue.is_empty()
                {
                    while seen[ranked[anchor]]
                    {
                        anchor += 1;
                    }
                    let root = ranked[anchor];
                    seen[root] = true;
                    queue.push_back(root);
                    starts += 1;
                }
                let node = queue.pop_front().expect("non-empty BFS queue");
                selected.push(node);
                if selected.len() == count
                {
                    break;
                }

                let mut neighbors: Vec<usize> = directed
                    .outgoing(node)
                    .expect("valid dense node")
                    .iter()
                    .map(|edge| edge.target)
                    .chain(reverse[node].iter().copied())
                    .collect();
                neighbors.sort_unstable_by_key(|&index| {
                    (splitmix64(ids[index].get() ^ seed), ids[index].get())
                });
                neighbors.dedup();
                for neighbor in neighbors
                {
                    if !seen[neighbor]
                    {
                        seen[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }

            selected.sort_unstable();
            Ok((selected, starts))
        },
    }
}

/// Induce a unit-adjacency reference over selected dense nodes.
///
/// Contact multiplicities are discarded: every retained directed pair becomes a
/// single unit edge. Self-loops are impossible in the qualified V8CSR001 graph.
///
/// `# Errors`
///
/// Returns an error when the selection is empty, contains duplicates, or points
/// outside the executable graph.
pub fn v888_growth_unit_reference(
    graph: &BancV888ExecutableGraph,
    selected_nodes: &[usize],
) -> Result<V888GrowthUnitEdges, V888GrowthControlError> {
    if selected_nodes.is_empty()
    {
        return Err(V888GrowthControlError::InvalidSelection);
    }
    let node_count = graph.node_count();
    let mut mapping = vec![usize::MAX; node_count];
    for (local, &global) in selected_nodes.iter().enumerate()
    {
        if global >= node_count || mapping[global] != usize::MAX
        {
            return Err(V888GrowthControlError::InvalidSelection);
        }
        mapping[global] = local;
    }

    let mut edges = V888GrowthUnitEdges::new();
    for edge in graph.graph().edges()
    {
        let source = mapping[edge.source];
        let target = mapping[edge.target];
        if source != usize::MAX && target != usize::MAX
        {
            edges.insert((source, target));
        }
    }
    Ok(edges)
}

/// Exact Floyd sample of `edge_count` distinct non-self directed pairs.
///
/// `# Errors`
///
/// Returns an error when the request is impossible on a simple loop-free digraph
/// or the pair universe overflows.
pub fn v888_growth_edge_count_control(
    node_count: usize,
    edge_count: usize,
    seed: u64,
) -> Result<V888GrowthUnitEdges, V888GrowthControlError> {
    let universe = node_count
        .checked_mul(node_count.saturating_sub(1))
        .ok_or(V888GrowthControlError::PairUniverseOverflow)?;
    if node_count < 2 || edge_count > universe
    {
        return Err(V888GrowthControlError::ImpossibleEdgeCount {
            nodes: node_count,
            edges: edge_count,
        });
    }

    let mut rng = ControlRng(seed);
    let mut indices = BTreeSet::new();
    for j in (universe - edge_count)..universe
    {
        let trial = rng.below(j + 1);
        let chosen = if indices.contains(&trial) { j } else { trial };
        indices.insert(chosen);
    }

    Ok(indices
        .into_iter()
        .map(|index| {
            let source = index / (node_count - 1);
            let remainder = index % (node_count - 1);
            let target = if remainder >= source
            {
                remainder + 1
            }
            else
            {
                remainder
            };
            (source, target)
        })
        .collect())
}

fn switch_allowed(
    edges: &V888GrowthUnitEdges,
    left: V888GrowthUnitEdge,
    right: V888GrowthUnitEdge,
    groups: &[u32],
    mode: V888GrowthControlMatch,
) -> bool {
    let (a, b) = left;
    let (c, d) = right;
    if a == b || c == d || a == c || a == d || b == c || b == d
    {
        return false;
    }
    if edges.contains(&(a, d)) || edges.contains(&(c, b))
    {
        return false;
    }
    if matches!(mode, V888GrowthControlMatch::Block)
        && groups[a] != groups[c]
        && groups[b] != groups[d]
    {
        return false;
    }
    if !matches!(mode, V888GrowthControlMatch::Degree)
    {
        let ab = edges.contains(&(b, a));
        let cd = edges.contains(&(d, c));
        let ad = edges.contains(&(d, a));
        let cb = edges.contains(&(b, c));
        if [ab, ab, cd, cd] != [ad, cb, cb, ad]
        {
            return false;
        }
    }
    true
}

/// Bounded directed double-edge switch rewiring under a declared match mode.
pub fn v888_growth_rewire_control(
    reference: &V888GrowthUnitEdges,
    groups: &[u32],
    seed: u64,
    mode: V888GrowthControlMatch,
    attempts: usize,
) -> (V888GrowthUnitEdges, usize) {
    let mut edges = reference.clone();
    let mut order: Vec<V888GrowthUnitEdge> = reference.iter().copied().collect();
    let mut rng = ControlRng(seed);
    let mut accepted = 0_usize;
    if order.len() < 2
    {
        return (edges, accepted);
    }
    for _ in 0..attempts
    {
        let i = rng.below(order.len());
        let j = rng.below(order.len());
        let left = order[i];
        let right = order[j];
        if !switch_allowed(&edges, left, right, groups, mode)
        {
            continue;
        }
        let new_left = (left.0, right.1);
        let new_right = (right.0, left.1);
        edges.remove(&left);
        edges.remove(&right);
        edges.insert(new_left);
        edges.insert(new_right);
        order[i] = new_left;
        order[j] = new_right;
        accepted += 1;
    }
    (edges, accepted)
}

/// Compute exact degree, reciprocity and block-matrix statistics.
pub fn v888_growth_control_stats(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    groups: &[u32],
) -> Result<V888GrowthControlStats, V888GrowthControlError> {
    if groups.len() != node_count
    {
        return Err(V888GrowthControlError::BlockGroupLengthMismatch {
            expected: node_count,
            observed: groups.len(),
        });
    }
    let mut stats = V888GrowthControlStats {
        incoming: vec![0; node_count],
        outgoing: vec![0; node_count],
        reciprocal: vec![0; node_count],
        blocks: BTreeMap::new(),
    };
    for &(source, target) in edges
    {
        if source >= node_count || target >= node_count || source == target
        {
            return Err(V888GrowthControlError::InvalidSelection);
        }
        stats.outgoing[source] += 1;
        stats.incoming[target] += 1;
        stats.reciprocal[source] += usize::from(edges.contains(&(target, source)));
        *stats
            .blocks
            .entry((groups[source], groups[target]))
            .or_insert(0) += 1;
    }
    Ok(stats)
}

/// Generate the four VG-3B matched control arms for one unit reference.
///
/// Arms are fail-closed: any generated graph that disagrees with its declared
/// invariants is rejected. A single distinct block label makes the block
/// constraint vacuous; that fact is exposed on the returned bundle.
///
/// `# Errors`
///
/// Returns an error when block groups are the wrong length or a generated arm
/// fails invariant verification.
///
/// `# Examples`
///
/// ~~~rust
/// use std::collections::BTreeSet;
/// use scirust_graph::v888_growth::{
///     v888_growth_edge_count_control, v888_growth_matched_controls,
/// };
///
/// let reference = v888_growth_edge_count_control(12, 40, 7)?;
/// let groups = vec![0_u32, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
/// let bundle = v888_growth_matched_controls(&reference, 12, &groups, 11)?;
/// assert_eq!(bundle.arms.len(), 4);
/// assert_eq!(bundle.arms[0].edges.len(), reference.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ~~~
pub fn v888_growth_matched_controls(
    reference: &V888GrowthUnitEdges,
    node_count: usize,
    groups: &[u32],
    seed: u64,
) -> Result<V888GrowthControlBundle, V888GrowthControlError> {
    if groups.len() != node_count
    {
        return Err(V888GrowthControlError::BlockGroupLengthMismatch {
            expected: node_count,
            observed: groups.len(),
        });
    }

    let expected = v888_growth_control_stats(node_count, reference, groups)?;
    let distinct_block_labels = groups.iter().copied().collect::<BTreeSet<_>>().len();
    let attempts = reference.len().saturating_mul(32).min(1_000_000);
    let ladder = [
        V888GrowthControlArm::EdgeCount,
        V888GrowthControlArm::Degree,
        V888GrowthControlArm::DegreeReciprocal,
        V888GrowthControlArm::DegreeReciprocalBlock,
    ];

    let mut arms = Vec::with_capacity(ladder.len());
    for (index, arm) in ladder.into_iter().enumerate()
    {
        let arm_seed = seed ^ (0x6a09e667f3bcc909_u64.wrapping_mul(index as u64 + 1));
        let (edges, accepted_swaps, arm_attempts) = match arm
        {
            V888GrowthControlArm::EdgeCount => (
                v888_growth_edge_count_control(node_count, reference.len(), arm_seed)?,
                0_usize,
                0_usize,
            ),
            V888GrowthControlArm::Degree =>
            {
                let (edges, accepted) = v888_growth_rewire_control(
                    reference,
                    groups,
                    arm_seed,
                    V888GrowthControlMatch::Degree,
                    attempts,
                );
                (edges, accepted, attempts)
            },
            V888GrowthControlArm::DegreeReciprocal =>
            {
                let (edges, accepted) = v888_growth_rewire_control(
                    reference,
                    groups,
                    arm_seed,
                    V888GrowthControlMatch::Reciprocal,
                    attempts,
                );
                (edges, accepted, attempts)
            },
            V888GrowthControlArm::DegreeReciprocalBlock =>
            {
                let (edges, accepted) = v888_growth_rewire_control(
                    reference,
                    groups,
                    arm_seed,
                    V888GrowthControlMatch::Block,
                    attempts,
                );
                (edges, accepted, attempts)
            },
        };

        let observed = v888_growth_control_stats(node_count, &edges, groups)?;
        let degree_mismatch_nodes = (0..node_count)
            .filter(|&node| {
                expected.incoming[node] != observed.incoming[node]
                    || expected.outgoing[node] != observed.outgoing[node]
            })
            .count();
        let reciprocal_mismatch_nodes = (0..node_count)
            .filter(|&node| expected.reciprocal[node] != observed.reciprocal[node])
            .count();
        let cells: BTreeSet<_> = expected
            .blocks
            .keys()
            .chain(observed.blocks.keys())
            .copied()
            .collect();
        let block_mismatch_cells = cells
            .iter()
            .filter(|key| {
                expected.blocks.get(key).unwrap_or(&0) != observed.blocks.get(key).unwrap_or(&0)
            })
            .count();

        let require_degree = !matches!(arm, V888GrowthControlArm::EdgeCount);
        let require_reciprocal = matches!(
            arm,
            V888GrowthControlArm::DegreeReciprocal | V888GrowthControlArm::DegreeReciprocalBlock
        );
        let require_block = matches!(arm, V888GrowthControlArm::DegreeReciprocalBlock);
        if edges.len() != reference.len()
            || (require_degree && degree_mismatch_nodes != 0)
            || (require_reciprocal && reciprocal_mismatch_nodes != 0)
            || (require_block && block_mismatch_cells != 0)
        {
            return Err(V888GrowthControlError::InvariantViolation { arm: arm.name() });
        }

        arms.push(V888GrowthControlArmResult {
            arm,
            seed: arm_seed,
            replaced_edges: reference.difference(&edges).count(),
            edges,
            accepted_swaps,
            attempts: arm_attempts,
            degree_mismatch_nodes,
            reciprocal_mismatch_nodes,
            block_mismatch_cells,
        });
    }

    Ok(V888GrowthControlBundle {
        node_count,
        reference_edges: reference.clone(),
        distinct_block_labels,
        block_constraint_vacuous: distinct_block_labels <= 1,
        arms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v888_executable::{BANC_V888_CSR_MAGIC, BancV888ExecutableGraph};

    fn fixture_graph() -> BancV888ExecutableGraph {
        let mut bytes = BANC_V888_CSR_MAGIC.to_vec();
        bytes.extend_from_slice(&6_u64.to_le_bytes());
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        for id in [10_u64, 20, 30, 40, 50, 60]
        {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        for offset in [0_u64, 2, 3, 5, 6, 7, 7]
        {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        // Edges: 0->1, 0->2, 1->0, 2->0, 2->3, 3->2, 4->5 (no loops).
        for target in [1_u32, 2, 0, 0, 3, 2, 5]
        {
            bytes.extend_from_slice(&target.to_le_bytes());
        }
        for contacts in [2_u64, 3, 5, 7, 11, 13, 17]
        {
            bytes.extend_from_slice(&contacts.to_le_bytes());
        }
        BancV888ExecutableGraph::from_v8csr001(&bytes).unwrap()
    }

    #[test]
    fn edge_count_control_is_exact_and_deterministic() {
        for nodes in 3..12
        {
            for edges in [0, 1, nodes, nodes * (nodes - 1)]
            {
                let first = v888_growth_edge_count_control(nodes, edges, 11).unwrap();
                let second = v888_growth_edge_count_control(nodes, edges, 11).unwrap();
                assert_eq!(first.len(), edges);
                assert_eq!(first, second);
                assert!(first.iter().all(|&(u, v)| u < nodes && v < nodes && u != v));
            }
        }
        assert!(v888_growth_edge_count_control(4, 13, 0).is_err());
    }

    #[test]
    fn rewiring_preserves_declared_invariants() {
        for seed in 0..24
        {
            let reference = v888_growth_edge_count_control(18, 70, seed).unwrap();
            let groups: Vec<u32> = (0..18).map(|index| (index / 6) as u32).collect();
            let before = v888_growth_control_stats(18, &reference, &groups).unwrap();
            for mode in [
                V888GrowthControlMatch::Degree,
                V888GrowthControlMatch::Reciprocal,
                V888GrowthControlMatch::Block,
            ]
            {
                let (result, _) = v888_growth_rewire_control(&reference, &groups, seed, mode, 4000);
                let after = v888_growth_control_stats(18, &result, &groups).unwrap();
                assert_eq!(reference.len(), result.len());
                assert_eq!(before.incoming, after.incoming);
                assert_eq!(before.outgoing, after.outgoing);
                if !matches!(mode, V888GrowthControlMatch::Degree)
                {
                    assert_eq!(before.reciprocal, after.reciprocal);
                }
                if matches!(mode, V888GrowthControlMatch::Block)
                {
                    assert_eq!(before.blocks, after.blocks);
                }
                assert_eq!(
                    result,
                    v888_growth_rewire_control(&reference, &groups, seed, mode, 4000).0
                );
            }
        }
    }

    #[test]
    fn reciprocal_constraint_rejects_degree_only_swap() {
        let edges: V888GrowthUnitEdges = [(0, 1), (1, 0), (2, 3)].into_iter().collect();
        assert!(switch_allowed(
            &edges,
            (0, 1),
            (2, 3),
            &[0; 4],
            V888GrowthControlMatch::Degree
        ));
        assert!(!switch_allowed(
            &edges,
            (0, 1),
            (2, 3),
            &[0; 4],
            V888GrowthControlMatch::Reciprocal
        ));
    }

    #[test]
    fn block_constraint_rejects_cross_block_change() {
        let edges: V888GrowthUnitEdges = [(0, 1), (2, 3)].into_iter().collect();
        assert!(switch_allowed(
            &edges,
            (0, 1),
            (2, 3),
            &[0, 0, 1, 1],
            V888GrowthControlMatch::Reciprocal
        ));
        assert!(!switch_allowed(
            &edges,
            (0, 1),
            (2, 3),
            &[0, 0, 1, 1],
            V888GrowthControlMatch::Block
        ));
    }

    #[test]
    fn matched_controls_bundle_is_fail_closed_and_deterministic() {
        let reference = v888_growth_edge_count_control(16, 48, 3).unwrap();
        let groups: Vec<u32> = (0..16).map(|index| (index % 4) as u32).collect();
        let first = v888_growth_matched_controls(&reference, 16, &groups, 19).unwrap();
        let second = v888_growth_matched_controls(&reference, 16, &groups, 19).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.arms.len(), 4);
        assert!(!first.block_constraint_vacuous);
        assert_eq!(first.arms[0].arm.name(), "edge_count");
        assert_eq!(first.arms[3].arm.name(), "degree_reciprocal_block");
        for arm in &first.arms
        {
            assert_eq!(arm.edges.len(), reference.len());
            if arm.arm != V888GrowthControlArm::EdgeCount
            {
                assert_eq!(arm.degree_mismatch_nodes, 0);
            }
            if matches!(
                arm.arm,
                V888GrowthControlArm::DegreeReciprocal
                    | V888GrowthControlArm::DegreeReciprocalBlock
            )
            {
                assert_eq!(arm.reciprocal_mismatch_nodes, 0);
            }
            if arm.arm == V888GrowthControlArm::DegreeReciprocalBlock
            {
                assert_eq!(arm.block_mismatch_cells, 0);
            }
        }
    }

    #[test]
    fn unit_reference_discards_multiplicity_and_maps_dense_indices() {
        let graph = fixture_graph();
        let selected = vec![0, 1, 2, 3];
        let edges = v888_growth_unit_reference(&graph, &selected).unwrap();
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(0, 2)));
        assert!(edges.contains(&(1, 0)));
        assert!(edges.contains(&(2, 0)));
        assert!(edges.contains(&(2, 3)));
        assert!(edges.contains(&(3, 2)));
        assert_eq!(edges.len(), 6);
    }

    #[test]
    fn subset_selection_is_deterministic() {
        let graph = fixture_graph();
        for policy in [
            V888GrowthSubsetPolicy::Ranked,
            V888GrowthSubsetPolicy::WeakBfs,
        ]
        {
            assert_eq!(
                v888_growth_select_subset(&graph, 4, 7, policy).unwrap(),
                v888_growth_select_subset(&graph, 4, 7, policy).unwrap()
            );
        }
        assert!(v888_growth_select_subset(&graph, 7, 0, V888GrowthSubsetPolicy::Ranked).is_err());
    }

    #[test]
    fn vacuous_block_constraint_is_exposed() {
        let reference = v888_growth_edge_count_control(10, 20, 5).unwrap();
        let groups = vec![0_u32; 10];
        let bundle = v888_growth_matched_controls(&reference, 10, &groups, 9).unwrap();
        assert!(bundle.block_constraint_vacuous);
        assert_eq!(bundle.distinct_block_labels, 1);
    }
}
