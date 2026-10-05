//! Deterministic reachability, centrality and rich-club descriptors for VG-3C.
//!
//! These helpers operate on unit-adjacency digraphs — either induced subsets of
//! the qualified V888 executable graph or VG-3B matched-control edge sets. They
//! compare adult structural descriptors across reference and control arms.
//!
//! Contact multiplicity is never treated as conductance. The slice does not
//! claim developmental causality, topology advantage, inferred modularity, or
//! morphology residual joins (those remain VG-3D / later VG-3 work).

use core::fmt;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::directed::{DirectedEdge, DirectedGraph, DirectedGraphError, DirectedGraphOptions};
use crate::v888_growth_controls::{V888GrowthControlArm, V888GrowthUnitEdges};

/// Degree axis used by the directed rich-club curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V888GrowthRichClubDegree {
    /// Threshold on total degree `in_degree + out_degree`.
    Total,
    /// Threshold on out-degree only.
    Out,
}

impl V888GrowthRichClubDegree {
    /// Stable machine name used in exporters.
    pub const fn name(self) -> &'static str {
        match self
        {
            Self::Total => "total",
            Self::Out => "out",
        }
    }
}

/// Exact directed reachability profile for an N-node unit digraph.
///
/// Out-reach counts include the source itself. In-reach counts are obtained by
/// BFS on the transpose (incoming adjacency) and also include the source.
/// Floating summaries are avoided: mean out-reach is retained as the exact
/// rational `(reachable_ordered_pairs / node_count)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthReachabilityProfile {
    /// Dense node count.
    pub node_count: usize,
    /// Per-node out-reachability size, including self.
    pub out_reach: Vec<usize>,
    /// Per-node in-reachability size, including self.
    pub in_reach: Vec<usize>,
    /// Sum of out-reach counts (= number of ordered pairs `u ⇝ v`, including
    /// `u = v`).
    pub reachable_ordered_pairs: u64,
    /// Denominator for mean out-reach (`node_count` as `u64`).
    pub mean_out_reach_denom: u64,
    /// Median of the sorted `out_reach` vector (lower median for even length).
    pub median_out_reach: usize,
    /// Ordered-pair universe `node_count * node_count`.
    pub ordered_pair_universe: u64,
}

/// Exact directed BFS distance profile for one source (or sink via transpose).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthDistanceProfile {
    /// Source (or sink) node.
    pub start: usize,
    /// Finite BFS distances; unreachable nodes are `None`.
    pub distance: Vec<Option<u32>>,
}

/// Directed harmonic closeness from exact BFS distances.
///
/// For each node `u`, `out_harmonic[u]` is the unnormalized sum
/// `Σ_{v ≠ u, d(u,v) < ∞} 1/d(u,v)`. `in_harmonic` uses distances on the
/// transpose. Values are IEEE-754 `f64` sums of exact reciprocal integers and
/// are therefore deterministic for a fixed graph. Self is excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthHarmonicCloseness {
    /// Dense node count.
    pub node_count: usize,
    /// Out-harmonic closeness (forward digraph).
    pub out_harmonic: Vec<f64>,
    /// In-harmonic closeness (transpose digraph).
    pub in_harmonic: Vec<f64>,
}

/// Exact directed Brandes betweenness on a unit digraph.
///
/// `raw[v]` accumulates dependency contributions over all sources. When
/// `node_count > 2`, `normalized[v] = raw[v] / ((n-1)(n-2))` so a star-like
/// directed hub can approach 1. The pilot exporter is expected to run this on
/// VG-3B-scale subsets (for example ≤512 nodes), not the full V888 graph.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthBetweenness {
    /// Dense node count.
    pub node_count: usize,
    /// Unnormalized Brandes scores.
    pub raw: Vec<f64>,
    /// Scores divided by `(n-1)(n-2)` when `n > 2`, else equal to `raw`.
    pub normalized: Vec<f64>,
}

/// One point on the directed rich-club curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V888GrowthRichClubPoint {
    /// Degree threshold `k` (nodes with degree ≥ `k`).
    pub degree_threshold: usize,
    /// Number of nodes in the club.
    pub club_nodes: usize,
    /// Directed unit edges with both endpoints inside the club.
    pub club_edges: usize,
    /// Maximum possible directed non-loop edges among club nodes (`n*(n-1)`).
    pub max_edges: usize,
}

impl V888GrowthRichClubPoint {
    /// Deterministic rich-club coefficient `phi = club_edges / max_edges`.
    ///
    /// Returns `0.0` when `max_edges == 0`. The division uses `f64` from exact
    /// integer counts and is therefore deterministic for a fixed point.
    #[must_use]
    pub fn phi(&self) -> f64 {
        if self.max_edges == 0
        {
            0.0
        }
        else
        {
            self.club_edges as f64 / self.max_edges as f64
        }
    }
}

/// Complete rich-club curve over sorted unique degree thresholds present in the
/// graph (plus threshold 0 when the graph is non-empty).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthRichClubCurve {
    /// Degree axis used for thresholds.
    pub degree_kind: V888GrowthRichClubDegree,
    /// Points ordered by increasing `degree_threshold`.
    pub points: Vec<V888GrowthRichClubPoint>,
}

/// Compact deterministic metric bundle for one unit-edge arm.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthMetricBundle {
    /// Arm label (`"reference"` or a control arm name).
    pub arm: String,
    /// Reachability profile.
    pub reachability: V888GrowthReachabilityProfile,
    /// Harmonic closeness.
    pub harmonic: V888GrowthHarmonicCloseness,
    /// Directed betweenness.
    pub betweenness: V888GrowthBetweenness,
    /// Rich-club curve on total degree.
    pub rich_club_total: V888GrowthRichClubCurve,
    /// Rich-club curve on out-degree.
    pub rich_club_out: V888GrowthRichClubCurve,
}

/// Exact integer deltas between a reference bundle and one control arm.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthMetricDelta {
    /// Control arm label.
    pub arm: String,
    /// `control.reachable_ordered_pairs - reference.reachable_ordered_pairs`.
    pub reachable_ordered_pairs_delta: i64,
    /// `control.median_out_reach - reference.median_out_reach`.
    pub median_out_reach_delta: i64,
    /// Sum of absolute per-node out-reach differences.
    pub out_reach_l1: u64,
    /// Sum of absolute per-node in-reach differences.
    pub in_reach_l1: u64,
    /// `round(1e9 * (mean_control_out_harmonic - mean_reference_out_harmonic))`
    /// using deterministic half-away-from-zero rounding of the f64 difference.
    pub mean_out_harmonic_nano_delta: i64,
    /// Same rounding for mean in-harmonic closeness.
    pub mean_in_harmonic_nano_delta: i64,
    /// Same rounding for mean normalized betweenness.
    pub mean_betweenness_nano_delta: i64,
    /// Max absolute betweenness (normalized) nano-delta across nodes.
    pub max_abs_betweenness_nano_delta: i64,
    /// Rich-club `phi` nano-deltas keyed by total-degree threshold present in
    /// either curve (missing points treated as `phi = 0`).
    pub rich_club_total_phi_nano_delta: BTreeMap<usize, i64>,
}

/// Fail-closed errors from VG-3C metric computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V888GrowthMetricsError {
    /// Directed graph construction or access failed.
    Graph(DirectedGraphError),
    /// Node count is zero.
    EmptyNodeSet,
    /// An edge endpoint is out of range for the declared node count.
    EdgeIndexOutOfBounds {
        /// Offending source.
        source: usize,
        /// Offending target.
        target: usize,
        /// Declared node count.
        node_count: usize,
    },
    /// A self-loop appeared in a unit-edge set (forbidden for VG-3C).
    SelfLoop {
        /// Node carrying the loop.
        node: usize,
    },
    /// Integer overflow while accumulating reachability or rich-club counts.
    AccountingOverflow {
        /// Counter that overflowed.
        field: &'static str,
    },
    /// Arm comparison received mismatched node counts.
    NodeCountMismatch {
        /// Expected node count from the reference.
        expected: usize,
        /// Observed node count on a control arm.
        observed: usize,
    },
}

impl From<DirectedGraphError> for V888GrowthMetricsError {
    fn from(value: DirectedGraphError) -> Self {
        Self::Graph(value)
    }
}

impl fmt::Display for V888GrowthMetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Graph(error) => write!(formatter, "{error}"),
            Self::EmptyNodeSet => formatter.write_str("metric digraph requires at least one node"),
            Self::EdgeIndexOutOfBounds {
                source,
                target,
                node_count,
            } => write!(
                formatter,
                "unit edge {source} -> {target} is out of bounds for {node_count} nodes"
            ),
            Self::SelfLoop { node } => write!(formatter, "self loop at node {node} is forbidden"),
            Self::AccountingOverflow { field } =>
            {
                write!(formatter, "metric accounting overflowed in `{field}`")
            },
            Self::NodeCountMismatch { expected, observed } => write!(
                formatter,
                "control node count {observed} does not match reference {expected}"
            ),
        }
    }
}

impl std::error::Error for V888GrowthMetricsError {}

/// Build a temporary unit digraph from a VG-3B-style edge set.
///
/// Payload is `()`; parallel edges are rejected by [`DirectedGraph::from_edges`].
pub fn v888_growth_unit_digraph(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
) -> Result<DirectedGraph<()>, V888GrowthMetricsError> {
    if node_count == 0
    {
        return Err(V888GrowthMetricsError::EmptyNodeSet);
    }
    let mut directed_edges = Vec::with_capacity(edges.len());
    for &(source, target) in edges
    {
        if source >= node_count || target >= node_count
        {
            return Err(V888GrowthMetricsError::EdgeIndexOutOfBounds {
                source,
                target,
                node_count,
            });
        }
        if source == target
        {
            return Err(V888GrowthMetricsError::SelfLoop { node: source });
        }
        directed_edges.push(DirectedEdge::new(source, target, ()));
    }
    Ok(DirectedGraph::from_edges(
        node_count,
        directed_edges,
        DirectedGraphOptions {
            allow_self_loops: false,
        },
    )?)
}

/// Exact out/in reachability profile, reusing [`DirectedGraph::reachable_from`].
pub fn v888_growth_reachability_profile(
    graph: &DirectedGraph<()>,
) -> Result<V888GrowthReachabilityProfile, V888GrowthMetricsError> {
    let node_count = graph.node_count();
    let mut out_reach = Vec::with_capacity(node_count);
    let mut reachable_ordered_pairs = 0_u64;
    for node in 0..node_count
    {
        let reached = graph.reachable_from(node)?;
        let count = reached.len();
        reachable_ordered_pairs = reachable_ordered_pairs.checked_add(count as u64).ok_or(
            V888GrowthMetricsError::AccountingOverflow {
                field: "reachable_ordered_pairs",
            },
        )?;
        out_reach.push(count);
    }

    let transpose = transpose_unit_graph(graph)?;
    let mut in_reach = Vec::with_capacity(node_count);
    for node in 0..node_count
    {
        in_reach.push(transpose.reachable_from(node)?.len());
    }

    let ordered_pair_universe = (node_count as u64).checked_mul(node_count as u64).ok_or(
        V888GrowthMetricsError::AccountingOverflow {
            field: "ordered_pair_universe",
        },
    )?;
    let median_out_reach = median_usize(&out_reach);

    Ok(V888GrowthReachabilityProfile {
        node_count,
        out_reach,
        in_reach,
        reachable_ordered_pairs,
        mean_out_reach_denom: node_count as u64,
        median_out_reach,
        ordered_pair_universe,
    })
}

/// Reachability profile from a unit-edge set.
pub fn v888_growth_reachability_from_edges(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
) -> Result<V888GrowthReachabilityProfile, V888GrowthMetricsError> {
    let graph = v888_growth_unit_digraph(node_count, edges)?;
    v888_growth_reachability_profile(&graph)
}

/// Exact BFS distances from `start` (unreachable = `None`).
pub fn v888_growth_bfs_distances(
    graph: &DirectedGraph<()>,
    start: usize,
) -> Result<V888GrowthDistanceProfile, V888GrowthMetricsError> {
    graph.check_reachable_node(start)?;
    let mut distance = vec![None; graph.node_count()];
    let mut queue = VecDeque::new();
    distance[start] = Some(0);
    queue.push_back(start);
    while let Some(node) = queue.pop_front()
    {
        let base: u32 = distance[node].expect("enqueued nodes carry a finite distance");
        for edge in graph.outgoing(node)?
        {
            if distance[edge.target].is_none()
            {
                let next =
                    base.checked_add(1)
                        .ok_or(V888GrowthMetricsError::AccountingOverflow {
                            field: "bfs_distance",
                        })?;
                distance[edge.target] = Some(next);
                queue.push_back(edge.target);
            }
        }
    }
    Ok(V888GrowthDistanceProfile { start, distance })
}

/// Directed harmonic closeness from exact BFS distances (self excluded).
pub fn v888_growth_harmonic_closeness(
    graph: &DirectedGraph<()>,
) -> Result<V888GrowthHarmonicCloseness, V888GrowthMetricsError> {
    let node_count = graph.node_count();
    let mut out_harmonic = vec![0.0_f64; node_count];
    for (start, slot) in out_harmonic.iter_mut().enumerate()
    {
        let profile = v888_growth_bfs_distances(graph, start)?;
        *slot = harmonic_from_distances(&profile.distance, start);
    }
    let transpose = transpose_unit_graph(graph)?;
    let mut in_harmonic = vec![0.0_f64; node_count];
    for (start, slot) in in_harmonic.iter_mut().enumerate()
    {
        let profile = v888_growth_bfs_distances(&transpose, start)?;
        *slot = harmonic_from_distances(&profile.distance, start);
    }
    Ok(V888GrowthHarmonicCloseness {
        node_count,
        out_harmonic,
        in_harmonic,
    })
}

/// Exact directed Brandes betweenness on the unit digraph.
pub fn v888_growth_directed_betweenness(
    graph: &DirectedGraph<()>,
) -> Result<V888GrowthBetweenness, V888GrowthMetricsError> {
    let node_count = graph.node_count();
    let mut raw = vec![0.0_f64; node_count];

    for source in 0..node_count
    {
        let mut distance = vec![-1_i32; node_count];
        let mut sigma = vec![0.0_f64; node_count];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        let mut order = Vec::with_capacity(node_count);
        let mut queue = VecDeque::new();

        distance[source] = 0;
        sigma[source] = 1.0;
        queue.push_back(source);

        while let Some(v) = queue.pop_front()
        {
            order.push(v);
            for edge in graph.outgoing(v)?
            {
                let w = edge.target;
                if distance[w] < 0
                {
                    distance[w] = distance[v] + 1;
                    queue.push_back(w);
                }
                if distance[w] == distance[v] + 1
                {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0_f64; node_count];
        for &w in order.iter().rev()
        {
            for &v in &predecessors[w]
            {
                if sigma[w] == 0.0
                {
                    continue;
                }
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != source
            {
                raw[w] += delta[w];
            }
        }
    }

    let norm = if node_count > 2
    {
        ((node_count - 1) * (node_count - 2)) as f64
    }
    else
    {
        1.0
    };
    let normalized = raw.iter().map(|&value| value / norm).collect();
    Ok(V888GrowthBetweenness {
        node_count,
        raw,
        normalized,
    })
}

/// Directed rich-club curve over unique degree thresholds present in the graph.
///
/// Thresholds are the sorted unique values of the chosen degree axis that appear
/// at least once, plus `0` when the node set is non-empty. For each threshold
/// `k`, club membership is `{v : degree(v) ≥ k}`.
pub fn v888_growth_rich_club_curve(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    degree_kind: V888GrowthRichClubDegree,
) -> Result<V888GrowthRichClubCurve, V888GrowthMetricsError> {
    if node_count == 0
    {
        return Err(V888GrowthMetricsError::EmptyNodeSet);
    }
    for &(source, target) in edges
    {
        if source >= node_count || target >= node_count
        {
            return Err(V888GrowthMetricsError::EdgeIndexOutOfBounds {
                source,
                target,
                node_count,
            });
        }
        if source == target
        {
            return Err(V888GrowthMetricsError::SelfLoop { node: source });
        }
    }

    let mut in_degree = vec![0_usize; node_count];
    let mut out_degree = vec![0_usize; node_count];
    for &(source, target) in edges
    {
        out_degree[source] = out_degree[source].checked_add(1).ok_or(
            V888GrowthMetricsError::AccountingOverflow {
                field: "out_degree",
            },
        )?;
        in_degree[target] = in_degree[target]
            .checked_add(1)
            .ok_or(V888GrowthMetricsError::AccountingOverflow { field: "in_degree" })?;
    }

    let degrees: Vec<usize> = (0..node_count)
        .map(|node| match degree_kind
        {
            V888GrowthRichClubDegree::Total => in_degree[node].checked_add(out_degree[node]).ok_or(
                V888GrowthMetricsError::AccountingOverflow {
                    field: "total_degree",
                },
            ),
            V888GrowthRichClubDegree::Out => Ok(out_degree[node]),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut thresholds: BTreeSet<usize> = degrees.iter().copied().collect();
    thresholds.insert(0);

    let mut points = Vec::with_capacity(thresholds.len());
    for &k in &thresholds
    {
        let mut club = Vec::new();
        for (node, &degree) in degrees.iter().enumerate()
        {
            if degree >= k
            {
                club.push(node);
            }
        }
        let club_nodes = club.len();
        let club_set: BTreeSet<usize> = club.into_iter().collect();
        let mut club_edges = 0_usize;
        for &(source, target) in edges
        {
            if club_set.contains(&source) && club_set.contains(&target)
            {
                club_edges = club_edges.checked_add(1).ok_or(
                    V888GrowthMetricsError::AccountingOverflow {
                        field: "rich_club_edges",
                    },
                )?;
            }
        }
        let max_edges = if club_nodes < 2
        {
            0
        }
        else
        {
            club_nodes.checked_mul(club_nodes - 1).ok_or(
                V888GrowthMetricsError::AccountingOverflow {
                    field: "rich_club_max_edges",
                },
            )?
        };
        points.push(V888GrowthRichClubPoint {
            degree_threshold: k,
            club_nodes,
            club_edges,
            max_edges,
        });
    }

    Ok(V888GrowthRichClubCurve {
        degree_kind,
        points,
    })
}

/// Compute the full VG-3C metric bundle for one named unit-edge arm.
pub fn v888_growth_metric_bundle(
    arm: &str,
    node_count: usize,
    edges: &V888GrowthUnitEdges,
) -> Result<V888GrowthMetricBundle, V888GrowthMetricsError> {
    let graph = v888_growth_unit_digraph(node_count, edges)?;
    Ok(V888GrowthMetricBundle {
        arm: arm.to_string(),
        reachability: v888_growth_reachability_profile(&graph)?,
        harmonic: v888_growth_harmonic_closeness(&graph)?,
        betweenness: v888_growth_directed_betweenness(&graph)?,
        rich_club_total: v888_growth_rich_club_curve(
            node_count,
            edges,
            V888GrowthRichClubDegree::Total,
        )?,
        rich_club_out: v888_growth_rich_club_curve(
            node_count,
            edges,
            V888GrowthRichClubDegree::Out,
        )?,
    })
}

/// Compare a reference unit-edge set against one or more control arms.
///
/// Control arms may be supplied as `(label, edges)` pairs. Labels should be
/// stable machine names (for example [`V888GrowthControlArm::name`]).
pub fn v888_growth_compare_metric_arms(
    node_count: usize,
    reference: &V888GrowthUnitEdges,
    controls: &[(&str, &V888GrowthUnitEdges)],
) -> Result<(V888GrowthMetricBundle, Vec<V888GrowthMetricDelta>), V888GrowthMetricsError> {
    let reference_bundle = v888_growth_metric_bundle("reference", node_count, reference)?;
    let mut deltas = Vec::with_capacity(controls.len());
    for &(label, edges) in controls
    {
        let control = v888_growth_metric_bundle(label, node_count, edges)?;
        if control.reachability.node_count != reference_bundle.reachability.node_count
        {
            return Err(V888GrowthMetricsError::NodeCountMismatch {
                expected: reference_bundle.reachability.node_count,
                observed: control.reachability.node_count,
            });
        }
        deltas.push(metric_delta(&reference_bundle, &control)?);
    }
    Ok((reference_bundle, deltas))
}

/// Convenience wrapper that compares the reference against a VG-3B control bundle.
pub fn v888_growth_compare_control_bundle(
    node_count: usize,
    reference: &V888GrowthUnitEdges,
    arms: &[(V888GrowthControlArm, &V888GrowthUnitEdges)],
) -> Result<(V888GrowthMetricBundle, Vec<V888GrowthMetricDelta>), V888GrowthMetricsError> {
    let labeled: Vec<(&str, &V888GrowthUnitEdges)> = arms
        .iter()
        .map(|(arm, edges)| (arm.name(), *edges))
        .collect();
    v888_growth_compare_metric_arms(node_count, reference, &labeled)
}

/// Round an `f64` to nano-units with half-away-from-zero deterministic rounding.
#[must_use]
pub fn v888_growth_f64_to_nano(value: f64) -> i64 {
    if !value.is_finite()
    {
        return 0;
    }
    let scaled = value * 1_000_000_000.0;
    if scaled >= 0.0
    {
        (scaled + 0.5).floor() as i64
    }
    else
    {
        (scaled - 0.5).ceil() as i64
    }
}

fn metric_delta(
    reference: &V888GrowthMetricBundle,
    control: &V888GrowthMetricBundle,
) -> Result<V888GrowthMetricDelta, V888GrowthMetricsError> {
    let reachable_ordered_pairs_delta = i64::try_from(control.reachability.reachable_ordered_pairs)
        .ok()
        .and_then(|control_pairs| {
            i64::try_from(reference.reachability.reachable_ordered_pairs)
                .ok()
                .map(|reference_pairs| control_pairs - reference_pairs)
        })
        .ok_or(V888GrowthMetricsError::AccountingOverflow {
            field: "reachable_ordered_pairs_delta",
        })?;
    let median_out_reach_delta = i64::try_from(control.reachability.median_out_reach)
        .ok()
        .and_then(|control_median| {
            i64::try_from(reference.reachability.median_out_reach)
                .ok()
                .map(|reference_median| control_median - reference_median)
        })
        .ok_or(V888GrowthMetricsError::AccountingOverflow {
            field: "median_out_reach_delta",
        })?;

    let out_reach_l1 = l1_diff_usize(
        &reference.reachability.out_reach,
        &control.reachability.out_reach,
    )?;
    let in_reach_l1 = l1_diff_usize(
        &reference.reachability.in_reach,
        &control.reachability.in_reach,
    )?;

    let mean_out_harmonic_nano_delta = v888_growth_f64_to_nano(
        mean_f64(&control.harmonic.out_harmonic) - mean_f64(&reference.harmonic.out_harmonic),
    );
    let mean_in_harmonic_nano_delta = v888_growth_f64_to_nano(
        mean_f64(&control.harmonic.in_harmonic) - mean_f64(&reference.harmonic.in_harmonic),
    );
    let mean_betweenness_nano_delta = v888_growth_f64_to_nano(
        mean_f64(&control.betweenness.normalized) - mean_f64(&reference.betweenness.normalized),
    );

    let mut max_abs_betweenness_nano_delta = 0_i64;
    for (left, right) in reference
        .betweenness
        .normalized
        .iter()
        .zip(control.betweenness.normalized.iter())
    {
        let delta = v888_growth_f64_to_nano(right - left).abs();
        max_abs_betweenness_nano_delta = max_abs_betweenness_nano_delta.max(delta);
    }

    let mut thresholds: BTreeSet<usize> = BTreeSet::new();
    for point in reference
        .rich_club_total
        .points
        .iter()
        .chain(control.rich_club_total.points.iter())
    {
        thresholds.insert(point.degree_threshold);
    }
    let reference_phi: BTreeMap<usize, f64> = reference
        .rich_club_total
        .points
        .iter()
        .map(|point| (point.degree_threshold, point.phi()))
        .collect();
    let control_phi: BTreeMap<usize, f64> = control
        .rich_club_total
        .points
        .iter()
        .map(|point| (point.degree_threshold, point.phi()))
        .collect();
    let mut rich_club_total_phi_nano_delta = BTreeMap::new();
    for k in thresholds
    {
        let left = reference_phi.get(&k).copied().unwrap_or(0.0);
        let right = control_phi.get(&k).copied().unwrap_or(0.0);
        rich_club_total_phi_nano_delta.insert(k, v888_growth_f64_to_nano(right - left));
    }

    Ok(V888GrowthMetricDelta {
        arm: control.arm.clone(),
        reachable_ordered_pairs_delta,
        median_out_reach_delta,
        out_reach_l1,
        in_reach_l1,
        mean_out_harmonic_nano_delta,
        mean_in_harmonic_nano_delta,
        mean_betweenness_nano_delta,
        max_abs_betweenness_nano_delta,
        rich_club_total_phi_nano_delta,
    })
}

fn transpose_unit_graph(
    graph: &DirectedGraph<()>,
) -> Result<DirectedGraph<()>, V888GrowthMetricsError> {
    let edges = graph
        .edges()
        .iter()
        .map(|edge| DirectedEdge::new(edge.target, edge.source, ()))
        .collect();
    Ok(DirectedGraph::from_edges(
        graph.node_count(),
        edges,
        DirectedGraphOptions {
            allow_self_loops: false,
        },
    )?)
}

fn harmonic_from_distances(distance: &[Option<u32>], start: usize) -> f64 {
    let mut sum = 0.0_f64;
    for (node, &maybe_distance) in distance.iter().enumerate()
    {
        if node == start
        {
            continue;
        }
        if let Some(hop) = maybe_distance
        {
            if hop > 0
            {
                sum += 1.0 / f64::from(hop);
            }
        }
    }
    sum
}

fn median_usize(values: &[usize]) -> usize {
    if values.is_empty()
    {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() - 1) / 2]
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn l1_diff_usize(left: &[usize], right: &[usize]) -> Result<u64, V888GrowthMetricsError> {
    if left.len() != right.len()
    {
        return Err(V888GrowthMetricsError::NodeCountMismatch {
            expected: left.len(),
            observed: right.len(),
        });
    }
    let mut total = 0_u64;
    for (&a, &b) in left.iter().zip(right.iter())
    {
        let diff = a.abs_diff(b) as u64;
        total = total
            .checked_add(diff)
            .ok_or(V888GrowthMetricsError::AccountingOverflow { field: "l1_diff" })?;
    }
    Ok(total)
}

trait CheckReachableNode {
    fn check_reachable_node(&self, node: usize) -> Result<(), V888GrowthMetricsError>;
}

impl CheckReachableNode for DirectedGraph<()> {
    fn check_reachable_node(&self, node: usize) -> Result<(), V888GrowthMetricsError> {
        // Trigger the same bounds check as reachable_from without exposing private helpers.
        self.outgoing(node)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v888_growth_controls::v888_growth_edge_count_control;

    fn path4() -> (usize, V888GrowthUnitEdges) {
        let edges: V888GrowthUnitEdges = [(0, 1), (1, 2), (2, 3)].into_iter().collect();
        (4, edges)
    }

    fn cycle3() -> (usize, V888GrowthUnitEdges) {
        let edges: V888GrowthUnitEdges = [(0, 1), (1, 2), (2, 0)].into_iter().collect();
        (3, edges)
    }

    #[test]
    fn reachability_on_directed_path_is_exact() {
        let (n, edges) = path4();
        let profile = v888_growth_reachability_from_edges(n, &edges).unwrap();
        assert_eq!(profile.out_reach, vec![4, 3, 2, 1]);
        assert_eq!(profile.in_reach, vec![1, 2, 3, 4]);
        assert_eq!(profile.reachable_ordered_pairs, 10);
        assert_eq!(profile.ordered_pair_universe, 16);
        assert_eq!(profile.median_out_reach, 2);
        assert_eq!(profile.mean_out_reach_denom, 4);
    }

    #[test]
    fn reachability_on_cycle_is_complete() {
        let (n, edges) = cycle3();
        let profile = v888_growth_reachability_from_edges(n, &edges).unwrap();
        assert_eq!(profile.out_reach, vec![3, 3, 3]);
        assert_eq!(profile.in_reach, vec![3, 3, 3]);
        assert_eq!(profile.reachable_ordered_pairs, 9);
    }

    #[test]
    fn harmonic_closeness_on_path_matches_hand_calculation() {
        let (n, edges) = path4();
        let graph = v888_growth_unit_digraph(n, &edges).unwrap();
        let harmonic = v888_growth_harmonic_closeness(&graph).unwrap();
        // From 0: d=1,2,3 -> 1 + 1/2 + 1/3 = 11/6
        assert!((harmonic.out_harmonic[0] - 11.0 / 6.0).abs() < 1e-12);
        // From 1: d=1,2 -> 1 + 1/2 = 1.5
        assert!((harmonic.out_harmonic[1] - 1.5).abs() < 1e-12);
        assert!((harmonic.out_harmonic[2] - 1.0).abs() < 1e-12);
        assert!((harmonic.out_harmonic[3] - 0.0).abs() < 1e-12);
        // In-harmonic is the reverse path.
        assert!((harmonic.in_harmonic[3] - 11.0 / 6.0).abs() < 1e-12);
        assert!((harmonic.in_harmonic[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn directed_betweenness_on_path_is_exact() {
        let (n, edges) = path4();
        let graph = v888_growth_unit_digraph(n, &edges).unwrap();
        let betweenness = v888_growth_directed_betweenness(&graph).unwrap();
        // Directed path 0->1->2->3. Intermediaries:
        // node 1 lies on (0,2) and (0,3) -> raw 2
        // node 2 lies on (0,3) and (1,3) -> raw 2
        // endpoints 0
        assert!((betweenness.raw[0] - 0.0).abs() < 1e-12);
        assert!((betweenness.raw[1] - 2.0).abs() < 1e-12);
        assert!((betweenness.raw[2] - 2.0).abs() < 1e-12);
        assert!((betweenness.raw[3] - 0.0).abs() < 1e-12);
        // norm = (4-1)*(4-2) = 6
        assert!((betweenness.normalized[1] - 2.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn rich_club_on_complete_digraph_is_one() {
        let mut edges = V888GrowthUnitEdges::new();
        for u in 0..4
        {
            for v in 0..4
            {
                if u != v
                {
                    edges.insert((u, v));
                }
            }
        }
        let curve =
            v888_growth_rich_club_curve(4, &edges, V888GrowthRichClubDegree::Total).unwrap();
        // Every node has total degree 6; thresholds include 0 and 6.
        let top = curve
            .points
            .iter()
            .find(|point| point.degree_threshold == 6)
            .unwrap();
        assert_eq!(top.club_nodes, 4);
        assert_eq!(top.club_edges, 12);
        assert_eq!(top.max_edges, 12);
        assert!((top.phi() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rich_club_rejects_bad_indexes() {
        let edges: V888GrowthUnitEdges = [(0, 5)].into_iter().collect();
        assert!(matches!(
            v888_growth_rich_club_curve(3, &edges, V888GrowthRichClubDegree::Out),
            Err(V888GrowthMetricsError::EdgeIndexOutOfBounds { .. })
        ));
        let loop_edges: V888GrowthUnitEdges = [(1, 1)].into_iter().collect();
        assert!(matches!(
            v888_growth_unit_digraph(2, &loop_edges),
            Err(V888GrowthMetricsError::SelfLoop { node: 1 })
        ));
    }

    #[test]
    fn arm_comparison_is_deterministic_and_zero_on_identical_arms() {
        let reference = v888_growth_edge_count_control(12, 30, 7).unwrap();
        let (bundle, deltas) =
            v888_growth_compare_metric_arms(12, &reference, &[("clone", &reference)]).unwrap();
        assert_eq!(bundle.arm, "reference");
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].reachable_ordered_pairs_delta, 0);
        assert_eq!(deltas[0].out_reach_l1, 0);
        assert_eq!(deltas[0].in_reach_l1, 0);
        assert_eq!(deltas[0].mean_out_harmonic_nano_delta, 0);
        assert_eq!(deltas[0].mean_betweenness_nano_delta, 0);
        assert!(
            deltas[0]
                .rich_club_total_phi_nano_delta
                .values()
                .all(|&delta| delta == 0)
        );
        let again =
            v888_growth_compare_metric_arms(12, &reference, &[("clone", &reference)]).unwrap();
        assert_eq!(bundle.reachability, again.0.reachability);
        assert_eq!(deltas[0].out_reach_l1, again.1[0].out_reach_l1);
    }

    #[test]
    fn empty_graph_fails_closed() {
        let edges = V888GrowthUnitEdges::new();
        assert!(matches!(
            v888_growth_unit_digraph(0, &edges),
            Err(V888GrowthMetricsError::EmptyNodeSet)
        ));
    }

    #[test]
    fn nano_rounding_is_half_away_from_zero() {
        assert_eq!(v888_growth_f64_to_nano(1.5e-9), 2);
        assert_eq!(v888_growth_f64_to_nano(-1.5e-9), -2);
        assert_eq!(v888_growth_f64_to_nano(0.0), 0);
    }
}
