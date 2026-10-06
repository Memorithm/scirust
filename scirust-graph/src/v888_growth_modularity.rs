//! Deterministic inferred modularity, label mixing and partition agreement for VG-3D.
//!
//! VG-3D adds the modular / mixing descriptors that VG-3C deferred. Every
//! helper operates on the same unit-adjacency digraphs as VG-3B / VG-3C (an
//! induced subset of the qualified V888 executable graph or a matched-control
//! arm) and keeps the key quantities as exact integers:
//!
//! - [`v888_growth_directed_modularity`] — Leicht–Newman directed modularity of
//!   a caller-supplied partition, retained as the exact rational
//!   `(m * intra - Σ_c out_c * in_c) / m²`;
//! - [`v888_growth_infer_communities`] — deterministic directed Louvain
//!   (local moving + aggregation) whose move decisions use exact integer gains,
//!   so the inferred partition is bit-for-bit reproducible;
//! - [`v888_growth_label_mixing`] — exact source-label × target-label edge
//!   counts and the nominal (Newman) assortativity coefficient for a
//!   caller-supplied annotation such as hemilineage or anatomical block;
//! - [`v888_growth_partition_agreement`] — exact pair counts and the adjusted
//!   Rand index between two partitions (for example inferred modules versus a
//!   hemilineage annotation);
//! - [`v888_growth_compare_modularity_arms`] — per-arm summaries and nano-scaled
//!   integer deltas against the reference arm.
//!
//! Inferred communities are a descriptive optimisation result, not a biological
//! module claim. Contact multiplicity is never treated as conductance. Nothing
//! here establishes developmental causality, topology advantage, a uniform null
//! ensemble or a mixing-time guarantee for the VG-3B rewiring chains.

use core::fmt;
use std::collections::BTreeMap;

use crate::v888_growth_controls::V888GrowthUnitEdges;
use crate::v888_growth_metrics::v888_growth_f64_to_nano;

/// Exact directed modularity of one partition of a unit digraph.
///
/// With `m` unit edges, intra-community edge count `intra` and per-community
/// out/in degree totals, Leicht–Newman directed modularity is
///
/// `Q = intra / m - Σ_c out_c * in_c / m²`
///
/// and is retained exactly as `q_numerator / q_denominator` with
/// `q_numerator = m * intra - Σ_c out_c * in_c` and `q_denominator = m²`.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthModularity {
    /// Dense node count.
    pub node_count: usize,
    /// Number of unit edges `m`.
    pub edge_count: u64,
    /// Number of distinct labels in the partition.
    pub community_count: usize,
    /// Edges whose endpoints share a label.
    pub intra_edges: u64,
    /// `Σ_c out_c * in_c` over communities.
    pub expected_product_sum: u128,
    /// Exact numerator `m * intra - Σ_c out_c * in_c`.
    pub q_numerator: i128,
    /// Exact denominator `m²` (zero for an edgeless graph).
    pub q_denominator: u128,
    /// `q_numerator / q_denominator`, or `None` for an edgeless graph.
    pub q: Option<f64>,
}

impl V888GrowthModularity {
    /// Deterministic nano-scaled modularity (`None` for an edgeless graph).
    #[must_use]
    pub fn q_nano(&self) -> Option<i64> {
        self.q.map(v888_growth_f64_to_nano)
    }
}

/// Bounds for the deterministic directed Louvain search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V888GrowthLouvainOptions {
    /// Maximum aggregation levels (each level = local moving + aggregation).
    pub max_levels: usize,
    /// Maximum full node sweeps per level.
    pub max_passes_per_level: usize,
}

impl Default for V888GrowthLouvainOptions {
    fn default() -> Self {
        Self {
            max_levels: 32,
            max_passes_per_level: 128,
        }
    }
}

/// Inferred community partition from [`v888_growth_infer_communities`].
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthCommunityPartition {
    /// Canonical labels: communities are numbered by their smallest member.
    pub labels: Vec<usize>,
    /// Community sizes indexed by canonical label.
    pub community_sizes: Vec<usize>,
    /// Aggregation levels that produced at least one move.
    pub levels: usize,
    /// Total accepted single-node moves across all levels.
    pub moves: u64,
    /// Whether the search stopped on a level or pass bound rather than at a
    /// local optimum.
    pub hit_bound: bool,
    /// Exact modularity of the final partition on the original unit edges.
    pub modularity: V888GrowthModularity,
}

/// Exact label mixing table for a caller-supplied annotation.
///
/// Labels are opaque caller annotations (hemilineage, anatomical block or an
/// inferred community id). Nominal assortativity follows Newman (2003):
/// `r = (Σ_a e_aa - Σ_a a_a b_a) / (1 - Σ_a a_a b_a)`, retained exactly as
/// `(m * same - S) / (m² - S)` where `S = Σ_a source_a * target_a`.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthLabelMixing {
    /// Number of unit edges `m`.
    pub edge_count: u64,
    /// Exact `(source_label, target_label) -> edge count` table.
    pub counts: BTreeMap<(usize, usize), u64>,
    /// Edges leaving each label.
    pub source_totals: BTreeMap<usize, u64>,
    /// Edges entering each label.
    pub target_totals: BTreeMap<usize, u64>,
    /// Edges whose endpoints share a label.
    pub same_label_edges: u64,
    /// Exact numerator `m * same - S`.
    pub assortativity_numerator: i128,
    /// Exact denominator `m² - S`.
    pub assortativity_denominator: i128,
    /// Nominal assortativity, or `None` when the denominator is zero
    /// (edgeless graph, or every edge confined to one label).
    pub assortativity: Option<f64>,
}

/// Exact pair-count agreement between two partitions of the same nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthPartitionAgreement {
    /// Dense node count.
    pub node_count: usize,
    /// Unordered node pairs `n * (n - 1) / 2`.
    pub pair_universe: u128,
    /// Pairs grouped together in both partitions.
    pub same_both: u128,
    /// Pairs grouped together in the left partition.
    pub same_left: u128,
    /// Pairs grouped together in the right partition.
    pub same_right: u128,
    /// Exact `2 * (pairs * same_both - same_left * same_right)`.
    pub ari_numerator: i128,
    /// Exact `pairs * (same_left + same_right) - 2 * same_left * same_right`.
    pub ari_denominator: i128,
    /// Adjusted Rand index, or `None` when the denominator is zero (both
    /// partitions trivial in the same way).
    pub adjusted_rand: Option<f64>,
}

/// Per-arm VG-3D modular / mixing summary.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthModularityArm {
    /// Stable arm name (`reference` or a VG-3B control name).
    pub arm: String,
    /// Inferred partition on this arm.
    pub inferred: V888GrowthCommunityPartition,
    /// Modularity of the caller annotation evaluated on this arm, if supplied.
    pub annotation_modularity: Option<V888GrowthModularity>,
    /// Label mixing of the caller annotation on this arm, if supplied.
    pub annotation_mixing: Option<V888GrowthLabelMixing>,
    /// Agreement between the inferred partition and the caller annotation.
    pub inferred_vs_annotation: Option<V888GrowthPartitionAgreement>,
    /// Agreement between this arm's inferred partition and the reference's.
    pub inferred_vs_reference: V888GrowthPartitionAgreement,
}

/// Nano-scaled integer deltas `control - reference` for one control arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthModularityDelta {
    /// Control arm name.
    pub arm: String,
    /// `control.community_count - reference.community_count`.
    pub community_count_delta: i64,
    /// Inferred modularity delta in nano-units.
    pub inferred_q_nano_delta: Option<i64>,
    /// Annotation modularity delta in nano-units.
    pub annotation_q_nano_delta: Option<i64>,
    /// Annotation assortativity delta in nano-units.
    pub annotation_assortativity_nano_delta: Option<i64>,
    /// Adjusted Rand index of the control's inferred partition against the
    /// reference's, in nano-units.
    pub inferred_vs_reference_ari_nano: Option<i64>,
}

/// Fail-closed errors from VG-3D computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V888GrowthModularityError {
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
    /// A self-loop appeared in a unit-edge set.
    SelfLoop {
        /// Node carrying the loop.
        node: usize,
    },
    /// A label vector does not cover exactly the declared nodes.
    LabelLengthMismatch {
        /// Declared node count.
        expected: usize,
        /// Supplied label count.
        observed: usize,
    },
    /// Integer overflow while accumulating exact counts.
    AccountingOverflow {
        /// Counter that overflowed.
        field: &'static str,
    },
}

impl fmt::Display for V888GrowthModularityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyNodeSet =>
            {
                formatter.write_str("modularity digraph requires at least one node")
            },
            Self::EdgeIndexOutOfBounds {
                source,
                target,
                node_count,
            } => write!(
                formatter,
                "unit edge {source} -> {target} is out of bounds for {node_count} nodes"
            ),
            Self::SelfLoop { node } => write!(formatter, "self loop at node {node} is forbidden"),
            Self::LabelLengthMismatch { expected, observed } => write!(
                formatter,
                "label vector has {observed} entries but the graph has {expected} nodes"
            ),
            Self::AccountingOverflow { field } =>
            {
                write!(formatter, "modularity accounting overflowed in `{field}`")
            },
        }
    }
}

impl std::error::Error for V888GrowthModularityError {}

/// Exact Leicht–Newman directed modularity of a caller-supplied partition.
///
/// `labels[i]` is the community of node `i`; labels are opaque and need not be
/// dense.
///
/// ```
/// use scirust_graph::v888_growth::{V888GrowthUnitEdges, v888_growth_directed_modularity};
///
/// // Two directed 3-cycles joined by one bridge edge.
/// let edges: V888GrowthUnitEdges =
///     [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)].into_iter().collect();
/// let q = v888_growth_directed_modularity(6, &edges, &[0, 0, 0, 1, 1, 1]).unwrap();
/// assert_eq!(q.intra_edges, 6);
/// assert_eq!(q.q_denominator, 49);
/// assert!(q.q.unwrap() > 0.35);
/// ```
pub fn v888_growth_directed_modularity(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    labels: &[usize],
) -> Result<V888GrowthModularity, V888GrowthModularityError> {
    validate_edges(node_count, edges)?;
    validate_labels(node_count, labels)?;
    let edge_count = edges.len() as u64;
    let mut out_totals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut in_totals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut intra_edges = 0_u64;
    for &(source, target) in edges
    {
        *out_totals.entry(labels[source]).or_insert(0) += 1;
        *in_totals.entry(labels[target]).or_insert(0) += 1;
        if labels[source] == labels[target]
        {
            intra_edges += 1;
        }
    }
    let mut expected_product_sum = 0_u128;
    for (label, &out_total) in &out_totals
    {
        let in_total = in_totals.get(label).copied().unwrap_or(0);
        expected_product_sum = expected_product_sum
            .checked_add(u128::from(out_total) * u128::from(in_total))
            .ok_or(V888GrowthModularityError::AccountingOverflow {
                field: "expected_product_sum",
            })?;
    }
    let mut distinct = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let q_numerator = i128::from(edge_count) * i128::from(intra_edges)
        - i128::try_from(expected_product_sum).map_err(|_| {
            V888GrowthModularityError::AccountingOverflow {
                field: "q_numerator",
            }
        })?;
    let q_denominator = u128::from(edge_count) * u128::from(edge_count);
    let q = ratio(q_numerator, q_denominator as i128);
    Ok(V888GrowthModularity {
        node_count,
        edge_count,
        community_count: distinct.len(),
        intra_edges,
        expected_product_sum,
        q_numerator,
        q_denominator,
        q,
    })
}

/// Deterministic directed Louvain community inference on a unit digraph.
///
/// Each level sweeps nodes in ascending index order. A node is removed from its
/// community and reinserted where the exact integer gain
/// `m * (w_to(C) + w_from(C)) - (k_out * in_C + k_in * out_C)` is largest;
/// it moves only on a strict improvement over staying, and ties go to the
/// smallest community label. Every accepted move strictly increases the exact
/// modularity, so the search terminates; `options` additionally bounds the
/// work. Communities are then aggregated into weighted super-nodes and the
/// process repeats until a level makes no move.
///
/// The final labels are canonical (numbered by smallest member), so identical
/// inputs give identical partitions on every platform.
///
/// ```
/// use scirust_graph::v888_growth::{
///     V888GrowthLouvainOptions, V888GrowthUnitEdges, v888_growth_infer_communities,
/// };
///
/// let edges: V888GrowthUnitEdges =
///     [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)].into_iter().collect();
/// let partition =
///     v888_growth_infer_communities(6, &edges, V888GrowthLouvainOptions::default()).unwrap();
/// assert_eq!(partition.labels, vec![0, 0, 0, 1, 1, 1]);
/// ```
pub fn v888_growth_infer_communities(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    options: V888GrowthLouvainOptions,
) -> Result<V888GrowthCommunityPartition, V888GrowthModularityError> {
    validate_edges(node_count, edges)?;
    let edge_count = edges.len() as u64;
    let mut membership: Vec<usize> = (0..node_count).collect();
    let mut levels = 0_usize;
    let mut moves = 0_u64;
    let mut hit_bound = false;

    if edge_count > 0
    {
        let mut graph = WeightedGraph::from_unit_edges(node_count, edges);
        let mut level = 0_usize;
        loop
        {
            if level >= options.max_levels
            {
                hit_bound = true;
                break;
            }
            let outcome = local_moving(&graph, edge_count, options.max_passes_per_level);
            hit_bound |= outcome.hit_bound;
            if outcome.moves == 0
            {
                break;
            }
            levels += 1;
            moves += outcome.moves;
            let (canonical, community_count) = canonicalize(&outcome.labels);
            for slot in &mut membership
            {
                *slot = canonical[*slot];
            }
            if community_count == graph.node_count
            {
                break;
            }
            graph = graph.aggregate(&canonical, community_count);
            level += 1;
        }
    }

    let (labels, community_count) = canonicalize(&membership);
    let mut community_sizes = vec![0_usize; community_count];
    for &label in &labels
    {
        community_sizes[label] += 1;
    }
    let modularity = v888_growth_directed_modularity(node_count, edges, &labels)?;
    Ok(V888GrowthCommunityPartition {
        labels,
        community_sizes,
        levels,
        moves,
        hit_bound,
        modularity,
    })
}

/// Exact label mixing table and nominal assortativity for an annotation.
///
/// ```
/// use scirust_graph::v888_growth::{V888GrowthUnitEdges, v888_growth_label_mixing};
///
/// // Every edge stays inside its label: perfectly assortative.
/// let edges: V888GrowthUnitEdges = [(0, 1), (1, 0), (2, 3), (3, 2)].into_iter().collect();
/// let mixing = v888_growth_label_mixing(4, &edges, &[7, 7, 9, 9]).unwrap();
/// assert_eq!(mixing.same_label_edges, 4);
/// assert_eq!(mixing.assortativity, Some(1.0));
/// ```
pub fn v888_growth_label_mixing(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    labels: &[usize],
) -> Result<V888GrowthLabelMixing, V888GrowthModularityError> {
    validate_edges(node_count, edges)?;
    validate_labels(node_count, labels)?;
    let edge_count = edges.len() as u64;
    let mut counts: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut source_totals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut target_totals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut same_label_edges = 0_u64;
    for &(source, target) in edges
    {
        let (left, right) = (labels[source], labels[target]);
        *counts.entry((left, right)).or_insert(0) += 1;
        *source_totals.entry(left).or_insert(0) += 1;
        *target_totals.entry(right).or_insert(0) += 1;
        if left == right
        {
            same_label_edges += 1;
        }
    }
    let mut product_sum = 0_i128;
    for (label, &source_total) in &source_totals
    {
        let target_total = target_totals.get(label).copied().unwrap_or(0);
        product_sum = product_sum
            .checked_add(i128::from(source_total) * i128::from(target_total))
            .ok_or(V888GrowthModularityError::AccountingOverflow {
                field: "assortativity_product_sum",
            })?;
    }
    let m = i128::from(edge_count);
    let assortativity_numerator = m * i128::from(same_label_edges) - product_sum;
    let assortativity_denominator = m * m - product_sum;
    Ok(V888GrowthLabelMixing {
        edge_count,
        counts,
        source_totals,
        target_totals,
        same_label_edges,
        assortativity_numerator,
        assortativity_denominator,
        assortativity: ratio(assortativity_numerator, assortativity_denominator),
    })
}

/// Exact pair counts and adjusted Rand index between two partitions.
///
/// ```
/// use scirust_graph::v888_growth::v888_growth_partition_agreement;
///
/// // Same grouping under different label names agrees perfectly.
/// let agreement = v888_growth_partition_agreement(&[0, 0, 1, 1], &[5, 5, 2, 2]).unwrap();
/// assert_eq!(agreement.adjusted_rand, Some(1.0));
/// ```
pub fn v888_growth_partition_agreement(
    left: &[usize],
    right: &[usize],
) -> Result<V888GrowthPartitionAgreement, V888GrowthModularityError> {
    if left.is_empty()
    {
        return Err(V888GrowthModularityError::EmptyNodeSet);
    }
    if left.len() != right.len()
    {
        return Err(V888GrowthModularityError::LabelLengthMismatch {
            expected: left.len(),
            observed: right.len(),
        });
    }
    let node_count = left.len();
    let mut joint: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut left_sizes: BTreeMap<usize, u64> = BTreeMap::new();
    let mut right_sizes: BTreeMap<usize, u64> = BTreeMap::new();
    for (&a, &b) in left.iter().zip(right)
    {
        *joint.entry((a, b)).or_insert(0) += 1;
        *left_sizes.entry(a).or_insert(0) += 1;
        *right_sizes.entry(b).or_insert(0) += 1;
    }
    let pairs =
        |count: u64| -> u128 { u128::from(count) * u128::from(count.saturating_sub(1)) / 2 };
    let same_both: u128 = joint.values().map(|&c| pairs(c)).sum();
    let same_left: u128 = left_sizes.values().map(|&c| pairs(c)).sum();
    let same_right: u128 = right_sizes.values().map(|&c| pairs(c)).sum();
    let pair_universe = pairs(node_count as u64);
    let overflow = |field| V888GrowthModularityError::AccountingOverflow { field };
    let to_i = |value: u128, field| i128::try_from(value).map_err(|_| overflow(field));
    let universe = to_i(pair_universe, "pair_universe")?;
    let both = to_i(same_both, "same_both")?;
    let sl = to_i(same_left, "same_left")?;
    let sr = to_i(same_right, "same_right")?;
    let cross = sl
        .checked_mul(sr)
        .ok_or(overflow("same_left_x_same_right"))?;
    let ari_numerator = universe
        .checked_mul(both)
        .and_then(|value| value.checked_sub(cross))
        .and_then(|value| value.checked_mul(2))
        .ok_or(overflow("ari_numerator"))?;
    let ari_denominator = universe
        .checked_mul(sl + sr)
        .and_then(|value| value.checked_sub(cross.checked_mul(2)?))
        .ok_or(overflow("ari_denominator"))?;
    Ok(V888GrowthPartitionAgreement {
        node_count,
        pair_universe,
        same_both,
        same_left,
        same_right,
        ari_numerator,
        ari_denominator,
        adjusted_rand: ratio(ari_numerator, ari_denominator),
    })
}

/// Per-arm VG-3D summaries plus integer deltas against the reference.
///
/// `annotation`, when supplied, is one caller label per node (for example
/// hemilineage or anatomical block); it is evaluated unchanged on every arm.
#[allow(clippy::type_complexity)]
pub fn v888_growth_compare_modularity_arms(
    node_count: usize,
    reference: &V888GrowthUnitEdges,
    controls: &[(&str, &V888GrowthUnitEdges)],
    annotation: Option<&[usize]>,
    options: V888GrowthLouvainOptions,
) -> Result<
    (
        V888GrowthModularityArm,
        Vec<(V888GrowthModularityArm, V888GrowthModularityDelta)>,
    ),
    V888GrowthModularityError,
> {
    let reference_partition = v888_growth_infer_communities(node_count, reference, options)?;
    let reference_arm = modularity_arm(
        "reference",
        node_count,
        reference,
        annotation,
        reference_partition.clone(),
        &reference_partition.labels,
    )?;
    let mut rows = Vec::with_capacity(controls.len());
    for &(label, edges) in controls
    {
        let inferred = v888_growth_infer_communities(node_count, edges, options)?;
        let arm = modularity_arm(
            label,
            node_count,
            edges,
            annotation,
            inferred,
            &reference_partition.labels,
        )?;
        let delta = modularity_delta(&reference_arm, &arm);
        rows.push((arm, delta));
    }
    Ok((reference_arm, rows))
}

fn modularity_arm(
    arm: &str,
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    annotation: Option<&[usize]>,
    inferred: V888GrowthCommunityPartition,
    reference_labels: &[usize],
) -> Result<V888GrowthModularityArm, V888GrowthModularityError> {
    let (annotation_modularity, annotation_mixing, inferred_vs_annotation) = match annotation
    {
        Some(labels) => (
            Some(v888_growth_directed_modularity(node_count, edges, labels)?),
            Some(v888_growth_label_mixing(node_count, edges, labels)?),
            Some(v888_growth_partition_agreement(&inferred.labels, labels)?),
        ),
        None => (None, None, None),
    };
    let inferred_vs_reference =
        v888_growth_partition_agreement(&inferred.labels, reference_labels)?;
    Ok(V888GrowthModularityArm {
        arm: arm.to_string(),
        inferred,
        annotation_modularity,
        annotation_mixing,
        inferred_vs_annotation,
        inferred_vs_reference,
    })
}

fn modularity_delta(
    reference: &V888GrowthModularityArm,
    control: &V888GrowthModularityArm,
) -> V888GrowthModularityDelta {
    let diff = |left: Option<i64>, right: Option<i64>| match (left, right)
    {
        (Some(control_value), Some(reference_value)) =>
        {
            Some(control_value.saturating_sub(reference_value))
        },
        _ => None,
    };
    let annotation_q = |arm: &V888GrowthModularityArm| {
        arm.annotation_modularity
            .as_ref()
            .and_then(V888GrowthModularity::q_nano)
    };
    let annotation_r = |arm: &V888GrowthModularityArm| {
        arm.annotation_mixing
            .as_ref()
            .and_then(|mixing| mixing.assortativity)
            .map(v888_growth_f64_to_nano)
    };
    V888GrowthModularityDelta {
        arm: control.arm.clone(),
        community_count_delta: control.inferred.community_sizes.len() as i64
            - reference.inferred.community_sizes.len() as i64,
        inferred_q_nano_delta: diff(
            control.inferred.modularity.q_nano(),
            reference.inferred.modularity.q_nano(),
        ),
        annotation_q_nano_delta: diff(annotation_q(control), annotation_q(reference)),
        annotation_assortativity_nano_delta: diff(annotation_r(control), annotation_r(reference)),
        inferred_vs_reference_ari_nano: control
            .inferred_vs_reference
            .adjusted_rand
            .map(v888_growth_f64_to_nano),
    }
}

/// Weighted digraph used internally by the Louvain aggregation levels.
///
/// Self-loop weights carry intra-community edges of the previous level.
struct WeightedGraph {
    node_count: usize,
    outgoing: Vec<Vec<(usize, u64)>>,
    incoming: Vec<Vec<(usize, u64)>>,
    out_degree: Vec<u64>,
    in_degree: Vec<u64>,
}

impl WeightedGraph {
    fn from_unit_edges(node_count: usize, edges: &V888GrowthUnitEdges) -> Self {
        let weighted: BTreeMap<(usize, usize), u64> = edges.iter().map(|&edge| (edge, 1)).collect();
        Self::from_weighted(node_count, &weighted)
    }

    fn from_weighted(node_count: usize, edges: &BTreeMap<(usize, usize), u64>) -> Self {
        let mut outgoing = vec![Vec::new(); node_count];
        let mut incoming = vec![Vec::new(); node_count];
        let mut out_degree = vec![0_u64; node_count];
        let mut in_degree = vec![0_u64; node_count];
        for (&(source, target), &weight) in edges
        {
            outgoing[source].push((target, weight));
            incoming[target].push((source, weight));
            out_degree[source] += weight;
            in_degree[target] += weight;
        }
        Self {
            node_count,
            outgoing,
            incoming,
            out_degree,
            in_degree,
        }
    }

    fn aggregate(&self, labels: &[usize], community_count: usize) -> Self {
        let mut edges: BTreeMap<(usize, usize), u64> = BTreeMap::new();
        for (source, row) in self.outgoing.iter().enumerate()
        {
            for &(target, weight) in row
            {
                *edges.entry((labels[source], labels[target])).or_insert(0) += weight;
            }
        }
        Self::from_weighted(community_count, &edges)
    }
}

struct LocalMovingOutcome {
    labels: Vec<usize>,
    moves: u64,
    hit_bound: bool,
}

fn local_moving(graph: &WeightedGraph, edge_count: u64, max_passes: usize) -> LocalMovingOutcome {
    let n = graph.node_count;
    let m = i128::from(edge_count);
    let mut labels: Vec<usize> = (0..n).collect();
    let mut community_out: Vec<u64> = graph.out_degree.clone();
    let mut community_in: Vec<u64> = graph.in_degree.clone();
    let mut moves = 0_u64;
    let mut hit_bound = false;
    let mut pass = 0_usize;
    loop
    {
        if pass >= max_passes
        {
            hit_bound = true;
            break;
        }
        let mut moved_this_pass = false;
        for node in 0..n
        {
            let current = labels[node];
            let k_out = graph.out_degree[node];
            let k_in = graph.in_degree[node];
            // Edge weight between `node` and each neighbouring community,
            // excluding the node's own self-loop.
            let mut links: BTreeMap<usize, u64> = BTreeMap::new();
            for &(target, weight) in &graph.outgoing[node]
            {
                if target != node
                {
                    *links.entry(labels[target]).or_insert(0) += weight;
                }
            }
            for &(source, weight) in &graph.incoming[node]
            {
                if source != node
                {
                    *links.entry(labels[source]).or_insert(0) += weight;
                }
            }
            community_out[current] -= k_out;
            community_in[current] -= k_in;
            let gain = |community: usize, link: u64| -> i128 {
                m * i128::from(link)
                    - (i128::from(k_out) * i128::from(community_in[community])
                        + i128::from(k_in) * i128::from(community_out[community]))
            };
            let mut best = current;
            let mut best_gain = gain(current, links.get(&current).copied().unwrap_or(0));
            for (&community, &link) in &links
            {
                if community == current
                {
                    continue;
                }
                let candidate = gain(community, link);
                if candidate > best_gain
                {
                    best = community;
                    best_gain = candidate;
                }
            }
            community_out[best] += k_out;
            community_in[best] += k_in;
            if best != current
            {
                labels[node] = best;
                moves += 1;
                moved_this_pass = true;
            }
        }
        pass += 1;
        if !moved_this_pass
        {
            break;
        }
    }
    LocalMovingOutcome {
        labels,
        moves,
        hit_bound,
    }
}

/// Renumber labels by first occurrence in node order.
fn canonicalize(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut mapping: BTreeMap<usize, usize> = BTreeMap::new();
    let mut canonical = Vec::with_capacity(labels.len());
    for &label in labels
    {
        let next = mapping.len();
        canonical.push(*mapping.entry(label).or_insert(next));
    }
    (canonical, mapping.len())
}

fn ratio(numerator: i128, denominator: i128) -> Option<f64> {
    if denominator == 0
    {
        None
    }
    else
    {
        Some(numerator as f64 / denominator as f64)
    }
}

fn validate_edges(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
) -> Result<(), V888GrowthModularityError> {
    if node_count == 0
    {
        return Err(V888GrowthModularityError::EmptyNodeSet);
    }
    for &(source, target) in edges
    {
        if source >= node_count || target >= node_count
        {
            return Err(V888GrowthModularityError::EdgeIndexOutOfBounds {
                source,
                target,
                node_count,
            });
        }
        if source == target
        {
            return Err(V888GrowthModularityError::SelfLoop { node: source });
        }
    }
    Ok(())
}

fn validate_labels(node_count: usize, labels: &[usize]) -> Result<(), V888GrowthModularityError> {
    if labels.len() != node_count
    {
        return Err(V888GrowthModularityError::LabelLengthMismatch {
            expected: node_count,
            observed: labels.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> (usize, V888GrowthUnitEdges) {
        let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
            .into_iter()
            .collect();
        (6, edges)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Planted partition: dense within blocks of `block`, sparse across.
    fn planted(blocks: usize, block: usize, seed: u64) -> (usize, V888GrowthUnitEdges, Vec<usize>) {
        let n = blocks * block;
        let mut state = seed;
        let mut edges = V888GrowthUnitEdges::new();
        for source in 0..n
        {
            for target in 0..n
            {
                if source == target
                {
                    continue;
                }
                let same = source / block == target / block;
                let draw = splitmix(&mut state) % 1000;
                if (same && draw < 600) || (!same && draw < 15)
                {
                    edges.insert((source, target));
                }
            }
        }
        let labels = (0..n).map(|node| node / block).collect();
        (n, edges, labels)
    }

    #[test]
    fn modularity_matches_hand_calculation() {
        let (n, edges) = two_triangles();
        let q = v888_growth_directed_modularity(n, &edges, &[0, 0, 0, 1, 1, 1]).unwrap();
        // m = 7, intra = 6; community 0: out 4, in 3; community 1: out 3, in 4.
        assert_eq!(q.edge_count, 7);
        assert_eq!(q.intra_edges, 6);
        assert_eq!(q.expected_product_sum, 24);
        assert_eq!(q.q_numerator, 7 * 6 - 24);
        assert_eq!(q.q_denominator, 49);
        assert_eq!(q.community_count, 2);
        assert!((q.q.unwrap() - 18.0 / 49.0).abs() < 1e-15);
    }

    #[test]
    fn single_community_has_zero_modularity() {
        let (n, edges) = two_triangles();
        let q = v888_growth_directed_modularity(n, &edges, &[3; 6]).unwrap();
        assert_eq!(q.q_numerator, 0);
        assert_eq!(q.q, Some(0.0));
    }

    #[test]
    fn edgeless_graph_has_undefined_modularity() {
        let edges = V888GrowthUnitEdges::new();
        let q = v888_growth_directed_modularity(3, &edges, &[0, 1, 2]).unwrap();
        assert_eq!(q.q, None);
        let partition =
            v888_growth_infer_communities(3, &edges, V888GrowthLouvainOptions::default()).unwrap();
        assert_eq!(partition.labels, vec![0, 1, 2]);
        assert_eq!(partition.moves, 0);
    }

    #[test]
    fn louvain_recovers_two_triangles() {
        let (n, edges) = two_triangles();
        let partition =
            v888_growth_infer_communities(n, &edges, V888GrowthLouvainOptions::default()).unwrap();
        assert_eq!(partition.labels, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(partition.community_sizes, vec![3, 3]);
        assert!(!partition.hit_bound);
        assert_eq!(partition.modularity.q_numerator, 18);
    }

    #[test]
    fn louvain_recovers_planted_blocks_deterministically() {
        let (n, edges, truth) = planted(4, 16, 0x5EED);
        let options = V888GrowthLouvainOptions::default();
        let first = v888_growth_infer_communities(n, &edges, options).unwrap();
        let second = v888_growth_infer_communities(n, &edges, options).unwrap();
        assert_eq!(first, second);
        let agreement = v888_growth_partition_agreement(&first.labels, &truth).unwrap();
        assert_eq!(agreement.adjusted_rand, Some(1.0));
        let truth_q = v888_growth_directed_modularity(n, &edges, &truth).unwrap();
        assert!(first.modularity.q_numerator >= truth_q.q_numerator);
    }

    #[test]
    fn louvain_never_scores_below_singletons() {
        let (n, edges, _) = planted(3, 10, 42);
        let partition =
            v888_growth_infer_communities(n, &edges, V888GrowthLouvainOptions::default()).unwrap();
        let singletons: Vec<usize> = (0..n).collect();
        let base = v888_growth_directed_modularity(n, &edges, &singletons).unwrap();
        assert!(partition.modularity.q_numerator > base.q_numerator);
        assert_eq!(partition.community_sizes.iter().sum::<usize>(), n);
    }

    #[test]
    fn zero_pass_bound_reports_hit_bound() {
        let (n, edges) = two_triangles();
        let options = V888GrowthLouvainOptions {
            max_levels: 4,
            max_passes_per_level: 0,
        };
        let partition = v888_growth_infer_communities(n, &edges, options).unwrap();
        assert!(partition.hit_bound);
        assert_eq!(partition.labels, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn label_mixing_counts_and_assortativity_are_exact() {
        let (n, edges) = two_triangles();
        let mixing = v888_growth_label_mixing(n, &edges, &[0, 0, 0, 1, 1, 1]).unwrap();
        assert_eq!(mixing.counts.get(&(0, 0)), Some(&3));
        assert_eq!(mixing.counts.get(&(0, 1)), Some(&1));
        assert_eq!(mixing.counts.get(&(1, 1)), Some(&3));
        assert_eq!(mixing.counts.get(&(1, 0)), None);
        assert_eq!(mixing.same_label_edges, 6);
        // S = 4*3 + 3*4 = 24; r = (7*6 - 24) / (49 - 24) = 18/25.
        assert_eq!(mixing.assortativity_numerator, 18);
        assert_eq!(mixing.assortativity_denominator, 25);
        assert!((mixing.assortativity.unwrap() - 0.72).abs() < 1e-15);
    }

    #[test]
    fn bipartite_mixing_is_disassortative() {
        let edges: V888GrowthUnitEdges = [(0, 2), (2, 0), (1, 3), (3, 1)].into_iter().collect();
        let mixing = v888_growth_label_mixing(4, &edges, &[0, 0, 1, 1]).unwrap();
        assert_eq!(mixing.same_label_edges, 0);
        assert_eq!(mixing.assortativity, Some(-1.0));
    }

    #[test]
    fn single_label_mixing_is_undefined() {
        let (n, edges) = two_triangles();
        let mixing = v888_growth_label_mixing(n, &edges, &[1; 6]).unwrap();
        assert_eq!(mixing.assortativity_denominator, 0);
        assert_eq!(mixing.assortativity, None);
    }

    #[test]
    fn adjusted_rand_matches_reference_values() {
        let identical =
            v888_growth_partition_agreement(&[0, 0, 1, 1, 2], &[4, 4, 9, 9, 1]).unwrap();
        assert_eq!(identical.adjusted_rand, Some(1.0));
        // ARI([0,0,1,1], [0,0,1,2]) = 4/7, kept as an exact ratio.
        let partial = v888_growth_partition_agreement(&[0, 0, 1, 1], &[0, 0, 1, 2]).unwrap();
        assert_eq!(partial.pair_universe, 6);
        assert_eq!(partial.same_both, 1);
        assert_eq!(partial.same_left, 2);
        assert_eq!(partial.same_right, 1);
        // 2 * (6*1 - 2) / (6*3 - 4) = 8/14.
        assert_eq!(partial.ari_numerator, 8);
        assert_eq!(partial.ari_denominator, 14);
        let trivial = v888_growth_partition_agreement(&[0, 0, 0], &[1, 1, 1]).unwrap();
        assert_eq!(trivial.adjusted_rand, None);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let (n, edges) = two_triangles();
        assert_eq!(
            v888_growth_directed_modularity(n, &edges, &[0; 5]),
            Err(V888GrowthModularityError::LabelLengthMismatch {
                expected: 6,
                observed: 5
            })
        );
        let looped: V888GrowthUnitEdges = [(1, 1)].into_iter().collect();
        assert_eq!(
            v888_growth_label_mixing(2, &looped, &[0, 0]),
            Err(V888GrowthModularityError::SelfLoop { node: 1 })
        );
        let out_of_range: V888GrowthUnitEdges = [(0, 9)].into_iter().collect();
        assert!(matches!(
            v888_growth_infer_communities(2, &out_of_range, V888GrowthLouvainOptions::default()),
            Err(V888GrowthModularityError::EdgeIndexOutOfBounds { .. })
        ));
        assert_eq!(
            v888_growth_infer_communities(
                0,
                &V888GrowthUnitEdges::new(),
                V888GrowthLouvainOptions::default()
            ),
            Err(V888GrowthModularityError::EmptyNodeSet)
        );
        assert_eq!(
            v888_growth_partition_agreement(&[], &[]),
            Err(V888GrowthModularityError::EmptyNodeSet)
        );
    }

    #[test]
    fn arm_comparison_is_zero_on_identical_arms() {
        let (n, edges, truth) = planted(3, 12, 7);
        let (reference, rows) = v888_growth_compare_modularity_arms(
            n,
            &edges,
            &[("copy", &edges)],
            Some(&truth),
            V888GrowthLouvainOptions::default(),
        )
        .unwrap();
        assert_eq!(reference.arm, "reference");
        assert_eq!(reference.inferred_vs_reference.adjusted_rand, Some(1.0));
        let (arm, delta) = &rows[0];
        assert_eq!(arm.arm, "copy");
        assert_eq!(delta.community_count_delta, 0);
        assert_eq!(delta.inferred_q_nano_delta, Some(0));
        assert_eq!(delta.annotation_q_nano_delta, Some(0));
        assert_eq!(delta.annotation_assortativity_nano_delta, Some(0));
        assert_eq!(delta.inferred_vs_reference_ari_nano, Some(1_000_000_000));
    }

    #[test]
    fn arm_comparison_without_annotation_omits_annotation_fields() {
        let (n, edges) = two_triangles();
        let shuffled: V888GrowthUnitEdges =
            [(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0), (0, 4)]
                .into_iter()
                .collect();
        let (reference, rows) = v888_growth_compare_modularity_arms(
            n,
            &edges,
            &[("rewired", &shuffled)],
            None,
            V888GrowthLouvainOptions::default(),
        )
        .unwrap();
        assert!(reference.annotation_modularity.is_none());
        let (arm, delta) = &rows[0];
        assert!(arm.inferred_vs_annotation.is_none());
        assert_eq!(delta.annotation_q_nano_delta, None);
        assert!(delta.inferred_q_nano_delta.unwrap() < 0);
    }
}
