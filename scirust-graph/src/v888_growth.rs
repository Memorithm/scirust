//! Exact topology descriptors, matched controls and VG-3C metrics for V888-GROWTH.
//!
//! This module converts the already-qualified executable BANC v888 graph into
//! per-node structural facts suitable for later joins with morphology or
//! annotation tables, exposes deterministic unit-adjacency matched controls for
//! VG-3B, and re-exports reachability / centrality / rich-club helpers for
//! VG-3C, plus inferred modularity, label mixing, partition agreement and
//! deterministic caller-annotation joins for VG-3D. It does not infer developmental causality, biological importance,
//! model quality, topology advantage, or growth mechanisms.

use core::fmt;

use crate::banc_v888::BancV888NodeId;
use crate::directed::DirectedGraphError;
use crate::v888_executable::BancV888ExecutableGraph;

pub use crate::v888_growth_annotation::{
    V888_GROWTH_UNLABELLED, V888GrowthAnnotation, V888GrowthAnnotationError,
    V888GrowthAnnotationJoin, v888_growth_join_annotation, v888_growth_parse_annotation_tsv,
};
pub use crate::v888_growth_controls::{
    V888GrowthControlArm, V888GrowthControlArmResult, V888GrowthControlBundle,
    V888GrowthControlError, V888GrowthControlMatch, V888GrowthControlStats, V888GrowthSubsetPolicy,
    V888GrowthUnitEdge, V888GrowthUnitEdges, v888_growth_control_mix, v888_growth_control_stats,
    v888_growth_edge_count_control, v888_growth_matched_controls, v888_growth_rewire_control,
    v888_growth_select_subset, v888_growth_unit_reference,
};
pub use crate::v888_growth_ensemble::{
    V888_GROWTH_ENSEMBLE_MAX_SEEDS, V888GrowthDispersion, V888GrowthEnsembleArmSummary,
    V888GrowthEnsembleDescriptor, V888GrowthEnsembleError, V888GrowthEnsembleReport,
    V888GrowthEnsembleSample, V888GrowthEnsembleValues, v888_growth_control_ensemble,
    v888_growth_dispersion, v888_growth_ensemble_seeds,
};
pub use crate::v888_growth_metrics::{
    V888GrowthBetweenness, V888GrowthDistanceProfile, V888GrowthHarmonicCloseness,
    V888GrowthMetricBundle, V888GrowthMetricDelta, V888GrowthMetricsError,
    V888GrowthReachabilityProfile, V888GrowthRichClubCurve, V888GrowthRichClubDegree,
    V888GrowthRichClubPoint, v888_growth_bfs_distances, v888_growth_compare_control_bundle,
    v888_growth_compare_metric_arms, v888_growth_directed_betweenness, v888_growth_f64_to_nano,
    v888_growth_harmonic_closeness, v888_growth_metric_bundle, v888_growth_reachability_from_edges,
    v888_growth_reachability_profile, v888_growth_rich_club_curve, v888_growth_unit_digraph,
};
pub use crate::v888_growth_modularity::{
    V888GrowthCommunityPartition, V888GrowthLabelMixing, V888GrowthLouvainOptions,
    V888GrowthModularity, V888GrowthModularityArm, V888GrowthModularityDelta,
    V888GrowthModularityError, V888GrowthPartitionAgreement, v888_growth_compare_modularity_arms,
    v888_growth_directed_modularity, v888_growth_infer_communities, v888_growth_label_mixing,
    v888_growth_partition_agreement,
};

/// One exact per-neuron topology row for V888-GROWTH VG-3.
///
/// Contact fields are integer multiplicity sums from the qualified pair graph;
/// they are not conductance, event counts, or physiological weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V888GrowthTopologyRow {
    /// Exact BANC root identifier.
    pub node_id: BancV888NodeId,
    /// Number of incoming directed neuron pairs.
    pub in_degree: usize,
    /// Number of outgoing directed neuron pairs.
    pub out_degree: usize,
    /// Sum of incoming contact multiplicities.
    pub incoming_contacts: u64,
    /// Sum of outgoing contact multiplicities.
    pub outgoing_contacts: u64,
    /// Number of distinct neighbours connected in both directions.
    pub reciprocal_neighbors: usize,
    /// Canonical strongly-connected-component label.
    pub scc_label: usize,
    /// Number of nodes in this node's strongly connected component.
    pub scc_size: usize,
}

/// Deterministic VG-3A profile over the complete executable graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthTopologyProfile {
    /// Rows in the same canonical root-ID order as the executable graph.
    pub rows: Vec<V888GrowthTopologyRow>,
    /// Exact directed-pair count.
    pub directed_pairs: usize,
    /// Exact sum of contact multiplicities.
    pub contacts: u64,
    /// Number of unordered reciprocal node pairs.
    pub reciprocal_pairs: usize,
    /// Number of strongly connected components.
    pub scc_count: usize,
    /// Size of the largest strongly connected component.
    pub largest_scc: usize,
}

/// Build exact per-neuron VG-3A topology facts from a qualified V888 graph.
///
/// The implementation is deterministic for canonical graph bytes. It computes
/// degree and contact-multiplicity sums in one edge pass, counts each reciprocal
/// node pair once, and uses the graph's canonical SCC labelling. The function
/// deliberately does not compute modularity or morphology joins; matched
/// controls live in the VG-3B helpers and reachability/centrality/rich-club
/// descriptors live in the VG-3C helpers re-exported from this module.
///
/// # Errors
///
/// Returns V888GrowthTopologyError if graph access fails, an exact integer
/// counter overflows, or the recomputed contact total disagrees with the
/// executable graph's retained total.
///
/// # Examples
///
/// ~~~rust
/// use scirust_graph::v888_executable::{BancV888ExecutableGraph, BANC_V888_CSR_MAGIC};
/// use scirust_graph::v888_growth::v888_growth_topology_profile;
///
/// let mut bytes = BANC_V888_CSR_MAGIC.to_vec();
/// bytes.extend_from_slice(&2_u64.to_le_bytes());
/// bytes.extend_from_slice(&2_u64.to_le_bytes());
/// for id in [10_u64, 20] {
///     bytes.extend_from_slice(&id.to_le_bytes());
/// }
/// for offset in [0_u64, 1, 2] {
///     bytes.extend_from_slice(&offset.to_le_bytes());
/// }
/// for target in [1_u32, 0] {
///     bytes.extend_from_slice(&target.to_le_bytes());
/// }
/// for contacts in [2_u64, 3] {
///     bytes.extend_from_slice(&contacts.to_le_bytes());
/// }
///
/// let graph = BancV888ExecutableGraph::from_v8csr001(&bytes)?;
/// let profile = v888_growth_topology_profile(&graph)?;
/// assert_eq!(profile.directed_pairs, 2);
/// assert_eq!(profile.contacts, 5);
/// assert_eq!(profile.reciprocal_pairs, 1);
/// assert_eq!(profile.rows[0].reciprocal_neighbors, 1);
/// assert_eq!(profile.rows[1].incoming_contacts, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ~~~
pub fn v888_growth_topology_profile(
    graph: &BancV888ExecutableGraph,
) -> Result<V888GrowthTopologyProfile, V888GrowthTopologyError> {
    let directed = graph.graph();
    let node_count = graph.node_count();

    let mut in_degree = vec![0_usize; node_count];
    let mut out_degree = vec![0_usize; node_count];
    let mut incoming_contacts = vec![0_u64; node_count];
    let mut outgoing_contacts = vec![0_u64; node_count];

    let mut observed_contacts = 0_u64;
    for edge in directed.edges()
    {
        out_degree[edge.source] = out_degree[edge.source].checked_add(1).ok_or(
            V888GrowthTopologyError::CounterOverflow {
                field: "out_degree",
                node: edge.source,
            },
        )?;
        in_degree[edge.target] = in_degree[edge.target].checked_add(1).ok_or(
            V888GrowthTopologyError::CounterOverflow {
                field: "in_degree",
                node: edge.target,
            },
        )?;
        outgoing_contacts[edge.source] = outgoing_contacts[edge.source]
            .checked_add(edge.value)
            .ok_or(V888GrowthTopologyError::CounterOverflow {
                field: "outgoing_contacts",
                node: edge.source,
            })?;
        incoming_contacts[edge.target] = incoming_contacts[edge.target]
            .checked_add(edge.value)
            .ok_or(V888GrowthTopologyError::CounterOverflow {
                field: "incoming_contacts",
                node: edge.target,
            })?;
        observed_contacts = observed_contacts
            .checked_add(edge.value)
            .ok_or(V888GrowthTopologyError::TotalContactOverflow)?;
    }

    if observed_contacts != graph.contact_count()
    {
        return Err(V888GrowthTopologyError::ContactTotalMismatch {
            retained: graph.contact_count(),
            recomputed: observed_contacts,
        });
    }

    let mut reciprocal_neighbors = vec![0_usize; node_count];
    let mut reciprocal_pairs = 0_usize;
    for edge in directed
        .edges()
        .iter()
        .filter(|edge| edge.source < edge.target)
    {
        if directed.has_edge(edge.target, edge.source)?
        {
            reciprocal_neighbors[edge.source] = reciprocal_neighbors[edge.source]
                .checked_add(1)
                .ok_or(V888GrowthTopologyError::CounterOverflow {
                    field: "reciprocal_neighbors",
                    node: edge.source,
                })?;
            reciprocal_neighbors[edge.target] = reciprocal_neighbors[edge.target]
                .checked_add(1)
                .ok_or(V888GrowthTopologyError::CounterOverflow {
                    field: "reciprocal_neighbors",
                    node: edge.target,
                })?;
            reciprocal_pairs = reciprocal_pairs
                .checked_add(1)
                .ok_or(V888GrowthTopologyError::ReciprocalPairOverflow)?;
        }
    }

    let scc_labels = directed.strongly_connected_components();
    let scc_count = scc_labels
        .iter()
        .copied()
        .max()
        .map_or(0_usize, |maximum| maximum + 1);
    let mut scc_sizes = vec![0_usize; scc_count];
    for &label in &scc_labels
    {
        scc_sizes[label] = scc_sizes[label]
            .checked_add(1)
            .ok_or(V888GrowthTopologyError::SccSizeOverflow { label })?;
    }
    let largest_scc = scc_sizes.iter().copied().max().unwrap_or(0);

    let rows = graph
        .node_ids()
        .iter()
        .copied()
        .enumerate()
        .map(|(node, node_id)| V888GrowthTopologyRow {
            node_id,
            in_degree: in_degree[node],
            out_degree: out_degree[node],
            incoming_contacts: incoming_contacts[node],
            outgoing_contacts: outgoing_contacts[node],
            reciprocal_neighbors: reciprocal_neighbors[node],
            scc_label: scc_labels[node],
            scc_size: scc_sizes[scc_labels[node]],
        })
        .collect();

    Ok(V888GrowthTopologyProfile {
        rows,
        directed_pairs: graph.directed_pair_count(),
        contacts: observed_contacts,
        reciprocal_pairs,
        scc_count,
        largest_scc,
    })
}

/// Fail-closed errors from exact VG-3A topology profiling.
#[derive(Debug)]
pub enum V888GrowthTopologyError {
    /// Directed graph access failed.
    DirectedGraph(DirectedGraphError),
    /// A per-node exact integer counter overflowed.
    CounterOverflow {
        /// Counter name.
        field: &'static str,
        /// Dense canonical node index.
        node: usize,
    },
    /// Total contact multiplicity overflowed.
    TotalContactOverflow,
    /// Reciprocal-pair count overflowed.
    ReciprocalPairOverflow,
    /// SCC-size accounting overflowed.
    SccSizeOverflow {
        /// Canonical SCC label.
        label: usize,
    },
    /// Recomputed contact total disagreed with the qualified graph.
    ContactTotalMismatch {
        /// Total retained by the executable graph.
        retained: u64,
        /// Total recomputed from edge payloads.
        recomputed: u64,
    },
}

impl From<DirectedGraphError> for V888GrowthTopologyError {
    fn from(value: DirectedGraphError) -> Self {
        Self::DirectedGraph(value)
    }
}

impl fmt::Display for V888GrowthTopologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::DirectedGraph(error) => write!(formatter, "{error}"),
            Self::CounterOverflow { field, node } =>
            {
                write!(formatter, "{field} overflowed for canonical node {node}")
            },
            Self::TotalContactOverflow => formatter.write_str("total contact count overflowed"),
            Self::ReciprocalPairOverflow => formatter.write_str("reciprocal-pair count overflowed"),
            Self::SccSizeOverflow { label } =>
            {
                write!(formatter, "SCC size overflowed for canonical label {label}")
            },
            Self::ContactTotalMismatch {
                retained,
                recomputed,
            } => write!(
                formatter,
                "recomputed contact total {recomputed} differs from retained total {retained}"
            ),
        }
    }
}

impl std::error::Error for V888GrowthTopologyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v888_executable::BANC_V888_CSR_MAGIC;

    fn fixture() -> BancV888ExecutableGraph {
        let mut bytes = BANC_V888_CSR_MAGIC.to_vec();
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        bytes.extend_from_slice(&5_u64.to_le_bytes());
        for id in [10_u64, 20, 30, 40]
        {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        for offset in [0_u64, 1, 3, 4, 5]
        {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for target in [1_u32, 0, 2, 3, 2]
        {
            bytes.extend_from_slice(&target.to_le_bytes());
        }
        for contacts in [2_u64, 3, 5, 7, 11]
        {
            bytes.extend_from_slice(&contacts.to_le_bytes());
        }
        BancV888ExecutableGraph::from_v8csr001(&bytes).unwrap()
    }

    #[test]
    fn vg3a_rows_preserve_exact_pair_contact_and_scc_semantics() {
        let profile = v888_growth_topology_profile(&fixture()).unwrap();

        assert_eq!(profile.directed_pairs, 5);
        assert_eq!(profile.contacts, 28);
        assert_eq!(profile.reciprocal_pairs, 2);
        assert_eq!(profile.scc_count, 2);
        assert_eq!(profile.largest_scc, 2);

        assert_eq!(profile.rows[0].in_degree, 1);
        assert_eq!(profile.rows[0].out_degree, 1);
        assert_eq!(profile.rows[0].incoming_contacts, 3);
        assert_eq!(profile.rows[0].outgoing_contacts, 2);
        assert_eq!(profile.rows[0].reciprocal_neighbors, 1);
        assert_eq!(profile.rows[0].scc_label, 0);
        assert_eq!(profile.rows[0].scc_size, 2);

        assert_eq!(profile.rows[2].in_degree, 2);
        assert_eq!(profile.rows[2].incoming_contacts, 16);
        assert_eq!(profile.rows[2].scc_label, 1);
        assert_eq!(profile.rows[2].scc_size, 2);
    }

    #[test]
    fn row_order_matches_canonical_root_identity_order() {
        let profile = v888_growth_topology_profile(&fixture()).unwrap();
        let ids: Vec<_> = profile.rows.iter().map(|row| row.node_id.get()).collect();
        assert_eq!(ids, vec![10, 20, 30, 40]);
    }
}
