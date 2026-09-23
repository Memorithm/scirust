//! Exact deterministic descriptors for directed sparse graphs.
//!
//! This module contains reusable graph observables only. It does not interpret
//! any descriptor as evidence of biological importance or model quality.

use std::collections::BTreeMap;
use std::fmt;

use crate::directed::{DirectedGraph, DirectedGraphError};

/// Exact in/out-degree information for every node plus deterministic histograms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedDegreeProfile {
    pub in_degree: Vec<usize>,
    pub out_degree: Vec<usize>,
    pub in_histogram: BTreeMap<usize, usize>,
    pub out_histogram: BTreeMap<usize, usize>,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
}

/// Compute exact directed degree sequences and histograms.
pub fn directed_degree_profile<E>(
    graph: &DirectedGraph<E>,
) -> Result<DirectedDegreeProfile, DirectedDescriptorError> {
    let mut in_degree = Vec::with_capacity(graph.node_count());
    let mut out_degree = Vec::with_capacity(graph.node_count());
    let mut in_histogram = BTreeMap::new();
    let mut out_histogram = BTreeMap::new();
    let mut max_in_degree = 0usize;
    let mut max_out_degree = 0usize;

    for node in 0..graph.node_count()
    {
        let inbound = graph.in_degree(node)?;
        let outbound = graph.out_degree(node)?;
        *in_histogram.entry(inbound).or_insert(0) += 1;
        *out_histogram.entry(outbound).or_insert(0) += 1;
        max_in_degree = max_in_degree.max(inbound);
        max_out_degree = max_out_degree.max(outbound);
        in_degree.push(inbound);
        out_degree.push(outbound);
    }

    Ok(DirectedDegreeProfile {
        in_degree,
        out_degree,
        in_histogram,
        out_histogram,
        max_in_degree,
        max_out_degree,
    })
}

/// Exact reciprocity counts.
///
/// reciprocal_pairs counts each unordered pair once; each such pair corresponds
/// to two directed edges in reciprocal_directed_edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocitySummary {
    pub directed_edges: usize,
    pub reciprocal_pairs: usize,
    pub reciprocal_directed_edges: usize,
}

/// Compute exact reciprocity counts without converting them into a floating
/// percentage.
pub fn reciprocity_summary<E>(
    graph: &DirectedGraph<E>,
) -> Result<ReciprocitySummary, DirectedDescriptorError> {
    let reciprocal_pairs = graph.reciprocal_pair_count();
    let reciprocal_directed_edges = reciprocal_pairs
        .checked_mul(2)
        .ok_or(DirectedDescriptorError::AccountingOverflow)?;

    Ok(ReciprocitySummary {
        directed_edges: graph.edge_count(),
        reciprocal_pairs,
        reciprocal_directed_edges,
    })
}

/// Canonical strongly-connected-component size profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrongComponentSummary {
    /// Number of strongly connected components.
    pub component_count: usize,
    /// Component sizes ordered by the canonical component labels returned by
    /// DirectedGraph::strongly_connected_components.
    pub component_sizes: Vec<usize>,
    /// Largest strongly connected component size.
    pub largest_component: usize,
}

/// Compute canonical strongly-connected-component sizes.
pub fn strong_component_summary<E>(graph: &DirectedGraph<E>) -> StrongComponentSummary {
    let labels = graph.strongly_connected_components();
    let component_count = labels
        .iter()
        .copied()
        .max()
        .map_or(0usize, |maximum| maximum + 1);
    let mut component_sizes = vec![0usize; component_count];
    for label in labels
    {
        component_sizes[label] += 1;
    }
    let largest_component = component_sizes.iter().copied().max().unwrap_or(0);

    StrongComponentSummary {
        component_count,
        component_sizes,
        largest_component,
    }
}

/// Exact weighted incoming/outgoing sums supplied by a caller-defined edge
/// projection.
///
/// The projection permits BANC synapse counts, normalized strengths, signed
/// artificial-model weights, or other finite scalar interpretations without
/// hard-coding a connectome-specific payload into scirust-graph.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedStrengthProfile {
    pub incoming: Vec<f64>,
    pub outgoing: Vec<f64>,
    pub total: f64,
}

/// Sum finite edge weights into deterministic incoming/outgoing node strengths.
///
/// Negative finite weights are accepted because signed artificial-model edges
/// are a legitimate downstream use. Non-finite source weights or accumulated
/// sums fail closed.
pub fn weighted_strength_profile<E, F>(
    graph: &DirectedGraph<E>,
    mut weight: F,
) -> Result<WeightedStrengthProfile, DirectedDescriptorError>
where
    F: FnMut(&E) -> f64,
{
    let mut incoming = vec![0.0f64; graph.node_count()];
    let mut outgoing = vec![0.0f64; graph.node_count()];
    let mut total = 0.0f64;

    for edge in graph.edges()
    {
        let value = weight(&edge.value);
        if !value.is_finite()
        {
            return Err(DirectedDescriptorError::NonFiniteWeight {
                source: edge.source,
                target: edge.target,
            });
        }

        outgoing[edge.source] += value;
        incoming[edge.target] += value;
        total += value;

        if !outgoing[edge.source].is_finite()
            || !incoming[edge.target].is_finite()
            || !total.is_finite()
        {
            return Err(DirectedDescriptorError::WeightAccumulationOverflow);
        }
    }

    Ok(WeightedStrengthProfile {
        incoming,
        outgoing,
        total,
    })
}

/// Exact descriptor failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectedDescriptorError {
    Graph(DirectedGraphError),
    AccountingOverflow,
    NonFiniteWeight { source: usize, target: usize },
    WeightAccumulationOverflow,
}

impl From<DirectedGraphError> for DirectedDescriptorError {
    fn from(value: DirectedGraphError) -> Self {
        Self::Graph(value)
    }
}

impl fmt::Display for DirectedDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Graph(error) => write!(formatter, "{error}"),
            Self::AccountingOverflow =>
            {
                formatter.write_str("directed descriptor accounting overflowed")
            },
            Self::NonFiniteWeight { source, target } => write!(
                formatter,
                "directed edge {source} -> {target} projected to a non-finite weight"
            ),
            Self::WeightAccumulationOverflow =>
            {
                formatter.write_str("directed weighted-strength accumulation became non-finite")
            },
        }
    }
}

impl std::error::Error for DirectedDescriptorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directed::{DirectedEdge, DirectedGraphOptions};

    fn graph() -> DirectedGraph<i32> {
        DirectedGraph::from_edges(
            5,
            vec![
                DirectedEdge::new(0, 1, 2),
                DirectedEdge::new(1, 0, 3),
                DirectedEdge::new(1, 2, 5),
                DirectedEdge::new(2, 0, 7),
                DirectedEdge::new(2, 3, 11),
                DirectedEdge::new(3, 4, 13),
            ],
            DirectedGraphOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn degree_profile_is_exact_and_histograms_reconcile() {
        let graph = graph();
        let profile = directed_degree_profile(&graph).unwrap();

        assert_eq!(profile.in_degree, vec![2, 1, 1, 1, 1]);
        assert_eq!(profile.out_degree, vec![1, 2, 2, 1, 0]);
        assert_eq!(profile.in_histogram.get(&1), Some(&4));
        assert_eq!(profile.in_histogram.get(&2), Some(&1));
        assert_eq!(profile.out_histogram.get(&0), Some(&1));
        assert_eq!(profile.out_histogram.get(&1), Some(&2));
        assert_eq!(profile.out_histogram.get(&2), Some(&2));
        assert_eq!(profile.in_histogram.values().sum::<usize>(), 5);
        assert_eq!(profile.out_histogram.values().sum::<usize>(), 5);
        assert_eq!(profile.max_in_degree, 2);
        assert_eq!(profile.max_out_degree, 2);
        assert_eq!(profile.in_degree.iter().sum::<usize>(), graph.edge_count());
        assert_eq!(profile.out_degree.iter().sum::<usize>(), graph.edge_count());
    }

    #[test]
    fn reciprocity_counts_bidirectional_pair_once() {
        let summary = reciprocity_summary(&graph()).unwrap();
        assert_eq!(
            summary,
            ReciprocitySummary {
                directed_edges: 6,
                reciprocal_pairs: 1,
                reciprocal_directed_edges: 2,
            }
        );
    }

    #[test]
    fn strongly_connected_component_sizes_follow_canonical_labels() {
        let summary = strong_component_summary(&graph());
        assert_eq!(summary.component_count, 3);
        assert_eq!(summary.component_sizes, vec![3, 1, 1]);
        assert_eq!(summary.largest_component, 3);
    }

    #[test]
    fn weighted_strengths_preserve_caller_defined_payload_semantics() {
        let profile = weighted_strength_profile(&graph(), |weight| f64::from(*weight)).unwrap();

        assert_eq!(profile.outgoing, vec![2.0, 8.0, 18.0, 13.0, 0.0]);
        assert_eq!(profile.incoming, vec![10.0, 2.0, 5.0, 11.0, 13.0]);
        assert_eq!(profile.total, 41.0);
        assert_eq!(profile.incoming.iter().sum::<f64>(), profile.total);
        assert_eq!(profile.outgoing.iter().sum::<f64>(), profile.total);
    }

    #[test]
    fn weighted_strengths_allow_finite_negative_edges() {
        let profile = weighted_strength_profile(&graph(), |weight| -f64::from(*weight)).unwrap();
        assert_eq!(profile.total, -41.0);
    }

    #[test]
    fn weighted_strengths_reject_non_finite_projection() {
        let error = weighted_strength_profile(&graph(), |weight| {
            if *weight == 11
            {
                f64::NAN
            }
            else
            {
                f64::from(*weight)
            }
        })
        .unwrap_err();

        assert_eq!(
            error,
            DirectedDescriptorError::NonFiniteWeight {
                source: 2,
                target: 3,
            }
        );
    }
}
