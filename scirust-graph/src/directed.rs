//! Deterministic directed graph with compact outgoing and incoming indexes.
//!
//! The representation owns one canonical edge array sorted by source and target.
//! Outgoing adjacency is therefore a zero-copy slice of that array. Incoming
//! adjacency stores only edge indices, avoiding payload duplication.

use std::collections::VecDeque;
use std::fmt;
use std::slice;

/// One directed edge with caller-owned payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedEdge<E> {
    pub source: usize,
    pub target: usize,
    pub value: E,
}

impl<E> DirectedEdge<E> {
    /// Construct one directed edge without validating graph-level bounds.
    #[must_use]
    pub const fn new(source: usize, target: usize, value: E) -> Self {
        Self {
            source,
            target,
            value,
        }
    }
}

/// Construction policy for directed graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectedGraphOptions {
    pub allow_self_loops: bool,
}

/// Deterministic directed graph.
///
/// Parallel edges are rejected. Self loops are rejected by default and require
/// explicit opt-in through DirectedGraphOptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedGraph<E> {
    node_count: usize,
    edges: Vec<DirectedEdge<E>>,
    out_offsets: Vec<usize>,
    in_offsets: Vec<usize>,
    in_edge_indices: Vec<usize>,
}

impl<E> DirectedGraph<E> {
    /// Validate edges and build deterministic outgoing and incoming indexes.
    pub fn from_edges(
        node_count: usize,
        mut edges: Vec<DirectedEdge<E>>,
        options: DirectedGraphOptions,
    ) -> Result<Self, DirectedGraphError> {
        if node_count == 0
        {
            return Err(DirectedGraphError::EmptyNodeSet);
        }

        for edge in &edges
        {
            if edge.source >= node_count || edge.target >= node_count
            {
                return Err(DirectedGraphError::NodeOutOfBounds {
                    source: edge.source,
                    target: edge.target,
                    node_count,
                });
            }
            if edge.source == edge.target && !options.allow_self_loops
            {
                return Err(DirectedGraphError::SelfLoop { node: edge.source });
            }
        }

        edges.sort_by_key(|edge| (edge.source, edge.target));
        for pair in edges.windows(2)
        {
            if pair[0].source == pair[1].source && pair[0].target == pair[1].target
            {
                return Err(DirectedGraphError::ParallelEdge {
                    source: pair[0].source,
                    target: pair[0].target,
                });
            }
        }

        let out_offsets =
            offsets_by_node(node_count, edges.iter().map(|edge| edge.source), edges.len());

        let mut in_edge_indices: Vec<usize> = (0..edges.len()).collect();
        in_edge_indices.sort_by_key(|&index| {
            let edge = &edges[index];
            (edge.target, edge.source)
        });
        let in_offsets = offsets_by_node(
            node_count,
            in_edge_indices.iter().map(|&index| edges[index].target),
            edges.len(),
        );

        Ok(Self {
            node_count,
            edges,
            out_offsets,
            in_offsets,
            in_edge_indices,
        })
    }

    /// Return the number of nodes in the fixed node universe.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Return the number of canonical directed edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Canonical edge array sorted by source and target.
    #[must_use]
    pub fn edges(&self) -> &[DirectedEdge<E>] {
        &self.edges
    }

    /// Zero-copy outgoing edges for one node, sorted by target.
    pub fn outgoing(&self, node: usize) -> Result<&[DirectedEdge<E>], DirectedGraphError> {
        self.check_node(node)?;
        let start = self.out_offsets[node];
        let end = self.out_offsets[node + 1];
        Ok(&self.edges[start..end])
    }

    /// Incoming edges for one node, sorted by source.
    pub fn incoming(&self, node: usize) -> Result<IncomingEdges<'_, E>, DirectedGraphError> {
        self.check_node(node)?;
        let start = self.in_offsets[node];
        let end = self.in_offsets[node + 1];
        Ok(IncomingEdges {
            graph: self,
            indices: self.in_edge_indices[start..end].iter(),
        })
    }

    /// Return the number of outgoing edges for one node.
    pub fn out_degree(&self, node: usize) -> Result<usize, DirectedGraphError> {
        self.check_node(node)?;
        Ok(self.out_offsets[node + 1] - self.out_offsets[node])
    }

    /// Return the number of incoming edges for one node.
    pub fn in_degree(&self, node: usize) -> Result<usize, DirectedGraphError> {
        self.check_node(node)?;
        Ok(self.in_offsets[node + 1] - self.in_offsets[node])
    }

    /// Test whether one exact directed edge is present.
    pub fn has_edge(&self, source: usize, target: usize) -> Result<bool, DirectedGraphError> {
        self.check_node(source)?;
        self.check_node(target)?;
        let outgoing = self.outgoing(source)?;
        Ok(outgoing
            .binary_search_by_key(&target, |edge| edge.target)
            .is_ok())
    }

    /// Deterministic breadth-first reachability including start.
    pub fn reachable_from(&self, start: usize) -> Result<Vec<usize>, DirectedGraphError> {
        self.check_node(start)?;
        let mut seen = vec![false; self.node_count];
        let mut queue = VecDeque::new();
        let mut order = Vec::new();

        seen[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front()
        {
            order.push(node);
            for edge in self.outgoing(node)?
            {
                if !seen[edge.target]
                {
                    seen[edge.target] = true;
                    queue.push_back(edge.target);
                }
            }
        }
        Ok(order)
    }

    /// Number of unordered node pairs connected in both directions.
    #[must_use]
    pub fn reciprocal_pair_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.source < edge.target)
            .filter(|edge| self.has_edge(edge.target, edge.source).unwrap_or(false))
            .count()
    }

    /// Strongly connected component labels canonicalized by smallest node.
    #[must_use]
    pub fn strongly_connected_components(&self) -> Vec<usize> {
        let mut seen = vec![false; self.node_count];
        let mut finish_order = Vec::with_capacity(self.node_count);

        for start in 0..self.node_count
        {
            if !seen[start]
            {
                self.finish_dfs(start, &mut seen, &mut finish_order);
            }
        }

        seen.fill(false);
        let mut components = Vec::new();
        for &start in finish_order.iter().rev()
        {
            if seen[start]
            {
                continue;
            }
            let mut component = Vec::new();
            self.reverse_collect(start, &mut seen, &mut component);
            component.sort_unstable();
            components.push(component);
        }

        components.sort_by_key(|component| component[0]);
        let mut labels = vec![usize::MAX; self.node_count];
        for (label, component) in components.iter().enumerate()
        {
            for &node in component
            {
                labels[node] = label;
            }
        }
        labels
    }

    fn finish_dfs(&self, node: usize, seen: &mut [bool], order: &mut Vec<usize>) {
        seen[node] = true;
        let start = self.out_offsets[node];
        let end = self.out_offsets[node + 1];
        for edge in &self.edges[start..end]
        {
            if !seen[edge.target]
            {
                self.finish_dfs(edge.target, seen, order);
            }
        }
        order.push(node);
    }

    fn reverse_collect(&self, node: usize, seen: &mut [bool], component: &mut Vec<usize>) {
        seen[node] = true;
        component.push(node);
        let start = self.in_offsets[node];
        let end = self.in_offsets[node + 1];
        for &edge_index in &self.in_edge_indices[start..end]
        {
            let source = self.edges[edge_index].source;
            if !seen[source]
            {
                self.reverse_collect(source, seen, component);
            }
        }
    }

    fn check_node(&self, node: usize) -> Result<(), DirectedGraphError> {
        if node >= self.node_count
        {
            return Err(DirectedGraphError::NodeIndexOutOfBounds {
                node,
                node_count: self.node_count,
            });
        }
        Ok(())
    }
}

fn offsets_by_node(
    node_count: usize,
    nodes: impl Iterator<Item = usize>,
    edge_count: usize,
) -> Vec<usize> {
    let mut counts = vec![0usize; node_count];
    for node in nodes
    {
        counts[node] += 1;
    }

    let mut offsets = Vec::with_capacity(node_count + 1);
    offsets.push(0);
    for count in counts
    {
        let next = offsets.last().copied().unwrap_or(0) + count;
        offsets.push(next);
    }
    debug_assert_eq!(offsets.last().copied(), Some(edge_count));
    offsets
}

/// Iterator over incoming edge references.
pub struct IncomingEdges<'a, E> {
    graph: &'a DirectedGraph<E>,
    indices: slice::Iter<'a, usize>,
}

impl<'a, E> Iterator for IncomingEdges<'a, E> {
    type Item = &'a DirectedEdge<E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|&index| &self.graph.edges[index])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<E> ExactSizeIterator for IncomingEdges<'_, E> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectedGraphError {
    EmptyNodeSet,
    NodeOutOfBounds {
        source: usize,
        target: usize,
        node_count: usize,
    },
    NodeIndexOutOfBounds {
        node: usize,
        node_count: usize,
    },
    SelfLoop {
        node: usize,
    },
    ParallelEdge {
        source: usize,
        target: usize,
    },
}

impl fmt::Display for DirectedGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyNodeSet => formatter.write_str("directed graph requires at least one node"),
            Self::NodeOutOfBounds {
                source,
                target,
                node_count,
            } => write!(
                formatter,
                "directed edge {source} -> {target} is out of bounds for {node_count} nodes"
            ),
            Self::NodeIndexOutOfBounds { node, node_count } => {
                write!(
                    formatter,
                    "node {node} is out of bounds for {node_count} nodes"
                )
            }
            Self::SelfLoop { node } => {
                write!(formatter, "self loop at node {node} is forbidden")
            }
            Self::ParallelEdge { source, target } => {
                write!(formatter, "parallel edge {source} -> {target} is forbidden")
            }
        }
    }
}

impl std::error::Error for DirectedGraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(source: usize, target: usize, value: i32) -> DirectedEdge<i32> {
        DirectedEdge::new(source, target, value)
    }

    fn graph() -> DirectedGraph<i32> {
        DirectedGraph::from_edges(
            5,
            vec![
                edge(3, 4, 34),
                edge(1, 2, 12),
                edge(0, 1, 1),
                edge(2, 0, 20),
                edge(2, 3, 23),
                edge(1, 0, 10),
            ],
            DirectedGraphOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn canonical_edge_order_is_input_order_independent() {
        let first = graph();
        let second = DirectedGraph::from_edges(
            5,
            vec![
                edge(1, 0, 10),
                edge(2, 3, 23),
                edge(2, 0, 20),
                edge(0, 1, 1),
                edge(1, 2, 12),
                edge(3, 4, 34),
            ],
            DirectedGraphOptions::default(),
        )
        .unwrap();
        assert_eq!(first, second);
        let keys: Vec<_> = first
            .edges()
            .iter()
            .map(|edge| (edge.source, edge.target))
            .collect();
        assert_eq!(
            keys,
            vec![(0, 1), (1, 0), (1, 2), (2, 0), (2, 3), (3, 4)]
        );
    }

    #[test]
    fn outgoing_and_incoming_indexes_agree() {
        let graph = graph();
        assert_eq!(graph.out_degree(1).unwrap(), 2);
        assert_eq!(graph.in_degree(1).unwrap(), 1);

        let outgoing: Vec<_> = graph
            .outgoing(1)
            .unwrap()
            .iter()
            .map(|edge| (edge.target, edge.value))
            .collect();
        assert_eq!(outgoing, vec![(0, 10), (2, 12)]);

        let incoming: Vec<_> = graph
            .incoming(0)
            .unwrap()
            .map(|edge| (edge.source, edge.value))
            .collect();
        assert_eq!(incoming, vec![(1, 10), (2, 20)]);
    }

    #[test]
    fn rejects_parallel_edges_and_self_loops_by_default() {
        assert_eq!(
            DirectedGraph::from_edges(
                2,
                vec![edge(0, 1, 1), edge(0, 1, 2)],
                DirectedGraphOptions::default(),
            ),
            Err(DirectedGraphError::ParallelEdge {
                source: 0,
                target: 1,
            })
        );
        assert_eq!(
            DirectedGraph::from_edges(
                2,
                vec![edge(1, 1, 1)],
                DirectedGraphOptions::default(),
            ),
            Err(DirectedGraphError::SelfLoop { node: 1 })
        );
    }

    #[test]
    fn self_loops_require_explicit_opt_in() {
        let graph = DirectedGraph::from_edges(
            2,
            vec![edge(1, 1, 7)],
            DirectedGraphOptions {
                allow_self_loops: true,
            },
        )
        .unwrap();
        assert!(graph.has_edge(1, 1).unwrap());
    }

    #[test]
    fn reachability_is_directed_and_deterministic() {
        let graph = graph();
        assert_eq!(graph.reachable_from(3).unwrap(), vec![3, 4]);
        assert_eq!(graph.reachable_from(0).unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn reciprocal_pairs_count_each_pair_once() {
        let graph = graph();
        assert_eq!(graph.reciprocal_pair_count(), 1);
    }

    #[test]
    fn strongly_connected_components_are_canonical() {
        let graph = graph();
        assert_eq!(graph.strongly_connected_components(), vec![0, 0, 0, 1, 2]);
    }

    #[test]
    fn node_bounds_fail_closed() {
        let graph = graph();
        assert!(matches!(
            graph.outgoing(5),
            Err(DirectedGraphError::NodeIndexOutOfBounds {
                node: 5,
                node_count: 5,
            })
        ));
        assert!(matches!(
            DirectedGraph::from_edges(
                2,
                vec![edge(0, 2, 1)],
                DirectedGraphOptions::default(),
            ),
            Err(DirectedGraphError::NodeOutOfBounds { .. })
        ));
    }
}
