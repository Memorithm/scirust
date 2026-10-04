//! Public executable-graph contract for the qualified BANC v888 pair graph.
//!
//! The format implemented here is the retained `V8CSR001` contract from
//! V888-BOOL-0.1. It contains only exact root identifiers, directed pair
//! topology and positive contact multiplicities. It deliberately contains no
//! neurotransmitter sign, conductance, threshold, delay, stimulation policy or
//! biological-time semantics.
//!
//! The raw connectome remains external to SciRust. The constants in
//! [`BANC_V888_BOOL01_QUALIFICATION`] bind the observed source/software evidence;
//! consumers still need the graph bytes whose digest matches that record.

use std::collections::HashSet;
use std::fmt;

use sha2::{Digest as _, Sha256};

use crate::banc_v888::{BancV888Error, BancV888NodeId};
use crate::directed::{DirectedEdge, DirectedGraph, DirectedGraphError, DirectedGraphOptions};

/// Versioned public interface identity for the executable BANC v888 graph.
pub const BANC_V888_EXECUTABLE_GRAPH_CONTRACT: &str = "scirust.banc-v888.executable-graph/v1";

/// Exact binary magic for the retained canonical CSR representation.
pub const BANC_V888_CSR_MAGIC: [u8; 8] = *b"V8CSR001";

/// Conservative parser ceiling for executable BANC node universes.
pub const BANC_V888_MAX_NODES: u64 = 1_000_000;

/// Conservative parser ceiling for executable directed-pair counts.
pub const BANC_V888_MAX_DIRECTED_PAIRS: u64 = 20_000_000;

/// Retained source/software identity for the successful V888-BOOL-0.1 run.
///
/// This is a reproducibility record, not a signature or a topology-advantage
/// result. The graph itself remains external to Git.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BancV888ExecutableGraphQualification {
    pub contract: &'static str,
    pub programme_revision: &'static str,
    pub workflow_run: u64,
    pub producer_commit: &'static str,
    pub rust_worker_sha256: &'static str,
    pub protocol_sha256: &'static str,
    pub metadata_sha256: &'static str,
    pub edgelist_sha256: &'static str,
    pub raw_synapses_sha256: &'static str,
    pub graph_sha256: &'static str,
    pub node_map_sha256: &'static str,
    pub graph_source_filter_identity: &'static str,
    pub logical_graph_path: &'static str,
    pub graph_bytes: u64,
    pub metadata_nodes: u64,
    pub directed_pairs: u64,
    pub induced_contacts: u64,
    pub raw_records: u64,
    pub pre_only_contacts: u64,
    pub post_only_contacts: u64,
    pub neither_contacts: u64,
}

/// Exact retained V888-BOOL-0.1 qualification evidence.
pub const BANC_V888_BOOL01_QUALIFICATION: BancV888ExecutableGraphQualification =
    BancV888ExecutableGraphQualification {
        contract: BANC_V888_EXECUTABLE_GRAPH_CONTRACT,
        programme_revision: "2026-09-22.1",
        workflow_run: 35_717_145_115,
        producer_commit: "d61ffb6a79ebe975c1d867ad1a728fd01126fde0",
        rust_worker_sha256: "dbf349dafcde9b4967bb514e8875f375801a2994315126c5f09458e2f799ed02",
        protocol_sha256: "497cdcd57ac8c1d9052b8df6a0b1fb5b2839cdb61079ac4ad719d4f7bb080ad7",
        metadata_sha256: "86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098",
        edgelist_sha256: "8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b",
        raw_synapses_sha256: "0dfb5cf89ba156d076beab2da38d87eaa63dcbe45d76f86b108570fb5b961dd0",
        graph_sha256: "385111a69cc8a1d748c0bdfd9b0b738c51fe83cde98f45762553435a2d15a2a4",
        node_map_sha256: "eaa481b3aef42f63fb9e2e8f9c27d1405c073c54d11cbcd656ca9432904f425a",
        graph_source_filter_identity: "3cfbed6392a097865147ac08fae65f36a986b2ba6d386b1f5d61b9cfbefd1dd9",
        logical_graph_path: "$HOME/datasets/banc_v888/analysis/bool01-35717145115-1/graph.csr",
        graph_bytes: 166_466_540,
        metadata_nodes: 188_508,
        directed_pairs: 13_620_865,
        induced_contacts: 42_309_621,
        raw_records: 198_816_365,
        pre_only_contacts: 141_812_486,
        post_only_contacts: 14_694_258,
        neither_contacts: 0,
    };

/// Canonical executable BANC v888 graph.
///
/// Node indices are dense implementation indices; [`Self::node_ids`] maps them
/// losslessly to the original BANC root IDs. Edge payloads are integer contact
/// multiplicities, not conductance or sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BancV888ExecutableGraph {
    node_ids: Vec<BancV888NodeId>,
    graph: DirectedGraph<u64>,
    contact_count: u64,
}

/// Exact pair/contact partition for a selected node subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BancV888BoundaryAccounting {
    pub outside_pairs: u64,
    pub outside_contacts: u64,
    pub incoming_pairs: u64,
    pub incoming_contacts: u64,
    pub outgoing_pairs: u64,
    pub outgoing_contacts: u64,
    pub internal_pairs: u64,
    pub internal_contacts: u64,
}

/// Exact graph-work counters for one declared active-node set.
///
/// `events` is supplied by the dynamics caller because the static graph does
/// not define event semantics. It is intentionally not inferred from contact
/// multiplicity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BancV888ExecutionAccounting {
    pub active_nodes: u64,
    pub directed_pairs_touched: u64,
    pub contact_multiplicity_touched: u64,
    pub events: u64,
}

impl BancV888ExecutableGraph {
    /// Decode and fully validate one canonical `V8CSR001` graph.
    ///
    /// Validation covers the exact byte length, strictly increasing non-zero
    /// root IDs, monotonic row offsets, in-range strictly increasing targets,
    /// self-loop rejection, positive contact multiplicities and checked contact
    /// summation. Input order is therefore canonical, not merely accepted.
    ///
    /// # Errors
    ///
    /// Returns [`BancV888ExecutableGraphError`] when any structural, range,
    /// length or integer invariant fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirust_graph::v888_executable::{BancV888ExecutableGraph, BANC_V888_CSR_MAGIC};
    ///
    /// let mut bytes = BANC_V888_CSR_MAGIC.to_vec();
    /// bytes.extend_from_slice(&2_u64.to_le_bytes());
    /// bytes.extend_from_slice(&1_u64.to_le_bytes());
    /// bytes.extend_from_slice(&10_u64.to_le_bytes());
    /// bytes.extend_from_slice(&20_u64.to_le_bytes());
    /// for offset in [0_u64, 1, 1] {
    ///     bytes.extend_from_slice(&offset.to_le_bytes());
    /// }
    /// bytes.extend_from_slice(&1_u32.to_le_bytes());
    /// bytes.extend_from_slice(&3_u64.to_le_bytes());
    ///
    /// let graph = BancV888ExecutableGraph::from_v8csr001(&bytes)?;
    /// assert_eq!(graph.node_count(), 2);
    /// assert_eq!(graph.directed_pair_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_v8csr001(bytes: &[u8]) -> Result<Self, BancV888ExecutableGraphError> {
        let mut cursor = 0_usize;
        if take::<8>(bytes, &mut cursor)? != BANC_V888_CSR_MAGIC
        {
            return Err(BancV888ExecutableGraphError::WrongMagic);
        }

        let node_count_u64 = read_u64(bytes, &mut cursor)?;
        let edge_count_u64 = read_u64(bytes, &mut cursor)?;
        if node_count_u64 == 0 || node_count_u64 > BANC_V888_MAX_NODES
        {
            return Err(BancV888ExecutableGraphError::NodeCountOutOfBounds(
                node_count_u64,
            ));
        }
        if edge_count_u64 > BANC_V888_MAX_DIRECTED_PAIRS
        {
            return Err(BancV888ExecutableGraphError::DirectedPairCountOutOfBounds(
                edge_count_u64,
            ));
        }

        let expected_len = canonical_len(node_count_u64, edge_count_u64)?;
        if bytes.len() as u64 != expected_len
        {
            return Err(BancV888ExecutableGraphError::ByteLengthMismatch {
                expected: expected_len,
                actual: bytes.len() as u64,
            });
        }

        let node_count = usize::try_from(node_count_u64)
            .map_err(|_| BancV888ExecutableGraphError::HostIndexOverflow)?;
        let edge_count = usize::try_from(edge_count_u64)
            .map_err(|_| BancV888ExecutableGraphError::HostIndexOverflow)?;

        let mut node_ids = Vec::with_capacity(node_count);
        for _ in 0..node_count
        {
            let value = read_u64(bytes, &mut cursor)?;
            let node =
                BancV888NodeId::new(value).map_err(BancV888ExecutableGraphError::NodeIdentity)?;
            if node_ids.last().is_some_and(|previous| *previous >= node)
            {
                return Err(BancV888ExecutableGraphError::NodeIdsNotStrictlyIncreasing);
            }
            node_ids.push(node);
        }

        let mut offsets = Vec::with_capacity(node_count + 1);
        for _ in 0..=node_count
        {
            let offset = read_u64(bytes, &mut cursor)?;
            if offset > edge_count_u64
            {
                return Err(BancV888ExecutableGraphError::RowOffsetOutOfBounds {
                    offset,
                    edge_count: edge_count_u64,
                });
            }
            offsets.push(
                usize::try_from(offset)
                    .map_err(|_| BancV888ExecutableGraphError::HostIndexOverflow)?,
            );
        }
        if offsets[0] != 0
            || offsets[node_count] != edge_count
            || offsets.windows(2).any(|window| window[0] > window[1])
        {
            return Err(BancV888ExecutableGraphError::InvalidRowOffsets);
        }

        let mut targets = Vec::with_capacity(edge_count);
        for _ in 0..edge_count
        {
            targets.push(read_u32(bytes, &mut cursor)?);
        }

        for source in 0..node_count
        {
            let row = &targets[offsets[source]..offsets[source + 1]];
            let mut previous = None;
            for &target in row
            {
                let target = target as usize;
                if target >= node_count
                {
                    return Err(BancV888ExecutableGraphError::TargetOutOfBounds {
                        source,
                        target,
                        node_count,
                    });
                }
                if target == source
                {
                    return Err(BancV888ExecutableGraphError::SelfLoop { node: source });
                }
                if previous.is_some_and(|value| value >= target)
                {
                    return Err(BancV888ExecutableGraphError::TargetsNotStrictlyIncreasing {
                        source,
                    });
                }
                previous = Some(target);
            }
        }

        let mut weights = Vec::with_capacity(edge_count);
        let mut contact_count = 0_u64;
        for edge_index in 0..edge_count
        {
            let weight = read_u64(bytes, &mut cursor)?;
            if weight == 0
            {
                return Err(BancV888ExecutableGraphError::ZeroContactMultiplicity { edge_index });
            }
            contact_count = contact_count
                .checked_add(weight)
                .ok_or(BancV888ExecutableGraphError::ContactCountOverflow)?;
            weights.push(weight);
        }
        if cursor != bytes.len()
        {
            return Err(BancV888ExecutableGraphError::TrailingBytes);
        }

        let mut edges = Vec::with_capacity(edge_count);
        for source in 0..node_count
        {
            for edge_index in offsets[source]..offsets[source + 1]
            {
                edges.push(DirectedEdge::new(
                    source,
                    targets[edge_index] as usize,
                    weights[edge_index],
                ));
            }
        }
        let graph = DirectedGraph::from_edges(node_count, edges, DirectedGraphOptions::default())
            .map_err(BancV888ExecutableGraphError::DirectedGraph)?;

        Ok(Self {
            node_ids,
            graph,
            contact_count,
        })
    }

    /// Return exact original BANC root IDs in canonical index order.
    #[must_use]
    pub fn node_ids(&self) -> &[BancV888NodeId] {
        &self.node_ids
    }

    /// Return the exact number of nodes in the executable graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    /// Return the exact number of directed pairs, distinct from contact count.
    #[must_use]
    pub fn directed_pair_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Return the sum of positive integer contact multiplicities.
    ///
    /// This quantity is not a conductance, event count or edge count.
    #[must_use]
    pub fn contact_count(&self) -> u64 {
        self.contact_count
    }

    /// Borrow the validated deterministic directed graph.
    ///
    /// The edge payload is the positive integer contact multiplicity.
    #[must_use]
    pub fn graph(&self) -> &DirectedGraph<u64> {
        &self.graph
    }

    /// Re-encode the validated graph as canonical `V8CSR001` bytes.
    ///
    /// The result is deterministic and byte-stable for the same validated graph.
    #[must_use]
    pub fn to_v8csr001(&self) -> Vec<u8> {
        let edge_count = self.graph.edge_count();
        let capacity = canonical_len(self.node_count() as u64, edge_count as u64)
            .ok()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&BANC_V888_CSR_MAGIC);
        bytes.extend_from_slice(&(self.node_count() as u64).to_le_bytes());
        bytes.extend_from_slice(&(edge_count as u64).to_le_bytes());
        for node in &self.node_ids
        {
            bytes.extend_from_slice(&node.get().to_le_bytes());
        }

        let mut offset = 0_u64;
        bytes.extend_from_slice(&offset.to_le_bytes());
        let mut edge_cursor = 0_usize;
        for node in 0..self.node_count()
        {
            while edge_cursor < self.graph.edges().len()
                && self.graph.edges()[edge_cursor].source == node
            {
                edge_cursor += 1;
            }
            offset = edge_cursor as u64;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for edge in self.graph.edges()
        {
            bytes.extend_from_slice(&(edge.target as u32).to_le_bytes());
        }
        for edge in self.graph.edges()
        {
            bytes.extend_from_slice(&edge.value.to_le_bytes());
        }
        bytes
    }

    /// Return the raw SHA-256 of the canonical `V8CSR001` bytes as lowercase hex.
    ///
    /// This digest is an integrity identity, not an authenticity signature.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        let digest = Sha256::digest(self.to_v8csr001());
        hex_digest(&digest)
    }

    /// Partition directed pairs and contact multiplicity around a node subset.
    ///
    /// The four pair/contact categories are outside→outside, outside→inside,
    /// inside→outside and inside→inside. This preserves cut-edge accounting
    /// instead of dropping boundary information.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range or duplicate selected node.
    pub fn subset_accounting(
        &self,
        selected_nodes: &[usize],
    ) -> Result<BancV888BoundaryAccounting, BancV888ExecutableGraphError> {
        let selected = selection_mask(self.node_count(), selected_nodes)?;
        let mut result = BancV888BoundaryAccounting::default();

        for edge in self.graph.edges()
        {
            let source_inside = selected[edge.source];
            let target_inside = selected[edge.target];
            let (pairs, contacts) = match (source_inside, target_inside)
            {
                (false, false) => (&mut result.outside_pairs, &mut result.outside_contacts),
                (false, true) => (&mut result.incoming_pairs, &mut result.incoming_contacts),
                (true, false) => (&mut result.outgoing_pairs, &mut result.outgoing_contacts),
                (true, true) => (&mut result.internal_pairs, &mut result.internal_contacts),
            };
            *pairs = pairs
                .checked_add(1)
                .ok_or(BancV888ExecutableGraphError::AccountingOverflow)?;
            *contacts = contacts
                .checked_add(edge.value)
                .ok_or(BancV888ExecutableGraphError::AccountingOverflow)?;
        }
        Ok(result)
    }

    /// Account graph work for a declared active-node set and caller-observed events.
    ///
    /// Directed pairs and contact multiplicities count outgoing adjacency touched
    /// by the active nodes. `events` is copied exactly from the dynamics caller
    /// and is not inferred from static topology.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range or duplicate active node, or if an
    /// exact counter overflows.
    pub fn execution_accounting(
        &self,
        active_nodes: &[usize],
        events: u64,
    ) -> Result<BancV888ExecutionAccounting, BancV888ExecutableGraphError> {
        let selected = selection_mask(self.node_count(), active_nodes)?;
        let mut result = BancV888ExecutionAccounting {
            active_nodes: active_nodes.len() as u64,
            events,
            ..BancV888ExecutionAccounting::default()
        };
        for edge in self.graph.edges()
        {
            if selected[edge.source]
            {
                result.directed_pairs_touched = result
                    .directed_pairs_touched
                    .checked_add(1)
                    .ok_or(BancV888ExecutableGraphError::AccountingOverflow)?;
                result.contact_multiplicity_touched = result
                    .contact_multiplicity_touched
                    .checked_add(edge.value)
                    .ok_or(BancV888ExecutableGraphError::AccountingOverflow)?;
            }
        }
        Ok(result)
    }
}

/// Structural and accounting failures for the executable graph contract.
#[derive(Debug)]
pub enum BancV888ExecutableGraphError {
    Truncated,
    WrongMagic,
    NodeCountOutOfBounds(u64),
    DirectedPairCountOutOfBounds(u64),
    CanonicalLengthOverflow,
    ByteLengthMismatch {
        expected: u64,
        actual: u64,
    },
    HostIndexOverflow,
    NodeIdentity(BancV888Error),
    NodeIdsNotStrictlyIncreasing,
    RowOffsetOutOfBounds {
        offset: u64,
        edge_count: u64,
    },
    InvalidRowOffsets,
    TargetOutOfBounds {
        source: usize,
        target: usize,
        node_count: usize,
    },
    SelfLoop {
        node: usize,
    },
    TargetsNotStrictlyIncreasing {
        source: usize,
    },
    ZeroContactMultiplicity {
        edge_index: usize,
    },
    ContactCountOverflow,
    TrailingBytes,
    DirectedGraph(DirectedGraphError),
    NodeSelectionOutOfBounds {
        node: usize,
        node_count: usize,
    },
    DuplicateNodeSelection(usize),
    AccountingOverflow,
}

impl fmt::Display for BancV888ExecutableGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Truncated => formatter.write_str("truncated V8CSR001 graph"),
            Self::WrongMagic => formatter.write_str("invalid V8CSR001 magic"),
            Self::NodeCountOutOfBounds(value) =>
            {
                write!(formatter, "node count {value} is outside executable bounds")
            },
            Self::DirectedPairCountOutOfBounds(value) =>
            {
                write!(
                    formatter,
                    "directed-pair count {value} is outside executable bounds"
                )
            },
            Self::CanonicalLengthOverflow => formatter.write_str("canonical byte length overflow"),
            Self::ByteLengthMismatch { expected, actual } =>
            {
                write!(
                    formatter,
                    "canonical byte length mismatch: expected {expected}, got {actual}"
                )
            },
            Self::HostIndexOverflow =>
            {
                formatter.write_str("graph dimensions do not fit host indices")
            },
            Self::NodeIdentity(error) => write!(formatter, "invalid BANC node identity: {error}"),
            Self::NodeIdsNotStrictlyIncreasing =>
            {
                formatter.write_str("BANC node IDs are not strictly increasing")
            },
            Self::RowOffsetOutOfBounds { offset, edge_count } =>
            {
                write!(
                    formatter,
                    "row offset {offset} exceeds directed-pair count {edge_count}"
                )
            },
            Self::InvalidRowOffsets => formatter.write_str("invalid V8CSR001 row offsets"),
            Self::TargetOutOfBounds {
                source,
                target,
                node_count,
            } => write!(
                formatter,
                "target {target} from source {source} is outside {node_count} nodes"
            ),
            Self::SelfLoop { node } => write!(formatter, "self loop at node {node} is forbidden"),
            Self::TargetsNotStrictlyIncreasing { source } =>
            {
                write!(
                    formatter,
                    "targets for source {source} are not strictly increasing"
                )
            },
            Self::ZeroContactMultiplicity { edge_index } =>
            {
                write!(formatter, "edge {edge_index} has zero contact multiplicity")
            },
            Self::ContactCountOverflow => formatter.write_str("contact count overflow"),
            Self::TrailingBytes => formatter.write_str("unexpected trailing V8CSR001 bytes"),
            Self::DirectedGraph(error) =>
            {
                write!(formatter, "directed graph validation failed: {error}")
            },
            Self::NodeSelectionOutOfBounds { node, node_count } =>
            {
                write!(
                    formatter,
                    "selected node {node} is outside {node_count} nodes"
                )
            },
            Self::DuplicateNodeSelection(node) =>
            {
                write!(formatter, "selected node {node} occurs more than once")
            },
            Self::AccountingOverflow => formatter.write_str("exact graph accounting overflow"),
        }
    }
}

impl std::error::Error for BancV888ExecutableGraphError {}

fn selection_mask(
    node_count: usize,
    selected_nodes: &[usize],
) -> Result<Vec<bool>, BancV888ExecutableGraphError> {
    let mut selected = vec![false; node_count];
    let mut seen = HashSet::with_capacity(selected_nodes.len());
    for &node in selected_nodes
    {
        if node >= node_count
        {
            return Err(BancV888ExecutableGraphError::NodeSelectionOutOfBounds {
                node,
                node_count,
            });
        }
        if !seen.insert(node)
        {
            return Err(BancV888ExecutableGraphError::DuplicateNodeSelection(node));
        }
        selected[node] = true;
    }
    Ok(selected)
}

fn canonical_len(node_count: u64, edge_count: u64) -> Result<u64, BancV888ExecutableGraphError> {
    let node_bytes = 16_u64
        .checked_mul(node_count)
        .ok_or(BancV888ExecutableGraphError::CanonicalLengthOverflow)?;
    let edge_bytes = 12_u64
        .checked_mul(edge_count)
        .ok_or(BancV888ExecutableGraphError::CanonicalLengthOverflow)?;
    32_u64
        .checked_add(node_bytes)
        .and_then(|value| value.checked_add(edge_bytes))
        .ok_or(BancV888ExecutableGraphError::CanonicalLengthOverflow)
}

fn take<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], BancV888ExecutableGraphError> {
    let end = cursor
        .checked_add(N)
        .ok_or(BancV888ExecutableGraphError::Truncated)?;
    let field = bytes
        .get(*cursor..end)
        .ok_or(BancV888ExecutableGraphError::Truncated)?;
    *cursor = end;
    field
        .try_into()
        .map_err(|_| BancV888ExecutableGraphError::Truncated)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, BancV888ExecutableGraphError> {
    Ok(u32::from_le_bytes(take(bytes, cursor)?))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, BancV888ExecutableGraphError> {
    Ok(u64::from_le_bytes(take(bytes, cursor)?))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &byte in bytes
    {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let mut bytes = BANC_V888_CSR_MAGIC.to_vec();
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        for id in [10_u64, 20, 30]
        {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        for offset in [0_u64, 2, 3, 3]
        {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for target in [1_u32, 2, 2]
        {
            bytes.extend_from_slice(&target.to_le_bytes());
        }
        for weight in [2_u64, 3, 5]
        {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn canonical_round_trip_and_hash_are_deterministic() {
        let bytes = fixture();
        let graph = BancV888ExecutableGraph::from_v8csr001(&bytes).unwrap();
        assert_eq!(graph.to_v8csr001(), bytes);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.directed_pair_count(), 3);
        assert_eq!(graph.contact_count(), 10);
        assert_eq!(
            graph.sha256_hex(),
            "1943623ed4cf01692158e37c5ab96f2103ae1cbb6dd86568a40c4c40dc9c17ea"
        );
    }

    #[test]
    fn subset_accounting_retains_cut_edges_and_contacts() {
        let graph = BancV888ExecutableGraph::from_v8csr001(&fixture()).unwrap();
        let accounting = graph.subset_accounting(&[0, 1]).unwrap();
        assert_eq!(
            accounting,
            BancV888BoundaryAccounting {
                outside_pairs: 0,
                outside_contacts: 0,
                incoming_pairs: 0,
                incoming_contacts: 0,
                outgoing_pairs: 2,
                outgoing_contacts: 8,
                internal_pairs: 1,
                internal_contacts: 2,
            }
        );
    }

    #[test]
    fn execution_accounting_separates_events_from_contact_multiplicity() {
        let graph = BancV888ExecutableGraph::from_v8csr001(&fixture()).unwrap();
        let accounting = graph.execution_accounting(&[0], 17).unwrap();
        assert_eq!(
            accounting,
            BancV888ExecutionAccounting {
                active_nodes: 1,
                directed_pairs_touched: 2,
                contact_multiplicity_touched: 5,
                events: 17,
            }
        );
    }

    #[test]
    fn malformed_identity_offsets_targets_and_weights_fail_closed() {
        let mut bytes = fixture();
        bytes[0] = b'X';
        assert!(matches!(
            BancV888ExecutableGraph::from_v8csr001(&bytes),
            Err(BancV888ExecutableGraphError::WrongMagic)
        ));

        let mut bytes = fixture();
        let second_id = 24 + 8;
        bytes[second_id..second_id + 8].copy_from_slice(&10_u64.to_le_bytes());
        assert!(matches!(
            BancV888ExecutableGraph::from_v8csr001(&bytes),
            Err(BancV888ExecutableGraphError::NodeIdsNotStrictlyIncreasing)
        ));

        let mut bytes = fixture();
        let offsets_start = 24 + 3 * 8;
        bytes[offsets_start + 8..offsets_start + 16].copy_from_slice(&4_u64.to_le_bytes());
        assert!(matches!(
            BancV888ExecutableGraph::from_v8csr001(&bytes),
            Err(BancV888ExecutableGraphError::RowOffsetOutOfBounds { .. })
        ));

        let mut bytes = fixture();
        let targets_start = 24 + 3 * 8 + 4 * 8;
        bytes[targets_start..targets_start + 4].copy_from_slice(&3_u32.to_le_bytes());
        assert!(matches!(
            BancV888ExecutableGraph::from_v8csr001(&bytes),
            Err(BancV888ExecutableGraphError::TargetOutOfBounds { .. })
        ));

        let mut bytes = fixture();
        let weights_start = targets_start + 3 * 4;
        bytes[weights_start..weights_start + 8].copy_from_slice(&0_u64.to_le_bytes());
        assert!(matches!(
            BancV888ExecutableGraph::from_v8csr001(&bytes),
            Err(BancV888ExecutableGraphError::ZeroContactMultiplicity { .. })
        ));
    }

    #[test]
    fn subset_selection_rejects_duplicates_and_out_of_range_indices() {
        let graph = BancV888ExecutableGraph::from_v8csr001(&fixture()).unwrap();
        assert!(matches!(
            graph.subset_accounting(&[1, 1]),
            Err(BancV888ExecutableGraphError::DuplicateNodeSelection(1))
        ));
        assert!(matches!(
            graph.execution_accounting(&[3], 0),
            Err(BancV888ExecutableGraphError::NodeSelectionOutOfBounds {
                node: 3,
                node_count: 3
            })
        ));
    }

    #[test]
    fn official_qualification_keeps_scopes_distinct() {
        let q = BANC_V888_BOOL01_QUALIFICATION;
        assert_eq!(q.metadata_nodes, 188_508);
        assert_eq!(q.directed_pairs, 13_620_865);
        assert_eq!(q.induced_contacts, 42_309_621);
        assert_eq!(q.raw_records, 198_816_365);
        assert_ne!(q.directed_pairs, q.induced_contacts);
        assert_ne!(q.induced_contacts, q.raw_records);
    }
}