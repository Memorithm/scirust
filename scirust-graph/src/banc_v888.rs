//! Storage-format-independent contract for the FlyWire BANC v888 connectivity snapshot.
//!
//! The contract deliberately performs no network access and does not depend on
//! Arrow/Feather. An external adapter decodes the published artifact and supplies
//! typed rows; this module owns provenance validation, fail-closed row validation,
//! and deterministic canonical ordering.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::str::FromStr;

pub const BANC_V888_CONTRACT: &str = "scirust.banc-v888.edgelist/v1";
pub const BANC_DATASET_ID: &str = "banc";
pub const BANC_MATERIALIZATION: u32 = 888;
pub const BANC_V888_V3_PRODUCT: &str = "banc_888_edgelist_simple_v3.feather";
pub const BANC_V888_V3_SCHEMA: &str =
    "pre:string,post:string,count:int32,norm:double,post_count:int32,pre_count:int32";
pub const BANC_V888_SOURCE_LICENSE: &str = "CC-BY-4.0";
pub const BANC_V888_DATAVERSE_DOI: &str = "10.7910/DVN/7WTH1N";

/// Lossless BANC root identifier.
///
/// Published tables represent root IDs as decimal strings. Serde therefore emits
/// and accepts string values only, avoiding precision loss in JSON consumers that
/// cannot exactly represent every `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BancV888NodeId(u64);

impl BancV888NodeId {
    pub fn new(value: u64) -> Result<Self, BancV888Error> {
        if value == 0
        {
            return Err(BancV888Error::InvalidNodeId("0".to_string()));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for BancV888NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for BancV888NodeId {
    type Err = BancV888Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let parsed = value
            .parse::<u64>()
            .map_err(|_| BancV888Error::InvalidNodeId(value.to_string()))?;
        Self::new(parsed)
    }
}

impl Serialize for BancV888NodeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for BancV888NodeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

/// Provenance envelope for the recommended v3 BANC v888 neuron-pair edgelist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BancV888Manifest {
    contract: String,
    dataset: String,
    materialization: u32,
    source_product: String,
    source_schema: String,
    source_uri: String,
    source_sha256: String,
    source_license: String,
    source_citation_doi: String,
    node_universe: String,
}

impl BancV888Manifest {
    pub fn v3(
        source_uri: impl Into<String>,
        source_sha256: impl Into<String>,
    ) -> Result<Self, BancV888Error> {
        let manifest = Self {
            contract: BANC_V888_CONTRACT.to_string(),
            dataset: BANC_DATASET_ID.to_string(),
            materialization: BANC_MATERIALIZATION,
            source_product: BANC_V888_V3_PRODUCT.to_string(),
            source_schema: BANC_V888_V3_SCHEMA.to_string(),
            source_uri: source_uri.into(),
            source_sha256: source_sha256.into().to_ascii_lowercase(),
            source_license: BANC_V888_SOURCE_LICENSE.to_string(),
            source_citation_doi: BANC_V888_DATAVERSE_DOI.to_string(),
            node_universe: "explicit_list".to_string(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_json(json: &str) -> Result<Self, BancV888Error> {
        let manifest: Self = serde_json::from_str(json)
            .map_err(|err| BancV888Error::InvalidManifestJson(err.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn to_json(&self) -> Result<String, BancV888Error> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|err| BancV888Error::InvalidManifestJson(err.to_string()))
    }

    pub fn validate(&self) -> Result<(), BancV888Error> {
        if self.contract != BANC_V888_CONTRACT
        {
            return Err(BancV888Error::WrongContract(self.contract.clone()));
        }
        if self.dataset != BANC_DATASET_ID
        {
            return Err(BancV888Error::WrongDataset(self.dataset.clone()));
        }
        if self.materialization != BANC_MATERIALIZATION
        {
            return Err(BancV888Error::WrongMaterialization(self.materialization));
        }
        if self.source_product != BANC_V888_V3_PRODUCT
        {
            return Err(BancV888Error::WrongProduct(self.source_product.clone()));
        }
        if self.source_schema != BANC_V888_V3_SCHEMA
        {
            return Err(BancV888Error::WrongSchema(self.source_schema.clone()));
        }
        if self.source_uri.trim().is_empty()
        {
            return Err(BancV888Error::EmptySourceUri);
        }
        if !is_sha256(&self.source_sha256)
        {
            return Err(BancV888Error::InvalidSha256(self.source_sha256.clone()));
        }
        if self.source_license != BANC_V888_SOURCE_LICENSE
        {
            return Err(BancV888Error::WrongLicense(self.source_license.clone()));
        }
        if self.source_citation_doi != BANC_V888_DATAVERSE_DOI
        {
            return Err(BancV888Error::WrongCitationDoi(
                self.source_citation_doi.clone(),
            ));
        }
        if self.node_universe != "explicit_list"
        {
            return Err(BancV888Error::WrongNodeUniverse(
                self.node_universe.clone(),
            ));
        }
        Ok(())
    }

    pub fn source_uri(&self) -> &str {
        &self.source_uri
    }

    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn source_product(&self) -> &str {
        &self.source_product
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// One row of `banc_888_edgelist_simple_v3.feather`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BancV888EdgeRow {
    pub pre: BancV888NodeId,
    pub post: BancV888NodeId,
    pub count: u32,
    pub norm: f64,
    pub post_count: u32,
    pub pre_count: u32,
}

impl BancV888EdgeRow {
    pub fn try_new(
        pre: BancV888NodeId,
        post: BancV888NodeId,
        count: u32,
        norm: f64,
        post_count: u32,
        pre_count: u32,
    ) -> Result<Self, BancV888Error> {
        let row = Self {
            pre,
            post,
            count,
            norm,
            post_count,
            pre_count,
        };
        row.validate()?;
        Ok(row)
    }

    pub fn validate(&self) -> Result<(), BancV888Error> {
        if self.pre == self.post
        {
            return Err(BancV888Error::Autapse(self.pre));
        }
        if self.count == 0
        {
            return Err(BancV888Error::ZeroCount {
                pre: self.pre,
                post: self.post,
            });
        }
        if self.pre_count < self.count || self.post_count < self.count
        {
            return Err(BancV888Error::InconsistentTotals {
                pre: self.pre,
                post: self.post,
                count: self.count,
                pre_count: self.pre_count,
                post_count: self.post_count,
            });
        }
        if !self.norm.is_finite() || !(0.0..=1.0).contains(&self.norm)
        {
            return Err(BancV888Error::InvalidNorm {
                pre: self.pre,
                post: self.post,
                norm: self.norm,
            });
        }
        Ok(())
    }
}

/// Canonical validated input for later directed-graph construction.
#[derive(Debug, Clone, PartialEq)]
pub struct BancV888Dataset {
    manifest: BancV888Manifest,
    nodes: Vec<BancV888NodeId>,
    edges: Vec<BancV888EdgeRow>,
}

impl BancV888Dataset {
    pub fn from_rows(
        manifest: BancV888Manifest,
        mut nodes: Vec<BancV888NodeId>,
        mut edges: Vec<BancV888EdgeRow>,
    ) -> Result<Self, BancV888Error> {
        manifest.validate()?;
        if nodes.is_empty()
        {
            return Err(BancV888Error::EmptyNodeUniverse);
        }

        nodes.sort_unstable();
        for pair in nodes.windows(2)
        {
            if pair[0] == pair[1]
            {
                return Err(BancV888Error::DuplicateNode(pair[0]));
            }
        }

        let known_nodes: HashSet<_> = nodes.iter().copied().collect();
        for edge in &edges
        {
            edge.validate()?;
            if !known_nodes.contains(&edge.pre)
            {
                return Err(BancV888Error::UnknownNode(edge.pre));
            }
            if !known_nodes.contains(&edge.post)
            {
                return Err(BancV888Error::UnknownNode(edge.post));
            }
        }

        edges.sort_by_key(|edge| (edge.pre, edge.post));
        for pair in edges.windows(2)
        {
            if pair[0].pre == pair[1].pre && pair[0].post == pair[1].post
            {
                return Err(BancV888Error::DuplicateEdge {
                    pre: pair[0].pre,
                    post: pair[0].post,
                });
            }
        }

        Ok(Self {
            manifest,
            nodes,
            edges,
        })
    }

    pub fn manifest(&self) -> &BancV888Manifest {
        &self.manifest
    }

    pub fn nodes(&self) -> &[BancV888NodeId] {
        &self.nodes
    }

    pub fn edges(&self) -> &[BancV888EdgeRow] {
        &self.edges
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BancV888Error {
    InvalidNodeId(String),
    InvalidManifestJson(String),
    WrongContract(String),
    WrongDataset(String),
    WrongMaterialization(u32),
    WrongProduct(String),
    WrongSchema(String),
    EmptySourceUri,
    InvalidSha256(String),
    WrongLicense(String),
    WrongCitationDoi(String),
    WrongNodeUniverse(String),
    EmptyNodeUniverse,
    DuplicateNode(BancV888NodeId),
    UnknownNode(BancV888NodeId),
    DuplicateEdge {
        pre: BancV888NodeId,
        post: BancV888NodeId,
    },
    Autapse(BancV888NodeId),
    ZeroCount {
        pre: BancV888NodeId,
        post: BancV888NodeId,
    },
    InconsistentTotals {
        pre: BancV888NodeId,
        post: BancV888NodeId,
        count: u32,
        pre_count: u32,
        post_count: u32,
    },
    InvalidNorm {
        pre: BancV888NodeId,
        post: BancV888NodeId,
        norm: f64,
    },
}

impl fmt::Display for BancV888Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidNodeId(value) => write!(f, "invalid BANC v888 node id: {value}"),
            Self::InvalidManifestJson(message) => {
                write!(f, "invalid BANC v888 manifest JSON: {message}")
            }
            Self::WrongContract(value) => write!(f, "unsupported BANC contract: {value}"),
            Self::WrongDataset(value) => write!(f, "unsupported dataset: {value}"),
            Self::WrongMaterialization(value) => {
                write!(f, "unsupported BANC materialization: {value}")
            }
            Self::WrongProduct(value) => write!(f, "unsupported BANC source product: {value}"),
            Self::WrongSchema(value) => write!(f, "unsupported BANC source schema: {value}"),
            Self::EmptySourceUri => write!(f, "BANC source URI must not be empty"),
            Self::InvalidSha256(value) => write!(f, "invalid BANC source SHA-256: {value}"),
            Self::WrongLicense(value) => write!(f, "unexpected BANC source license: {value}"),
            Self::WrongCitationDoi(value) => write!(f, "unexpected BANC citation DOI: {value}"),
            Self::WrongNodeUniverse(value) => {
                write!(f, "unsupported BANC node-universe policy: {value}")
            }
            Self::EmptyNodeUniverse => write!(f, "BANC node universe must not be empty"),
            Self::DuplicateNode(node) => write!(f, "duplicate BANC node id: {node}"),
            Self::UnknownNode(node) => write!(f, "BANC edge references unknown node: {node}"),
            Self::DuplicateEdge { pre, post } => {
                write!(f, "duplicate BANC edge: {pre} -> {post}")
            }
            Self::Autapse(node) => {
                write!(f, "BANC v3 edgelist must not contain autapse at node {node}")
            }
            Self::ZeroCount { pre, post } => {
                write!(f, "BANC edge {pre} -> {post} has zero synapse count")
            }
            Self::InconsistentTotals {
                pre,
                post,
                count,
                pre_count,
                post_count,
            } => write!(
                f,
                "BANC edge {pre} -> {post} count {count} exceeds pre/post totals ({pre_count}, {post_count})"
            ),
            Self::InvalidNorm { pre, post, norm } => {
                write!(f, "BANC edge {pre} -> {post} has invalid norm {norm}")
            }
        }
    }
}

impl Error for BancV888Error {}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: u64) -> BancV888NodeId {
        BancV888NodeId::new(value).unwrap()
    }

    fn manifest() -> BancV888Manifest {
        BancV888Manifest::v3(
            "https://example.invalid/banc_888_edgelist_simple_v3.feather",
            "ab".repeat(32),
        )
        .unwrap()
    }

    fn edge(pre: u64, post: u64, count: u32) -> BancV888EdgeRow {
        BancV888EdgeRow::try_new(id(pre), id(post), count, 0.25, count * 4, count * 5)
            .unwrap()
    }

    #[test]
    fn node_ids_round_trip_as_decimal_strings() {
        let node = id(720_575_941_521_131_930);
        let json = serde_json::to_string(&node).unwrap();
        assert_eq!(json, "\"720575941521131930\"");
        assert_eq!(
            serde_json::from_str::<BancV888NodeId>(&json).unwrap(),
            node
        );
        assert!(serde_json::from_str::<BancV888NodeId>("720575941521131930").is_err());
    }

    #[test]
    fn manifest_round_trip_is_strict_and_versioned() {
        let manifest = manifest();
        let json = manifest.to_json().unwrap();
        assert_eq!(BancV888Manifest::from_json(&json).unwrap(), manifest);

        let with_unknown = json.replacen('{', "{\"unexpected\":true,", 1);
        assert!(BancV888Manifest::from_json(&with_unknown).is_err());

        let wrong_materialization =
            json.replace("\"materialization\":888", "\"materialization\":626");
        assert!(matches!(
            BancV888Manifest::from_json(&wrong_materialization),
            Err(BancV888Error::WrongMaterialization(626))
        ));
    }

    #[test]
    fn manifest_requires_exact_sha256_and_provenance() {
        assert!(BancV888Manifest::v3("file:///tmp/v888.feather", "abc").is_err());
        let manifest = manifest();
        assert_eq!(manifest.source_product(), BANC_V888_V3_PRODUCT);
        assert_eq!(manifest.source_sha256(), "ab".repeat(32));
    }

    #[test]
    fn dataset_order_is_independent_of_input_order() {
        let first = BancV888Dataset::from_rows(
            manifest(),
            vec![id(30), id(10), id(20)],
            vec![edge(20, 30, 3), edge(10, 20, 2)],
        )
        .unwrap();
        let second = BancV888Dataset::from_rows(
            manifest(),
            vec![id(20), id(30), id(10)],
            vec![edge(10, 20, 2), edge(20, 30, 3)],
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.nodes(), &[id(10), id(20), id(30)]);
        assert_eq!(
            (first.edges()[0].pre, first.edges()[0].post),
            (id(10), id(20))
        );
    }

    #[test]
    fn dataset_rejects_duplicate_nodes_and_unknown_endpoints() {
        assert!(matches!(
            BancV888Dataset::from_rows(manifest(), vec![id(10), id(10)], vec![]),
            Err(BancV888Error::DuplicateNode(node)) if node == id(10)
        ));
        assert!(matches!(
            BancV888Dataset::from_rows(manifest(), vec![id(10)], vec![edge(10, 20, 1)]),
            Err(BancV888Error::UnknownNode(node)) if node == id(20)
        ));
    }

    #[test]
    fn dataset_rejects_duplicate_directed_edges() {
        assert!(matches!(
            BancV888Dataset::from_rows(
                manifest(),
                vec![id(10), id(20)],
                vec![edge(10, 20, 1), edge(10, 20, 2)],
            ),
            Err(BancV888Error::DuplicateEdge { pre, post })
                if pre == id(10) && post == id(20)
        ));
    }

    #[test]
    fn edge_validation_matches_v3_boundaries() {
        assert!(matches!(
            BancV888EdgeRow::try_new(id(10), id(10), 1, 1.0, 1, 1),
            Err(BancV888Error::Autapse(node)) if node == id(10)
        ));
        assert!(matches!(
            BancV888EdgeRow::try_new(id(10), id(20), 0, 0.0, 1, 1),
            Err(BancV888Error::ZeroCount { .. })
        ));
        assert!(matches!(
            BancV888EdgeRow::try_new(id(10), id(20), 4, 0.5, 3, 4),
            Err(BancV888Error::InconsistentTotals { .. })
        ));
        assert!(matches!(
            BancV888EdgeRow::try_new(id(10), id(20), 1, f64::NAN, 2, 2),
            Err(BancV888Error::InvalidNorm { .. })
        ));
        assert!(matches!(
            BancV888EdgeRow::try_new(id(10), id(20), 1, 1.01, 2, 2),
            Err(BancV888Error::InvalidNorm { .. })
        ));
    }
}
