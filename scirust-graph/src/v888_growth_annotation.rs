//! Deterministic caller-annotation joins for V888-GROWTH VG-3D exporters.
//!
//! VG-3D scores how strongly unit edges follow an external node annotation
//! (for example hemilineage or an anatomical block) on the reference and the
//! VG-3B control arms. The annotation itself lives outside Git (Thor-side
//! tables); this module only owns the reusable, fail-closed join contract:
//!
//! - [`v888_growth_parse_annotation_tsv`] reads `node_id<TAB>label` rows
//!   (optional `#` comments and blank lines), rejecting malformed ids, empty
//!   labels, extra fields and duplicate ids;
//! - [`v888_growth_join_annotation`] maps a selected node-id list onto dense
//!   label indices, numbering distinct labels in lexicographic order and
//!   placing every unannotated node into one explicit trailing
//!   [`V888_GROWTH_UNLABELLED`] bucket, so the dense labels are identical on
//!   every platform and independent of file row order.
//!
//! Labels are opaque strings. Nothing here interprets them biologically or
//! makes any developmental-causality claim.

use core::fmt;
use std::collections::BTreeMap;
use std::io::BufRead;

/// Name of the explicit bucket holding selected nodes absent from the annotation.
pub const V888_GROWTH_UNLABELLED: &str = "<unlabelled>";

/// Parsed `node_id -> label` annotation table.
pub type V888GrowthAnnotation = BTreeMap<u64, String>;

/// Dense annotation labels for a selected node list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V888GrowthAnnotationJoin {
    /// One dense label index per selected node, in selection order.
    pub labels: Vec<usize>,
    /// Label name for each dense index (lexicographic, then the unlabelled
    /// bucket last when present).
    pub names: Vec<String>,
    /// Number of selected nodes carried by each dense index.
    pub sizes: Vec<usize>,
    /// Selected nodes that had an annotation row.
    pub labelled_nodes: usize,
    /// Selected nodes placed in the [`V888_GROWTH_UNLABELLED`] bucket.
    pub unlabelled_nodes: usize,
}

impl V888GrowthAnnotationJoin {
    /// Dense index of the unlabelled bucket, when any node fell into it.
    pub fn unlabelled_index(&self) -> Option<usize> {
        (self.unlabelled_nodes > 0).then(|| self.names.len() - 1)
    }
}

/// Fail-closed errors from annotation parsing and joins.
#[derive(Debug)]
pub enum V888GrowthAnnotationError {
    /// Underlying reader failed.
    Io(std::io::Error),
    /// A row is malformed.
    MalformedRow {
        /// 1-based line number.
        line: usize,
        /// What was wrong.
        reason: &'static str,
    },
    /// The same node id appeared twice.
    DuplicateNode {
        /// 1-based line number of the repeat.
        line: usize,
        /// Repeated node id.
        node_id: u64,
    },
    /// A label collides with the reserved unlabelled bucket name.
    ReservedLabel {
        /// 1-based line number.
        line: usize,
    },
    /// The selected node list is empty.
    EmptySelection,
    /// The selected node list repeats a node id.
    DuplicateSelection {
        /// Repeated node id.
        node_id: u64,
    },
}

impl fmt::Display for V888GrowthAnnotationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Io(error) => write!(formatter, "annotation read failed: {error}"),
            Self::MalformedRow { line, reason } =>
            {
                write!(formatter, "annotation line {line}: {reason}")
            },
            Self::DuplicateNode { line, node_id } => write!(
                formatter,
                "annotation line {line}: node id {node_id} already annotated"
            ),
            Self::ReservedLabel { line } => write!(
                formatter,
                "annotation line {line}: label `{V888_GROWTH_UNLABELLED}` is reserved"
            ),
            Self::EmptySelection => formatter.write_str("annotation join needs at least one node"),
            Self::DuplicateSelection { node_id } =>
            {
                write!(formatter, "selected node id {node_id} appears twice")
            },
        }
    }
}

impl std::error::Error for V888GrowthAnnotationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for V888GrowthAnnotationError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Parse a `node_id<TAB>label` annotation table.
///
/// Blank lines and lines starting with `#` are skipped. Labels are trimmed of
/// surrounding whitespace and must be non-empty.
///
/// ```
/// use scirust_graph::v888_growth::v888_growth_parse_annotation_tsv;
///
/// let table = "# hemilineage\n10\tLB23\n7\tDM1\n";
/// let annotation = v888_growth_parse_annotation_tsv(table.as_bytes()).unwrap();
/// assert_eq!(annotation.len(), 2);
/// assert_eq!(annotation[&7], "DM1");
/// ```
pub fn v888_growth_parse_annotation_tsv(
    reader: impl BufRead,
) -> Result<V888GrowthAnnotation, V888GrowthAnnotationError> {
    let mut annotation = V888GrowthAnnotation::new();
    for (index, line) in reader.lines().enumerate()
    {
        let line_number = index + 1;
        let line = line?;
        let trimmed = line.trim_end_matches('\r');
        if trimmed.trim().is_empty() || trimmed.starts_with('#')
        {
            continue;
        }
        let mut fields = trimmed.split('\t');
        let raw_id = fields.next().unwrap_or_default().trim();
        let node_id =
            raw_id
                .parse::<u64>()
                .map_err(|_| V888GrowthAnnotationError::MalformedRow {
                    line: line_number,
                    reason: "node id is not a u64",
                })?;
        let label = fields
            .next()
            .map(str::trim)
            .filter(|label| !label.is_empty())
            .ok_or(V888GrowthAnnotationError::MalformedRow {
                line: line_number,
                reason: "missing or empty label",
            })?;
        if fields.next().is_some()
        {
            return Err(V888GrowthAnnotationError::MalformedRow {
                line: line_number,
                reason: "unexpected extra field",
            });
        }
        if label == V888_GROWTH_UNLABELLED
        {
            return Err(V888GrowthAnnotationError::ReservedLabel { line: line_number });
        }
        if annotation.insert(node_id, label.to_string()).is_some()
        {
            return Err(V888GrowthAnnotationError::DuplicateNode {
                line: line_number,
                node_id,
            });
        }
    }
    Ok(annotation)
}

/// Join selected node ids onto dense annotation labels.
///
/// Distinct labels present among the selected nodes are numbered in
/// lexicographic order; nodes without an annotation row share one trailing
/// [`V888_GROWTH_UNLABELLED`] bucket. Annotation rows for unselected nodes are
/// ignored.
///
/// ```
/// use scirust_graph::v888_growth::{
///     V888_GROWTH_UNLABELLED, v888_growth_join_annotation, v888_growth_parse_annotation_tsv,
/// };
///
/// let annotation = v888_growth_parse_annotation_tsv("1\tB\n2\tA\n3\tB\n".as_bytes()).unwrap();
/// let join = v888_growth_join_annotation(&[3, 9, 2, 1], &annotation).unwrap();
/// assert_eq!(join.labels, vec![1, 2, 0, 1]);
/// assert_eq!(join.names, vec!["A", "B", V888_GROWTH_UNLABELLED]);
/// assert_eq!(join.unlabelled_nodes, 1);
/// ```
pub fn v888_growth_join_annotation(
    selected_ids: &[u64],
    annotation: &V888GrowthAnnotation,
) -> Result<V888GrowthAnnotationJoin, V888GrowthAnnotationError> {
    if selected_ids.is_empty()
    {
        return Err(V888GrowthAnnotationError::EmptySelection);
    }
    let mut seen = std::collections::BTreeSet::new();
    for &node_id in selected_ids
    {
        if !seen.insert(node_id)
        {
            return Err(V888GrowthAnnotationError::DuplicateSelection { node_id });
        }
    }
    let mut dense: BTreeMap<&str, usize> = selected_ids
        .iter()
        .filter_map(|id| annotation.get(id).map(String::as_str))
        .map(|label| (label, 0))
        .collect();
    let mut names = Vec::with_capacity(dense.len() + 1);
    for (index, (label, slot)) in dense.iter_mut().enumerate()
    {
        *slot = index;
        names.push((*label).to_string());
    }
    let unlabelled_index = names.len();
    let mut labels = Vec::with_capacity(selected_ids.len());
    let mut unlabelled_nodes = 0_usize;
    for id in selected_ids
    {
        match annotation.get(id)
        {
            Some(label) => labels.push(dense[label.as_str()]),
            None =>
            {
                unlabelled_nodes += 1;
                labels.push(unlabelled_index);
            },
        }
    }
    if unlabelled_nodes > 0
    {
        names.push(V888_GROWTH_UNLABELLED.to_string());
    }
    let mut sizes = vec![0_usize; names.len()];
    for &label in &labels
    {
        sizes[label] += 1;
    }
    Ok(V888GrowthAnnotationJoin {
        labels,
        names,
        sizes,
        labelled_nodes: selected_ids.len() - unlabelled_nodes,
        unlabelled_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_comments_blank_lines_and_crlf() {
        let table = "# header\n\n5\tLB7\r\n6\t  DM2 \n";
        let annotation = v888_growth_parse_annotation_tsv(table.as_bytes()).unwrap();
        assert_eq!(annotation.len(), 2);
        assert_eq!(annotation[&5], "LB7");
        assert_eq!(annotation[&6], "DM2");
    }

    #[test]
    fn parse_rejects_malformed_rows() {
        for (table, expected_line) in [
            ("x\tA\n", 1),
            ("1\n", 1),
            ("1\t \n", 1),
            ("1\tA\tB\n", 1),
            ("1\tA\n-2\tB\n", 2),
        ]
        {
            match v888_growth_parse_annotation_tsv(table.as_bytes())
            {
                Err(V888GrowthAnnotationError::MalformedRow { line, .. }) =>
                {
                    assert_eq!(line, expected_line, "{table:?}")
                },
                other => panic!("expected malformed row for {table:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_rejects_duplicates_and_reserved_label() {
        assert!(matches!(
            v888_growth_parse_annotation_tsv("1\tA\n1\tA\n".as_bytes()),
            Err(V888GrowthAnnotationError::DuplicateNode {
                line: 2,
                node_id: 1
            })
        ));
        let reserved = format!("1\t{V888_GROWTH_UNLABELLED}\n");
        assert!(matches!(
            v888_growth_parse_annotation_tsv(reserved.as_bytes()),
            Err(V888GrowthAnnotationError::ReservedLabel { line: 1 })
        ));
    }

    #[test]
    fn join_is_independent_of_row_order() {
        let forward = v888_growth_parse_annotation_tsv("1\tZ\n2\tA\n3\tM\n".as_bytes()).unwrap();
        let reverse = v888_growth_parse_annotation_tsv("3\tM\n2\tA\n1\tZ\n".as_bytes()).unwrap();
        let selected = [1, 2, 3];
        let left = v888_growth_join_annotation(&selected, &forward).unwrap();
        let right = v888_growth_join_annotation(&selected, &reverse).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.labels, vec![2, 0, 1]);
        assert_eq!(left.unlabelled_index(), None);
        assert_eq!(left.sizes, vec![1, 1, 1]);
    }

    #[test]
    fn join_ignores_unselected_rows_and_buckets_missing_nodes() {
        let annotation =
            v888_growth_parse_annotation_tsv("1\tA\n2\tA\n100\tUNUSED\n".as_bytes()).unwrap();
        let join = v888_growth_join_annotation(&[2, 5, 1, 6], &annotation).unwrap();
        assert_eq!(join.names, vec!["A", V888_GROWTH_UNLABELLED]);
        assert_eq!(join.labels, vec![0, 1, 0, 1]);
        assert_eq!(join.sizes, vec![2, 2]);
        assert_eq!(join.labelled_nodes, 2);
        assert_eq!(join.unlabelled_nodes, 2);
        assert_eq!(join.unlabelled_index(), Some(1));
    }

    #[test]
    fn join_with_empty_annotation_is_single_bucket() {
        let join = v888_growth_join_annotation(&[4, 8], &V888GrowthAnnotation::new()).unwrap();
        assert_eq!(join.labels, vec![0, 0]);
        assert_eq!(join.names, vec![V888_GROWTH_UNLABELLED]);
    }

    #[test]
    fn join_rejects_empty_and_duplicate_selection() {
        let annotation = V888GrowthAnnotation::new();
        assert!(matches!(
            v888_growth_join_annotation(&[], &annotation),
            Err(V888GrowthAnnotationError::EmptySelection)
        ));
        assert!(matches!(
            v888_growth_join_annotation(&[3, 3], &annotation),
            Err(V888GrowthAnnotationError::DuplicateSelection { node_id: 3 })
        ));
    }
}
