//! Pre-tokenization provenance and leakage checks. Not a legal or quality audit.
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::artifact_provenance::artifact_sha256;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Domain {
    Finance {
        available_at: u64,
        decision_at: u64,
        label_end: u64,
    },
    Rust,
}

/// Times are UTC Unix milliseconds; intervals are half-open at split boundaries.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub id: String,
    pub source_uri: String,
    pub revision: String,
    pub license: String,
    /// Reference to a completed rights review, not an automatically verified license.
    pub rights_review: String,
    /// Canonical upstream project (Rust) or event identity (finance), not a fork ID.
    pub group: String,
    pub text: String,
    pub sha256: String,
    pub split: Split,
    pub domain: Domain,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub train_end: u64,
    pub validation_end: u64,
    pub embargo_ms: u64,
    pub records: Vec<Record>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct CheckedRecord {
    pub id: String,
    /// None means the financial label horizon/embargo requires purging this row.
    pub split: Option<Split>,
}

/// Validate caller-declared metadata and exact UTF-8 bytes before tokenization.
/// Reject duplicate contents even within one split. No text normalization,
/// near-duplicate detection, source retrieval or training is performed here.
/// Returns no accepted result when any row is invalid.
pub fn validate(manifest: &Manifest) -> Result<Vec<CheckedRecord>, String> {
    if manifest.schema_version != 1
        || manifest.records.is_empty()
        || manifest.records.len() > 100_000
    {
        return Err("unsupported version or record count".into());
    }
    let validation_start = manifest
        .train_end
        .checked_add(manifest.embargo_ms)
        .ok_or("embargo overflow")?;
    let test_start = manifest
        .validation_end
        .checked_add(manifest.embargo_ms)
        .ok_or("embargo overflow")?;
    if manifest.train_end >= manifest.validation_end || validation_start >= manifest.validation_end
    {
        return Err("invalid chronological boundaries".into());
    }
    let mut ids = HashSet::new();
    let mut hashes = HashSet::new();
    let mut groups = HashMap::new();
    let mut result = Vec::new();
    for record in &manifest.records
    {
        if [
            &record.id,
            &record.source_uri,
            &record.revision,
            &record.license,
            &record.rights_review,
            &record.group,
            &record.text,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
        {
            return Err(format!("{}: missing provenance or content", record.id));
        }
        if record.text.len() > 1_048_576 || !ids.insert(&record.id)
        {
            return Err(format!("{}: duplicate ID or oversized content", record.id));
        }
        let hash = artifact_sha256(record.text.as_bytes());
        if hash.as_str() != record.sha256 || !hashes.insert(record.sha256.clone())
        {
            return Err(format!("{}: hash mismatch or duplicate content", record.id));
        }
        let (domain, split) = match record.domain
        {
            Domain::Rust => ("rust", Some(record.split)),
            Domain::Finance {
                available_at,
                decision_at,
                label_end,
            } =>
            {
                if available_at > decision_at || decision_at > label_end
                {
                    return Err(format!("{}: noncausal time ordering", record.id));
                }
                let split = if decision_at < manifest.train_end && label_end < manifest.train_end
                {
                    Some(Split::Train)
                }
                else if decision_at >= validation_start
                    && decision_at < manifest.validation_end
                    && label_end < manifest.validation_end
                {
                    Some(Split::Validation)
                }
                else if decision_at >= test_start
                {
                    Some(Split::Test)
                }
                else
                {
                    None
                };
                if split.is_some_and(|s| s != record.split)
                {
                    return Err(format!(
                        "{}: declared split disagrees with chronology",
                        record.id
                    ));
                }
                ("finance", split)
            },
        };
        if let Some(split) = split
        {
            if groups
                .insert((domain, &record.group), split)
                .is_some_and(|old| old != split)
            {
                return Err(format!("{}: origin/event crosses splits", record.id));
            }
        }
        result.push(CheckedRecord {
            id: record.id.clone(),
            split,
        });
    }
    Ok(result)
}
