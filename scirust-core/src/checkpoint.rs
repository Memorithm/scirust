//! Training checkpoint persistence for model parameters and optimizer state.
//!
//! A checkpoint directory contains one `checkpoint.json` file. The file embeds
//! model weights, optimizer state, epoch/step counters, the SciRust crate version,
//! and free-form metadata. The writer does **not** create a separate
//! `weights.json` file.
//!
//! Checkpoint JSON is a persistence representation of the current Rust data
//! structures; this module does not currently enforce a checkpoint schema
//! version or reject a checkpoint solely because `scirust_version` differs from
//! the running crate version.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Optimizer state stored in a training checkpoint.
///
/// The moment maps use parameter names as keys and flattened `f32` buffers as
/// values. This type records the state supplied by the caller; it does not
/// validate that moment-buffer lengths match the corresponding model weights.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizerState {
    /// Learning rate at checkpoint time.
    pub learning_rate: f32,
    /// Optimizer/global training step recorded in the optimizer state.
    pub step: usize,
    /// Epoch recorded in the optimizer state.
    pub epoch: usize,
    /// Adam first-moment buffers, keyed by parameter name.
    pub m: HashMap<String, Vec<f32>>,
    /// Adam second-moment buffers, keyed by parameter name.
    pub v: HashMap<String, Vec<f32>>,
    /// `beta1^t` value used for Adam bias correction.
    pub beta1_t: f32,
    /// `beta2^t` value used for Adam bias correction.
    pub beta2_t: f32,
}

/// Deserialized contents of one SciRust training checkpoint.
///
/// `epoch` and `step` at the top level are stored independently from the same
/// named fields in [`OptimizerState`]; this module does not require them to be
/// equal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// SciRust crate version recorded by the writer.
    pub scirust_version: String,
    /// Top-level epoch supplied to [`save_checkpoint`].
    pub epoch: usize,
    /// Top-level global step supplied to [`save_checkpoint`].
    pub step: usize,
    /// Serialized optimizer state.
    pub optimizer_state: OptimizerState,
    /// Model parameters keyed by name and stored as flattened `f32` buffers.
    pub weights: HashMap<String, Vec<f32>>,
    /// Free-form key/value metadata.
    pub metadata: HashMap<String, String>,
}

/// Saves model and optimizer state in `{dir}/checkpoint.json`.
///
/// The directory is created recursively when necessary. Existing
/// `checkpoint.json` content is replaced. The write uses [`std::fs::write`]; it
/// is not an atomic rename/transaction and this function does not fsync the file
/// or parent directory.
///
/// The stored [`Checkpoint::metadata`] map is empty. The top-level `epoch` and
/// `step` arguments are stored independently from the fields already present in
/// `opt_state`.
///
/// # Errors
///
/// Returns [`crate::error::SciRustError::IoError`] when directory creation or
/// file writing fails. JSON serialization errors are converted through the
/// crate error type.
///
/// # Examples
///
/// Save and load a small deterministic checkpoint:
///
/// ```
/// use scirust_core::checkpoint::{load_checkpoint, save_checkpoint, OptimizerState};
/// use std::collections::HashMap;
///
/// let dir = std::env::temp_dir().join(format!("scirust-doc-checkpoint-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// let weights = HashMap::from([("w".to_owned(), vec![1.0_f32, 2.0])]);
/// let state = OptimizerState {
///     learning_rate: 0.01,
///     step: 7,
///     epoch: 2,
///     m: HashMap::new(),
///     v: HashMap::new(),
///     beta1_t: 0.9,
///     beta2_t: 0.999,
/// };
/// save_checkpoint(&dir, &weights, &state, 2, 7).unwrap();
/// let loaded = load_checkpoint(&dir).unwrap();
/// assert_eq!(loaded.weights["w"], vec![1.0, 2.0]);
/// assert_eq!(loaded.epoch, 2);
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
///
/// The top-level counters are independent of the counters inside the optimizer
/// snapshot:
///
/// ```
/// use scirust_core::checkpoint::{load_checkpoint, save_checkpoint, OptimizerState};
/// use std::collections::HashMap;
///
/// let dir = std::env::temp_dir().join(format!("scirust-doc-checkpoint-counters-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// let state = OptimizerState {
///     learning_rate: 0.1, step: 3, epoch: 1,
///     m: HashMap::new(), v: HashMap::new(), beta1_t: 0.9, beta2_t: 0.999,
/// };
/// save_checkpoint(&dir, &HashMap::new(), &state, 9, 99).unwrap();
/// let loaded = load_checkpoint(&dir).unwrap();
/// assert_eq!((loaded.epoch, loaded.step), (9, 99));
/// assert_eq!((loaded.optimizer_state.epoch, loaded.optimizer_state.step), (1, 3));
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
pub fn save_checkpoint(
    dir: impl AsRef<Path>,
    weights: &HashMap<String, Vec<f32>>,
    opt_state: &OptimizerState,
    epoch: usize,
    step: usize,
) -> crate::error::Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let checkpoint = Checkpoint {
        scirust_version: env!("CARGO_PKG_VERSION").to_string(),
        epoch,
        step,
        optimizer_state: opt_state.clone(),
        weights: weights.clone(),
        metadata: HashMap::new(),
    };

    let json = serde_json::to_string_pretty(&checkpoint)?;
    fs::write(dir.join("checkpoint.json"), json)?;
    Ok(())
}

/// Loads and deserializes `{dir}/checkpoint.json`.
///
/// The stored `scirust_version` is returned as data and is not currently used as
/// a compatibility gate.
///
/// # Errors
///
/// A missing or unreadable file is returned as
/// [`crate::error::SciRustError::IoError`]. Malformed or incompatible JSON is
/// returned as [`crate::error::SciRustError::InvalidFormat`] through the crate's
/// `serde_json::Error` conversion.
///
/// # Examples
///
/// ```
/// use scirust_core::checkpoint::{load_checkpoint, save_checkpoint, OptimizerState};
/// use std::collections::HashMap;
/// let dir = std::env::temp_dir().join(format!("scirust-doc-load-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// let state = OptimizerState {
///     learning_rate: 0.01, step: 1, epoch: 1,
///     m: HashMap::new(), v: HashMap::new(), beta1_t: 0.9, beta2_t: 0.999,
/// };
/// save_checkpoint(&dir, &HashMap::new(), &state, 1, 1).unwrap();
/// assert_eq!(load_checkpoint(&dir).unwrap().step, 1);
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
///
/// Malformed JSON is rejected as a format error:
///
/// ```
/// use scirust_core::checkpoint::load_checkpoint;
/// use scirust_core::error::SciRustError;
/// let dir = std::env::temp_dir().join(format!("scirust-doc-load-bad-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// std::fs::create_dir_all(&dir).unwrap();
/// std::fs::write(dir.join("checkpoint.json"), "{not-json]").unwrap();
/// let result = load_checkpoint(&dir);
/// assert!(matches!(result, Err(SciRustError::InvalidFormat { .. })));
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
pub fn load_checkpoint(dir: impl AsRef<Path>) -> crate::error::Result<Checkpoint> {
    let dir = dir.as_ref();
    let json = fs::read_to_string(dir.join("checkpoint.json"))?;
    let checkpoint: Checkpoint = serde_json::from_str(&json)?;
    Ok(checkpoint)
}

/// Lists readable checkpoint subdirectories, sorted by stored epoch.
///
/// Only direct child directories containing a readable, valid
/// `checkpoint.json` are returned. Missing parent directories yield an empty
/// list. Child checkpoints whose JSON cannot be read or deserialized are
/// silently skipped. Equal epochs are not otherwise ordered by this contract.
///
/// # Errors
///
/// Returns [`crate::error::SciRustError::IoError`] if reading the parent
/// directory or one of its directory entries fails. Errors while reading or
/// parsing an individual `checkpoint.json` are deliberately skipped.
///
/// # Examples
///
/// ```
/// use scirust_core::checkpoint::{list_checkpoints, save_checkpoint, OptimizerState};
/// use std::collections::HashMap;
/// let parent = std::env::temp_dir().join(format!("scirust-doc-list-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&parent);
/// let state = OptimizerState {
///     learning_rate: 0.01, step: 0, epoch: 0,
///     m: HashMap::new(), v: HashMap::new(), beta1_t: 0.9, beta2_t: 0.999,
/// };
/// save_checkpoint(parent.join("third"), &HashMap::new(), &state, 3, 30).unwrap();
/// save_checkpoint(parent.join("first"), &HashMap::new(), &state, 1, 10).unwrap();
/// let epochs: Vec<_> = list_checkpoints(&parent).unwrap().into_iter().map(|x| x.0).collect();
/// assert_eq!(epochs, vec![1, 3]);
/// let _ = std::fs::remove_dir_all(&parent);
/// ```
///
/// A missing parent is an empty collection, not an error:
///
/// ```
/// use scirust_core::checkpoint::list_checkpoints;
/// let parent = std::env::temp_dir().join(format!("scirust-doc-list-missing-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&parent);
/// assert!(list_checkpoints(&parent).unwrap().is_empty());
/// ```
pub fn list_checkpoints(
    parent_dir: impl AsRef<Path>,
) -> crate::error::Result<Vec<(usize, std::path::PathBuf)>> {
    let parent = parent_dir.as_ref();
    let mut checkpoints = Vec::new();

    if !parent.exists()
    {
        return Ok(checkpoints);
    }

    for entry in fs::read_dir(parent)?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir()
        {
            let ckpt_file = path.join("checkpoint.json");
            if ckpt_file.exists()
                && let Ok(json) = fs::read_to_string(&ckpt_file)
                && let Ok(ckpt) = serde_json::from_str::<Checkpoint>(&json)
            {
                checkpoints.push((ckpt.epoch, path));
            }
        }
    }

    checkpoints.sort_by_key(|(epoch, _)| *epoch);
    Ok(checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("scirust_ckpt_test_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);

        let mut weights = HashMap::new();
        weights.insert("fc1.weight".into(), vec![0.1, 0.2, 0.3]);
        weights.insert("fc1.bias".into(), vec![0.01, 0.02]);

        let opt_state = OptimizerState {
            learning_rate: 0.001,
            step: 100,
            epoch: 5,
            m: HashMap::new(),
            v: HashMap::new(),
            beta1_t: 0.9f32.powi(100),
            beta2_t: 0.999f32.powi(100),
        };

        save_checkpoint(&dir, &weights, &opt_state, 5, 100).unwrap();

        let loaded = load_checkpoint(&dir).unwrap();
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.optimizer_state.learning_rate, 0.001);
        assert_eq!(
            loaded.weights.get("fc1.weight").unwrap(),
            &vec![0.1, 0.2, 0.3]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_malformed_json_as_invalid_format() {
        let dir = std::env::temp_dir().join("scirust_ckpt_test_badjson");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("checkpoint.json"), b"{ not valid json ]").unwrap();

        let err = load_checkpoint(&dir).unwrap_err();
        assert_eq!(err.code(), "E_FORMAT");
        assert!(matches!(
            err,
            crate::error::SciRustError::InvalidFormat { .. }
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_list_checkpoints() {
        let parent = std::env::temp_dir().join("scirust_ckpt_test_list");
        let _ = std::fs::remove_dir_all(&parent);

        let mut weights = HashMap::new();
        weights.insert("w".into(), vec![1.0]);

        let opt = OptimizerState {
            learning_rate: 0.01,
            step: 0,
            epoch: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            beta1_t: 0.9,
            beta2_t: 0.999,
        };

        save_checkpoint(parent.join("ckpt_epoch1"), &weights, &opt, 1, 10).unwrap();
        save_checkpoint(parent.join("ckpt_epoch3"), &weights, &opt, 3, 30).unwrap();
        save_checkpoint(parent.join("ckpt_epoch2"), &weights, &opt, 2, 20).unwrap();

        let list = list_checkpoints(&parent).unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].0, 1);
        assert_eq!(list[1].0, 2);
        assert_eq!(list[2].0, 3);

        let _ = std::fs::remove_dir_all(&parent);
    }
}
