//! Read-only bounded manifest validator. Never downloads, writes shards or trains.
use std::fs::File;
use std::io::Read;

use scirust_sciagent::artifact_provenance::artifact_sha256;
use scirust_sciagent::train::corpus_manifest::{Manifest, validate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: sciagent-corpus-check MANIFEST.json")?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 32 * 1024 * 1024
    {
        return Err("manifest exceeds 32 MiB".into());
    }
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    let records = validate(&manifest).map_err(std::io::Error::other)?;
    println!(
        "{}",
        serde_json::json!({"schema_version":1,
        "manifest_sha256":artifact_sha256(&bytes).as_str(),
        "records":records,"rights_verified":false,"training_performed":false})
    );
    Ok(())
}
