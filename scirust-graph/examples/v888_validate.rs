use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use scirust_graph::v888_executable::{
    BANC_V888_BOOL01_QUALIFICATION, BancV888ExecutableGraph,
};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args_os()
        .nth(1)
        .ok_or("usage: v888_validate <graph.csr>")?;
    let path = Path::new(&path);
    let metadata = fs::metadata(path)?;

    let expected = BANC_V888_BOOL01_QUALIFICATION;
    if metadata.len() != expected.graph_bytes {
        return Err(format!(
            "graph byte length mismatch: expected {}, got {}",
            expected.graph_bytes,
            metadata.len()
        )
        .into());
    }

    let bytes = fs::read(path)?;
    let graph = BancV888ExecutableGraph::from_v8csr001(&bytes)?;
    if graph.node_count() as u64 != expected.metadata_nodes {
        return Err(format!(
            "node count mismatch: expected {}, got {}",
            expected.metadata_nodes,
            graph.node_count()
        )
        .into());
    }
    if graph.directed_pair_count() as u64 != expected.directed_pairs {
        return Err(format!(
            "directed-pair count mismatch: expected {}, got {}",
            expected.directed_pairs,
            graph.directed_pair_count()
        )
        .into());
    }
    if graph.contact_count() != expected.induced_contacts {
        return Err(format!(
            "contact count mismatch: expected {}, got {}",
            expected.induced_contacts,
            graph.contact_count()
        )
        .into());
    }

    let digest = graph.sha256_hex();
    if digest != expected.graph_sha256 {
        return Err(format!(
            "graph SHA-256 mismatch: expected {}, got {}",
            expected.graph_sha256, digest
        )
        .into());
    }
    if graph.to_v8csr001() != bytes {
        return Err("canonical V8CSR001 replay changed bytes".into());
    }

    println!("contract={}", expected.contract);
    println!("programme_revision={}", expected.programme_revision);
    println!("producer_commit={}", expected.producer_commit);
    println!("graph_sha256={digest}");
    println!("nodes={}", graph.node_count());
    println!("directed_pairs={}", graph.directed_pair_count());
    println!("contacts={}", graph.contact_count());
    println!("status=qualified_exact_graph_identity");
    Ok(())
}
