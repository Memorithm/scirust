use scirust_graph::v888_executable::{
    BancV888ExecutableGraph, BANC_V888_BOOL01_QUALIFICATION,
};
use scirust_graph::v888_growth::v888_growth_topology_profile;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let graph_path = PathBuf::from(args.next().ok_or(
        "usage: v888_growth_vg3a <qualified-graph.csr> <output.csv>",
    )?);
    let output_path = PathBuf::from(args.next().ok_or(
        "usage: v888_growth_vg3a <qualified-graph.csr> <output.csv>",
    )?);
    if args.next().is_some() {
        return Err("unexpected extra argument".into());
    }

    let bytes = fs::read(&graph_path)?;
    let graph = BancV888ExecutableGraph::from_v8csr001(&bytes)?;
    let observed_sha = graph.sha256_hex();
    if observed_sha != BANC_V888_BOOL01_QUALIFICATION.graph_sha256 {
        return Err(format!(
            "qualified graph SHA-256 mismatch: expected {}, got {}",
            BANC_V888_BOOL01_QUALIFICATION.graph_sha256, observed_sha
        )
        .into());
    }

    let profile = v888_growth_topology_profile(&graph)?;
    let file = fs::File::create(&output_path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "node_id,in_degree,out_degree,incoming_contacts,outgoing_contacts,reciprocal_neighbors,scc_label,scc_size"
    )?;
    for row in &profile.rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{}",
            row.node_id,
            row.in_degree,
            row.out_degree,
            row.incoming_contacts,
            row.outgoing_contacts,
            row.reciprocal_neighbors,
            row.scc_label,
            row.scc_size
        )?;
    }
    writer.flush()?;

    eprintln!(
        "rows={} directed_pairs={} contacts={} reciprocal_pairs={} scc_count={} largest_scc={}",
        profile.rows.len(),
        profile.directed_pairs,
        profile.contacts,
        profile.reciprocal_pairs,
        profile.scc_count,
        profile.largest_scc
    );
    Ok(())
}
