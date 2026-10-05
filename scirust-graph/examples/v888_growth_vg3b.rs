use scirust_graph::v888_executable::{BANC_V888_BOOL01_QUALIFICATION, BancV888ExecutableGraph};
use scirust_graph::v888_growth::{
    V888GrowthSubsetPolicy, v888_growth_matched_controls, v888_growth_select_subset,
    v888_growth_unit_reference,
};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

fn parse_policy(raw: &str) -> Result<V888GrowthSubsetPolicy, String> {
    match raw
    {
        "ranked" => Ok(V888GrowthSubsetPolicy::Ranked),
        "weak_bfs" => Ok(V888GrowthSubsetPolicy::WeakBfs),
        other => Err(format!(
            "unknown subset policy `{other}`; expected ranked|weak_bfs"
        )),
    }
}

fn read_block_groups(
    path: &PathBuf,
    node_ids: &[u64],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut groups = Vec::with_capacity(node_ids.len());
    for (line_number, line) in reader.lines().enumerate()
    {
        let line = line?;
        let mut fields = line.split('\t');
        let id = fields
            .next()
            .ok_or_else(|| format!("blocks.tsv line {}: missing node id", line_number + 1))?
            .parse::<u64>()?;
        let group = fields
            .next()
            .ok_or_else(|| format!("blocks.tsv line {}: missing block id", line_number + 1))?
            .parse::<u32>()?;
        if fields.next().is_some()
        {
            return Err(format!(
                "blocks.tsv line {}: unexpected extra field",
                line_number + 1
            )
            .into());
        }
        if line_number >= node_ids.len() || id != node_ids[line_number]
        {
            return Err(format!(
                "blocks.tsv line {}: node id/order mismatch against qualified graph",
                line_number + 1
            )
            .into());
        }
        groups.push(group);
    }
    if groups.len() != node_ids.len()
    {
        return Err(format!(
            "blocks.tsv has {} rows but qualified graph has {} nodes",
            groups.len(),
            node_ids.len()
        )
        .into());
    }
    Ok(groups)
}

fn write_edges(
    path: &PathBuf,
    edges: &BTreeSet<(usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &(source, target) in edges
    {
        writeln!(writer, "{source}\t{target}")?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let graph_path = PathBuf::from(
        args.next()
            .ok_or("usage: v888_growth_vg3b <graph.csr> <out_dir> <size> <seed> <ranked|weak_bfs> [blocks.tsv]")?,
    );
    let output_dir = PathBuf::from(
        args.next()
            .ok_or("usage: v888_growth_vg3b <graph.csr> <out_dir> <size> <seed> <ranked|weak_bfs> [blocks.tsv]")?,
    );
    let size: usize = args
        .next()
        .ok_or("missing subset size")?
        .parse()
        .map_err(|_| "subset size must be an integer")?;
    let seed: u64 = args
        .next()
        .ok_or("missing seed")?
        .parse()
        .map_err(|_| "seed must be a u64")?;
    let policy = parse_policy(&args.next().ok_or("missing subset policy")?)?;
    let blocks_path = args.next().map(PathBuf::from);
    if args.next().is_some()
    {
        return Err("unexpected extra argument".into());
    }

    let bytes = fs::read(&graph_path)?;
    let graph = BancV888ExecutableGraph::from_v8csr001(&bytes)?;
    let observed_sha = graph.sha256_hex();
    if observed_sha != BANC_V888_BOOL01_QUALIFICATION.graph_sha256
    {
        return Err(format!(
            "qualified graph SHA-256 mismatch: expected {}, got {}",
            BANC_V888_BOOL01_QUALIFICATION.graph_sha256, observed_sha
        )
        .into());
    }

    fs::create_dir_all(&output_dir)?;

    let source_ids: Vec<u64> = graph.node_ids().iter().map(|id| id.get()).collect();
    let source_groups = match blocks_path
    {
        Some(path) => read_block_groups(&path, &source_ids)?,
        None => vec![0_u32; graph.node_count()],
    };

    let (selected, bfs_starts) = v888_growth_select_subset(&graph, size, seed, policy)?;
    let local_groups: Vec<u32> = selected.iter().map(|&index| source_groups[index]).collect();
    let reference = v888_growth_unit_reference(&graph, &selected)?;
    let bundle = v888_growth_matched_controls(&reference, selected.len(), &local_groups, seed)?;

    {
        let nodes_path = output_dir.join("nodes.tsv");
        let file = fs::File::create(&nodes_path)?;
        let mut writer = BufWriter::new(file);
        for (local, &global) in selected.iter().enumerate()
        {
            writeln!(
                writer,
                "{local}\t{}\t{}",
                source_ids[global], local_groups[local]
            )?;
        }
        writer.flush()?;
    }

    write_edges(
        &output_dir.join("reference.edges.tsv"),
        &bundle.reference_edges,
    )?;
    for arm in &bundle.arms
    {
        write_edges(
            &output_dir.join(format!("{}.edges.tsv", arm.arm.name())),
            &arm.edges,
        )?;
    }

    let policy_name = match policy
    {
        V888GrowthSubsetPolicy::Ranked => "ranked",
        V888GrowthSubsetPolicy::WeakBfs => "weak_bfs",
    };
    let arm_json: Vec<String> = bundle
        .arms
        .iter()
        .map(|arm| {
            format!(
                "{{\"arm\":\"{}\",\"seed\":{},\"edge_count\":{},\"attempts\":{},\"accepted_swaps\":{},\"replaced_edges\":{},\"degree_mismatch_nodes\":{},\"reciprocal_mismatch_nodes\":{},\"block_mismatch_cells\":{}}}",
                arm.arm.name(),
                arm.seed,
                arm.edges.len(),
                arm.attempts,
                arm.accepted_swaps,
                arm.replaced_edges,
                arm.degree_mismatch_nodes,
                arm.reciprocal_mismatch_nodes,
                arm.block_mismatch_cells
            )
        })
        .collect();

    let summary = format!(
        "{{\"schema_version\":1,\"programme\":\"V888-GROWTH-VG-3B\",\"source_nodes\":{},\"source_pairs\":{},\"selection\":\"{policy_name}\",\"seed\":{seed},\"nodes\":{},\"pairs\":{},\"bfs_component_starts\":{bfs_starts},\"block_labels\":{},\"block_constraint_vacuous\":{},\"developmental_causality_claim\":false,\"topology_advantage_claim\":false,\"arms\":[{}]}}\n",
        graph.node_count(),
        graph.directed_pair_count(),
        bundle.node_count,
        bundle.reference_edges.len(),
        bundle.distinct_block_labels,
        bundle.block_constraint_vacuous,
        arm_json.join(",")
    );
    fs::write(output_dir.join("summary.json"), &summary)?;

    eprintln!(
        "nodes={} pairs={} arms={} block_labels={} vacuous_block={} bfs_starts={}",
        bundle.node_count,
        bundle.reference_edges.len(),
        bundle.arms.len(),
        bundle.distinct_block_labels,
        bundle.block_constraint_vacuous,
        bfs_starts
    );
    Ok(())
}
