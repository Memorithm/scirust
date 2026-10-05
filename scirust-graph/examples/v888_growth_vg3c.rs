use scirust_graph::v888_executable::{BANC_V888_BOOL01_QUALIFICATION, BancV888ExecutableGraph};
use scirust_graph::v888_growth::{
    V888GrowthMetricBundle, V888GrowthSubsetPolicy, v888_growth_compare_metric_arms,
    v888_growth_matched_controls, v888_growth_metric_bundle, v888_growth_select_subset,
    v888_growth_unit_reference,
};
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

fn write_tsv_row(writer: &mut impl Write, fields: &[String]) -> std::io::Result<()> {
    writeln!(writer, "{}", fields.join("\t"))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        0.0
    }
    else
    {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn emit_arm_row(
    writer: &mut impl Write,
    metrics: &V888GrowthMetricBundle,
    edge_count: usize,
) -> std::io::Result<()> {
    write_tsv_row(
        writer,
        &[
            metrics.arm.clone(),
            metrics.reachability.node_count.to_string(),
            edge_count.to_string(),
            metrics.reachability.reachable_ordered_pairs.to_string(),
            metrics.reachability.ordered_pair_universe.to_string(),
            metrics.reachability.median_out_reach.to_string(),
            format!("{:.9}", mean(&metrics.harmonic.out_harmonic)),
            format!("{:.9}", mean(&metrics.harmonic.in_harmonic)),
            format!("{:.9}", mean(&metrics.betweenness.normalized)),
            metrics.rich_club_total.points.len().to_string(),
            metrics.rich_club_out.points.len().to_string(),
        ],
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let usage = "usage: v888_growth_vg3c <graph.csr> <out_dir> <size> <seed> <ranked|weak_bfs> [blocks.tsv]";
    let graph_path = PathBuf::from(args.next().ok_or(usage)?);
    let output_dir = PathBuf::from(args.next().ok_or(usage)?);
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
    if size > 512
    {
        return Err("VG-3C pilot exporter rejects size > 512 (Brandes scale bound)".into());
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
    let control_bundle =
        v888_growth_matched_controls(&reference, selected.len(), &local_groups, seed)?;

    let controls: Vec<(&str, _)> = control_bundle
        .arms
        .iter()
        .map(|arm| (arm.arm.name(), &arm.edges))
        .collect();
    let (reference_metrics, deltas) = v888_growth_compare_metric_arms(
        selected.len(),
        &control_bundle.reference_edges,
        &controls,
    )?;

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

    {
        let path = output_dir.join("metrics_per_arm.tsv");
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_tsv_row(
            &mut writer,
            &[
                "arm".into(),
                "nodes".into(),
                "edges".into(),
                "reachable_ordered_pairs".into(),
                "ordered_pair_universe".into(),
                "median_out_reach".into(),
                "mean_out_harmonic".into(),
                "mean_in_harmonic".into(),
                "mean_betweenness_normalized".into(),
                "rich_club_total_points".into(),
                "rich_club_out_points".into(),
            ],
        )?;
        emit_arm_row(
            &mut writer,
            &reference_metrics,
            control_bundle.reference_edges.len(),
        )?;
        for arm in &control_bundle.arms
        {
            let metrics = v888_growth_metric_bundle(arm.arm.name(), selected.len(), &arm.edges)?;
            emit_arm_row(&mut writer, &metrics, arm.edges.len())?;
        }
        writer.flush()?;
    }

    {
        let path = output_dir.join("deltas.tsv");
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_tsv_row(
            &mut writer,
            &[
                "arm".into(),
                "reachable_ordered_pairs_delta".into(),
                "median_out_reach_delta".into(),
                "out_reach_l1".into(),
                "in_reach_l1".into(),
                "mean_out_harmonic_nano_delta".into(),
                "mean_in_harmonic_nano_delta".into(),
                "mean_betweenness_nano_delta".into(),
                "max_abs_betweenness_nano_delta".into(),
            ],
        )?;
        for delta in &deltas
        {
            write_tsv_row(
                &mut writer,
                &[
                    delta.arm.clone(),
                    delta.reachable_ordered_pairs_delta.to_string(),
                    delta.median_out_reach_delta.to_string(),
                    delta.out_reach_l1.to_string(),
                    delta.in_reach_l1.to_string(),
                    delta.mean_out_harmonic_nano_delta.to_string(),
                    delta.mean_in_harmonic_nano_delta.to_string(),
                    delta.mean_betweenness_nano_delta.to_string(),
                    delta.max_abs_betweenness_nano_delta.to_string(),
                ],
            )?;
        }
        writer.flush()?;
    }

    {
        let path = output_dir.join("rich_club_reference_total.tsv");
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_tsv_row(
            &mut writer,
            &[
                "degree_threshold".into(),
                "club_nodes".into(),
                "club_edges".into(),
                "max_edges".into(),
                "phi".into(),
            ],
        )?;
        for point in &reference_metrics.rich_club_total.points
        {
            write_tsv_row(
                &mut writer,
                &[
                    point.degree_threshold.to_string(),
                    point.club_nodes.to_string(),
                    point.club_edges.to_string(),
                    point.max_edges.to_string(),
                    format!("{:.9}", point.phi()),
                ],
            )?;
        }
        writer.flush()?;
    }

    let policy_name = match policy
    {
        V888GrowthSubsetPolicy::Ranked => "ranked",
        V888GrowthSubsetPolicy::WeakBfs => "weak_bfs",
    };
    let delta_json: Vec<String> = deltas
        .iter()
        .map(|delta| {
            format!(
                "{{\"arm\":\"{}\",\"reachable_ordered_pairs_delta\":{},\"median_out_reach_delta\":{},\"out_reach_l1\":{},\"in_reach_l1\":{},\"mean_out_harmonic_nano_delta\":{},\"mean_in_harmonic_nano_delta\":{},\"mean_betweenness_nano_delta\":{},\"max_abs_betweenness_nano_delta\":{}}}",
                delta.arm,
                delta.reachable_ordered_pairs_delta,
                delta.median_out_reach_delta,
                delta.out_reach_l1,
                delta.in_reach_l1,
                delta.mean_out_harmonic_nano_delta,
                delta.mean_in_harmonic_nano_delta,
                delta.mean_betweenness_nano_delta,
                delta.max_abs_betweenness_nano_delta
            )
        })
        .collect();

    let summary = format!(
        "{{\"schema_version\":1,\"programme\":\"V888-GROWTH-VG-3C\",\"source_nodes\":{},\"source_pairs\":{},\"selection\":\"{policy_name}\",\"seed\":{seed},\"nodes\":{},\"pairs\":{},\"bfs_component_starts\":{bfs_starts},\"block_labels\":{},\"block_constraint_vacuous\":{},\"reachable_ordered_pairs\":{},\"median_out_reach\":{},\"developmental_causality_claim\":false,\"topology_advantage_claim\":false,\"deltas\":[{}]}}\n",
        graph.node_count(),
        graph.directed_pair_count(),
        selected.len(),
        control_bundle.reference_edges.len(),
        control_bundle.distinct_block_labels,
        control_bundle.block_constraint_vacuous,
        reference_metrics.reachability.reachable_ordered_pairs,
        reference_metrics.reachability.median_out_reach,
        delta_json.join(",")
    );
    fs::write(output_dir.join("summary.json"), &summary)?;

    eprintln!(
        "nodes={} pairs={} reachable_pairs={} median_out_reach={} deltas={}",
        selected.len(),
        control_bundle.reference_edges.len(),
        reference_metrics.reachability.reachable_ordered_pairs,
        reference_metrics.reachability.median_out_reach,
        deltas.len()
    );
    Ok(())
}
