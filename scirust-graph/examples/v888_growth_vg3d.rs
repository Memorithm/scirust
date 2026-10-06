//! VG-3D pilot exporter: inferred modularity, annotation mixing and partition
//! agreement on the qualified V888 subset and its VG-3B matched controls.
//!
//! Emits compact TSV / JSON summaries only. The optional annotation table
//! (`node_id<TAB>label`, e.g. hemilineage) stays outside Git; selected nodes
//! without a row fall into one explicit `<unlabelled>` bucket. Descriptive
//! output only: no biological module, topology-advantage or
//! developmental-causality claim.
//!
//! With `--ensemble <count>` the VG-3B ladder is regenerated for the seeds
//! `seed, seed + 1, …, seed + count - 1` and every VG-3C / VG-3D scalar
//! descriptor is summarised per arm (min / max / mean / sample SD / median and
//! nano-rounded below / equal / above-reference counts). The counts are
//! descriptive, not p-values.

use scirust_graph::v888_executable::{BANC_V888_BOOL01_QUALIFICATION, BancV888ExecutableGraph};
use scirust_graph::v888_growth::{
    V888GrowthAnnotationJoin, V888GrowthEnsembleDescriptor, V888GrowthEnsembleReport,
    V888GrowthLouvainOptions, V888GrowthModularityArm, V888GrowthSubsetPolicy,
    v888_growth_compare_modularity_arms, v888_growth_control_ensemble, v888_growth_ensemble_seeds,
    v888_growth_f64_to_nano, v888_growth_join_annotation, v888_growth_matched_controls,
    v888_growth_parse_annotation_tsv, v888_growth_select_subset, v888_growth_unit_reference,
};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

const USAGE: &str = "usage: v888_growth_vg3d <graph.csr> <out_dir> <size> <seed> <ranked|weak_bfs> [--blocks blocks.tsv] [--labels labels.tsv] [--ensemble count]";

/// Pilot bound on `--ensemble`; every draw reruns betweenness and Louvain on four arms.
const MAX_PILOT_ENSEMBLE: usize = 64;

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

/// Same strict, graph-ordered `node_id<TAB>block` contract as the VG-3C exporter.
fn read_block_groups(
    path: &PathBuf,
    node_ids: &[u64],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let reader = BufReader::new(fs::File::open(path)?);
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

fn opt_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |value| format!("{value:.9}"))
}

fn opt_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "NA".to_string(), |value| value.to_string())
}

fn json_opt_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 2);
    out.push('"');
    for ch in raw.chars()
    {
        match ch
        {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn usize_flag(raw: &str, flag: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| format!("`{flag}` needs a non-negative integer, got `{raw}`"))
}

fn write_ensemble_dispersion(
    path: &PathBuf,
    report: &V888GrowthEnsembleReport,
) -> std::io::Result<()> {
    let mut writer = BufWriter::new(fs::File::create(path)?);
    write_tsv_row(
        &mut writer,
        &[
            "arm".into(),
            "descriptor".into(),
            "draws".into(),
            "distinct_graphs".into(),
            "reference".into(),
            "defined".into(),
            "undefined".into(),
            "min".into(),
            "max".into(),
            "mean".into(),
            "std_dev".into(),
            "median".into(),
            "below_reference".into(),
            "equal_reference".into(),
            "above_reference".into(),
        ],
    )?;
    for arm in &report.arms
    {
        for descriptor in V888GrowthEnsembleDescriptor::ALL
        {
            if descriptor.requires_annotation() && !report.annotated
            {
                continue;
            }
            let summary = &arm.dispersion[&descriptor];
            write_tsv_row(
                &mut writer,
                &[
                    arm.arm.name().to_string(),
                    descriptor.name().to_string(),
                    arm.draws.to_string(),
                    arm.distinct_graphs.to_string(),
                    opt_f64(summary.reference),
                    summary.defined.to_string(),
                    summary.undefined.to_string(),
                    opt_f64(summary.min),
                    opt_f64(summary.max),
                    opt_f64(summary.mean),
                    opt_f64(summary.std_dev),
                    opt_f64(summary.median),
                    summary.below_reference.to_string(),
                    summary.equal_reference.to_string(),
                    summary.above_reference.to_string(),
                ],
            )?;
        }
    }
    writer.flush()
}

fn write_ensemble_samples(
    path: &PathBuf,
    report: &V888GrowthEnsembleReport,
) -> std::io::Result<()> {
    let descriptors: Vec<V888GrowthEnsembleDescriptor> = V888GrowthEnsembleDescriptor::ALL
        .into_iter()
        .filter(|descriptor| report.annotated || !descriptor.requires_annotation())
        .collect();
    let mut writer = BufWriter::new(fs::File::create(path)?);
    let mut header: Vec<String> = vec!["seed".into(), "arm".into(), "accepted_swaps".into()];
    header.extend(
        descriptors
            .iter()
            .map(|descriptor| descriptor.name().to_string()),
    );
    write_tsv_row(&mut writer, &header)?;
    for sample in &report.samples
    {
        let mut row = vec![
            sample.seed.to_string(),
            sample.arm.name().to_string(),
            sample.accepted_swaps.to_string(),
        ];
        row.extend(
            descriptors
                .iter()
                .map(|descriptor| opt_f64(sample.values.get(descriptor).copied().flatten())),
        );
        write_tsv_row(&mut writer, &row)?;
    }
    writer.flush()
}

fn ensemble_json(report: &V888GrowthEnsembleReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|arm| {
            let modularity = &arm.dispersion[&V888GrowthEnsembleDescriptor::InferredModularity];
            format!(
                "{{\"arm\":{},\"draws\":{},\"distinct_graphs\":{},\"inferred_q_mean_nano\":{},\"inferred_q_std_dev_nano\":{}}}",
                json_string(arm.arm.name()),
                arm.draws,
                arm.distinct_graphs,
                json_opt_i64(modularity.mean_nano()),
                json_opt_i64(modularity.std_dev_nano()),
            )
        })
        .collect();
    format!(
        "{{\"seeds\":{},\"first_seed\":{},\"last_seed\":{},\"seed_rule\":\"base_plus_offset_wrapping\",\"samples\":{},\"comparison_counts_are_p_values\":false,\"arms\":[{}]}}",
        report.seeds.len(),
        report.seeds.first().copied().unwrap_or_default(),
        report.seeds.last().copied().unwrap_or_default(),
        report.samples.len(),
        arms.join(",")
    )
}

fn emit_arm_row(
    writer: &mut impl Write,
    arm: &V888GrowthModularityArm,
    node_count: usize,
) -> std::io::Result<()> {
    let modularity = &arm.inferred.modularity;
    write_tsv_row(
        writer,
        &[
            arm.arm.clone(),
            node_count.to_string(),
            modularity.edge_count.to_string(),
            arm.inferred.community_sizes.len().to_string(),
            arm.inferred
                .community_sizes
                .iter()
                .max()
                .copied()
                .unwrap_or(0)
                .to_string(),
            modularity.q_numerator.to_string(),
            modularity.q_denominator.to_string(),
            opt_f64(modularity.q),
            arm.inferred.levels.to_string(),
            arm.inferred.moves.to_string(),
            arm.inferred.hit_bound.to_string(),
            opt_f64(arm.annotation_modularity.as_ref().and_then(|q| q.q)),
            opt_f64(arm.annotation_mixing.as_ref().and_then(|m| m.assortativity)),
            arm.annotation_mixing
                .as_ref()
                .map_or_else(|| "NA".to_string(), |m| m.same_label_edges.to_string()),
            opt_f64(
                arm.inferred_vs_annotation
                    .as_ref()
                    .and_then(|a| a.adjusted_rand),
            ),
            opt_f64(arm.inferred_vs_reference.adjusted_rand),
        ],
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let graph_path = PathBuf::from(args.next().ok_or(USAGE)?);
    let output_dir = PathBuf::from(args.next().ok_or(USAGE)?);
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
    let mut blocks_path: Option<PathBuf> = None;
    let mut labels_path: Option<PathBuf> = None;
    let mut ensemble_count: Option<usize> = None;
    while let Some(flag) = args.next()
    {
        let value = args
            .next()
            .ok_or_else(|| format!("`{flag}` needs a value"))?;
        let duplicate = match flag.as_str()
        {
            "--blocks" => blocks_path.replace(PathBuf::from(value)).is_some(),
            "--labels" => labels_path.replace(PathBuf::from(value)).is_some(),
            "--ensemble" => ensemble_count.replace(usize_flag(&value, &flag)?).is_some(),
            other => return Err(format!("unexpected argument `{other}`; {USAGE}").into()),
        };
        if duplicate
        {
            return Err(format!("`{flag}` given twice").into());
        }
    }
    if let Some(count) = ensemble_count
    {
        if !(2..=MAX_PILOT_ENSEMBLE).contains(&count)
        {
            return Err(format!(
                "`--ensemble` must be between 2 and {MAX_PILOT_ENSEMBLE} for the pilot exporter, got {count}"
            )
            .into());
        }
    }
    if size > 512
    {
        return Err(
            "VG-3D pilot exporter rejects size > 512 (same pilot subset as VG-3B/3C)".into(),
        );
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
    let source_groups = match &blocks_path
    {
        Some(path) => read_block_groups(path, &source_ids)?,
        None => vec![0_u32; graph.node_count()],
    };

    let (selected, bfs_starts) = v888_growth_select_subset(&graph, size, seed, policy)?;
    let selected_ids: Vec<u64> = selected.iter().map(|&index| source_ids[index]).collect();
    let local_groups: Vec<u32> = selected.iter().map(|&index| source_groups[index]).collect();
    let join: Option<V888GrowthAnnotationJoin> = match &labels_path
    {
        Some(path) =>
        {
            let annotation =
                v888_growth_parse_annotation_tsv(BufReader::new(fs::File::open(path)?))?;
            Some(v888_growth_join_annotation(&selected_ids, &annotation)?)
        },
        None => None,
    };

    let reference = v888_growth_unit_reference(&graph, &selected)?;
    let control_bundle =
        v888_growth_matched_controls(&reference, selected.len(), &local_groups, seed)?;
    let controls: Vec<(&str, _)> = control_bundle
        .arms
        .iter()
        .map(|arm| (arm.arm.name(), &arm.edges))
        .collect();
    let options = V888GrowthLouvainOptions::default();
    let (reference_arm, rows) = v888_growth_compare_modularity_arms(
        selected.len(),
        &control_bundle.reference_edges,
        &controls,
        join.as_ref().map(|join| join.labels.as_slice()),
        options,
    )?;

    let ensemble = match ensemble_count
    {
        Some(count) =>
        {
            let seeds = v888_growth_ensemble_seeds(seed, count)?;
            let report = v888_growth_control_ensemble(
                &control_bundle.reference_edges,
                selected.len(),
                &local_groups,
                &seeds,
                join.as_ref().map(|join| join.labels.as_slice()),
                options,
            )?;
            // Fail closed if the ensemble and the single-draw comparison disagree
            // on the shared reference arm.
            let ensemble_q = report
                .reference
                .get(&V888GrowthEnsembleDescriptor::InferredModularity)
                .copied()
                .flatten()
                .map(v888_growth_f64_to_nano);
            if ensemble_q
                != reference_arm
                    .inferred
                    .modularity
                    .q
                    .map(v888_growth_f64_to_nano)
            {
                return Err(
                    "ensemble reference modularity disagrees with the single-draw reference arm"
                        .into(),
                );
            }
            write_ensemble_dispersion(&output_dir.join("ensemble_dispersion.tsv"), &report)?;
            write_ensemble_samples(&output_dir.join("ensemble_samples.tsv"), &report)?;
            Some(report)
        },
        None => None,
    };

    {
        let mut writer = BufWriter::new(fs::File::create(output_dir.join("nodes.tsv"))?);
        write_tsv_row(
            &mut writer,
            &[
                "local".into(),
                "node_id".into(),
                "block".into(),
                "annotation_index".into(),
                "reference_community".into(),
            ],
        )?;
        for (local, &node_id) in selected_ids.iter().enumerate()
        {
            write_tsv_row(
                &mut writer,
                &[
                    local.to_string(),
                    node_id.to_string(),
                    local_groups[local].to_string(),
                    join.as_ref()
                        .map_or_else(|| "NA".to_string(), |join| join.labels[local].to_string()),
                    reference_arm.inferred.labels[local].to_string(),
                ],
            )?;
        }
        writer.flush()?;
    }

    if let Some(join) = &join
    {
        let mut writer =
            BufWriter::new(fs::File::create(output_dir.join("annotation_labels.tsv"))?);
        write_tsv_row(
            &mut writer,
            &[
                "annotation_index".into(),
                "label".into(),
                "selected_nodes".into(),
            ],
        )?;
        for (index, (name, size)) in join.names.iter().zip(&join.sizes).enumerate()
        {
            write_tsv_row(
                &mut writer,
                &[index.to_string(), name.clone(), size.to_string()],
            )?;
        }
        writer.flush()?;

        if let Some(mixing) = &reference_arm.annotation_mixing
        {
            let mut writer =
                BufWriter::new(fs::File::create(output_dir.join("mixing_reference.tsv"))?);
            write_tsv_row(
                &mut writer,
                &["source_label".into(), "target_label".into(), "edges".into()],
            )?;
            for (&(source, target), &count) in &mixing.counts
            {
                write_tsv_row(
                    &mut writer,
                    &[
                        join.names[source].clone(),
                        join.names[target].clone(),
                        count.to_string(),
                    ],
                )?;
            }
            writer.flush()?;
        }
    }

    {
        let mut writer =
            BufWriter::new(fs::File::create(output_dir.join("modularity_per_arm.tsv"))?);
        write_tsv_row(
            &mut writer,
            &[
                "arm".into(),
                "nodes".into(),
                "edges".into(),
                "inferred_communities".into(),
                "largest_community".into(),
                "inferred_q_numerator".into(),
                "inferred_q_denominator".into(),
                "inferred_q".into(),
                "louvain_levels".into(),
                "louvain_moves".into(),
                "louvain_hit_bound".into(),
                "annotation_q".into(),
                "annotation_assortativity".into(),
                "annotation_same_label_edges".into(),
                "inferred_vs_annotation_ari".into(),
                "inferred_vs_reference_ari".into(),
            ],
        )?;
        emit_arm_row(&mut writer, &reference_arm, selected.len())?;
        for (arm, _) in &rows
        {
            emit_arm_row(&mut writer, arm, selected.len())?;
        }
        writer.flush()?;
    }

    {
        let mut writer = BufWriter::new(fs::File::create(output_dir.join("deltas.tsv"))?);
        write_tsv_row(
            &mut writer,
            &[
                "arm".into(),
                "community_count_delta".into(),
                "inferred_q_nano_delta".into(),
                "annotation_q_nano_delta".into(),
                "annotation_assortativity_nano_delta".into(),
                "inferred_vs_reference_ari_nano".into(),
            ],
        )?;
        for (_, delta) in &rows
        {
            write_tsv_row(
                &mut writer,
                &[
                    delta.arm.clone(),
                    delta.community_count_delta.to_string(),
                    opt_i64(delta.inferred_q_nano_delta),
                    opt_i64(delta.annotation_q_nano_delta),
                    opt_i64(delta.annotation_assortativity_nano_delta),
                    opt_i64(delta.inferred_vs_reference_ari_nano),
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
    let delta_json: Vec<String> = rows
        .iter()
        .map(|(_, delta)| {
            format!(
                "{{\"arm\":{},\"community_count_delta\":{},\"inferred_q_nano_delta\":{},\"annotation_q_nano_delta\":{},\"annotation_assortativity_nano_delta\":{},\"inferred_vs_reference_ari_nano\":{}}}",
                json_string(&delta.arm),
                delta.community_count_delta,
                json_opt_i64(delta.inferred_q_nano_delta),
                json_opt_i64(delta.annotation_q_nano_delta),
                json_opt_i64(delta.annotation_assortativity_nano_delta),
                json_opt_i64(delta.inferred_vs_reference_ari_nano),
            )
        })
        .collect();
    let annotation_json = match &join
    {
        Some(join) => format!(
            "{{\"labels\":{},\"labelled_nodes\":{},\"unlabelled_nodes\":{}}}",
            join.names.len(),
            join.labelled_nodes,
            join.unlabelled_nodes
        ),
        None => "null".to_string(),
    };
    let ensemble_summary = ensemble
        .as_ref()
        .map_or_else(|| "null".to_string(), ensemble_json);
    let reference_q = &reference_arm.inferred.modularity;
    let summary = format!(
        "{{\"schema_version\":1,\"programme\":\"V888-GROWTH-VG-3D\",\"source_nodes\":{},\"source_pairs\":{},\"selection\":\"{policy_name}\",\"seed\":{seed},\"nodes\":{},\"pairs\":{},\"bfs_component_starts\":{bfs_starts},\"block_labels\":{},\"block_constraint_vacuous\":{},\"louvain_max_levels\":{},\"louvain_max_passes_per_level\":{},\"reference_communities\":{},\"reference_q_numerator\":{},\"reference_q_denominator\":{},\"reference_louvain_hit_bound\":{},\"annotation\":{annotation_json},\"ensemble\":{ensemble_summary},\"biological_module_claim\":false,\"developmental_causality_claim\":false,\"topology_advantage_claim\":false,\"deltas\":[{}]}}\n",
        graph.node_count(),
        graph.directed_pair_count(),
        selected.len(),
        control_bundle.reference_edges.len(),
        control_bundle.distinct_block_labels,
        control_bundle.block_constraint_vacuous,
        options.max_levels,
        options.max_passes_per_level,
        reference_arm.inferred.community_sizes.len(),
        reference_q.q_numerator,
        reference_q.q_denominator,
        reference_arm.inferred.hit_bound,
        delta_json.join(",")
    );
    fs::write(output_dir.join("summary.json"), &summary)?;

    eprintln!(
        "nodes={} pairs={} reference_communities={} reference_q={} annotation_labels={} deltas={} ensemble_seeds={}",
        selected.len(),
        control_bundle.reference_edges.len(),
        reference_arm.inferred.community_sizes.len(),
        opt_f64(reference_q.q),
        join.as_ref()
            .map_or_else(|| "none".to_string(), |join| join.names.len().to_string()),
        rows.len(),
        ensemble.as_ref().map_or_else(
            || "none".to_string(),
            |report| report.seeds.len().to_string()
        )
    );
    Ok(())
}
