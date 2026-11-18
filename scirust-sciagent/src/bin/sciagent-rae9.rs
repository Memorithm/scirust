//! Operator-gated real-checkpoint SciAgent + FLAT RAE-9 CPU experiment.
//! No change to default SciAgent generation or the resident WGPU route.
use clap::Parser;
use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::rae9_research::Rae9ResearchBridge;
use scirust_sciagent::train::checkpoint::{load_checkpoint, read_meta};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Existing trained SciAgent safetensors checkpoint directory.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Exact-version built FLAT RAE-9 executable, supplied by RemoteOps.
    #[arg(long)]
    bridge: PathBuf,
    /// Original SciAgent query-head index, not an already repeated KV head.
    #[arg(long, default_value_t = 1)]
    head: usize,
    #[arg(long, default_value_t = 8)]
    page_size: usize,
    /// Comma-separated token IDs for a bounded, non-padded test sequence.
    #[arg(long, default_value = "4,18,61,7,19,24,9,42,3,80,121,5,8,22,31,63")]
    ids: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let meta = read_meta(&args.checkpoint)?;
    let ids: Vec<usize> = args.ids.split(',')
        .map(|t| t.parse::<usize>())
        .collect::<Result<_,_>>()?;
    if ids.len() < 2 || ids.len() > meta.config.max_seq_len
        || ids.iter().any(|&id| id >= meta.config.vocab_size)
    {
        return Err("RAE9_INVALID_TRAINED_MODEL_TOKEN_IDS".into());
    }
    let bridge = Rae9ResearchBridge::new(args.bridge, args.page_size, args.head)?;
    let mut model = SciAgentModel::new(&meta.config);
    load_checkpoint(&mut model, &args.checkpoint)?;
    let original = Tape::new();
    let reference = model.forward(&original, &ids, ids.len());
    let reference_logits = original.value(reference.idx());
    let vocab = meta.config.vocab_size;
    let reference_last = &reference_logits.data[reference_logits.data.len() - vocab..];
    let candidate_tape = Tape::new();
    let candidate = model.forward_research_rae9(&candidate_tape, &ids, &bridge)?;
    let candidate_logits = candidate_tape.value(candidate.idx());
    let candidate_last = &candidate_logits.data[candidate_logits.data.len() - vocab..];
    let max_logit_delta = reference_last.iter().zip(candidate_last)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0f64, f64::max);
    let winner = |row: &[f32]| -> usize {
        row.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1))
            .map(|x| x.0).expect("nonempty vocab")
    };
    let evidence = bridge.observations()?;
    if evidence.len() != 1 {
        return Err(format!("RAE9_EXPECTED_ONE_MODEL_HEAD got={}", evidence.len()).into());
    }
    let head = &evidence[0];
    if !head.injected {
        return Err("RAE9_MODEL_HEAD_NOT_INJECTED".into());
    }
    if head.executed_keys == head.keys &&
        (max_logit_delta > 0.001 || winner(reference_last) != winner(candidate_last))
    {
        return Err("RAE9_FULL_KEY_NUMERICAL_PARITY_FAILED".into());
    }
    println!("RAE9_SCIAGENT_MODEL_RESULT {}", json!({
        "model": "sciagent",
        "checkpoint": args.checkpoint.to_string_lossy(),
        "head": args.head,
        "source_query_heads": meta.config.n_heads,
        "source_kv_heads": meta.config.n_kv_heads,
        "context_tokens": ids.len(),
        "n_layers": meta.config.n_layers,
        "d_model": meta.config.d_model,
        "route": head.route,
        "original_keys": head.keys,
        "executed_keys": head.executed_keys,
        "scored_keys": head.scored_keys,
        "logical_key_reduction": head.executed_keys < head.keys,
        "native_head_abs_difference": head.native_component_delta,
        "maximum_output_error": head.maximum_output_error,
        "actual_injected_heads": 1,
        "max_abs_logit_delta": max_logit_delta,
        "reference_argmax": winner(reference_last),
        "candidate_argmax": winner(candidate_last),
        "physical_bandwidth_measured": false,
        "production_wgpu_backend_changed": false,
        "training_semantics_changed": false,
    }));
    Ok(())
}
