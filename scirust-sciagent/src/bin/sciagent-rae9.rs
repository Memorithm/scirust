//! Operator-gated real-checkpoint SciAgent + FLAT RAE-9 CPU experiment.
//! No change to default SciAgent generation or the resident WGPU route.
use clap::Parser;
use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::bpe::BpeTokenizer;
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
    /// Independent absolute error cap vs the native SciAgent head output.
    /// Exceeding it retains dense attention; this experiment then fails closed.
    #[arg(long)]
    max_native_head_delta: Option<f64>,
    /// Comma-separated token IDs; default remains the original numeric fixture.
    #[arg(long, conflicts_with = "prompt")]
    ids: Option<String>,
    /// Run a real Rust text prompt through SciAgent's embedded BPE tokenizer.
    #[arg(long, conflicts_with = "ids")]
    prompt: Option<String>,
    /// Permit truthful, observable native dense fallback during research replay.
    /// Without this opt-in, a non-injected head remains a hard failure.
    #[arg(long)]
    allow_dense_fallback: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let meta = read_meta(&args.checkpoint)?;
    let input_mode = if args.prompt.is_some()
    {
        "bpe_prompt"
    }
    else
    {
        "token_ids"
    };
    let ids: Vec<usize> = if let Some(prompt) = &args.prompt
    {
        if prompt.is_empty() || prompt.len() > 4096
        {
            return Err("RAE9_INVALID_PROMPT_LENGTH".into());
        }
        BpeTokenizer::from_embedded()?.encode_with_special(prompt, true, false)
    }
    else
    {
        args.ids
            .as_deref()
            .unwrap_or("4,18,61,7,19,24,9,42,3,80,121,5,8,22,31,63")
            .split(',')
            .map(|t| t.parse::<usize>())
            .collect::<Result<_, _>>()?
    };
    if ids.len() < 2
        || ids.len() > meta.config.max_seq_len
        || ids.iter().any(|&id| id >= meta.config.vocab_size)
    {
        return Err("RAE9_INVALID_TRAINED_MODEL_TOKEN_IDS".into());
    }
    let mut bridge = Rae9ResearchBridge::new(args.bridge, args.page_size, args.head)?;
    if let Some(budget) = args.max_native_head_delta
    {
        bridge = bridge.with_max_native_head_delta(budget)?;
    }
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
    if reference_last
        .iter()
        .chain(candidate_last.iter())
        .any(|value| !value.is_finite())
    {
        return Err("RAE9_NONFINITE_MODEL_LOGITS".into());
    }
    let max_logit_delta = reference_last
        .iter()
        .zip(candidate_last)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0f64, f64::max);
    let winner = |row: &[f32]| -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|x| x.0)
            .expect("nonempty vocab")
    };
    let evidence = bridge.observations()?;
    if evidence.len() != 1
    {
        return Err(format!("RAE9_EXPECTED_ONE_MODEL_HEAD got={}", evidence.len()).into());
    }
    let head = &evidence[0];
    if !head.injected && !args.allow_dense_fallback
    {
        return Err("RAE9_MODEL_HEAD_NOT_INJECTED".into());
    }
    if (!head.injected || head.executed_keys == head.keys)
        && (max_logit_delta > 0.001 || winner(reference_last) != winner(candidate_last))
    {
        return Err("RAE9_FULL_KEY_NUMERICAL_PARITY_FAILED".into());
    }
    println!(
        "RAE9_SCIAGENT_MODEL_RESULT {}",
        json!({
            "model": "sciagent",
            "checkpoint": args.checkpoint.to_string_lossy(),
            "head": args.head,
            "source_query_heads": meta.config.n_heads,
            "source_kv_heads": meta.config.n_kv_heads,
            "context_tokens": ids.len(),
            "input_mode": input_mode,
            "n_layers": meta.config.n_layers,
            "d_model": meta.config.d_model,
            "route": head.route,
            "effective_route": if head.native_budget_vetoed {
                "native_budget_fallback"
            } else if head.injected {
                "bounded_sparse"
            } else {
                "native_dense_fallback"
            },
            "original_keys": head.keys,
            "executed_keys": head.executed_keys,
            "scored_keys": head.scored_keys,
            "logical_key_reduction": head.injected && head.executed_keys < head.keys,
            "native_head_abs_difference": head.native_component_delta,
            "native_head_budget": args.max_native_head_delta,
            "native_budget_vetoed": head.native_budget_vetoed,
            "maximum_output_error": head.maximum_output_error,
            "actual_injected_heads": usize::from(head.injected),
            "dense_fallback_accepted": !head.injected && args.allow_dense_fallback,
            "max_abs_logit_delta": max_logit_delta,
            "reference_argmax": winner(reference_last),
            "candidate_argmax": winner(candidate_last),
            "physical_bandwidth_measured": false,
            "production_wgpu_backend_changed": false,
            "training_semantics_changed": false,
        })
    );
    Ok(())
}
