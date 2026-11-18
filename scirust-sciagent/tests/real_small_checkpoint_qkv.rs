//! Real-weight CPU gate for the shipped SciAgent small checkpoint.
//!
//! Unlike debug/random-weight parity tests, this loads the repository's
//! safetensors checkpoint and extracts actual RoPE-rotated GQA Q/K and raw V
//! for one causal decode query. It is the real-model fixture boundary for a
//! later FLAT RAE-7/RAE-8 replay, not itself a sparse-attention speedup claim.

use std::path::Path;

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::module::Module;
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{load_checkpoint_with_provenance, read_meta};

fn rotate_rope_head(head: &[f32], position: usize, theta: f32) -> Vec<f64> {
    assert_eq!(head.len() % 2, 0);
    let mut result = Vec::with_capacity(head.len());
    for pair in 0..head.len() / 2
    {
        // Match SciAgent's interleaved-pair f32 RoPE convention.
        let frequency = theta.powf(-2.0 * pair as f32 / head.len() as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        result.push((head[2 * pair] * cos - head[2 * pair + 1] * sin) as f64);
        result.push((head[2 * pair] * sin + head[2 * pair + 1] * cos) as f64);
    }
    result
}

#[test]
fn shipped_small_checkpoint_provides_real_causal_gqa_qkv() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checkpoint = root.join("checkpoints/small-20M/final");
    let metadata = read_meta(&checkpoint).expect("shipped checkpoint metadata must be readable");
    let config = SciAgentConfig::small();
    assert_eq!(metadata.config.vocab_size, config.vocab_size);
    assert_eq!(metadata.config.d_model, config.d_model);
    assert_eq!(metadata.config.n_layers, config.n_layers);
    assert_eq!(metadata.config.n_heads, config.n_heads);
    assert_eq!(metadata.config.n_kv_heads, config.n_kv_heads);
    assert_eq!(metadata.config.max_seq_len, config.max_seq_len);
    assert_eq!(metadata.step, 2000);

    let mut model = SciAgentModel::new(&metadata.config);
    let loaded = load_checkpoint_with_provenance(&mut model, &checkpoint)
        .expect("the real safetensors weights must load successfully");
    assert_eq!(loaded.meta.step, metadata.step);

    let tokenizer = BpeTokenizer::from_embedded().expect("embedded SciAgent BPE must load");
    let input = tokenizer.encode_with_special("pub fn f(x: u32) -> u32 {", true, false);
    assert!(
        (2..=metadata.config.max_seq_len).contains(&input.len()),
        "real BPE prompt must include causal history"
    );
    assert!(input.iter().all(|&id| id < metadata.config.vocab_size));

    // Exercise all four layers and the trained vocabulary output projection.
    let tape = Tape::new();
    let logits = model.forward(&tape, &input, input.len());
    let actual = tape.value(logits.idx());
    assert_eq!(actual.rows, input.len());
    assert_eq!(actual.cols, metadata.config.vocab_size);
    assert!(actual.data.iter().all(|x| x.is_finite()));
    drop(tape);

    // Recover the first layer's *actual trained-weight* attention inputs,
    // rather than synthetic/random Q/K/V. Same norm and projection modules
    // as the ordinary model's forward; extraction does not mutate weights.
    let tape = Tape::new();
    let tokens = tape.input(Tensor::from_vec(
        input.iter().map(|&id| id as f32).collect(),
        input.len(),
        1,
    ));
    let embedded = model.embed.forward(&tape, tokens);
    let attention_input = model.layers[0].rms_attn.forward(&tape, embedded);
    let q = model.layers[0].attn.w_q.forward(&tape, attention_input);
    let k = model.layers[0].attn.w_k.forward(&tape, attention_input);
    let v = model.layers[0].attn.w_v.forward(&tape, attention_input);

    let q = tape.value(q.idx());
    let k = tape.value(k.idx());
    let v = tape.value(v.idx());
    let head_dim = metadata.config.d_model / metadata.config.n_heads;
    let kv_dim = head_dim * metadata.config.n_kv_heads;
    assert_eq!((q.rows, q.cols), (input.len(), metadata.config.d_model));
    assert_eq!((k.rows, k.cols), (input.len(), kv_dim));
    assert_eq!((v.rows, v.cols), (input.len(), kv_dim));
    assert!(q.data.iter().all(|x| x.is_finite()));
    assert!(k.data.iter().all(|x| x.is_finite()));
    assert!(v.data.iter().all(|x| x.is_finite()));

    // Compute an independent causal softmax on the real first Q head and its
    // mapped first KV head. This is a numerical fixture for FLAT's later
    // selective replay; it does NOT replace or requalify SciAgent attention.
    let last_position = input.len() - 1;
    let query_start = last_position * q.cols;
    let rotated_query = rotate_rope_head(
        &q.data[query_start..query_start + head_dim],
        last_position,
        metadata.config.rope_theta,
    );
    let mut scores = Vec::with_capacity(input.len());
    let mut values = Vec::with_capacity(input.len());
    for position in 0..input.len()
    {
        let key_start = position * k.cols;
        let value_start = position * v.cols;
        let rotated_key = rotate_rope_head(
            &k.data[key_start..key_start + head_dim],
            position,
            metadata.config.rope_theta,
        );
        let score: f64 = rotated_query
            .iter()
            .zip(rotated_key.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / (head_dim as f64).sqrt();
        scores.push(score);
        values.push(
            v.data[value_start..value_start + head_dim]
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>(),
        );
    }
    assert!(scores.iter().all(|x| x.is_finite()));
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|&x| (x - max_score).exp()).collect();
    let denominator: f64 = weights.iter().sum();
    assert!(denominator.is_finite() && denominator > 0.0);
    let mut output = vec![0.0_f64; head_dim];
    for (weight, row) in weights.iter().zip(values.iter())
    {
        for (sum, &value) in output.iter_mut().zip(row.iter())
        {
            *sum += weight * value;
        }
    }
    for component in &mut output
    {
        *component /= denominator;
    }
    assert!(output.iter().all(|x| x.is_finite()));
    assert!(output.iter().any(|&x| x != 0.0));

    println!(
        "sciagent_real_qkv=PASS checkpoint_step={} model_params={} token_count={} layers={} q_heads={} kv_heads={} head_dim={} causal_keys={}",
        loaded.meta.step,
        metadata.config.total_parameters(),
        input.len(),
        metadata.config.n_layers,
        metadata.config.n_heads,
        metadata.config.n_kv_heads,
        head_dim,
        scores.len()
    );
    println!("performance_claim=none");
}
