#[cfg(feature = "flat-attention")]
use std::time::{Duration, Instant};

#[cfg(feature = "flat-attention")]
use scirust_sciagent::config::SciAgentConfig;
#[cfg(feature = "flat-attention")]
use scirust_sciagent::gpu::ResidentModel;
#[cfg(feature = "flat-attention")]
use scirust_sciagent::model::SciAgentModel;

#[cfg(feature = "flat-attention")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "flat-attention")]
fn prompt(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 17 + 3) % vocab) as u32)
        .collect()
}

#[cfg(feature = "flat-attention")]
fn elapsed(mut operation: impl FnMut()) -> Duration {
    let started = Instant::now();
    operation();
    started.elapsed()
}

#[cfg(feature = "flat-attention")]
fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

#[cfg(feature = "flat-attention")]
fn checked_incremental_decode_ms(
    prefill: Duration,
    total: Duration,
    decode_steps: usize,
) -> Result<f64, &'static str> {
    if decode_steps == 0 {
        return Err("zero incremental decode steps");
    }
    let incremental = total
        .checked_sub(prefill)
        .ok_or("total generation time is less than prefill time; reject noisy pair")?;
    if incremental.is_zero() {
        return Err("no measurable incremental decode time; reject invalid pair");
    }
    let per_token_ms = incremental.as_secs_f64() * 1_000.0 / decode_steps as f64;
    if !per_token_ms.is_finite() || per_token_ms <= 0.0 {
        return Err("invalid incremental decode latency");
    }
    Ok(per_token_ms)
}

#[cfg(all(test, feature = "flat-attention"))]
mod benchmark_contract_tests {
    use super::checked_incremental_decode_ms;
    use std::time::Duration;

    #[test]
    fn valid_paired_measurement_preserves_real_delta() {
        let actual = checked_incremental_decode_ms(
            Duration::from_millis(10),
            Duration::from_millis(20),
            5,
        )
        .expect("valid timing pair");
        assert!((actual - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_nonpositive_decode_latency_instead_of_fabricating_throughput() {
        for total in [Duration::from_millis(9), Duration::from_millis(10)] {
            assert!(
                checked_incremental_decode_ms(Duration::from_millis(10), total, 5).is_err()
            );
        }
        assert!(
            checked_incremental_decode_ms(
                Duration::from_millis(10),
                Duration::from_millis(20),
                0,
            )
            .is_err()
        );
    }
}

#[cfg(feature = "flat-attention")]
fn main() {
    let vocab = 256usize;
    let prompt_len = env_usize("SCIAGENT_FLAT_M33_PROMPT", 128);
    let new_tokens = env_usize("SCIAGENT_FLAT_M33_NEW", 32).max(2);
    let warmups = env_usize("SCIAGENT_FLAT_M33_WARMUPS", 2);
    let repeats = env_usize("SCIAGENT_FLAT_M33_REPEATS", 7).max(1);
    let config = SciAgentConfig {
        vocab_size: vocab,
        d_model: env_usize("SCIAGENT_FLAT_M33_D_MODEL", 512),
        n_layers: env_usize("SCIAGENT_FLAT_M33_LAYERS", 8),
        n_heads: env_usize("SCIAGENT_FLAT_M33_Q_HEADS", 8),
        n_kv_heads: env_usize("SCIAGENT_FLAT_M33_KV_HEADS", 2),
        d_ff: env_usize("SCIAGENT_FLAT_M33_D_FF", 1408),
        max_seq_len: prompt_len + new_tokens + 8,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let model = SciAgentModel::new(&config);
    let Some(resident) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("FLAT M33 requires a real WGPU adapter; none was available");
        std::process::exit(2);
    };
    let prompt = prompt(prompt_len, vocab);

    for _ in 0..warmups
    {
        let _ = resident.generate_cached(&prompt, 1);
        let _ = resident.generate_cached(&prompt, new_tokens);
    }

    let mut prefill_ms = Vec::with_capacity(repeats);
    let mut decode_ms_per_token = Vec::with_capacity(repeats);
    for index in 0..repeats
    {
        let (prefill, total) = if index.is_multiple_of(2)
        {
            let prefill = elapsed(|| {
                let _ = resident.generate_cached(&prompt, 1);
            });
            let total = elapsed(|| {
                let _ = resident.generate_cached(&prompt, new_tokens);
            });
            (prefill, total)
        }
        else
        {
            let total = elapsed(|| {
                let _ = resident.generate_cached(&prompt, new_tokens);
            });
            let prefill = elapsed(|| {
                let _ = resident.generate_cached(&prompt, 1);
            });
            (prefill, total)
        };
        let decode = checked_incremental_decode_ms(
            prefill,
            total,
            new_tokens - 1,
        )
        .unwrap_or_else(|reason| {
            eprintln!("FLAT M33 measurement rejected: pair {index}: {reason}");
            std::process::exit(3);
        });
        let prefill = prefill.as_secs_f64() * 1e3;
        prefill_ms.push(prefill);
        decode_ms_per_token.push(decode);
    }

    let mut prefill_for_median = prefill_ms.clone();
    let mut decode_for_median = decode_ms_per_token.clone();
    let mut decode_for_p95 = decode_ms_per_token;
    let prefill_median_ms = percentile(&mut prefill_for_median, 0.5);
    let decode_median_ms = percentile(&mut decode_for_median, 0.5);
    let decode_p95_ms = percentile(&mut decode_for_p95, 0.95);
    let tokens_per_second = 1_000.0 / decode_median_ms;
    let revision = std::env::var("SCIRUST_SOURCE_REVISION")
        .or_else(|_| std::env::var("GITHUB_SHA"))
        .unwrap_or_else(|_| "unknown".to_owned());

    println!(
        "revision,adapter,d_model,layers,q_heads,kv_heads,prompt_len,new_tokens,warmups,repeats,prefill_median_ms,decode_median_ms_per_token,decode_p95_ms_per_token,decode_tokens_per_second,performance_claim"
    );
    println!(
        "{revision},\"{}\",{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},none",
        resident.adapter_name().replace('"', "\"\""),
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        prompt_len,
        new_tokens,
        warmups,
        repeats,
        prefill_median_ms,
        decode_median_ms,
        decode_p95_ms,
        tokens_per_second,
    );
}

#[cfg(not(feature = "flat-attention"))]
fn main() {
    eprintln!("enable the flat-attention feature to run the FLAT M33 benchmark");
    std::process::exit(2);
}
