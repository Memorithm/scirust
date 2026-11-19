#![cfg(feature = "flat-attention")]

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;

fn config() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 32,
        d_model: 128,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        d_ff: 256,
        max_seq_len: 32,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

#[test]
fn flat_m33_cache_reset_replay_and_eos_are_deterministic() {
    let model = SciAgentModel::new(&config());
    let Some(resident) = ResidentModel::from_model(&model)
    else
    {
        assert_ne!(
            std::env::var("FLAT_REQUIRE_WGPU").as_deref(),
            Ok("1"),
            "FLAT M33 qualification requires a WGPU adapter; refusing a skipped lifecycle test"
        );
        eprintln!("wgpu: no adapter; optional FLAT M33 lifecycle test skipped");
        return;
    };

    if let Ok(required) = std::env::var("FLAT_EXPECTED_WGPU_ADAPTER")
    {
        assert!(
            !required.is_empty() && resident.adapter_name().contains(&required),
            "FLAT qualification selected WGPU adapter {:?}, expected a name containing {:?}",
            resident.adapter_name(),
            required
        );
    }

    eprintln!("FLAT_M33_LIFECYCLE_ADAPTER={}", resident.adapter_name());

    let prompt = vec![3u32, 7, 1, 4];
    let replay_a = resident.generate_cached(&prompt, 6);
    let replay_b = resident.generate_cached(&prompt, 6);
    assert_eq!(
        replay_b, replay_a,
        "a fresh resident KV cache must replay deterministically"
    );

    let first = resident.generate_cached(&prompt, 1);
    let eos = *first.last().expect("one generated token");
    let stopped = resident.generate_cached_until_eos(&prompt, 6, &[eos]);
    assert_eq!(
        stopped, first,
        "EOS policy must stop immediately after the matching decoded token"
    );

    let without_eos = resident.generate_cached_until_eos(&prompt, 6, &[]);
    assert_eq!(
        without_eos, replay_a,
        "an empty EOS set must preserve ordinary cached greedy generation"
    );
}
