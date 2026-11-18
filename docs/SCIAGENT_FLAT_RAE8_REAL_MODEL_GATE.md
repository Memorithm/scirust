# SciAgent shipped-small — real model Q/K/V readiness for FLAT RAE-7/RAE-8

Status: **real-model fixture extraction and CPU qualification gate**; not an RAE sparse-attention integration, latency win, or model-quality improvement.

## Verified repository model identity

- Source: `Memorithm/scirust`, `scirust-sciagent/checkpoints/small-20M/final`.
- `meta.json` records step 2000, 4 transformer blocks, `d_model=128`, 4 query heads, 2 GQA KV heads, head width 32, a maximum context of 256, BPE vocab 8192, and tied embeddings.
- `small-20M` refers to its training-corpus/token budget, **not a 20-million-parameter checkpoint**. `SciAgentConfig::small().total_parameters()` is the configuration-derived parameter estimate (roughly 1.6 million).
- The tracked `model.safetensors` contains pretrained model parameters. No new training, model download, GPU, vendor SDK or production runtime mutation is required.

## Exact CPU test

From the repository root:

```bash
cargo test --locked -p scirust-sciagent --test real_small_checkpoint_qkv -- --nocapture
```

The test fails if the checkpoint is missing or cannot be parsed. It reads the shipped metadata, loads the actual safetensors checkpoint using `load_checkpoint_with_provenance`, tokenizes a short Rust prompt with the embedded BPE, executes the entire pretrained CPU model and checks finite vocabulary logits. It then extracts the first attention block's trained `w_q`, `w_k` and `w_v` activations, applies SciAgent's interleaved-pair positional RoPE convention to a single causal query and its eligible keys, and accumulates its real value vectors with stable max-shifted softmax. It checks actual GQA dimensions and reports logical token/head geometry.

The fixture uses a *single real trained-weight query/head*, not artificial Q/K/V samples. Its CPU calculations are independent scalar reference results; this stage does not route K/V pages through FLAT or prove matching full-model logits after sparse attention.

## RAE-7 / RAE-8 integration contract

For the subsequent model-backed RAE replay, preserve the same:
- checkpoint byte SHA-256, tokenizer, input token IDs, layer, query head, mapped KV head and key-generation identity;
- interleaved RoPE convention on Q and K, **raw V**, score scale `1/sqrt(32)`, causal eligibility and original page identity;
- per-page maximum rotated-K L2 norms and exact causal key counts;
- a version-pinned absolute bound covering **all** active V components, including omitted pages, before any sparse output is accepted.

The future FLAT execution must compare its RAE-8 output with a dense SciAgent reference on the same Q/K/V, then separately verify vocabulary-logit and task-level effects. Synthetic RAE-8 unit results do not qualify physical read, power, throughput, or end-to-end model quality claims.

CI: `.github/workflows/sciagent-real-qkv.yml` explicitly runs the real checkpoint test on Linux CPU; compilation, rustfmt, Clippy and the test's results must be inspected on the exact PR HEAD before merge.
