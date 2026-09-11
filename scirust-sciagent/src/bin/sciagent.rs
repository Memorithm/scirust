use std::path::PathBuf;

use clap::{Parser, Subcommand};
use scirust_sciagent::agentic::{AgentAction, AgentRouter, Tool};
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::generate::Generator;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{load_checkpoint, read_meta};

type CliResult<T> = Result<T, (i32, String)>;

const MAX_AGENT_STEPS: usize = 32;
const MAX_AGENT_OBSERVATION_CHARS: usize = 16 * 1024;

#[derive(Parser)]
#[command(
    name = "sciagent",
    about = "SCIAGENT — determinist SLM for scirust ecosystem"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(global = true, long, default_value = "350m")]
    model: String,

    #[arg(global = true, long, default_value_t = 42)]
    seed: u64,

    #[arg(global = true, long, default_value_t = 2048)]
    max_tokens: usize,

    #[arg(global = true, long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(global = true, long, default_value_t = 0)]
    top_k: usize,

    #[arg(global = true, long, default_value_t = 1.0)]
    top_p: f32,

    #[arg(global = true, long, default_value_t = 1.0)]
    repetition_penalty: f32,

    #[arg(global = true, long)]
    json: bool,

    #[arg(global = true, long)]
    checkpoint: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    Ask {
        prompt: String,
    },
    Chat,
    /// Run a bounded repository-aware tool loop for one maintenance task.
    Agent {
        task: String,
        #[arg(long, default_value_t = 8)]
        max_steps: usize,
    },
    Explain {
        path: PathBuf,
        #[arg(long)]
        lines: Option<String>,
    },
    Generate {
        description: String,
    },
    Info,
}

fn report_cli_error((code, message): (i32, String)) -> ! {
    eprintln!("error: {message}");
    std::process::exit(code);
}

/// Build a model only from a valid, explicitly supplied checkpoint.
///
/// A freshly initialized model contains random weights. All inference paths
/// must fail closed when the checkpoint is missing or invalid.
fn build_model(cli: &Cli) -> CliResult<SciAgentModel> {
    let checkpoint = cli.checkpoint.as_ref().ok_or_else(|| {
        (
            2,
            String::from(
                "missing required `--checkpoint PATH`; refusing to run with random weights",
            ),
        )
    })?;

    eprintln!("Loading checkpoint from {:?} ...", checkpoint);

    let meta = read_meta(checkpoint).map_err(|error| {
        (
            1,
            format!(
                "cannot read checkpoint metadata from `{}`: {error}",
                checkpoint.display()
            ),
        )
    })?;

    let mut model = SciAgentModel::new(&meta.config);
    load_checkpoint(&mut model, checkpoint).map_err(|error| {
        (
            1,
            format!(
                "cannot load checkpoint from `{}`: {error}",
                checkpoint.display()
            ),
        )
    })?;

    Ok(model)
}

/// Resolve `info` configuration without allocating random model weights.
fn info_config(cli: &Cli) -> CliResult<SciAgentConfig> {
    if let Some(checkpoint) = cli.checkpoint.as_ref()
    {
        read_meta(checkpoint)
            .map(|meta| meta.config)
            .map_err(|error| {
                (
                    1,
                    format!(
                        "cannot read checkpoint metadata from `{}`: {error}",
                        checkpoint.display()
                    ),
                )
            })
    }
    else
    {
        Ok(get_config(&cli.model))
    }
}

fn main() {
    let cli = Cli::parse();

    if matches!(&cli.command, Command::Info)
    {
        let config = info_config(&cli).unwrap_or_else(|error| report_cli_error(error));
        cmd_info(&config, &cli);
        return;
    }

    let mut model = build_model(&cli).unwrap_or_else(|error| report_cli_error(error));

    match &cli.command
    {
        Command::Ask { prompt } => cmd_ask(&mut model, prompt, &cli),
        Command::Chat => cmd_chat(&mut model, &cli),
        Command::Agent { task, max_steps } =>
        {
            cmd_agent(&mut model, task, *max_steps, &cli)
                .unwrap_or_else(|error| report_cli_error(error));
        },
        Command::Explain { path, lines } => cmd_explain(&mut model, path, lines.as_deref(), &cli),
        Command::Generate { description } => cmd_generate(&mut model, description, &cli),
        Command::Info => unreachable!("info returns before model construction"),
    }
}

fn get_config(model_name: &str) -> SciAgentConfig {
    match model_name
    {
        "debug" => SciAgentConfig::debug(),
        "small" | "Small" => SciAgentConfig::small(),
        "350m" | "350M" => SciAgentConfig::sciagent_350m(),
        "7b" | "7B" => SciAgentConfig::sciagent_7b(),
        _ =>
        {
            eprintln!("Unknown model '{model_name}', using 350M");
            SciAgentConfig::sciagent_350m()
        },
    }
}

fn generator(cli: &Cli, config: &SciAgentConfig) -> Generator {
    Generator::new(config)
        .with_temperature(cli.temperature)
        .with_top_k(cli.top_k)
        .with_top_p(cli.top_p)
        .with_repetition_penalty(cli.repetition_penalty)
}

/// `Generator::generate` returns the prompt followed by newly generated ids.
/// Keep that contract internal to the CLI so callers only see the continuation.
fn generate_continuation(
    model: &mut SciAgentModel,
    prompt: &[usize],
    max_tokens: usize,
    seed: u64,
    cli: &Cli,
) -> Vec<usize> {
    let generated = generator(cli, &model.config).generate(model, prompt, max_tokens, seed);
    generated.get(prompt.len()..).unwrap_or_default().to_vec()
}

fn cmd_ask(model: &mut SciAgentModel, prompt: &str, cli: &Cli) {
    let vocab = model.config.vocab_size;
    let tokens = tokenize_with_vocab(prompt, vocab);
    let continuation = generate_continuation(model, &tokens, cli.max_tokens, cli.seed, cli);
    let text = detokenize_with_vocab(&continuation, vocab);

    if cli.json
    {
        let output = serde_json::json!({
            "prompt": prompt,
            "response": text,
            "tokens": continuation.len(),
            "seed": cli.seed,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    }
    else
    {
        println!("{text}");
    }
}

fn cmd_chat(model: &mut SciAgentModel, cli: &Cli) {
    let vocab = model.config.vocab_size;
    let max_seq = model.config.max_seq_len;
    println!("SCIAGENT chat (Ctrl+D to exit)");
    let mut history: Vec<usize> = Vec::new();

    loop
    {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        print!("> ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let mut line = String::new();
        match stdin.lock().read_line(&mut line)
        {
            Ok(0) | Err(_) => break,
            Ok(_) =>
            {},
        }
        let line = line.trim();
        if line.is_empty()
        {
            continue;
        }

        history.extend(tokenize_with_vocab(
            &format!("user: {line}\nassistant: "),
            vocab,
        ));
        if history.len() > max_seq
        {
            let drain = history.len() - max_seq;
            history.drain(..drain);
        }

        let continuation =
            generate_continuation(model, &history, cli.max_tokens.min(512), cli.seed, cli);
        let text = detokenize_with_vocab(&continuation, vocab);
        println!("{text}");

        history.extend(continuation);
        history.extend(tokenize_with_vocab("\n", vocab));
        if history.len() > max_seq
        {
            let drain = history.len() - max_seq;
            history.drain(..drain);
        }
    }
}

fn agent_tool_catalog() -> String {
    let mut catalog = String::new();
    for tool in Tool::builtins()
    {
        catalog.push_str(tool.name);
        catalog.push('(');
        for (index, parameter) in tool.parameters.iter().enumerate()
        {
            if index > 0
            {
                catalog.push_str(", ");
            }
            catalog.push_str(parameter.name);
            if !parameter.required
            {
                catalog.push('?');
            }
        }
        catalog.push_str("): ");
        catalog.push_str(tool.description);
        catalog.push('\n');
    }
    catalog
}

fn bounded_observation(text: &str) -> String {
    let mut chars = text.chars();
    let bounded: String = chars.by_ref().take(MAX_AGENT_OBSERVATION_CHARS).collect();
    if chars.next().is_some()
    {
        format!("{bounded}\n[observation truncated]")
    }
    else
    {
        bounded
    }
}

fn agent_prompt(task: &str, transcript: &str, tools: &str) -> String {
    format!(
        "{transcript}\n\
You are SCIAGENT operating on the live SciRust workspace. Ground every repository claim in tool observations. Never claim a build, test, file read, or search succeeded unless its observation says so. Use at most one tool call per turn. To call a tool, output only compact JSON: {{\"name\":\"TOOL\",\"params\":{{\"key\":\"value\"}}}}. When the task is complete or cannot be completed with the available tools, answer with plain text instead of JSON.\n\
Available tools:\n{tools}\
Task: {task}\n\
Next action:"
    )
}

fn cmd_agent(model: &mut SciAgentModel, task: &str, max_steps: usize, cli: &Cli) -> CliResult<()> {
    if max_steps == 0 || max_steps > MAX_AGENT_STEPS
    {
        return Err((2, format!("--max-steps must be in 1..={MAX_AGENT_STEPS}")));
    }

    let router = AgentRouter::new();
    let tools = agent_tool_catalog();
    let vocab = model.config.vocab_size;
    let mut transcript = String::new();
    let mut trace = Vec::new();

    for step in 0..max_steps
    {
        let prompt = agent_prompt(task, &transcript, &tools);
        let prompt_tokens = tokenize_with_vocab(&prompt, vocab);
        let continuation = generate_continuation(
            model,
            &prompt_tokens,
            cli.max_tokens.min(512),
            cli.seed.wrapping_add(step as u64),
            cli,
        );
        let model_text = detokenize_with_vocab(&continuation, vocab);
        let action = router.parse_action(model_text.trim());

        match action
        {
            AgentAction::Call { tool, params } =>
            {
                let executable = AgentAction::Call {
                    tool: tool.clone(),
                    params: params.clone(),
                };
                let observation = router.execute(&executable);
                let observation = bounded_observation(&observation);
                trace.push(serde_json::json!({
                    "step": step + 1,
                    "action": "tool",
                    "tool": tool,
                    "params": params,
                    "observation": observation,
                }));
                transcript.push_str("\nassistant: ");
                transcript.push_str(model_text.trim());
                transcript.push_str("\ntool observation: ");
                transcript.push_str(&observation);
                transcript.push('\n');
            },
            AgentAction::Respond { text } =>
            {
                if cli.json
                {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "task": task,
                            "completed": true,
                            "response": text,
                            "turns": trace,
                        }))
                        .unwrap()
                    );
                }
                else
                {
                    println!("{text}");
                }
                return Ok(());
            },
            AgentAction::Abstain =>
            {
                let text = "I abstain — confidence below threshold.";
                if cli.json
                {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "task": task,
                            "completed": false,
                            "abstained": true,
                            "response": text,
                            "turns": trace,
                        }))
                        .unwrap()
                    );
                }
                else
                {
                    println!("{text}");
                }
                return Ok(());
            },
        }
    }

    Err((
        3,
        format!("agent stopped after reaching the {max_steps}-step safety bound"),
    ))
}

fn cmd_explain(model: &mut SciAgentModel, path: &PathBuf, lines: Option<&str>, cli: &Cli) {
    let content = match std::fs::read_to_string(path)
    {
        Ok(c) => c,
        Err(e) =>
        {
            eprintln!("Cannot read {:?}: {e}", path);
            return;
        },
    };

    let excerpt = match lines
    {
        Some(range) =>
        {
            let (start, end) = match parse_line_range(range)
            {
                Ok(bounds) => bounds,
                Err(error) =>
                {
                    eprintln!("Invalid --lines value `{range}`: {error}");
                    return;
                },
            };
            content
                .lines()
                .skip(start.saturating_sub(1))
                .take(end - start + 1)
                .collect::<Vec<_>>()
                .join("\n")
        },
        None => content.chars().take(2000).collect::<String>(),
    };

    let prompt = format!("Explain this code:\n```rust\n{excerpt}\n```");
    cmd_ask(model, &prompt, cli);
}

fn parse_line_range(range: &str) -> Result<(usize, usize), &'static str> {
    let (start, end) = match range.split_once('-')
    {
        Some((start, end)) =>
        {
            if start.is_empty() || end.is_empty() || end.contains('-')
            {
                return Err("expected START-END with two positive line numbers");
            }
            let start = start
                .parse::<usize>()
                .map_err(|_| "START is not a positive line number")?;
            let end = end
                .parse::<usize>()
                .map_err(|_| "END is not a positive line number")?;
            (start, end)
        },
        None =>
        {
            let start = range
                .parse::<usize>()
                .map_err(|_| "expected a positive line number or START-END")?;
            let end = start
                .checked_add(30)
                .ok_or("line range overflows this platform")?;
            (start, end)
        },
    };

    if start == 0 || end == 0
    {
        return Err("line numbers are one-based and must be greater than zero");
    }
    if end < start
    {
        return Err("END must be greater than or equal to START");
    }
    Ok((start, end))
}

fn cmd_generate(model: &mut SciAgentModel, description: &str, cli: &Cli) {
    let prompt = format!("Write Rust code for: {description}");
    cmd_ask(model, &prompt, cli);
}

fn cmd_info(config: &SciAgentConfig, _cli: &Cli) {
    println!("=== SCIAGENT Model Info ===");
    println!("Name: scirust-sciagent");
    println!("Architecture: GQA + SwiGLU + RoPE + RMSNorm");
    println!("Vocab size: {}", config.vocab_size);
    println!("d_model: {}", config.d_model);
    println!("n_layers: {}", config.n_layers);
    println!(
        "n_heads: {} ({} KV heads)",
        config.n_heads, config.n_kv_heads
    );
    println!("d_ff: {}", config.d_ff);
    println!("max_seq_len: {}", config.max_seq_len);
    println!(
        "Total parameters: {}",
        fmt_params(config.total_parameters())
    );
    println!("Tie embeddings: {}", config.tie_embeddings);
}

fn tokenize_with_vocab(text: &str, vocab_size: usize) -> Vec<usize> {
    if let Ok(tok) = BpeTokenizer::from_embedded()
    {
        if tok.vocab_size() <= vocab_size
        {
            tok.encode_with_special(text, true, false)
        }
        else
        {
            text.bytes().map(|b| b as usize).collect()
        }
    }
    else
    {
        text.bytes().map(|b| b as usize).collect()
    }
}

fn detokenize_with_vocab(ids: &[usize], vocab_size: usize) -> String {
    if let Ok(tok) = BpeTokenizer::from_embedded()
    {
        if tok.vocab_size() <= vocab_size
        {
            tok.decode(ids)
        }
        else
        {
            ids.iter()
                .filter_map(|&id| char::from_u32(id as u32))
                .collect()
        }
    }
    else
    {
        ids.iter()
            .filter_map(|&id| char::from_u32(id as u32))
            .collect()
    }
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000_000
    {
        format!("{:.1}B", n as f64 / 1e9)
    }
    else if n >= 1_000_000
    {
        format!("{:.1}M", n as f64 / 1e6)
    }
    else
    {
        format!("{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cli(checkpoint: Option<PathBuf>) -> Cli {
        Cli {
            command: Command::Ask {
                prompt: String::from("fn main() {}"),
            },
            model: String::from("debug"),
            seed: 42,
            max_tokens: 16,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            json: false,
            checkpoint,
        }
    }

    #[test]
    fn line_ranges_are_validated() {
        assert_eq!(parse_line_range("4-9"), Ok((4, 9)));
        assert_eq!(parse_line_range("4"), Ok((4, 34)));
        assert!(parse_line_range("0-2").is_err());
        assert!(parse_line_range("9-4").is_err());
        assert!(parse_line_range("bad").is_err());
        assert!(parse_line_range("1-2-3").is_err());
    }

    #[test]
    fn inference_requires_an_explicit_checkpoint() {
        let error = match build_model(&test_cli(None))
        {
            Ok(_) => panic!("randomly initialized model must not be accepted"),
            Err(error) => error,
        };
        assert_eq!(error.0, 2);
        assert!(error.1.contains("--checkpoint PATH"));
        assert!(error.1.contains("random weights"));
    }

    #[test]
    fn invalid_checkpoint_fails_closed() {
        let cli = test_cli(Some(PathBuf::from("/path/that/does/not/exist")));
        let error = match build_model(&cli)
        {
            Ok(_) => panic!("invalid checkpoint must not produce a model"),
            Err(error) => error,
        };
        assert_eq!(error.0, 1);
        assert!(error.1.contains("cannot read checkpoint metadata"));
    }

    #[test]
    fn info_config_does_not_allocate_model_weights() {
        let config = info_config(&test_cli(None)).expect("info config should resolve");
        assert_eq!(config.d_model, SciAgentConfig::debug().d_model);
    }

    #[test]
    fn tool_catalog_is_generated_from_builtin_contracts() {
        let catalog = agent_tool_catalog();
        assert!(catalog.contains("search(pattern, path?)"));
        assert!(catalog.contains("read(path, lines?)"));
        assert!(catalog.contains("build(crate)"));
        assert!(catalog.contains("test(crate, test?)"));
        assert!(catalog.contains("status()"));
    }

    #[test]
    fn observation_bound_is_enforced() {
        let oversized = "x".repeat(MAX_AGENT_OBSERVATION_CHARS + 10);
        let bounded = bounded_observation(&oversized);
        assert!(bounded.ends_with("[observation truncated]"));
        assert!(bounded.len() < oversized.len() + 32);
    }

    #[test]
    fn agent_step_bounds_are_rejected_before_inference() {
        let cli = test_cli(None);
        let config = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&config);
        assert!(cmd_agent(&mut model, "inspect", 0, &cli).is_err());
        assert!(cmd_agent(&mut model, "inspect", MAX_AGENT_STEPS + 1, &cli).is_err());
    }
}
