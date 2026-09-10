use std::collections::BTreeMap;
use std::fs;
use std::io;

use crate::bpe::BpeTokenizer;
use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;
use crate::train::optimizer::TrainOptimizer;
use crate::train::scheduler::WarmupCosineSchedule;

use scirust_core::autodiff::reverse::Tape;
use scirust_core::autodiff::scheduler::LrSchedule;

#[derive(Debug)]
pub struct SftExample {
    pub instruction: String,
    pub output: String,
    pub tool_calls: Option<Vec<ToolCallExample>>,
}

#[derive(Debug)]
pub struct ToolCallExample {
    pub name: String,
    pub params: Vec<(String, String)>,
}

pub struct SftDataset {
    examples: Vec<SftExample>,
    #[allow(dead_code)]
    position: usize,
}

impl SftDataset {
    pub fn from_jsonl(path: &str) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut examples = Vec::new();

        for (line_number, line) in std::io::BufRead::lines(reader).enumerate()
        {
            let line = line?;
            if line.trim().is_empty()
            {
                continue;
            }
            let example = parse_sft_example(&line).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("invalid SFT example on line {}: {error}", line_number + 1),
                )
            })?;
            examples.push(example);
        }

        Ok(Self {
            examples,
            position: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Parse one SFT record and enforce the same one-action-per-turn contract that
/// [`crate::agentic::AgentRouter`] consumes at inference time.
///
/// A training target is either a final textual response or exactly one tool
/// call. Mixing text with a call, or teaching several calls in one assistant
/// turn, would produce output that `AgentRouter::parse_action` cannot execute.
fn parse_sft_example(line: &str) -> io::Result<SftExample> {
    let json: serde_json::Value =
        serde_json::from_str(line).map_err(|error| invalid_data(error.to_string()))?;
    let object = json
        .as_object()
        .ok_or_else(|| invalid_data("SFT record must be a JSON object"))?;

    let instruction = object
        .get("instruction")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_data("SFT `instruction` must be a string"))?
        .to_string();
    if instruction.trim().is_empty()
    {
        return Err(invalid_data("SFT `instruction` must not be empty"));
    }

    let output = match object.get("output")
    {
        Some(value) => value
            .as_str()
            .ok_or_else(|| invalid_data("SFT `output` must be a string"))?
            .to_string(),
        None => String::new(),
    };

    let tool_calls = match object.get("tool_calls")
    {
        None | Some(serde_json::Value::Null) => None,
        Some(value) =>
        {
            let calls = value
                .as_array()
                .ok_or_else(|| invalid_data("SFT `tool_calls` must be an array"))?;
            if calls.len() > 1
            {
                return Err(invalid_data(
                    "SFT records may contain at most one tool call per assistant turn",
                ));
            }
            if calls.is_empty()
            {
                None
            }
            else
            {
                let call = calls[0]
                    .as_object()
                    .ok_or_else(|| invalid_data("SFT tool call must be a JSON object"))?;
                let name = call
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| invalid_data("SFT tool call `name` must be a string"))?;
                if name.trim().is_empty()
                {
                    return Err(invalid_data("SFT tool call `name` must not be empty"));
                }
                let params = call
                    .get("params")
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| invalid_data("SFT tool call `params` must be an object"))?
                    .iter()
                    .map(|(key, value)| {
                        value
                            .as_str()
                            .map(|value| (key.clone(), value.to_string()))
                            .ok_or_else(|| {
                                invalid_data(format!("SFT tool parameter `{key}` must be a string"))
                            })
                    })
                    .collect::<io::Result<Vec<_>>>()?;

                Some(vec![ToolCallExample {
                    name: name.to_string(),
                    params,
                }])
            }
        },
    };

    if tool_calls.is_some() && !output.trim().is_empty()
    {
        return Err(invalid_data(
            "SFT assistant turn cannot contain both `output` text and a tool call",
        ));
    }
    if tool_calls.is_none() && output.trim().is_empty()
    {
        return Err(invalid_data(
            "SFT assistant turn must contain either `output` text or one tool call",
        ));
    }

    Ok(SftExample {
        instruction,
        output,
        tool_calls,
    })
}

/// Render the exact assistant target used for SFT.
///
/// Tool-call examples use the same compact JSON shape accepted by
/// `AgentRouter`: `{\"name\": ..., \"params\": {...}}`. Parameters are sorted so
/// an identical logical call always has byte-identical training text.
pub fn format_sft_target(example: &SftExample) -> String {
    let Some(tool_calls) = example.tool_calls.as_ref()
    else
    {
        return example.output.clone();
    };
    let Some(call) = tool_calls.first()
    else
    {
        return example.output.clone();
    };

    let params = call
        .params
        .iter()
        .cloned()
        .collect::<BTreeMap<String, String>>();
    serde_json::to_string(&serde_json::json!({
        "name": call.name.as_str(),
        "params": params,
    }))
    .expect("serializing a validated SFT tool call cannot fail")
}

pub fn format_sft_prompt(example: &SftExample, tokenizer: &BpeTokenizer) -> Vec<usize> {
    let bos = tokenizer.special_id("<bos>");
    let eos = tokenizer.special_id("<eos>");

    let mut tokens = vec![bos];
    tokens.extend(tokenizer.encode("user: "));
    tokens.extend(tokenizer.encode(&example.instruction));
    tokens.push(eos);

    tokens.push(bos);
    tokens.extend(tokenizer.encode("assistant: "));
    tokens.extend(tokenizer.encode(&format_sft_target(example)));
    tokens.push(eos);

    tokens
}

#[allow(clippy::too_many_arguments)] // training entry point mirrors the CLI knobs
pub fn sft_train(
    model: &mut SciAgentModel,
    dataset: &SftDataset,
    _config: &SciAgentConfig,
    tokenizer: &BpeTokenizer,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    max_seq_len: usize,
) {
    let total_steps = dataset.len() * epochs / batch_size;
    let mut opt = TrainOptimizer::new_muon(lr);
    let scheduler = WarmupCosineSchedule::new(lr, lr * 0.1, total_steps / 20, total_steps);

    let mut step = 0usize;
    for epoch in 0..epochs
    {
        let mut epoch_loss = 0.0f64;

        for chunk in dataset.examples.chunks(batch_size)
        {
            let tape = Tape::new();
            let mut all_inputs = Vec::new();
            let mut all_targets = Vec::new();

            for ex in chunk
            {
                let tokens = format_sft_prompt(ex, tokenizer);
                if tokens.len() < 2
                {
                    continue;
                }
                let seq = &tokens[..tokens.len().min(max_seq_len)];
                let inputs: Vec<usize> = seq[..seq.len() - 1].to_vec();
                let targets: Vec<usize> = seq[1..].to_vec();
                all_inputs.extend(inputs);
                all_targets.extend(targets);
            }

            if all_inputs.is_empty()
            {
                continue;
            }

            let seq_len = all_inputs.len() / chunk.len();
            let logits = model.forward(&tape, &all_inputs, seq_len);
            let loss = crate::train::cross_entropy_loss(&tape, logits, &all_targets);
            tape.backward(loss.idx());
            let loss_val = tape.value(loss.idx()).data[0] as f64;
            epoch_loss += loss_val;

            let lr = scheduler.lr_at(step);
            opt.set_lr(lr);
            opt.clip_grad_norm(&tape, 1.0);
            let params = model.parameter_indices();
            opt.step(&params, &tape);
            model.sync(&tape);

            step += 1;

            if step.is_multiple_of(10)
            {
                println!("[SFT Epoch {epoch} Step {step}] loss: {loss_val:.4} | lr: {lr:.8}");
            }
        }

        println!(
            "[SFT Epoch {epoch}] avg loss: {:.4}",
            epoch_loss / dataset.len() as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BpeTrainer;

    #[test]
    fn test_format_prompt() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec!["user: hello world assistant: hi".to_string()];
        let tok = trainer.train(&texts);

        let ex = SftExample {
            instruction: "hello".to_string(),
            output: "world".to_string(),
            tool_calls: None,
        };
        let tokens = format_sft_prompt(&ex, &tok);
        assert!(tokens.len() > 2);
        assert_eq!(tokens[0], 1); // <bos>
    }

    #[test]
    fn tool_call_target_matches_agent_router_protocol() {
        let ex = SftExample {
            instruction: "find Muon".to_string(),
            output: String::new(),
            tool_calls: Some(vec![ToolCallExample {
                name: "search".to_string(),
                params: vec![
                    ("pattern".to_string(), "Muon".to_string()),
                    ("path".to_string(), "scirust-core".to_string()),
                ],
            }]),
        };

        assert_eq!(
            format_sft_target(&ex),
            r#"{"name":"search","params":{"path":"scirust-core","pattern":"Muon"}}"#
        );
    }

    #[test]
    fn parser_rejects_unexecutable_multi_action_targets() {
        let two_calls = r#"{"instruction":"inspect","output":"","tool_calls":[{"name":"status","params":{}},{"name":"search","params":{"pattern":"TODO"}}]}"#;
        assert!(parse_sft_example(two_calls).is_err());

        let mixed = r#"{"instruction":"inspect","output":"done","tool_calls":[{"name":"status","params":{}}]}"#;
        assert!(parse_sft_example(mixed).is_err());
    }

    #[test]
    fn parser_rejects_non_string_tool_parameters() {
        let line = r#"{"instruction":"test","output":"","tool_calls":[{"name":"test","params":{"crate":"scirust-core","retries":2}}]}"#;
        let error = parse_sft_example(line).unwrap_err();
        assert!(error.to_string().contains("retries"));
    }
}
