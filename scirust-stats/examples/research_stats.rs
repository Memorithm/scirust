//! One bounded JSON request on stdin, one versioned JSON response on stdout.
//!
//! Example (build with `cargo build -p scirust-stats --example research_stats`):
//! `echo '{"schema":1,"operation":"holm","p_values":[0.01,0.04,0.03]}' | target/debug/examples/research_stats`
//! Input is at most 1 MiB; errors exit 21 without returning numeric results.
//! No files, network, product protocols, subprocesses or model execution.

use scirust_stats::{resampling, sensitivity};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    PairedMeanPercentile {
        schema: u8,
        contrasts: Vec<f64>,
        resamples: usize,
        confidence: f64,
        seed: String,
    },
    Holm {
        schema: u8,
        p_values: Vec<f64>,
    },
    Morris {
        schema: u8,
        inputs: Vec<Vec<f64>>,
        outputs: Vec<f64>,
    },
    Sobol {
        schema: u8,
        a: Vec<f64>,
        b: Vec<f64>,
        ab: Vec<Vec<f64>>,
    },
}

fn evaluate(request: Request) -> Result<Value, Box<dyn std::error::Error>> {
    let version = match &request
    {
        Request::PairedMeanPercentile { schema, .. }
        | Request::Holm { schema, .. }
        | Request::Morris { schema, .. }
        | Request::Sobol { schema, .. } => *schema,
    };
    if version != 1
    {
        return Err("unsupported schema".into());
    }
    let result = match request
    {
        Request::PairedMeanPercentile {
            contrasts,
            resamples,
            confidence,
            seed,
            ..
        } =>
        {
            let parsed = seed.parse::<u64>()?;
            if parsed.to_string() != seed
            {
                return Err("seed must be canonical unsigned decimal".into());
            }
            let value =
                resampling::paired_mean_percentile(&contrasts, resamples, confidence, parsed)?;
            json!({"estimate":value.estimate,"lower":value.lower,"upper":value.upper,"confidence":value.confidence,"units":value.units,"resamples":value.resamples,"method":"paired-mean-percentile-type7-splitmix64/v1"})
        },
        Request::Holm { p_values, .. } =>
        {
            json!({"adjusted":resampling::holm_adjust(&p_values)?,"method":"holm-step-down/v1"})
        },
        Request::Morris {
            inputs, outputs, ..
        } =>
        {
            let value = sensitivity::morris_effects(&inputs, &outputs)?;
            json!({"mu":value.mu,"mu_star":value.mu_star,"sigma":value.sigma,"trajectories":value.trajectories,"method":"morris-unscaled-elementary-effects/v1"})
        },
        Request::Sobol { a, b, ab, .. } =>
        {
            let value = sensitivity::sobol_first_total(&a, &b, &ab)?;
            json!({"first":value.first,"total":value.total,"base_samples":value.base_samples,"method":"saltelli-2010-first-total-centered/v1"})
        },
    };
    Ok(json!({"schema":1,"status":"computed","result":result}))
}

fn run() -> Result<Value, Box<dyn std::error::Error>> {
    let mut raw = Vec::new();
    io::stdin().take(1024 * 1024 + 1).read_to_end(&mut raw)?;
    if raw.len() > 1024 * 1024
    {
        return Err("input exceeds 1 MiB".into());
    }
    evaluate(serde_json::from_slice(&raw)?)
}

fn main() {
    match run()
    {
        Ok(result) => println!("{result}"),
        Err(error) =>
        {
            println!(
                "{}",
                json!({"schema":1,"status":"rejected","error":error.to_string()})
            );
            std::process::exit(21);
        },
    }
}
