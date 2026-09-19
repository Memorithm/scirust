//! Reproducible process-level optimizer overhead microbenchmark.
//!
//! One invocation runs one SciRust TPE configuration and emits one CSV record.
//! External orchestration should launch each repeat in a fresh process so Linux
//! peak RSS (VmHWM) is interpretable.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use scirust_opt_core::{
    Condition, Distribution, ParamId, ParamValue, ParameterSpec, SearchSpace, Study, TrialOutcome,
};
use scirust_opt_tpe::{Direction, TpeConfig, TpeSampler};

const LOW: f64 = -5.0;
const HIGH: f64 = 5.0;

#[derive(Debug, Clone, Copy)]
struct Config {
    dims: usize,
    trials: usize,
    seed: u64,
}

fn parse_usize(flag: &str, value: Option<String>) -> usize {
    value
        .unwrap_or_else(|| panic!("missing value for {flag}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid integer for {flag}"))
}

fn parse_u64(flag: &str, value: Option<String>) -> u64 {
    value
        .unwrap_or_else(|| panic!("missing value for {flag}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid integer for {flag}"))
}

fn parse_args() -> Config {
    let mut dims = None;
    let mut trials = None;
    let mut seed = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dims" => dims = Some(parse_usize("--dims", args.next())),
            "--trials" => trials = Some(parse_usize("--trials", args.next())),
            "--seed" => seed = Some(parse_u64("--seed", args.next())),
            "--header" => {
                println!(
                    "engine,version,dims,trials,seed,proposal_ns,tell_ns,total_ns,proposal_ns_per_trial,tell_ns_per_trial,total_ns_per_trial,rss_start_kib,rss_end_kib,peak_rss_kib,best"
                );
                std::process::exit(0);
            },
            "--help" | "-h" => {
                println!(
                    "usage: scirust-opt-bench --dims N --trials N --seed N\n       scirust-opt-bench --header"
                );
                std::process::exit(0);
            },
            other => panic!("unknown argument: {other}"),
        }
    }

    let dims = dims.unwrap_or(10);
    let trials = trials.unwrap_or(1_000);
    let seed = seed.unwrap_or(0x5eed);
    assert!(dims > 0, "--dims must be greater than zero");
    assert!(trials > 0, "--trials must be greater than zero");
    Config { dims, trials, seed }
}

fn search_space(dims: usize) -> SearchSpace {
    let params = (0..dims)
        .map(|index| {
            ParameterSpec::new(
                ParamId::new(u32::try_from(index).expect("dimension fits ParamId")),
                format!("x{index}"),
                Distribution::Uniform {
                    low: LOW,
                    high: HIGH,
                },
                Condition::Always,
            )
        })
        .collect();
    SearchSpace::compile(params).expect("benchmark search space is valid")
}

fn target(index: usize) -> f64 {
    ((index % 5) as f64 - 2.0) * 0.5
}

fn objective(candidate: &scirust_opt_core::Candidate, dims: usize) -> f64 {
    let mut total = 0.0;
    for index in 0..dims {
        let id = ParamId::new(u32::try_from(index).expect("dimension fits ParamId"));
        let ParamValue::Float(value) = candidate.value(id).expect("benchmark value is assigned")
        else {
            panic!("benchmark parameter must be Float");
        };
        let delta = value - target(index);
        total += delta * delta;
    }
    black_box(total)
}

fn proc_status_kib(field: &str) -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find_map(|line| {
            let (name, rest) = line.split_once(':')?;
            if name != field {
                return None;
            }
            rest.split_whitespace().next()?.parse().ok()
        })
        .unwrap_or(0)
}

fn main() {
    let config = parse_args();
    let rss_start_kib = proc_status_kib("VmRSS");

    let mut study = Study::new(search_space(config.dims), 1).expect("study");
    let mut sampler = TpeSampler::with_config(
        Direction::Minimize,
        config.seed,
        TpeConfig {
            n_startup_trials: 10,
            n_ei_candidates: 24,
            prior_weight: 1.0,
            consider_magic_clip: true,
            consider_endpoints: false,
            constant_liar: true,
        },
    )
    .expect("TPE config");

    let total_start = Instant::now();
    let mut proposal_ns = 0_u128;
    let mut tell_ns = 0_u128;
    let mut best = f64::INFINITY;

    for _ in 0..config.trials {
        let proposal_start = Instant::now();
        let proposal = study.ask(&mut sampler).expect("proposal");
        proposal_ns += proposal_start.elapsed().as_nanos();

        let value = objective(&proposal.candidate, config.dims);
        best = best.min(value);

        let tell_start = Instant::now();
        study
            .tell(proposal.trial, TrialOutcome::Complete(vec![value]))
            .expect("tell");
        tell_ns += tell_start.elapsed().as_nanos();
    }

    let total_ns = total_start.elapsed().as_nanos();
    let rss_end_kib = proc_status_kib("VmRSS");
    let peak_rss_kib = proc_status_kib("VmHWM");
    let trials = config.trials as u128;

    println!(
        "scirust,{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{},{},{},{:.17e}",
        env!("CARGO_PKG_VERSION"),
        config.dims,
        config.trials,
        config.seed,
        proposal_ns,
        tell_ns,
        total_ns,
        proposal_ns as f64 / trials as f64,
        tell_ns as f64 / trials as f64,
        total_ns as f64 / trials as f64,
        rss_start_kib,
        rss_end_kib,
        peak_rss_kib,
        best
    );
}
