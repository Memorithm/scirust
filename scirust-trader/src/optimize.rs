//! Parameter optimization with a bounded, deterministic grid search.
//!
//! The holdout protocol remains unchanged: parameter selection uses the train
//! split, finalists are then evaluated on the untouched holdout. Grid generation
//! is lazy and budgeted: the full Cartesian product is never materialized before
//! `max_combos` is applied.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::backtest::{BacktestConfig, run_backtest};
use crate::market::Candle;
use crate::robustness::walk_forward;
use crate::strategy::strategy_from_spec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamAxis {
    pub name: String,
    pub values: Vec<f32>,
}

impl ParamAxis {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    ReturnConsistency,
    Consistency,
    MeanReturn,
    WorstWindow,
    Sharpe,
}

impl Objective {
    pub fn parse(s: &str) -> Option<Objective> {
        match s.trim().to_lowercase().as_str()
        {
            "return_consistency" | "default" => Some(Objective::ReturnConsistency),
            "consistency" => Some(Objective::Consistency),
            "mean_return" | "return" => Some(Objective::MeanReturn),
            "worst_window" | "worst" | "minimax" => Some(Objective::WorstWindow),
            "sharpe" => Some(Objective::Sharpe),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self
        {
            Objective::ReturnConsistency => "return×consistency (out-of-sample)",
            Objective::Consistency => "walk-forward consistency",
            Objective::MeanReturn => "mean walk-forward return",
            Objective::WorstWindow => "worst walk-forward window",
            Objective::Sharpe => "train Sharpe",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeConfig {
    pub train_frac: f32,
    pub wf_windows: usize,
    pub objective: Objective,
    pub top_k: usize,
    /// Maximum number of parameter combinations evaluated.
    pub max_combos: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            train_frac: 0.7,
            wf_windows: 4,
            objective: Objective::ReturnConsistency,
            top_k: 10,
            max_combos: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub params: BTreeMap<String, f32>,
    pub train_return: f32,
    pub train_sharpe: f32,
    pub train_consistency: f32,
    pub train_mean_window: f32,
    pub train_worst_window: f32,
    pub objective_score: f32,
    pub holdout_return: f32,
    pub holdout_sharpe: f32,
    pub holdout_max_drawdown: f32,
    pub holdout_trades: usize,
    pub overfit_gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeReport {
    pub strategy: String,
    pub objective: String,
    pub train_bars: usize,
    pub holdout_bars: usize,
    /// Total valid grid combinations before `max_combos` sampling.
    pub grid_size: usize,
    /// Combinations that reached strategy evaluation on the train split.
    pub num_evaluated: usize,
    pub truncated: bool,
    pub best: Candidate,
    pub leaderboard: Vec<Candidate>,
    pub verdict: String,
}

pub fn default_axes(strategy_name: &str) -> Vec<ParamAxis> {
    match strategy_name
    {
        "sma_cross" | "ema_cross" => vec![
            ParamAxis::new("fast", vec![5.0, 10.0, 15.0, 20.0]),
            ParamAxis::new("slow", vec![30.0, 50.0, 100.0, 200.0]),
        ],
        "rsi_reversion" => vec![
            ParamAxis::new("period", vec![7.0, 14.0, 21.0]),
            ParamAxis::new("oversold", vec![20.0, 30.0]),
            ParamAxis::new("overbought", vec![70.0, 80.0]),
        ],
        "macd" => vec![
            ParamAxis::new("fast", vec![8.0, 12.0]),
            ParamAxis::new("slow", vec![21.0, 26.0]),
            ParamAxis::new("signal", vec![9.0]),
        ],
        "bollinger_breakout" => vec![
            ParamAxis::new("period", vec![14.0, 20.0, 30.0]),
            ParamAxis::new("k", vec![1.5, 2.0, 2.5]),
        ],
        "donchian_breakout" => vec![ParamAxis::new("period", vec![10.0, 20.0, 55.0])],
        "supertrend" => vec![
            ParamAxis::new("period", vec![7.0, 10.0, 14.0]),
            ParamAxis::new("mult", vec![2.0, 3.0, 4.0]),
        ],
        "momentum" => vec![ParamAxis::new("lookback", vec![10.0, 20.0, 40.0])],
        _ => Vec::new(),
    }
}

/// Non-empty axes participating in the product. Empty axes historically had no
/// effect and retain that behavior.
fn active_axes(axes: &[ParamAxis]) -> Vec<&ParamAxis> {
    axes.iter().filter(|axis| !axis.values.is_empty()).collect()
}

/// Compute the raw Cartesian-product size without allocating combinations.
fn raw_grid_size(axes: &[&ParamAxis]) -> Option<usize> {
    axes.iter()
        .try_fold(1usize, |acc, axis| acc.checked_mul(axis.values.len()))
}

/// Exact valid-grid count for the only cross-axis rule currently defined by the
/// optimizer (`fast < slow`). This remains O(number of fast×slow values), not
/// O(full Cartesian product). Other axes contribute by multiplication only.
fn valid_grid_size(axes: &[&ParamAxis]) -> Option<usize> {
    let raw = raw_grid_size(axes)?;
    let fast = axes.iter().position(|axis| axis.name == "fast");
    let slow = axes.iter().position(|axis| axis.name == "slow");
    let (Some(fast_index), Some(slow_index)) = (fast, slow)
    else
    {
        return Some(raw);
    };
    if fast_index == slow_index
    {
        return Some(raw);
    }

    let fast_axis = axes[fast_index];
    let slow_axis = axes[slow_index];
    let valid_pairs = fast_axis
        .values
        .iter()
        .flat_map(|fast| slow_axis.values.iter().map(move |slow| (*fast, *slow)))
        .filter(|(fast, slow)| fast < slow)
        .count();
    if valid_pairs == 0
    {
        return Some(0);
    }

    let pair_product = fast_axis.values.len().checked_mul(slow_axis.values.len())?;
    let other_product = raw / pair_product;
    valid_pairs.checked_mul(other_product)
}

/// Build one combination directly from its mixed-radix product index.
/// Memory is O(number of axes), irrespective of total grid size.
fn combo_at(axes: &[&ParamAxis], mut index: usize) -> Vec<(String, f32)> {
    let mut combo = Vec::with_capacity(axes.len());
    let mut selected = vec![0usize; axes.len()];
    for axis_index in (0..axes.len()).rev()
    {
        let radix = axes[axis_index].values.len();
        selected[axis_index] = index % radix;
        index /= radix;
    }
    for (axis, value_index) in axes.iter().zip(selected)
    {
        combo.push((axis.name.clone(), axis.values[value_index]));
    }
    combo
}

fn combo_is_valid(combo: &[(String, f32)]) -> bool {
    let get = |key: &str| {
        combo
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| *value)
    };
    match (get("fast"), get("slow"))
    {
        (Some(fast), Some(slow)) => fast < slow,
        _ => true,
    }
}

type ParamCombination = Vec<(String, f32)>;

#[derive(Debug)]
struct SampledGrid {
    grid_size: usize,
    truncated: bool,
    combinations: Vec<ParamCombination>,
}

fn sampled_combos(axes: &[ParamAxis], max_combos: usize) -> Option<SampledGrid> {
    let active = active_axes(axes);
    let raw_size = raw_grid_size(&active)?;
    let grid_size = valid_grid_size(&active)?;
    if grid_size == 0 || raw_size == 0
    {
        return None;
    }

    let budget = max_combos.max(1);
    let stride = raw_size.div_ceil(budget);
    let mut combinations = Vec::with_capacity(budget.min(grid_size));

    // Evenly sample raw product indices and filter invalid fast/slow pairs.
    // No intermediate product vector exists. If a stride lands on an invalid
    // pair, deterministically probe forward only inside that stride bucket.
    let mut start = 0usize;
    while start < raw_size && combinations.len() < budget
    {
        let end = start.saturating_add(stride).min(raw_size);
        let mut probe = start;
        while probe < end
        {
            let combo = combo_at(&active, probe);
            if combo_is_valid(&combo)
            {
                combinations.push(combo);
                break;
            }
            probe += 1;
        }
        start = end;
    }

    Ok::<_, ()>(SampledGrid {
        grid_size,
        truncated: grid_size > combinations.len(),
        combinations,
    })
    .ok()
}

fn objective_score(
    objective: Objective,
    consistency: f32,
    mean_window: f32,
    worst_window: f32,
    train_sharpe: f32,
) -> f32 {
    match objective
    {
        Objective::ReturnConsistency =>
        {
            if mean_window > 0.0
            {
                mean_window * consistency
            }
            else
            {
                mean_window
            }
        },
        Objective::Consistency => consistency + 1e-6 * mean_window,
        Objective::MeanReturn => mean_window,
        Objective::WorstWindow => worst_window,
        Objective::Sharpe => train_sharpe,
    }
}

fn verdict(best: &Candidate) -> String {
    let ret = best.holdout_return * 100.0;
    if best.holdout_return <= 0.0
    {
        format!(
            "OVERFIT / NO EDGE — the best in-sample parameters lose out-of-sample \
             (holdout return {ret:+.2}%, Sharpe {:.2}). Do not trade this.",
            best.holdout_sharpe
        )
    }
    else if best.overfit_gap > 1.0 || best.holdout_sharpe < 0.5 * best.train_sharpe.max(0.0)
    {
        format!(
            "PARTIAL — positive out-of-sample but materially degraded from in-sample \
             (Sharpe {:.2}→{:.2}, holdout return {ret:+.2}%). Size down and re-validate.",
            best.train_sharpe, best.holdout_sharpe
        )
    }
    else
    {
        format!(
            "ROBUST — holds up out-of-sample (holdout return {ret:+.2}%, Sharpe {:.2}); \
             in-sample→holdout degradation is modest ({:.2} Sharpe).",
            best.holdout_sharpe, best.overfit_gap
        )
    }
}

pub fn optimize(
    strategy_name: &str,
    axes: &[ParamAxis],
    base_params: &BTreeMap<String, f32>,
    candles: &[Candle],
    cfg: &BacktestConfig,
    opt: &OptimizeConfig,
) -> Option<OptimizeReport> {
    let n = candles.len();
    let train_frac = opt.train_frac.clamp(0.3, 0.9);
    let split = ((n as f32) * train_frac).round() as usize;
    if n < 40 || split < 20 || n - split < 8
    {
        return None;
    }
    let train = &candles[..split];
    let holdout = &candles[split..];

    let sampled = sampled_combos(axes, opt.max_combos)?;
    let mut candidates = Vec::with_capacity(sampled.combinations.len());

    for combo in &sampled.combinations
    {
        let mut params = base_params.clone();
        for (key, value) in combo
        {
            params.insert(key.clone(), *value);
        }
        let Some(strategy) = strategy_from_spec(strategy_name, &params)
        else
        {
            continue;
        };
        let wf = walk_forward(strategy.as_ref(), train, opt.wf_windows, cfg);
        let train_bt = run_backtest(strategy.as_ref(), train, cfg);
        let score = objective_score(
            opt.objective,
            wf.consistency,
            wf.mean_return,
            wf.worst_window_return,
            train_bt.performance.sharpe,
        );
        candidates.push(Candidate {
            params,
            train_return: train_bt.total_return,
            train_sharpe: train_bt.performance.sharpe,
            train_consistency: wf.consistency,
            train_mean_window: wf.mean_return,
            train_worst_window: wf.worst_window_return,
            objective_score: score,
            holdout_return: 0.0,
            holdout_sharpe: 0.0,
            holdout_max_drawdown: 0.0,
            holdout_trades: 0,
            overfit_gap: 0.0,
        });
    }
    if candidates.is_empty()
    {
        return None;
    }

    let num_evaluated = candidates.len();
    candidates.sort_by(|a, b| b.objective_score.total_cmp(&a.objective_score));

    let top_k = opt.top_k.clamp(1, candidates.len());
    let mut leaderboard: Vec<Candidate> = candidates.into_iter().take(top_k).collect();
    for candidate in &mut leaderboard
    {
        if let Some(strategy) = strategy_from_spec(strategy_name, &candidate.params)
        {
            let bt = run_backtest(strategy.as_ref(), holdout, cfg);
            candidate.holdout_return = bt.total_return;
            candidate.holdout_sharpe = bt.performance.sharpe;
            candidate.holdout_max_drawdown = bt.performance.max_drawdown;
            candidate.holdout_trades = bt.num_trades;
            candidate.overfit_gap = candidate.train_sharpe - bt.performance.sharpe;
        }
    }

    let best = leaderboard[0].clone();
    let verdict_text = verdict(&best);
    Some(OptimizeReport {
        strategy: strategy_name.to_string(),
        objective: opt.objective.label().to_string(),
        train_bars: train.len(),
        holdout_bars: holdout.len(),
        grid_size: sampled.grid_size,
        num_evaluated,
        truncated: sampled.truncated,
        best,
        leaderboard,
        verdict: verdict_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BacktestConfig {
        BacktestConfig {
            fees: crate::orders::FeeSchedule {
                maker_bps: 0.0,
                taker_bps: 0.0,
            },
            slippage: crate::orders::SlippageModel {
                base_bps: 0.0,
                impact_bps: 0.0,
                ref_liquidity: 1.0,
            },
            ..Default::default()
        }
    }

    fn candle(ts: i64, close: f32) -> Candle {
        Candle {
            ts_ms: ts,
            open: close,
            high: close * 1.003,
            low: close * 0.997,
            close,
            volume: 100.0,
        }
    }

    fn trend(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| candle(i as i64 * 60_000, 100.0 + i as f32))
            .collect()
    }

    #[test]
    fn mixed_radix_access_matches_small_cartesian_product() {
        let axes = vec![
            ParamAxis::new("a", vec![1.0, 2.0]),
            ParamAxis::new("b", vec![10.0, 20.0, 30.0]),
        ];
        let active = active_axes(&axes);
        assert_eq!(raw_grid_size(&active), Some(6));
        assert_eq!(
            combo_at(&active, 0),
            vec![("a".into(), 1.0), ("b".into(), 10.0)]
        );
        assert_eq!(
            combo_at(&active, 5),
            vec![("a".into(), 2.0), ("b".into(), 30.0)]
        );
    }

    #[test]
    fn huge_grid_is_bounded_before_any_product_vector_exists() {
        let axes: Vec<ParamAxis> = (0..8)
            .map(|i| ParamAxis::new(format!("p{i}"), (0..100).map(|x| x as f32).collect()))
            .collect();
        let active = active_axes(&axes);
        assert_eq!(raw_grid_size(&active), Some(10_000_000_000_000_000));
        let sampled = sampled_combos(&axes, 17).unwrap();
        assert_eq!(sampled.grid_size, 10_000_000_000_000_000);
        assert!(sampled.truncated);
        assert!(sampled.combinations.len() <= 17);
    }

    #[test]
    fn overflowed_grid_is_rejected_before_enumeration() {
        let axes: Vec<ParamAxis> = (0..32)
            .map(|i| ParamAxis::new(format!("p{i}"), vec![0.0, 1.0, 2.0, 3.0]))
            .collect();
        let active = active_axes(&axes);
        assert!(raw_grid_size(&active).is_none());
        assert!(sampled_combos(&axes, 10).is_none());
    }

    #[test]
    fn fast_slow_valid_count_is_exact_without_full_product() {
        let axes = vec![
            ParamAxis::new("fast", vec![5.0, 10.0, 30.0]),
            ParamAxis::new("slow", vec![10.0, 20.0]),
            ParamAxis::new("x", vec![1.0, 2.0]),
        ];
        let active = active_axes(&axes);
        assert_eq!(valid_grid_size(&active), Some(6));
    }

    #[test]
    fn objective_parse_roundtrip() {
        assert_eq!(
            Objective::parse("consistency"),
            Some(Objective::Consistency)
        );
        assert_eq!(Objective::parse("worst"), Some(Objective::WorstWindow));
        assert_eq!(Objective::parse("nope"), None);
    }

    #[test]
    fn none_when_too_little_data() {
        let candles = trend(30);
        assert!(
            optimize(
                "sma_cross",
                &default_axes("sma_cross"),
                &BTreeMap::new(),
                &candles,
                &cfg(),
                &OptimizeConfig::default(),
            )
            .is_none()
        );
    }

    #[test]
    fn optimizes_and_confirms_on_holdout() {
        let candles = trend(400);
        let report = optimize(
            "sma_cross",
            &default_axes("sma_cross"),
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        assert!(report.train_bars > 0 && report.holdout_bars > 0);
        assert!(report.grid_size > 1);
        assert!(!report.leaderboard.is_empty());
        assert!(report.best.params.contains_key("fast"));
        assert!(report.best.params.contains_key("slow"));
        assert!(report.best.holdout_return > 0.0);
    }

    #[test]
    fn max_combos_bounds_actual_evaluation() {
        let candles = trend(400);
        let opt = OptimizeConfig {
            max_combos: 3,
            ..Default::default()
        };
        let report = optimize(
            "sma_cross",
            &default_axes("sma_cross"),
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &opt,
        )
        .unwrap();
        assert!(report.truncated);
        assert!(report.num_evaluated <= 3);
        assert!(report.leaderboard.len() <= 3);
    }

    #[test]
    fn deterministic_result() {
        let candles = trend(400);
        let axes = default_axes("supertrend");
        let a = optimize(
            "supertrend",
            &axes,
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        let b = optimize(
            "supertrend",
            &axes,
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        assert_eq!(a.best.params, b.best.params);
        assert_eq!(a.best.holdout_return, b.best.holdout_return);
        assert_eq!(a.best.objective_score, b.best.objective_score);
    }
}
