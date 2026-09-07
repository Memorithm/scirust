//! Leakage-aware reinforcement-learning market environment.
//!
//! The environment deliberately separates the mechanics of a market episode
//! from a particular RL algorithm.  It implements [`scirust_learning::rl::Env`]
//! so the existing SciRust tabular/PPO/deep agents can be evaluated against the
//! same deterministic contract.  Training and holdout execution are exposed as
//! separate functions; holdout evaluation never calls `Agent::update`.

use serde::{Deserialize, Serialize};
use std::ops::Range;

use scirust_learning::rl::{Agent, Env};

use crate::ml_dataset::{MlDatasetError, TimeSeriesMlDataset, TimeSplit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketAction {
    Short,
    Flat,
    Long,
}

impl MarketAction {
    #[inline]
    pub fn exposure(self) -> f64 {
        match self
        {
            Self::Short => -1.0,
            Self::Flat => 0.0,
            Self::Long => 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketRlState {
    pub row_index: usize,
    pub ts_ms: i64,
    pub features: Vec<f32>,
    pub current_exposure: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodeKind {
    Train,
    Validation,
    Holdout,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlExperimentPlan {
    pub seed: u64,
    pub transaction_cost_bps_per_unit_turnover: f64,
    pub train: Range<usize>,
    pub validation: Range<usize>,
    pub holdout: Range<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RlMarketError {
    Dataset(MlDatasetError),
    EmptyRange,
    RangeOutOfBounds,
    InvalidTransactionCost,
    PartitionOrderInvalid,
    LabelOverlapAcrossPartitions,
    InsufficientRowsForOneStepReward,
    RewardNotAvailableAtNextObservation {
        row: usize,
        target_ts_ms: i64,
        next_observation_ts_ms: i64,
    },
}

impl From<MlDatasetError> for RlMarketError {
    fn from(value: MlDatasetError) -> Self {
        Self::Dataset(value)
    }
}

impl RlExperimentPlan {
    pub fn from_time_split(
        dataset: &TimeSeriesMlDataset,
        split: TimeSplit,
        seed: u64,
        transaction_cost_bps_per_unit_turnover: f64,
    ) -> Result<Self, RlMarketError> {
        dataset.validate()?;
        if !transaction_cost_bps_per_unit_turnover.is_finite()
            || transaction_cost_bps_per_unit_turnover < 0.0
        {
            return Err(RlMarketError::InvalidTransactionCost);
        }

        let train = split.train();
        let validation = split.validation();
        let holdout = split.test();
        if train.is_empty() || validation.is_empty() || holdout.is_empty()
        {
            return Err(RlMarketError::EmptyRange);
        }
        if holdout.end > dataset.rows.len()
        {
            return Err(RlMarketError::RangeOutOfBounds);
        }
        if train.start != 0
            || train.end > validation.start
            || validation.end > holdout.start
            || train.end != validation.start
            || validation.end != holdout.start
        {
            return Err(RlMarketError::PartitionOrderInvalid);
        }

        let validation_first_ts = dataset.rows[validation.start].ts_ms;
        let holdout_first_ts = dataset.rows[holdout.start].ts_ms;
        if dataset.target_overlaps_boundary(train.clone(), validation_first_ts)
            || dataset.target_overlaps_boundary(validation.clone(), holdout_first_ts)
        {
            return Err(RlMarketError::LabelOverlapAcrossPartitions);
        }

        Ok(Self {
            seed,
            transaction_cost_bps_per_unit_turnover,
            train,
            validation,
            holdout,
        })
    }

    pub fn env(
        &self,
        dataset: &TimeSeriesMlDataset,
        kind: EpisodeKind,
    ) -> Result<MarketRlEnv, RlMarketError> {
        let range = match kind
        {
            EpisodeKind::Train => self.train.clone(),
            EpisodeKind::Validation => self.validation.clone(),
            EpisodeKind::Holdout => self.holdout.clone(),
        };
        MarketRlEnv::new(dataset, range, self.transaction_cost_bps_per_unit_turnover)
    }
}

/// Deterministic one-step target-exposure environment.
///
/// `MlRow::target` is interpreted as the forward fractional return associated
/// with the observation. The target is never exposed in [`MarketRlState`]. A
/// row is actionable only when its target is observable no later than the next
/// observation timestamp. The final row in the selected range is therefore a
/// terminal observation and its own target is not consumed. This prevents a
/// multi-step or overlapping target from being credited to the agent before the
/// simulated information clock reaches that target.
///
/// Transaction costs are charged on absolute exposure change. The terminal
/// transition also charges the cost required to flatten the remaining exposure,
/// so an episode cannot silently finish with a free open position.
#[derive(Debug, Clone)]
pub struct MarketRlEnv {
    rows: Vec<crate::ml_dataset::MlRow>,
    source_start: usize,
    cursor: usize,
    current_exposure: f64,
    transaction_cost_rate_per_unit_turnover: f64,
}

impl MarketRlEnv {
    pub fn new(
        dataset: &TimeSeriesMlDataset,
        range: Range<usize>,
        transaction_cost_bps_per_unit_turnover: f64,
    ) -> Result<Self, RlMarketError> {
        dataset.validate()?;
        if !transaction_cost_bps_per_unit_turnover.is_finite()
            || transaction_cost_bps_per_unit_turnover < 0.0
        {
            return Err(RlMarketError::InvalidTransactionCost);
        }
        if range.is_empty()
        {
            return Err(RlMarketError::EmptyRange);
        }
        if range.end > dataset.rows.len()
        {
            return Err(RlMarketError::RangeOutOfBounds);
        }
        if range.len() < 2
        {
            return Err(RlMarketError::InsufficientRowsForOneStepReward);
        }

        for source_index in range.start..range.end - 1
        {
            let row = &dataset.rows[source_index];
            let next = &dataset.rows[source_index + 1];
            if row.target_ts_ms > next.ts_ms
            {
                return Err(RlMarketError::RewardNotAvailableAtNextObservation {
                    row: source_index,
                    target_ts_ms: row.target_ts_ms,
                    next_observation_ts_ms: next.ts_ms,
                });
            }
        }

        let source_start = range.start;
        Ok(Self {
            rows: dataset.rows[range].to_vec(),
            source_start,
            cursor: 0,
            current_exposure: 0.0,
            transaction_cost_rate_per_unit_turnover: transaction_cost_bps_per_unit_turnover
                / 10_000.0,
        })
    }

    fn state(&self) -> MarketRlState {
        let index = self.cursor.min(self.rows.len() - 1);
        let row = &self.rows[index];
        MarketRlState {
            row_index: self.source_start + index,
            ts_ms: row.ts_ms,
            features: row.features.clone(),
            current_exposure: self.current_exposure,
        }
    }

    /// Number of actionable one-step transitions. The final row is the terminal
    /// next observation and is not itself acted upon.
    pub fn len(&self) -> usize {
        self.rows.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Env for MarketRlEnv {
    type State = MarketRlState;
    type Action = MarketAction;

    fn reset(&mut self) -> Self::State {
        self.cursor = 0;
        self.current_exposure = 0.0;
        self.state()
    }

    fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool) {
        if self.cursor + 1 >= self.rows.len()
        {
            return (self.state(), 0.0, true);
        }

        let row = &self.rows[self.cursor];
        let next_exposure = action.exposure();
        let turnover = (next_exposure - self.current_exposure).abs();
        let gross_reward = next_exposure * f64::from(row.target);
        let mut cost = turnover * self.transaction_cost_rate_per_unit_turnover;

        self.cursor += 1;
        let done = self.cursor + 1 >= self.rows.len();
        if done
        {
            cost += next_exposure.abs() * self.transaction_cost_rate_per_unit_turnover;
            self.current_exposure = 0.0;
        }
        else
        {
            self.current_exposure = next_exposure;
        }

        let reward = gross_reward - cost;
        (self.state(), reward, done)
    }

    fn available_actions(&self, _state: &Self::State) -> Vec<Self::Action> {
        vec![MarketAction::Short, MarketAction::Flat, MarketAction::Long]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpisodeReport {
    pub steps: usize,
    /// Additive RL objective score across transitions. This is not a compounded
    /// portfolio return and must not be reported as one.
    pub cumulative_reward: f64,
    pub mean_reward: f64,
}

/// Run one training episode.  This is the only helper in this module that calls
/// `Agent::update`.
pub fn run_training_episode<A>(agent: &mut A, env: &mut MarketRlEnv) -> EpisodeReport
where
    A: Agent<MarketRlEnv>,
{
    let mut state = env.reset();
    let mut steps = 0usize;
    let mut cumulative_reward = 0.0f64;
    loop
    {
        let action = agent.act(&state);
        let (next_state, reward, done) = env.step(&action);
        agent.update(&state, &action, reward, &next_state, done);
        steps += 1;
        cumulative_reward += reward;
        state = next_state;
        if done
        {
            break;
        }
    }
    EpisodeReport {
        steps,
        cumulative_reward,
        mean_reward: cumulative_reward / steps as f64,
    }
}

/// Evaluate a frozen agent.  The immutable agent reference makes holdout
/// updates impossible through the generic [`Agent`] API.
pub fn evaluate_frozen_agent<A>(agent: &A, env: &mut MarketRlEnv) -> EpisodeReport
where
    A: Agent<MarketRlEnv>,
{
    let mut state = env.reset();
    let mut steps = 0usize;
    let mut cumulative_reward = 0.0f64;
    loop
    {
        let action = agent.act(&state);
        let (next_state, reward, done) = env.step(&action);
        steps += 1;
        cumulative_reward += reward;
        state = next_state;
        if done
        {
            break;
        }
    }
    EpisodeReport {
        steps,
        cumulative_reward,
        mean_reward: cumulative_reward / steps as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_dataset::{FeatureProvenance, MlRow};

    fn dataset() -> TimeSeriesMlDataset {
        TimeSeriesMlDataset {
            feature_provenance: vec![FeatureProvenance {
                name: "lag_return".into(),
                source: "close".into(),
                transformation: "lag-1".into(),
            }],
            rows: (0..12)
                .map(|i| MlRow {
                    ts_ms: i * 10,
                    feature_available_ts_ms: i * 10,
                    target_ts_ms: i * 10 + 5,
                    features: vec![i as f32 / 10.0],
                    target: if i % 2 == 0 { 0.01 } else { -0.01 },
                })
                .collect(),
        }
    }

    #[derive(Default)]
    struct CountingAgent {
        updates: usize,
    }

    impl Agent<MarketRlEnv> for CountingAgent {
        fn act(&self, state: &MarketRlState) -> MarketAction {
            if state.features[0] >= 0.0
            {
                MarketAction::Long
            }
            else
            {
                MarketAction::Flat
            }
        }

        fn update(
            &mut self,
            _state: &MarketRlState,
            _action: &MarketAction,
            _reward: f64,
            _next_state: &MarketRlState,
            _done: bool,
        ) {
            self.updates += 1;
        }
    }

    #[test]
    fn plan_preserves_train_validation_holdout_order() {
        let d = dataset();
        let split = d.time_split(0.5, 0.25).unwrap();
        let plan = RlExperimentPlan::from_time_split(&d, split, 42, 5.0).unwrap();
        assert_eq!(plan.train, 0..6);
        assert_eq!(plan.validation, 6..9);
        assert_eq!(plan.holdout, 9..12);
    }

    #[test]
    fn plan_checks_every_label_at_partition_boundaries() {
        let mut d = dataset();
        d.rows[0].target_ts_ms = 100;
        let split = TimeSplit {
            train_start: 0,
            train_end: 6,
            validation_start: 6,
            validation_end: 9,
            test_start: 9,
            test_end: 12,
        };
        assert!(matches!(
            RlExperimentPlan::from_time_split(&d, split, 42, 0.0),
            Err(RlMarketError::LabelOverlapAcrossPartitions)
        ));
    }

    #[test]
    fn target_is_not_part_of_observation_state() {
        let d = dataset();
        let mut env = MarketRlEnv::new(&d, 0..2, 0.0).unwrap();
        let state = env.reset();
        assert_eq!(state.features, d.rows[0].features);
        assert_eq!(state.ts_ms, d.rows[0].ts_ms);
    }

    #[test]
    fn future_reward_is_rejected_before_episode_start() {
        let mut d = dataset();
        d.rows[0].target_ts_ms = 25;
        assert!(matches!(
            MarketRlEnv::new(&d, 0..3, 0.0),
            Err(RlMarketError::RewardNotAvailableAtNextObservation {
                row: 0,
                target_ts_ms: 25,
                next_observation_ts_ms: 10,
            })
        ));
    }

    #[test]
    fn reward_available_exactly_at_next_observation_is_allowed() {
        let mut d = dataset();
        d.rows[0].target_ts_ms = d.rows[1].ts_ms;
        assert!(MarketRlEnv::new(&d, 0..3, 0.0).is_ok());
    }

    #[test]
    fn turnover_cost_is_charged_deterministically() {
        let d = dataset();
        let mut env = MarketRlEnv::new(&d, 0..3, 10.0).unwrap();
        env.reset();
        let (_, reward, done) = env.step(&MarketAction::Long);
        let expected = f64::from(d.rows[0].target) - 10.0 / 10_000.0;
        assert!(!done);
        assert!((reward - expected).abs() < 1e-12);
    }

    #[test]
    fn terminal_transition_charges_close_cost_once() {
        let d = dataset();
        let mut env = MarketRlEnv::new(&d, 0..2, 10.0).unwrap();
        env.reset();
        let (next_state, reward, done) = env.step(&MarketAction::Long);
        let expected = f64::from(d.rows[0].target) - 2.0 * 10.0 / 10_000.0;
        assert!(done);
        assert_eq!(next_state.ts_ms, d.rows[1].ts_ms);
        assert_eq!(next_state.current_exposure, 0.0);
        assert!((reward - expected).abs() < 1e-12);
    }

    #[test]
    fn training_updates_but_frozen_holdout_does_not() {
        let d = dataset();
        let split = d.time_split(0.5, 0.25).unwrap();
        let plan = RlExperimentPlan::from_time_split(&d, split, 7, 0.0).unwrap();
        let mut agent = CountingAgent::default();
        let mut train = plan.env(&d, EpisodeKind::Train).unwrap();
        let train_report = run_training_episode(&mut agent, &mut train);
        assert_eq!(agent.updates, train_report.steps);
        assert_eq!(train_report.steps, plan.train.len() - 1);
        let before = agent.updates;
        let mut holdout = plan.env(&d, EpisodeKind::Holdout).unwrap();
        let holdout_report = evaluate_frozen_agent(&agent, &mut holdout);
        assert_eq!(agent.updates, before);
        assert_eq!(holdout_report.steps, plan.holdout.len() - 1);
    }
}
