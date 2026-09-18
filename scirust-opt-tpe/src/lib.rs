//! Reference Tree-structured Parzen Estimator (TPE) for SciRust.
//!
//! This crate is the correctness-oriented Phase 3 baseline from the
//! Optuna/Rustuna comparison roadmap. It intentionally starts with the
//! single-objective, independent/univariate TPE regime before multivariate and
//! streaming variants are introduced.
//!
//! The default constants follow the audited Optuna 5 development snapshot:
//! ten startup trials, twenty-four acquisition candidates, ten-percent gamma
//! capped at twenty-five, unit prior weight, magic-clip enabled and endpoint
//! influence disabled.
//!
//! The implementation supports SciRust's continuous, log-continuous, integer,
//! categorical and conditional parameter domains. Running proposals can be
//! included in the "above" model (constant-liar substrate). Pruned-trial
//! intermediate-value ranking, constraints, multivariate grouping and
//! multi-objective hypervolume weighting are deliberately not claimed by this
//! first reference slice.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::cmp::Ordering;
use core::fmt;

use scirust_opt_core::{
    Candidate, CandidateError, Distribution, ParamId, ParamValue, Sampler, StudyView, TrialId,
    TrialState,
};
use scirust_special::{erfc, erfinv};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const SQRT_2PI: f64 = 2.506_628_274_631_000_2;

/// Single-objective optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Lower objective values are preferred.
    Minimize,
    /// Higher objective values are preferred.
    Maximize,
}

/// Configuration of the reference univariate TPE sampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TpeConfig {
    /// Number of completed trials sampled randomly before fitting Parzen models.
    pub n_startup_trials: usize,
    /// Number of candidates drawn from the good-density model per parameter.
    pub n_ei_candidates: usize,
    /// Weight of the broad prior kernel.
    pub prior_weight: f64,
    /// Clamp numerical kernel bandwidths away from zero.
    pub consider_magic_clip: bool,
    /// Retain search-space endpoint distances in first/last observed bandwidths.
    pub consider_endpoints: bool,
    /// Include already-running proposals in the "above" density model.
    pub constant_liar: bool,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            n_ei_candidates: 24,
            prior_weight: 1.0,
            consider_magic_clip: true,
            consider_endpoints: false,
            constant_liar: true,
        }
    }
}

/// Error returned by the reference TPE sampler.
#[derive(Debug, Clone, PartialEq)]
pub enum TpeError {
    /// Configuration value was invalid.
    InvalidConfig(&'static str),
    /// This reference slice only supports one objective column.
    UnsupportedObjectiveCount(usize),
    /// Historical parameter value did not match the compiled distribution.
    InvalidObservation(ParamId),
    /// Candidate assignment failed.
    Candidate(CandidateError),
    /// Numerical Parzen construction produced an invalid finite range.
    InvalidNumericalRange(ParamId),
}

impl fmt::Display for TpeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidConfig(message) => write!(f, "invalid TPE configuration: {message}"),
            Self::UnsupportedObjectiveCount(found) =>
            {
                write!(f, "reference TPE requires one objective, found {found}")
            },
            Self::InvalidObservation(param) =>
            {
                write!(f, "historical value does not match parameter {param:?}")
            },
            Self::Candidate(source) => write!(f, "candidate assignment failed: {source}"),
            Self::InvalidNumericalRange(param) =>
            {
                write!(f, "invalid numerical Parzen range for {param:?}")
            },
        }
    }
}

impl std::error::Error for TpeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Candidate(source) => Some(source),
            _ => None,
        }
    }
}

/// Optuna-v5-compatible default split size for single-objective TPE.
///
/// Returns `min(ceil(0.1 * n), 25)`.
pub fn default_gamma(n: usize) -> usize {
    ((n + 9) / 10).min(25)
}

/// Default chronological observation weights used by the audited TPE.
///
/// Histories shorter than twenty-five observations receive unit weights. Older
/// observations are linearly ramped when the history exceeds twenty-five.
pub fn default_weights(n: usize) -> Vec<f64> {
    if n == 0
    {
        Vec::new()
    }
    else if n < 25
    {
        vec![1.0; n]
    }
    else
    {
        let ramp_len = n - 25;
        let mut weights = Vec::with_capacity(n);
        if ramp_len == 1
        {
            weights.push(1.0 / n as f64);
        }
        else if ramp_len > 1
        {
            let start = 1.0 / n as f64;
            let denominator = (ramp_len - 1) as f64;
            for i in 0..ramp_len
            {
                let t = i as f64 / denominator;
                weights.push(start + (1.0 - start) * t);
            }
        }
        weights.extend(std::iter::repeat_n(1.0, 25));
        weights
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn index(&mut self, n: usize) -> usize {
        if n <= 1
        {
            return 0;
        }
        (((self.next_u64() as u128) * (n as u128)) >> 64) as usize
    }

    fn weighted_index(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        if !(total > 0.0 && total.is_finite())
        {
            return 0;
        }
        let target = self.next_f64() * total;
        let mut cumulative = 0.0;
        for (index, weight) in weights.iter().copied().enumerate()
        {
            cumulative += weight;
            if target < cumulative
            {
                return index;
            }
        }
        weights.len().saturating_sub(1)
    }
}

#[derive(Debug, Clone, Copy)]
enum NumericalKind {
    LinearFloat,
    LogFloat,
    Integer { low: i64, high: i64 },
}

#[derive(Debug, Clone)]
struct NumericalParzen {
    weights: Vec<f64>,
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    adapted_low: f64,
    adapted_high: f64,
    kind: NumericalKind,
}

impl NumericalParzen {
    fn new(
        param: ParamId,
        observations: &[ParamValue],
        distribution: &Distribution,
        config: TpeConfig,
    ) -> Result<Self, TpeError> {
        let (adapted_low, adapted_high, kind) = match distribution
        {
            Distribution::Uniform { low, high } => (*low, *high, NumericalKind::LinearFloat),
            Distribution::LogUniform { low, high } =>
            {
                (low.ln(), high.ln(), NumericalKind::LogFloat)
            },
            Distribution::IntRange { low, high } => (
                *low as f64 - 0.5,
                *high as f64 + 0.5,
                NumericalKind::Integer {
                    low: *low,
                    high: *high,
                },
            ),
            Distribution::Categorical { .. } => return Err(TpeError::InvalidObservation(param)),
        };
        let range = adapted_high - adapted_low;
        if !(range > 0.0 && range.is_finite())
        {
            return Err(TpeError::InvalidNumericalRange(param));
        }

        let mut transformed = Vec::with_capacity(observations.len());
        for value in observations
        {
            let x = match (distribution, value)
            {
                (Distribution::Uniform { .. }, ParamValue::Float(value)) => *value,
                (Distribution::LogUniform { .. }, ParamValue::Float(value)) => value.ln(),
                (Distribution::IntRange { .. }, ParamValue::Int(value)) => *value as f64,
                _ => return Err(TpeError::InvalidObservation(param)),
            };
            if !x.is_finite()
            {
                return Err(TpeError::InvalidObservation(param));
            }
            transformed.push(x);
        }

        let n_observations = transformed.len();
        let mut sigmas = vec![range; n_observations];
        if n_observations != 0
        {
            let mut sorted_indices: Vec<usize> = (0..n_observations).collect();
            sorted_indices
                .sort_by(|left, right| transformed[*left].total_cmp(&transformed[*right]));

            let mut sorted = Vec::with_capacity(n_observations + 2);
            sorted.push(adapted_low);
            sorted.extend(sorted_indices.iter().map(|index| transformed[*index]));
            sorted.push(adapted_high);

            let mut sorted_sigmas = Vec::with_capacity(n_observations);
            for index in 1..=n_observations
            {
                sorted_sigmas.push(
                    (sorted[index] - sorted[index - 1]).max(sorted[index + 1] - sorted[index]),
                );
            }

            if !config.consider_endpoints && sorted.len() >= 4
            {
                sorted_sigmas[0] = sorted[2] - sorted[1];
                let last = sorted_sigmas.len() - 1;
                sorted_sigmas[last] = sorted[sorted.len() - 2] - sorted[sorted.len() - 3];
            }

            let min_sigma = if config.consider_magic_clip
            {
                let n_kernels = n_observations + 1;
                range / (100.0_f64.min(1.0 + n_kernels as f64))
            }
            else
            {
                f64::EPSILON
            };
            for (sorted_position, original_index) in sorted_indices.into_iter().enumerate()
            {
                sigmas[original_index] = sorted_sigmas[sorted_position].clamp(min_sigma, range);
            }
        }

        let mut mus = transformed;
        mus.push(0.5 * (adapted_low + adapted_high));
        sigmas.push(range);

        let weights = mixture_weights(n_observations, config.prior_weight);
        Ok(Self {
            weights,
            mus,
            sigmas,
            adapted_low,
            adapted_high,
            kind,
        })
    }

    fn sample(&self, rng: &mut SplitMix64) -> ParamValue {
        let component = rng.weighted_index(&self.weights);
        let transformed = sample_truncated_normal(
            rng,
            self.mus[component],
            self.sigmas[component],
            self.adapted_low,
            self.adapted_high,
        );
        match self.kind
        {
            NumericalKind::LinearFloat =>
            {
                ParamValue::Float(transformed.clamp(self.adapted_low, self.adapted_high))
            },
            NumericalKind::LogFloat => ParamValue::Float(
                transformed
                    .exp()
                    .clamp(self.adapted_low.exp(), self.adapted_high.exp()),
            ),
            NumericalKind::Integer { low, high } =>
            {
                let rounded = transformed.round().clamp(low as f64, high as f64) as i64;
                ParamValue::Int(rounded)
            },
        }
    }

    fn log_pdf(&self, value: ParamValue) -> f64 {
        let transformed = match (self.kind, value)
        {
            (NumericalKind::LinearFloat, ParamValue::Float(value)) => value,
            (NumericalKind::LogFloat, ParamValue::Float(value)) if value > 0.0 => value.ln(),
            (NumericalKind::Integer { .. }, ParamValue::Int(value)) => value as f64,
            _ => return f64::NEG_INFINITY,
        };

        let mut terms = Vec::with_capacity(self.weights.len());
        for index in 0..self.weights.len()
        {
            let weight = self.weights[index];
            if weight <= 0.0
            {
                continue;
            }
            let probability = match self.kind
            {
                NumericalKind::Integer { .. } => truncated_discrete_mass(
                    transformed,
                    self.mus[index],
                    self.sigmas[index],
                    self.adapted_low,
                    self.adapted_high,
                ),
                NumericalKind::LinearFloat | NumericalKind::LogFloat => truncated_normal_pdf(
                    transformed,
                    self.mus[index],
                    self.sigmas[index],
                    self.adapted_low,
                    self.adapted_high,
                ),
            };
            if probability > 0.0
            {
                terms.push(weight.ln() + probability.ln());
            }
        }
        logsumexp(&terms)
    }
}

#[derive(Debug, Clone)]
struct CategoricalParzen {
    weights: Vec<f64>,
    component_probabilities: Vec<Vec<f64>>,
    cardinality: usize,
}

impl CategoricalParzen {
    fn new(
        param: ParamId,
        observations: &[ParamValue],
        cardinality: u32,
        config: TpeConfig,
    ) -> Result<Self, TpeError> {
        let cardinality = cardinality as usize;
        if observations.is_empty()
        {
            return Ok(Self {
                weights: vec![1.0],
                component_probabilities: vec![vec![1.0 / cardinality as f64; cardinality]],
                cardinality,
            });
        }

        let n_kernels = observations.len() + 1;
        let base = config.prior_weight / n_kernels as f64;
        let mut rows = vec![vec![base; cardinality]; n_kernels];
        for (row, observation) in observations.iter().enumerate()
        {
            let ParamValue::Categorical(choice) = observation
            else
            {
                return Err(TpeError::InvalidObservation(param));
            };
            let choice = *choice as usize;
            if choice >= cardinality
            {
                return Err(TpeError::InvalidObservation(param));
            }
            rows[row][choice] += 1.0;
        }
        for row in &mut rows
        {
            let sum: f64 = row.iter().sum();
            if sum > 0.0
            {
                for value in row
                {
                    *value /= sum;
                }
            }
        }
        Ok(Self {
            weights: mixture_weights(observations.len(), config.prior_weight),
            component_probabilities: rows,
            cardinality,
        })
    }

    fn sample(&self, rng: &mut SplitMix64) -> ParamValue {
        let component = rng.weighted_index(&self.weights);
        let choice = rng.weighted_index(&self.component_probabilities[component]);
        ParamValue::Categorical(choice as u32)
    }

    fn log_pdf(&self, value: ParamValue) -> f64 {
        let ParamValue::Categorical(choice) = value
        else
        {
            return f64::NEG_INFINITY;
        };
        let choice = choice as usize;
        if choice >= self.cardinality
        {
            return f64::NEG_INFINITY;
        }
        let mut terms = Vec::with_capacity(self.weights.len());
        for index in 0..self.weights.len()
        {
            let weight = self.weights[index];
            let probability = self.component_probabilities[index][choice];
            if weight > 0.0 && probability > 0.0
            {
                terms.push(weight.ln() + probability.ln());
            }
        }
        logsumexp(&terms)
    }
}

#[derive(Debug, Clone)]
enum ParzenModel {
    Numerical(NumericalParzen),
    Categorical(CategoricalParzen),
}

impl ParzenModel {
    fn new(
        param: ParamId,
        observations: &[ParamValue],
        distribution: &Distribution,
        config: TpeConfig,
    ) -> Result<Self, TpeError> {
        match distribution
        {
            Distribution::Categorical { cardinality } => Ok(Self::Categorical(
                CategoricalParzen::new(param, observations, *cardinality, config)?,
            )),
            _ => Ok(Self::Numerical(NumericalParzen::new(
                param,
                observations,
                distribution,
                config,
            )?)),
        }
    }

    fn sample(&self, rng: &mut SplitMix64) -> ParamValue {
        match self
        {
            Self::Numerical(model) => model.sample(rng),
            Self::Categorical(model) => model.sample(rng),
        }
    }

    fn log_pdf(&self, value: ParamValue) -> f64 {
        match self
        {
            Self::Numerical(model) => model.log_pdf(value),
            Self::Categorical(model) => model.log_pdf(value),
        }
    }
}

fn mixture_weights(n_observations: usize, prior_weight: f64) -> Vec<f64> {
    if n_observations == 0
    {
        return vec![1.0];
    }
    let mut weights = default_weights(n_observations);
    weights.push(prior_weight);
    let sum: f64 = weights.iter().sum();
    if sum > 0.0
    {
        for weight in &mut weights
        {
            *weight /= sum;
        }
    }
    weights
}

fn normal_cdf(value: f64) -> f64 {
    if value < 0.0
    {
        0.5 * erfc(-value / SQRT_2)
    }
    else
    {
        1.0 - 0.5 * erfc(value / SQRT_2)
    }
}

fn normal_inverse_cdf(probability: f64) -> f64 {
    let probability = probability.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    SQRT_2 * erfinv(2.0 * probability - 1.0)
}

fn sample_truncated_normal(rng: &mut SplitMix64, mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    let lower = normal_cdf((low - mu) / sigma);
    let upper = normal_cdf((high - mu) / sigma);
    let mass = upper - lower;
    if !(mass > f64::MIN_POSITIVE)
    {
        return mu.clamp(low, high);
    }
    let probability = lower + mass * rng.next_f64();
    (mu + sigma * normal_inverse_cdf(probability)).clamp(low, high)
}

fn truncated_normal_pdf(value: f64, mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    if value < low || value > high || !(sigma > 0.0)
    {
        return 0.0;
    }
    let denominator = normal_cdf((high - mu) / sigma) - normal_cdf((low - mu) / sigma);
    if !(denominator > f64::MIN_POSITIVE)
    {
        return 0.0;
    }
    let z = (value - mu) / sigma;
    (-0.5 * z * z).exp() / (SQRT_2PI * sigma * denominator)
}

fn truncated_discrete_mass(value: f64, mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    if !(sigma > 0.0)
    {
        return 0.0;
    }
    let left = value - 0.5;
    let right = value + 0.5;
    let numerator = normal_cdf((right - mu) / sigma) - normal_cdf((left - mu) / sigma);
    let denominator = normal_cdf((high - mu) / sigma) - normal_cdf((low - mu) / sigma);
    if !(numerator > 0.0 && denominator > f64::MIN_POSITIVE)
    {
        return 0.0;
    }
    numerator / denominator
}

fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() && max.is_sign_negative()
    {
        return max;
    }
    max + values
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f64>()
        .ln()
}

fn random_value(rng: &mut SplitMix64, distribution: &Distribution) -> ParamValue {
    match distribution
    {
        Distribution::Uniform { low, high } =>
        {
            ParamValue::Float(low + (high - low) * rng.next_f64())
        },
        Distribution::LogUniform { low, high } =>
        {
            let log_value = low.ln() + (high.ln() - low.ln()) * rng.next_f64();
            ParamValue::Float(log_value.exp())
        },
        Distribution::IntRange { low, high } =>
        {
            let width = (*high - *low + 1) as usize;
            ParamValue::Int(*low + rng.index(width) as i64)
        },
        Distribution::Categorical { cardinality } =>
        {
            ParamValue::Categorical(rng.index(*cardinality as usize) as u32)
        },
    }
}

#[derive(Debug, Clone, Copy)]
struct RankedTrial {
    id: TrialId,
    objective: f64,
}

/// Correctness-oriented independent TPE sampler.
#[derive(Debug, Clone)]
pub struct TpeSampler {
    direction: Direction,
    config: TpeConfig,
    rng: SplitMix64,
}

impl TpeSampler {
    /// Construct the reference sampler with Optuna-v5-compatible defaults.
    pub fn new(direction: Direction, seed: u64) -> Self {
        Self {
            direction,
            config: TpeConfig::default(),
            rng: SplitMix64::new(seed),
        }
    }

    /// Construct with an explicit configuration.
    pub fn with_config(
        direction: Direction,
        seed: u64,
        config: TpeConfig,
    ) -> Result<Self, TpeError> {
        if config.n_ei_candidates == 0
        {
            return Err(TpeError::InvalidConfig(
                "n_ei_candidates must be greater than zero",
            ));
        }
        if !config.prior_weight.is_finite() || config.prior_weight < 0.0
        {
            return Err(TpeError::InvalidConfig(
                "prior_weight must be finite and non-negative",
            ));
        }
        Ok(Self {
            direction,
            config,
            rng: SplitMix64::new(seed),
        })
    }

    /// Return the active configuration.
    pub const fn config(&self) -> TpeConfig {
        self.config
    }

    /// Return the optimization direction.
    pub const fn direction(&self) -> Direction {
        self.direction
    }

    fn ranked_complete(&self, study: StudyView<'_>) -> Vec<RankedTrial> {
        let trials = study.trials();
        let mut ranked = trials
            .trial_ids()
            .filter_map(|id| {
                trials
                    .objective_value(id, 0)
                    .map(|objective| RankedTrial { id, objective })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            let objective_order = left.objective.total_cmp(&right.objective);
            let objective_order = match self.direction
            {
                Direction::Minimize => objective_order,
                Direction::Maximize => objective_order.reverse(),
            };
            objective_order.then_with(|| left.id.cmp(&right.id))
        });
        ranked
    }

    fn observations_for(
        &self,
        study: StudyView<'_>,
        ids: impl IntoIterator<Item = TrialId>,
        param: ParamId,
    ) -> Vec<ParamValue> {
        ids.into_iter()
            .filter_map(|id| study.trials().param_value(id, param))
            .collect()
    }

    fn sample_tpe_value(
        &mut self,
        study: StudyView<'_>,
        current_trial: TrialId,
        ranked: &[RankedTrial],
        param: ParamId,
        distribution: &Distribution,
    ) -> Result<ParamValue, TpeError> {
        let n_below = default_gamma(ranked.len()).min(ranked.len());
        let below_ids = ranked[..n_below].iter().map(|trial| trial.id);
        let above_complete = ranked[n_below..].iter().map(|trial| trial.id);

        let below = self.observations_for(study, below_ids, param);
        let mut above = self.observations_for(study, above_complete, param);
        if self.config.constant_liar
        {
            above.extend(
                study
                    .trials()
                    .trial_ids()
                    .filter(|id| *id != current_trial)
                    .filter(|id| study.trials().state(*id) == Some(TrialState::Running))
                    .filter_map(|id| study.trials().param_value(id, param)),
            );
        }

        let below_model = ParzenModel::new(param, &below, distribution, self.config)?;
        let above_model = ParzenModel::new(param, &above, distribution, self.config)?;

        let mut best_value = below_model.sample(&mut self.rng);
        let mut best_score = below_model.log_pdf(best_value) - above_model.log_pdf(best_value);
        for _ in 1..self.config.n_ei_candidates
        {
            let value = below_model.sample(&mut self.rng);
            let score = below_model.log_pdf(value) - above_model.log_pdf(value);
            if score.total_cmp(&best_score) == Ordering::Greater
            {
                best_value = value;
                best_score = score;
            }
        }
        Ok(best_value)
    }
}

impl Sampler for TpeSampler {
    type Error = TpeError;

    fn sample(&mut self, study: StudyView<'_>, trial: TrialId) -> Result<Candidate, Self::Error> {
        let objective_count = study.trials().objective_count();
        if objective_count != 1
        {
            return Err(TpeError::UnsupportedObjectiveCount(objective_count));
        }

        let ranked = self.ranked_complete(study);
        let use_random = ranked.len() < self.config.n_startup_trials;
        let space = study.search_space();
        let mut candidate = Candidate::empty(space);

        for spec in space.parameters()
        {
            if !space.is_active(spec.id, &candidate)
            {
                continue;
            }
            let value = if use_random
            {
                random_value(&mut self.rng, &spec.distribution)
            }
            else
            {
                self.sample_tpe_value(study, trial, &ranked, spec.id, &spec.distribution)?
            };
            candidate
                .set(space, spec.id, value)
                .map_err(TpeError::Candidate)?;
        }

        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_opt_core::{Condition, ParameterSpec, SearchSpace, Study, TrialOutcome};

    fn close(left: f64, right: f64, tolerance: f64) -> bool {
        (left - right).abs() <= tolerance
    }

    fn model(values: Vec<ParamValue>, distribution: Distribution) -> ParzenModel {
        ParzenModel::new(
            ParamId::new(0),
            &values,
            &distribution,
            TpeConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn defaults_match_audited_optuna_v5_slice() {
        let config = TpeConfig::default();
        assert_eq!(config.n_startup_trials, 10);
        assert_eq!(config.n_ei_candidates, 24);
        assert_eq!(config.prior_weight, 1.0);
        assert!(config.consider_magic_clip);
        assert!(!config.consider_endpoints);
        assert!(config.constant_liar);
        assert_eq!(default_gamma(1), 1);
        assert_eq!(default_gamma(250), 25);
        assert_eq!(default_gamma(1_000), 25);
    }

    #[test]
    fn default_weight_shape_matches_reference_rule() {
        assert_eq!(default_weights(0), Vec::<f64>::new());
        assert_eq!(default_weights(3), vec![1.0, 1.0, 1.0]);
        let weights = default_weights(30);
        assert_eq!(weights.len(), 30);
        assert!(close(weights[0], 1.0 / 30.0, 1e-15));
        assert_eq!(&weights[5..], vec![1.0; 25].as_slice());
    }

    #[test]
    fn continuous_parzen_log_pdf_matches_optuna_oracle() {
        let parzen = model(
            vec![
                ParamValue::Float(0.2),
                ParamValue::Float(0.5),
                ParamValue::Float(0.8),
            ],
            Distribution::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        let expected = [
            -0.169_822_673_404_147_03,
            0.080_648_168_482_729_71,
            0.138_563_428_843_004_2,
            -0.169_822_673_404_147_14,
        ];
        for (value, expected) in [0.1, 0.3, 0.6, 0.9].into_iter().zip(expected)
        {
            assert!(
                close(parzen.log_pdf(ParamValue::Float(value)), expected, 2e-11),
                "value={value}"
            );
        }
    }

    #[test]
    fn log_parzen_log_pdf_matches_optuna_oracle() {
        let parzen = model(
            vec![
                ParamValue::Float(1e-3),
                ParamValue::Float(1e-2),
                ParamValue::Float(1e-1),
            ],
            Distribution::LogUniform {
                low: 1e-4,
                high: 1.0,
            },
        );
        let expected = [
            -2.765_475_482_806_371,
            -2.176_158_111_874_251_2,
            -2.009_332_934_960_555_3,
            -2.547_251_324_163_88,
        ];
        for (value, expected) in [1e-4, 1e-3, 2e-2, 0.5].into_iter().zip(expected)
        {
            assert!(
                close(parzen.log_pdf(ParamValue::Float(value)), expected, 2e-10),
                "value={value}"
            );
        }
    }

    #[test]
    fn integer_parzen_log_pdf_matches_optuna_oracle() {
        let parzen = model(
            vec![ParamValue::Int(1), ParamValue::Int(4), ParamValue::Int(8)],
            Distribution::IntRange { low: 1, high: 10 },
        );
        let expected = [
            -2.166_379_418_467_155,
            -2.117_897_884_418_275,
            -2.218_051_499_663_914,
            -2.715_901_021_465_812,
        ];
        for (value, expected) in [1, 3, 5, 10].into_iter().zip(expected)
        {
            assert!(
                close(parzen.log_pdf(ParamValue::Int(value)), expected, 2e-10),
                "value={value}"
            );
        }
    }

    #[test]
    fn categorical_parzen_log_pdf_matches_optuna_oracle() {
        let parzen = model(
            vec![
                ParamValue::Categorical(0),
                ParamValue::Categorical(2),
                ParamValue::Categorical(2),
            ],
            Distribution::Categorical { cardinality: 3 },
        );
        let expected = [
            -1.098_612_288_668_109_8,
            -1.658_228_076_603_532_4,
            -0.741_937_344_729_377_2,
        ];
        for (choice, expected) in [0, 1, 2].into_iter().zip(expected)
        {
            assert!(
                close(
                    parzen.log_pdf(ParamValue::Categorical(choice)),
                    expected,
                    2e-12
                ),
                "choice={choice}"
            );
        }
    }

    fn one_dimensional_space() -> SearchSpace {
        SearchSpace::compile(vec![ParameterSpec::new(
            ParamId::new(0),
            "x",
            Distribution::Uniform {
                low: -5.0,
                high: 5.0,
            },
            Condition::Always,
        )])
        .unwrap()
    }

    #[test]
    fn startup_and_tpe_are_reproducible() {
        fn run() -> Vec<f64> {
            let mut study = Study::new(one_dimensional_space(), 1).unwrap();
            let mut sampler = TpeSampler::new(Direction::Minimize, 0x5eed);
            let mut values = Vec::new();
            for _ in 0..30
            {
                let proposal = study.ask(&mut sampler).unwrap();
                let ParamValue::Float(x) = proposal.candidate.value(ParamId::new(0)).unwrap()
                else
                {
                    panic!("x must be float");
                };
                values.push(x);
                study
                    .tell(
                        proposal.trial,
                        TrialOutcome::Complete(vec![(x - 1.25) * (x - 1.25)]),
                    )
                    .unwrap();
            }
            values
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn reference_tpe_improves_simple_quadratic() {
        let mut study = Study::new(one_dimensional_space(), 1).unwrap();
        let mut sampler = TpeSampler::new(Direction::Minimize, 1234);
        let mut best = f64::INFINITY;
        let mut best_x = f64::NAN;
        for _ in 0..80
        {
            let proposal = study.ask(&mut sampler).unwrap();
            let ParamValue::Float(x) = proposal.candidate.value(ParamId::new(0)).unwrap()
            else
            {
                panic!("x must be float");
            };
            let objective = (x - 1.25) * (x - 1.25);
            if objective < best
            {
                best = objective;
                best_x = x;
            }
            study
                .tell(proposal.trial, TrialOutcome::Complete(vec![objective]))
                .unwrap();
        }
        assert!(best < 1e-3, "best={best} best_x={best_x}");
    }

    #[test]
    fn conditional_space_proposals_validate() {
        let space = SearchSpace::compile(vec![
            ParameterSpec::new(
                ParamId::new(0),
                "kind",
                Distribution::Categorical { cardinality: 2 },
                Condition::Always,
            ),
            ParameterSpec::new(
                ParamId::new(1),
                "depth",
                Distribution::IntRange { low: 1, high: 8 },
                Condition::CategoricalEquals {
                    parent: ParamId::new(0),
                    choice: 1,
                },
            ),
            ParameterSpec::new(
                ParamId::new(2),
                "lr",
                Distribution::LogUniform {
                    low: 1e-5,
                    high: 1e-1,
                },
                Condition::Always,
            ),
        ])
        .unwrap();
        let mut study = Study::new(space, 1).unwrap();
        let mut sampler = TpeSampler::new(Direction::Minimize, 77);
        for index in 0..40
        {
            let proposal = study.ask(&mut sampler).unwrap();
            study
                .search_space()
                .validate_candidate(&proposal.candidate)
                .unwrap();
            study
                .tell(proposal.trial, TrialOutcome::Complete(vec![index as f64]))
                .unwrap();
        }
    }

    #[test]
    fn rolling_batch_is_supported_after_startup() {
        let mut study = Study::new(one_dimensional_space(), 1).unwrap();
        let mut sampler = TpeSampler::new(Direction::Minimize, 99);
        for _ in 0..12
        {
            let proposal = study.ask(&mut sampler).unwrap();
            let ParamValue::Float(x) = proposal.candidate.value(ParamId::new(0)).unwrap()
            else
            {
                panic!("x must be float");
            };
            study
                .tell(proposal.trial, TrialOutcome::Complete(vec![x * x]))
                .unwrap();
        }

        let pending = study.ask_batch(&mut sampler, 4).unwrap();
        assert_eq!(pending.len(), 4);
        assert_eq!(study.trials().count_state(TrialState::Running), 4);
    }

    #[test]
    fn multiobjective_study_is_rejected_explicitly() {
        let mut study = Study::new(one_dimensional_space(), 2).unwrap();
        let mut sampler = TpeSampler::new(Direction::Minimize, 1);
        let error = study.ask(&mut sampler).unwrap_err();
        match error
        {
            scirust_opt_core::AskError::Sampler {
                source: TpeError::UnsupportedObjectiveCount(2),
                ..
            } =>
            {},
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
