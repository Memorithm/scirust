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
use std::collections::BTreeSet;

use scirust_opt_core::{
    Candidate, CandidateError, Distribution, ParamId, ParamValue, Sampler, StudyEvent, StudyView,
    TrialId,
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
    n.div_ceil(10).min(25)
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
        if !total.is_finite() || total <= 0.0
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

#[derive(Debug, Clone, Copy)]
struct SortedNumericObservation {
    transformed: f64,
    trial: TrialId,
}

#[derive(Debug, Clone, Default)]
struct ParamHistoryCache {
    chronological: Vec<(TrialId, ParamValue)>,
    numeric_sorted: Vec<SortedNumericObservation>,
}

impl ParamHistoryCache {
    fn transform(
        param: ParamId,
        value: ParamValue,
        distribution: &Distribution,
    ) -> Result<Option<f64>, TpeError> {
        let transformed = match (distribution, value)
        {
            (Distribution::Uniform { .. }, ParamValue::Float(value)) => Some(value),
            (Distribution::LogUniform { .. }, ParamValue::Float(value)) if value > 0.0 =>
            {
                Some(value.ln())
            },
            (Distribution::IntRange { .. }, ParamValue::Int(value)) => Some(value as f64),
            (Distribution::Categorical { cardinality }, ParamValue::Categorical(choice))
                if choice < *cardinality =>
            {
                None
            },
            _ => return Err(TpeError::InvalidObservation(param)),
        };
        if transformed.is_some_and(|value| !value.is_finite())
        {
            return Err(TpeError::InvalidObservation(param));
        }
        Ok(transformed)
    }

    fn sorted_numeric_position(&self, transformed: f64, trial: TrialId) -> Result<usize, usize> {
        self.numeric_sorted.binary_search_by(|probe| {
            probe
                .transformed
                .total_cmp(&transformed)
                .then_with(|| probe.trial.cmp(&trial))
        })
    }

    fn remove_numeric(&mut self, transformed: f64, trial: TrialId) {
        if let Ok(position) = self.sorted_numeric_position(transformed, trial)
        {
            self.numeric_sorted.remove(position);
        }
    }

    fn upsert(
        &mut self,
        param: ParamId,
        trial: TrialId,
        value: ParamValue,
        distribution: &Distribution,
    ) -> Result<(), TpeError> {
        let transformed = Self::transform(param, value, distribution)?;
        match self
            .chronological
            .binary_search_by_key(&trial, |(stored, _)| *stored)
        {
            Ok(position) =>
            {
                let old = self.chronological[position].1;
                if let Some(old_transformed) = Self::transform(param, old, distribution)?
                {
                    self.remove_numeric(old_transformed, trial);
                }
                self.chronological[position].1 = value;
            },
            Err(position) => self.chronological.insert(position, (trial, value)),
        }

        if let Some(transformed) = transformed
        {
            let position = self
                .sorted_numeric_position(transformed, trial)
                .unwrap_or_else(|position| position);
            self.numeric_sorted
                .insert(position, SortedNumericObservation { transformed, trial });
        }
        Ok(())
    }

    fn selected_count(&self, mask: &[bool]) -> usize {
        self.chronological
            .iter()
            .filter(|(trial, _)| mask_contains(mask, *trial))
            .count()
    }
}

fn mask_contains(mask: &[bool], trial: TrialId) -> bool {
    usize::try_from(trial.get())
        .ok()
        .and_then(|index| mask.get(index))
        .copied()
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
struct NumericalParzen {
    weights: Vec<f64>,
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    log_component_factors: Vec<f64>,
    adapted_low: f64,
    adapted_high: f64,
    kind: NumericalKind,
}

impl NumericalParzen {
    fn from_components(
        weights: Vec<f64>,
        mus: Vec<f64>,
        sigmas: Vec<f64>,
        adapted_low: f64,
        adapted_high: f64,
        kind: NumericalKind,
    ) -> Self {
        let log_component_factors = weights
            .iter()
            .copied()
            .zip(mus.iter().copied())
            .zip(sigmas.iter().copied())
            .map(|((weight, mu), sigma)| {
                if weight <= 0.0 || !sigma.is_finite() || sigma <= 0.0
                {
                    return f64::NEG_INFINITY;
                }
                let denominator =
                    normal_cdf((adapted_high - mu) / sigma)
                        - normal_cdf((adapted_low - mu) / sigma);
                if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE
                {
                    return f64::NEG_INFINITY;
                }
                let base = weight.ln() - denominator.ln();
                match kind
                {
                    NumericalKind::Integer { .. } => base,
                    NumericalKind::LinearFloat | NumericalKind::LogFloat =>
                    {
                        base - (SQRT_2PI * sigma).ln()
                    },
                }
            })
            .collect();

        Self {
            weights,
            mus,
            sigmas,
            log_component_factors,
            adapted_low,
            adapted_high,
            kind,
        }
    }

    #[cfg(test)]
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
        if !range.is_finite() || range <= 0.0
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

    fn new_cached(
        param: ParamId,
        history: &ParamHistoryCache,
        mask: &[bool],
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
        if !range.is_finite() || range <= 0.0
        {
            return Err(TpeError::InvalidNumericalRange(param));
        }

        let n_observations = history.selected_count(mask);
        if n_observations == 0
        {
            return Ok(Self {
                weights: vec![1.0],
                mus: vec![0.5 * (adapted_low + adapted_high)],
                sigmas: vec![range],
                adapted_low,
                adapted_high,
                kind,
            });
        }

        let chronological_weights = default_weights(n_observations);
        let mut weight_by_trial = vec![0.0; mask.len()];
        let mut chronological_index = 0;
        for (trial, _) in &history.chronological
        {
            if !mask_contains(mask, *trial)
            {
                continue;
            }
            let trial_index =
                usize::try_from(trial.get()).map_err(|_| TpeError::InvalidObservation(param))?;
            let Some(slot) = weight_by_trial.get_mut(trial_index)
            else
            {
                return Err(TpeError::InvalidObservation(param));
            };
            *slot = chronological_weights[chronological_index];
            chronological_index += 1;
        }

        let selected_sorted = history
            .numeric_sorted
            .iter()
            .copied()
            .filter(|observation| mask_contains(mask, observation.trial))
            .collect::<Vec<_>>();
        if selected_sorted.len() != n_observations
        {
            return Err(TpeError::InvalidObservation(param));
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
        let mut sigma_by_trial = vec![range; mask.len()];
        for (position, observation) in selected_sorted.iter().enumerate()
        {
            let left_gap = if position == 0
            {
                observation.transformed - adapted_low
            }
            else
            {
                observation.transformed - selected_sorted[position - 1].transformed
            };
            let right_gap = if position + 1 == n_observations
            {
                adapted_high - observation.transformed
            }
            else
            {
                selected_sorted[position + 1].transformed - observation.transformed
            };
            let mut sigma = left_gap.max(right_gap);
            if !config.consider_endpoints && n_observations >= 2
            {
                if position == 0
                {
                    sigma = selected_sorted[1].transformed - observation.transformed;
                }
                else if position + 1 == n_observations
                {
                    sigma =
                        observation.transformed - selected_sorted[n_observations - 2].transformed;
                }
            }
            let trial_index = usize::try_from(observation.trial.get())
                .map_err(|_| TpeError::InvalidObservation(param))?;
            let Some(slot) = sigma_by_trial.get_mut(trial_index)
            else
            {
                return Err(TpeError::InvalidObservation(param));
            };
            *slot = sigma.clamp(min_sigma, range);
        }

        let mut mus = Vec::with_capacity(n_observations + 1);
        let mut sigmas = Vec::with_capacity(n_observations + 1);
        let mut weights = Vec::with_capacity(n_observations + 1);
        for (trial, value) in &history.chronological
        {
            if !mask_contains(mask, *trial)
            {
                continue;
            }
            let Some(transformed) = ParamHistoryCache::transform(param, *value, distribution)?
            else
            {
                return Err(TpeError::InvalidObservation(param));
            };
            let trial_index =
                usize::try_from(trial.get()).map_err(|_| TpeError::InvalidObservation(param))?;
            mus.push(transformed);
            sigmas.push(sigma_by_trial[trial_index]);
            weights.push(weight_by_trial[trial_index]);
        }

        mus.push(0.5 * (adapted_low + adapted_high));
        sigmas.push(range);
        weights.push(config.prior_weight);
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0
        {
            for weight in &mut weights
            {
                *weight /= weight_sum;
            }
        }

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
    #[cfg(test)]
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

    fn new_cached(
        param: ParamId,
        history: &ParamHistoryCache,
        mask: &[bool],
        cardinality: u32,
        config: TpeConfig,
    ) -> Result<Self, TpeError> {
        let cardinality = cardinality as usize;
        let n_observations = history.selected_count(mask);
        if n_observations == 0
        {
            return Ok(Self {
                weights: vec![1.0],
                component_probabilities: vec![vec![1.0 / cardinality as f64; cardinality]],
                cardinality,
            });
        }

        let n_kernels = n_observations + 1;
        let base = config.prior_weight / n_kernels as f64;
        let mut rows = Vec::with_capacity(n_kernels);
        for (trial, observation) in &history.chronological
        {
            if !mask_contains(mask, *trial)
            {
                continue;
            }
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
            let mut row = vec![base; cardinality];
            row[choice] += 1.0;
            let sum: f64 = row.iter().sum();
            if sum > 0.0
            {
                for value in &mut row
                {
                    *value /= sum;
                }
            }
            rows.push(row);
        }

        let mut prior = vec![base; cardinality];
        let prior_sum: f64 = prior.iter().sum();
        if prior_sum > 0.0
        {
            for value in &mut prior
            {
                *value /= prior_sum;
            }
        }
        rows.push(prior);

        Ok(Self {
            weights: mixture_weights(n_observations, config.prior_weight),
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
    #[cfg(test)]
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

    fn new_cached(
        param: ParamId,
        history: &ParamHistoryCache,
        mask: &[bool],
        distribution: &Distribution,
        config: TpeConfig,
    ) -> Result<Self, TpeError> {
        match distribution
        {
            Distribution::Categorical { cardinality } => Ok(Self::Categorical(
                CategoricalParzen::new_cached(param, history, mask, *cardinality, config)?,
            )),
            _ => Ok(Self::Numerical(NumericalParzen::new_cached(
                param,
                history,
                mask,
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
    if !mass.is_finite() || mass <= f64::MIN_POSITIVE
    {
        return mu.clamp(low, high);
    }
    let probability = lower + mass * rng.next_f64();
    (mu + sigma * normal_inverse_cdf(probability)).clamp(low, high)
}

fn truncated_normal_pdf(value: f64, mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    if value < low || value > high || !sigma.is_finite() || sigma <= 0.0
    {
        return 0.0;
    }
    let denominator = normal_cdf((high - mu) / sigma) - normal_cdf((low - mu) / sigma);
    if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE
    {
        return 0.0;
    }
    let z = (value - mu) / sigma;
    (-0.5 * z * z).exp() / (SQRT_2PI * sigma * denominator)
}

fn truncated_discrete_mass(value: f64, mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0
    {
        return 0.0;
    }
    let left = value - 0.5;
    let right = value + 0.5;
    let numerator = normal_cdf((right - mu) / sigma) - normal_cdf((left - mu) / sigma);
    let denominator = normal_cdf((high - mu) / sigma) - normal_cdf((low - mu) / sigma);
    if !numerator.is_finite()
        || numerator <= 0.0
        || !denominator.is_finite()
        || denominator <= f64::MIN_POSITIVE
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

fn compare_ranked(direction: Direction, left: &RankedTrial, right: &RankedTrial) -> Ordering {
    let objective_order = left.objective.total_cmp(&right.objective);
    let objective_order = match direction
    {
        Direction::Minimize => objective_order,
        Direction::Maximize => objective_order.reverse(),
    };
    objective_order.then_with(|| left.id.cmp(&right.id))
}

/// Correctness-oriented independent TPE sampler.
#[derive(Debug, Clone)]
pub struct TpeSampler {
    direction: Direction,
    config: TpeConfig,
    rng: SplitMix64,
    event_cursor: usize,
    last_event: Option<StudyEvent>,
    ranked_complete: Vec<RankedTrial>,
    below_complete: BTreeSet<TrialId>,
    above_complete: BTreeSet<TrialId>,
    running: BTreeSet<TrialId>,
    param_history: Vec<ParamHistoryCache>,
}

impl TpeSampler {
    /// Construct the reference sampler with Optuna-v5-compatible defaults.
    pub fn new(direction: Direction, seed: u64) -> Self {
        Self::with_validated_config(direction, seed, TpeConfig::default())
    }

    fn with_validated_config(direction: Direction, seed: u64, config: TpeConfig) -> Self {
        Self {
            direction,
            config,
            rng: SplitMix64::new(seed),
            event_cursor: 0,
            last_event: None,
            ranked_complete: Vec::new(),
            below_complete: BTreeSet::new(),
            above_complete: BTreeSet::new(),
            running: BTreeSet::new(),
            param_history: Vec::new(),
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
        Ok(Self::with_validated_config(direction, seed, config))
    }

    /// Return the active configuration.
    pub const fn config(&self) -> TpeConfig {
        self.config
    }

    /// Return the optimization direction.
    pub const fn direction(&self) -> Direction {
        self.direction
    }

    fn reset_history(&mut self) {
        self.event_cursor = 0;
        self.last_event = None;
        self.ranked_complete.clear();
        self.below_complete.clear();
        self.above_complete.clear();
        self.running.clear();
        self.param_history.clear();
    }

    fn insert_complete(&mut self, trial: RankedTrial) {
        if self
            .ranked_complete
            .iter()
            .any(|existing| existing.id == trial.id)
        {
            return;
        }

        let position = self
            .ranked_complete
            .binary_search_by(|probe| compare_ranked(self.direction, probe, &trial))
            .unwrap_or_else(|position| position);
        self.ranked_complete.insert(position, trial);
        self.above_complete.insert(trial.id);
        self.rebalance_complete_groups();
    }

    fn rebalance_complete_groups(&mut self) {
        let n_below = default_gamma(self.ranked_complete.len()).min(self.ranked_complete.len());
        let desired = self.ranked_complete[..n_below]
            .iter()
            .map(|trial| trial.id)
            .collect::<BTreeSet<_>>();

        let to_above = self
            .below_complete
            .difference(&desired)
            .copied()
            .collect::<Vec<_>>();
        for trial in to_above
        {
            self.below_complete.remove(&trial);
            self.above_complete.insert(trial);
        }

        let to_below = desired
            .difference(&self.below_complete)
            .copied()
            .collect::<Vec<_>>();
        for trial in to_below
        {
            self.above_complete.remove(&trial);
            self.below_complete.insert(trial);
        }
    }

    fn ensure_param_history(&mut self, study: StudyView<'_>) {
        if self.param_history.len() != study.search_space().len()
        {
            self.param_history = vec![ParamHistoryCache::default(); study.search_space().len()];
        }
    }

    fn apply_event(&mut self, study: StudyView<'_>, event: &StudyEvent) -> Result<(), TpeError> {
        match event
        {
            StudyEvent::TrialStarted { trial } =>
            {
                self.running.insert(*trial);
            },
            StudyEvent::TrialCompleted { trial, values } =>
            {
                self.running.remove(trial);
                if let Some(objective) = values.first().copied()
                {
                    self.insert_complete(RankedTrial {
                        id: *trial,
                        objective,
                    });
                }
            },
            StudyEvent::TrialPruned { trial } | StudyEvent::TrialFailed { trial } =>
            {
                self.running.remove(trial);
            },
            StudyEvent::ParameterAssigned {
                trial,
                param,
                value,
            } =>
            {
                let Some(spec) = study.search_space().parameter(*param)
                else
                {
                    return Err(TpeError::InvalidObservation(*param));
                };
                let Some(history) = self.param_history.get_mut(param.index())
                else
                {
                    return Err(TpeError::InvalidObservation(*param));
                };
                history.upsert(*param, *trial, *value, &spec.distribution)?;
            },
            StudyEvent::TrialReserved { .. } =>
            {},
        }
        Ok(())
    }

    fn sync_history(&mut self, study: StudyView<'_>) -> Result<(), TpeError> {
        let prefix_matches = self.event_cursor == 0
            || study.events().get(self.event_cursor.saturating_sub(1)) == self.last_event.as_ref();
        if study.events().len() < self.event_cursor || !prefix_matches
        {
            self.reset_history();
        }
        self.ensure_param_history(study);

        if let Some(events) = study.events_since(self.event_cursor)
        {
            for event in events
            {
                self.apply_event(study, event)?;
            }
            self.event_cursor = study.events().len();
            self.last_event = study.events().last().cloned();
        }
        Ok(())
    }

    fn sample_tpe_value(
        config: TpeConfig,
        rng: &mut SplitMix64,
        history: &ParamHistoryCache,
        below_mask: &[bool],
        above_mask: &[bool],
        param: ParamId,
        distribution: &Distribution,
    ) -> Result<ParamValue, TpeError> {
        let below_model =
            ParzenModel::new_cached(param, history, below_mask, distribution, config)?;
        let above_model =
            ParzenModel::new_cached(param, history, above_mask, distribution, config)?;

        let mut best_value = below_model.sample(rng);
        let mut best_score = below_model.log_pdf(best_value) - above_model.log_pdf(best_value);
        for _ in 1..config.n_ei_candidates
        {
            let value = below_model.sample(rng);
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

    fn sample(&mut self, study: StudyView<'_>, _trial: TrialId) -> Result<Candidate, Self::Error> {
        let objective_count = study.trials().objective_count();
        if objective_count != 1
        {
            return Err(TpeError::UnsupportedObjectiveCount(objective_count));
        }

        self.sync_history(study)?;
        let use_random = self.ranked_complete.len() < self.config.n_startup_trials;
        let trial_count = study.trials().len();
        let mut below_mask = vec![false; trial_count];
        let mut above_mask = vec![false; trial_count];
        for trial in &self.below_complete
        {
            if let Ok(index) = usize::try_from(trial.get())
                && let Some(slot) = below_mask.get_mut(index)
            {
                *slot = true;
            }
        }
        for trial in &self.above_complete
        {
            if let Ok(index) = usize::try_from(trial.get())
                && let Some(slot) = above_mask.get_mut(index)
            {
                *slot = true;
            }
        }
        if self.config.constant_liar
        {
            for trial in &self.running
            {
                if let Ok(index) = usize::try_from(trial.get())
                    && let Some(slot) = above_mask.get_mut(index)
                {
                    *slot = true;
                }
            }
        }

        let space = study.search_space();
        let mut candidate = Candidate::empty(space);
        let config = self.config;
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
                let Some(history) = self.param_history.get(spec.id.index())
                else
                {
                    return Err(TpeError::InvalidObservation(spec.id));
                };
                Self::sample_tpe_value(
                    config,
                    &mut self.rng,
                    history,
                    &below_mask,
                    &above_mask,
                    spec.id,
                    &spec.distribution,
                )?
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
    use scirust_opt_core::{
        Condition, ParameterSpec, SearchSpace, Study, TrialOutcome, TrialState,
    };

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

    fn cached_model(
        values: &[ParamValue],
        selected: &[bool],
        distribution: Distribution,
    ) -> (ParzenModel, ParzenModel) {
        let param = ParamId::new(0);
        let mut history = ParamHistoryCache::default();
        for (index, value) in values.iter().copied().enumerate()
        {
            history
                .upsert(param, TrialId::new(index as u64), value, &distribution)
                .unwrap();
        }
        let selected_values = values
            .iter()
            .copied()
            .zip(selected.iter().copied())
            .filter_map(|(value, selected)| selected.then_some(value))
            .collect::<Vec<_>>();
        let reference =
            ParzenModel::new(param, &selected_values, &distribution, TpeConfig::default()).unwrap();
        let cached = ParzenModel::new_cached(
            param,
            &history,
            selected,
            &distribution,
            TpeConfig::default(),
        )
        .unwrap();
        (reference, cached)
    }

    #[test]
    fn cached_numerical_parzen_preserves_reference_sampling_sequence() {
        let values = (0..30)
            .map(|index| {
                let raw = ((index * 17) % 29) as f64 / 29.0;
                ParamValue::Float(raw)
            })
            .collect::<Vec<_>>();
        let selected = (0..30).map(|index| index % 4 != 1).collect::<Vec<_>>();
        let (reference, cached) = cached_model(
            &values,
            &selected,
            Distribution::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );

        for probe in [0.01, 0.17, 0.43, 0.77, 0.99]
        {
            assert!(
                close(
                    reference.log_pdf(ParamValue::Float(probe)),
                    cached.log_pdf(ParamValue::Float(probe)),
                    1e-14,
                ),
                "probe={probe}"
            );
        }

        let mut reference_rng = SplitMix64::new(0x1234);
        let mut cached_rng = SplitMix64::new(0x1234);
        for _ in 0..64
        {
            assert_eq!(
                reference.sample(&mut reference_rng),
                cached.sample(&mut cached_rng)
            );
        }
    }

    #[test]
    fn cached_categorical_parzen_preserves_reference_sampling_sequence() {
        let values = (0..36)
            .map(|index| ParamValue::Categorical(((index * 5) % 4) as u32))
            .collect::<Vec<_>>();
        let selected = (0..36).map(|index| index % 5 != 2).collect::<Vec<_>>();
        let (reference, cached) = cached_model(
            &values,
            &selected,
            Distribution::Categorical { cardinality: 4 },
        );

        for choice in 0..4
        {
            assert!(
                close(
                    reference.log_pdf(ParamValue::Categorical(choice)),
                    cached.log_pdf(ParamValue::Categorical(choice)),
                    1e-14,
                ),
                "choice={choice}"
            );
        }

        let mut reference_rng = SplitMix64::new(0x5678);
        let mut cached_rng = SplitMix64::new(0x5678);
        for _ in 0..64
        {
            assert_eq!(
                reference.sample(&mut reference_rng),
                cached.sample(&mut cached_rng)
            );
        }
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

    #[test]
    fn event_watermark_updates_ranked_history_incrementally() {
        let mut study = Study::new(one_dimensional_space(), 1).unwrap();
        let mut sampler = TpeSampler::new(Direction::Minimize, 9);

        for (x, objective) in [(0.0, 4.0), (1.0, 1.0), (2.0, 3.0), (3.0, 2.0)]
        {
            let trial = study.reserve().unwrap();
            study
                .set_param(trial, ParamId::new(0), ParamValue::Float(x))
                .unwrap();
            study.start(trial).unwrap();
            study
                .tell(trial, TrialOutcome::Complete(vec![objective]))
                .unwrap();
        }

        let first_pending = study.ask(&mut sampler).unwrap();
        assert!(sampler.event_cursor <= study.events().len());
        assert_eq!(
            sampler
                .ranked_complete
                .iter()
                .map(|trial| trial.id.get())
                .collect::<Vec<_>>(),
            vec![1, 3, 2, 0]
        );
        assert_eq!(
            sampler
                .below_complete
                .iter()
                .map(|trial| trial.get())
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert!(sampler.running.is_empty());

        let _second_pending = study.ask(&mut sampler).unwrap();
        assert_eq!(
            sampler
                .running
                .iter()
                .map(|trial| trial.get())
                .collect::<Vec<_>>(),
            vec![first_pending.trial.get()]
        );
    }

    #[test]
    fn split_groups_are_chronological_before_parzen_weighting() {
        let mut study = Study::new(one_dimensional_space(), 1).unwrap();
        let mut sampler = TpeSampler::new(Direction::Minimize, 11);

        for number in 0..30_u64
        {
            let trial = study.reserve().unwrap();
            study
                .set_param(
                    trial,
                    ParamId::new(0),
                    ParamValue::Float(number as f64 / 10.0 - 1.5),
                )
                .unwrap();
            study.start(trial).unwrap();
            study
                .tell(trial, TrialOutcome::Complete(vec![(29 - number) as f64]))
                .unwrap();
        }

        let _pending = study.ask(&mut sampler).unwrap();

        assert_eq!(
            sampler
                .below_complete
                .iter()
                .map(|trial| trial.get())
                .collect::<Vec<_>>(),
            vec![27, 28, 29]
        );
        assert_eq!(
            sampler
                .above_complete
                .iter()
                .take(5)
                .map(|trial| trial.get())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );

        let above_ids = sampler.above_complete.iter().copied().collect::<Vec<_>>();
        let above = above_ids
            .iter()
            .copied()
            .filter_map(|trial| study.trials().param_value(trial, ParamId::new(0)))
            .collect::<Vec<_>>();
        assert_eq!(above.len(), 27);
        assert_eq!(above[0], ParamValue::Float(-1.5));
        assert_eq!(above[26], ParamValue::Float(1.1));
    }
}
