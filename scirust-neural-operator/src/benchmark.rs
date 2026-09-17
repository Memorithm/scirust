//! Measured exact-vs-surrogate qualification harness.
//!
//! Operation counts are useful for explaining where work is removed, but an
//! acceleration claim requires elapsed-time evidence. This module times the
//! authoritative solver and learned operator on the same input, alternates their
//! execution order to reduce first/second-position bias, and reports accuracy
//! independently from timing.

use std::hint::black_box;
use std::time::Instant;

use crate::{
    LearnedOperator, NeuralOperatorError, OperatorMetrics, Result, SurrogateEconomics, relative_l2,
};

/// Reproducible benchmark schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkConfig {
    /// Untimed calls of each path before measurement.
    pub warmup_calls: usize,
    /// Timed calls of each path. Must be positive.
    pub repeats: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_calls: 1,
            repeats: 10,
        }
    }
}

impl BenchmarkConfig {
    /// Validate that the benchmark contains at least one measured repeat.
    pub fn validate(self) -> Result<Self> {
        if self.repeats == 0
        {
            return Err(NeuralOperatorError::InvalidBenchmarkRepeats { repeats: 0 });
        }
        Ok(self)
    }
}

/// Distribution summary of measured wall-clock calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingStats {
    pub calls: usize,
    pub total_seconds: f64,
    pub mean_seconds: f64,
    pub median_seconds: f64,
    pub min_seconds: f64,
    pub max_seconds: f64,
}

impl TimingStats {
    fn from_samples(samples: &[f64]) -> Self {
        debug_assert!(!samples.is_empty());
        let total_seconds = samples.iter().sum::<f64>();
        let mut ordered = samples.to_vec();
        ordered.sort_by(f64::total_cmp);
        let median_seconds = if ordered.len() % 2 == 1
        {
            ordered[ordered.len() / 2]
        }
        else
        {
            let upper = ordered.len() / 2;
            (ordered[upper - 1] + ordered[upper]) * 0.5
        };
        Self {
            calls: samples.len(),
            total_seconds,
            mean_seconds: total_seconds / samples.len() as f64,
            median_seconds,
            min_seconds: ordered[0],
            max_seconds: ordered[ordered.len() - 1],
        }
    }
}

/// Accuracy accumulated across repeated paired exact/surrogate evaluations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkAccuracy {
    /// Mean per-repeat relative L2 error of the surrogate against the exact path.
    pub mean_relative_l2: f64,
    /// Largest per-repeat relative L2 error.
    pub max_relative_l2: f64,
    /// Mean per-repeat RMSE.
    pub mean_rmse: f64,
    /// Largest absolute scalar error observed over all repeats.
    pub max_abs: f64,
    /// Largest relative L2 drift of repeated exact outputs against the first exact output.
    pub exact_repeat_drift: f64,
    /// Largest relative L2 drift of repeated surrogate outputs against the first surrogate output.
    pub surrogate_repeat_drift: f64,
}

/// Measured qualification report for one fixed input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurrogateBenchmarkReport {
    pub exact_timing: TimingStats,
    pub surrogate_timing: TimingStats,
    pub accuracy: BenchmarkAccuracy,
}

impl SurrogateBenchmarkReport {
    /// Ratio of exact mean wall time to surrogate mean wall time.
    ///
    /// This number describes only this measured benchmark environment/input.
    pub fn measured_online_speedup(self) -> f64 {
        if self.surrogate_timing.mean_seconds == 0.0
        {
            f64::INFINITY
        }
        else
        {
            self.exact_timing.mean_seconds / self.surrogate_timing.mean_seconds
        }
    }

    /// Convert measured mean wall times plus an externally measured training cost
    /// into the standard amortisation model.
    pub fn economics(self, training_seconds: f64) -> Result<SurrogateEconomics> {
        SurrogateEconomics::new(
            self.exact_timing.mean_seconds,
            self.surrogate_timing.mean_seconds,
            training_seconds,
        )
    }
}

fn validate_input<O: LearnedOperator>(operator: &O, input: &[f32]) -> Result<()> {
    if input.len() != operator.input_len()
    {
        return Err(NeuralOperatorError::ShapeMismatch {
            what: "surrogate benchmark input",
            expected: operator.input_len(),
            got: input.len(),
        });
    }
    if let Some((index, _)) = input
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(NeuralOperatorError::NonFinite {
            what: "surrogate benchmark input",
            index,
        });
    }
    Ok(())
}

fn validate_output(what: &'static str, output: Vec<f32>, expected: usize) -> Result<Vec<f32>> {
    if output.len() != expected
    {
        return Err(NeuralOperatorError::ShapeMismatch {
            what,
            expected,
            got: output.len(),
        });
    }
    if let Some((index, _)) = output
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(NeuralOperatorError::NonFinite { what, index });
    }
    Ok(output)
}

fn time_call<F>(mut call: F) -> Result<(Vec<f32>, f64)>
where
    F: FnMut() -> Result<Vec<f32>>,
{
    let start = Instant::now();
    let output = call()?;
    black_box(&output);
    Ok((output, start.elapsed().as_secs_f64()))
}

/// Benchmark an authoritative solver and learned operator on the same fixed input.
///
/// After explicit warmup, each repeat executes both paths. Even repeats execute
/// exact first; odd repeats execute surrogate first. Accuracy is computed for
/// every paired repeat. Repeated-output drift is retained so nondeterministic or
/// stateful paths are visible instead of silently averaged away.
pub fn benchmark_exact_vs_surrogate<O, F>(
    operator: &mut O,
    input: &[f32],
    config: BenchmarkConfig,
    mut exact_solver: F,
) -> Result<SurrogateBenchmarkReport>
where
    O: LearnedOperator,
    F: FnMut(&[f32]) -> Result<Vec<f32>>,
{
    let config = config.validate()?;
    validate_input(operator, input)?;
    let output_len = operator.output_len();

    for warmup in 0..config.warmup_calls
    {
        if warmup % 2 == 0
        {
            let exact =
                validate_output("benchmark exact output", exact_solver(input)?, output_len)?;
            black_box(exact);
            let surrogate = validate_output(
                "benchmark surrogate output",
                operator.predict(input)?,
                output_len,
            )?;
            black_box(surrogate);
        }
        else
        {
            let surrogate = validate_output(
                "benchmark surrogate output",
                operator.predict(input)?,
                output_len,
            )?;
            black_box(surrogate);
            let exact =
                validate_output("benchmark exact output", exact_solver(input)?, output_len)?;
            black_box(exact);
        }
    }

    let mut exact_times = Vec::with_capacity(config.repeats);
    let mut surrogate_times = Vec::with_capacity(config.repeats);
    let mut first_exact: Option<Vec<f32>> = None;
    let mut first_surrogate: Option<Vec<f32>> = None;
    let mut exact_repeat_drift = 0.0f64;
    let mut surrogate_repeat_drift = 0.0f64;
    let mut relative_sum = 0.0f64;
    let mut max_relative = 0.0f64;
    let mut rmse_sum = 0.0f64;
    let mut max_abs = 0.0f64;

    for repeat in 0..config.repeats
    {
        let (exact, exact_seconds, surrogate, surrogate_seconds) = if repeat % 2 == 0
        {
            let (exact, exact_seconds) = time_call(|| exact_solver(input))?;
            let exact = validate_output("benchmark exact output", exact, output_len)?;
            let (surrogate, surrogate_seconds) = time_call(|| operator.predict(input))?;
            let surrogate = validate_output("benchmark surrogate output", surrogate, output_len)?;
            (exact, exact_seconds, surrogate, surrogate_seconds)
        }
        else
        {
            let (surrogate, surrogate_seconds) = time_call(|| operator.predict(input))?;
            let surrogate = validate_output("benchmark surrogate output", surrogate, output_len)?;
            let (exact, exact_seconds) = time_call(|| exact_solver(input))?;
            let exact = validate_output("benchmark exact output", exact, output_len)?;
            (exact, exact_seconds, surrogate, surrogate_seconds)
        };

        exact_times.push(exact_seconds);
        surrogate_times.push(surrogate_seconds);
        if let Some(reference) = &first_exact
        {
            exact_repeat_drift = exact_repeat_drift.max(relative_l2(&exact, reference)?);
        }
        else
        {
            first_exact = Some(exact.clone());
        }
        if let Some(reference) = &first_surrogate
        {
            surrogate_repeat_drift =
                surrogate_repeat_drift.max(relative_l2(&surrogate, reference)?);
        }
        else
        {
            first_surrogate = Some(surrogate.clone());
        }
        let metrics = OperatorMetrics::from_fields(&surrogate, &exact)?;
        relative_sum += metrics.relative_l2;
        max_relative = max_relative.max(metrics.relative_l2);
        rmse_sum += metrics.rmse;
        max_abs = max_abs.max(metrics.max_abs);
    }

    Ok(SurrogateBenchmarkReport {
        exact_timing: TimingStats::from_samples(&exact_times),
        surrogate_timing: TimingStats::from_samples(&surrogate_times),
        accuracy: BenchmarkAccuracy {
            mean_relative_l2: relative_sum / config.repeats as f64,
            max_relative_l2: max_relative,
            mean_rmse: rmse_sum / config.repeats as f64,
            max_abs,
            exact_repeat_drift,
            surrogate_repeat_drift,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ScaleOperator(f32);

    impl LearnedOperator for ScaleOperator {
        fn input_len(&self) -> usize {
            4
        }
        fn output_len(&self) -> usize {
            4
        }
        fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>> {
            Ok(input.iter().map(|value| value * self.0).collect())
        }
    }

    #[test]
    fn benchmark_reports_zero_error_for_matching_paths() {
        let mut operator = ScaleOperator(2.0);
        let mut exact_calls = 0usize;
        let report = benchmark_exact_vs_surrogate(
            &mut operator,
            &[1.0, 2.0, 3.0, 4.0],
            BenchmarkConfig {
                warmup_calls: 2,
                repeats: 4,
            },
            |input| {
                exact_calls += 1;
                Ok(input.iter().map(|value| value * 2.0).collect())
            },
        )
        .unwrap();
        assert_eq!(exact_calls, 6);
        assert_eq!(report.exact_timing.calls, 4);
        assert_eq!(report.surrogate_timing.calls, 4);
        assert_eq!(report.accuracy.mean_relative_l2, 0.0);
        assert_eq!(report.accuracy.max_abs, 0.0);
        assert_eq!(report.accuracy.exact_repeat_drift, 0.0);
        assert_eq!(report.accuracy.surrogate_repeat_drift, 0.0);
        assert!(report.exact_timing.total_seconds >= 0.0);
        assert!(report.surrogate_timing.total_seconds >= 0.0);
        let economics = report.economics(1.0).unwrap();
        assert_eq!(economics.training_cost, 1.0);
    }

    #[test]
    fn benchmark_retains_surrogate_error_and_exact_drift() {
        let mut operator = ScaleOperator(1.0);
        let mut call = 0usize;
        let report = benchmark_exact_vs_surrogate(
            &mut operator,
            &[1.0, 2.0, 3.0, 4.0],
            BenchmarkConfig {
                warmup_calls: 0,
                repeats: 3,
            },
            |input| {
                call += 1;
                let drift = call as f32 * 0.01;
                Ok(input.iter().map(|value| value * 2.0 + drift).collect())
            },
        )
        .unwrap();
        assert!(report.accuracy.mean_relative_l2 > 0.0);
        assert!(report.accuracy.max_relative_l2 >= report.accuracy.mean_relative_l2);
        assert!(report.accuracy.exact_repeat_drift > 0.0);
        assert_eq!(report.accuracy.surrogate_repeat_drift, 0.0);
    }

    #[test]
    fn benchmark_rejects_zero_repeats_and_bad_exact_shape() {
        let mut operator = ScaleOperator(1.0);
        assert_eq!(
            benchmark_exact_vs_surrogate(
                &mut operator,
                &[1.0; 4],
                BenchmarkConfig {
                    warmup_calls: 0,
                    repeats: 0
                },
                |input| Ok(input.to_vec()),
            )
            .unwrap_err(),
            NeuralOperatorError::InvalidBenchmarkRepeats { repeats: 0 }
        );
        assert!(matches!(
            benchmark_exact_vs_surrogate(
                &mut operator,
                &[1.0; 4],
                BenchmarkConfig {
                    warmup_calls: 0,
                    repeats: 1
                },
                |_| Ok(vec![1.0]),
            ),
            Err(NeuralOperatorError::ShapeMismatch {
                what: "benchmark exact output",
                ..
            })
        ));
    }
}
