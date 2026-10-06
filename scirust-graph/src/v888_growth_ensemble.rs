//! Multi-seed matched-control ensembles with per-arm dispersion for VG-3D.
//!
//! VG-3B generates one matched-control bundle per seed, and VG-3C / VG-3D
//! compare a reference against that single draw. A single draw cannot tell a
//! reader how much a descriptor moves from one control realisation to the
//! next. This module regenerates the four-arm ladder for a caller-declared,
//! duplicate-free list of seeds and summarises every scalar descriptor per arm:
//!
//! - [`v888_growth_control_ensemble`] — runs the VG-3B ladder once per seed,
//!   evaluates the VG-3C reachability / closeness / betweenness descriptors and
//!   the VG-3D inferred-modularity / annotation descriptors on every arm, and
//!   returns the raw per-seed samples plus one [`V888GrowthDispersion`] per
//!   `(arm, descriptor)`;
//! - [`v888_growth_ensemble_seeds`] — the deterministic consecutive seed list
//!   used by the VG-3D pilot exporter's `--ensemble` option;
//! - [`v888_growth_dispersion`] — the deterministic summary itself (minimum,
//!   maximum, mean, sample standard deviation, median, and nano-rounded counts
//!   of samples below / equal to / above the reference value).
//!
//! The ensemble also reports, per arm, how many distinct control graphs the
//! seeds actually produced, so a stuck rewiring chain is visible instead of
//! masquerading as zero dispersion.
//!
//! The summaries are descriptive. The below / equal / above counts are not
//! p-values: the VG-3B rewiring chains carry no uniform-null or mixing-time
//! guarantee, and nothing here establishes topology advantage, biological
//! modules or developmental causality. Dispersion is a precondition for any
//! later qualification, not a qualification by itself.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::v888_growth_controls::{
    V888GrowthControlArm, V888GrowthControlError, V888GrowthUnitEdges, v888_growth_matched_controls,
};
use crate::v888_growth_metrics::{
    V888GrowthMetricBundle, V888GrowthMetricsError, v888_growth_f64_to_nano,
    v888_growth_metric_bundle,
};
use crate::v888_growth_modularity::{
    V888GrowthLouvainOptions, V888GrowthModularityError, v888_growth_directed_modularity,
    v888_growth_infer_communities, v888_growth_label_mixing, v888_growth_partition_agreement,
};

/// Upper bound on the number of seeds accepted by one ensemble call.
pub const V888_GROWTH_ENSEMBLE_MAX_SEEDS: usize = 1024;

/// Scalar descriptor tracked across a control ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum V888GrowthEnsembleDescriptor {
    /// Fraction of reference edges absent from the control arm (zero for the reference).
    ReplacedEdgeFraction,
    /// Ordered pairs `(u, v)`, `u != v`, with `v` reachable from `u`.
    ReachableOrderedPairs,
    /// Median out-reach over nodes.
    MedianOutReach,
    /// Mean outgoing harmonic closeness.
    MeanOutHarmonic,
    /// Mean incoming harmonic closeness.
    MeanInHarmonic,
    /// Mean normalised directed betweenness.
    MeanBetweenness,
    /// Maximum normalised directed betweenness.
    MaxBetweenness,
    /// Number of communities inferred by deterministic directed Louvain.
    InferredCommunityCount,
    /// Directed modularity of the inferred partition.
    InferredModularity,
    /// Adjusted Rand index of the inferred partition against the reference's inferred partition.
    InferredVsReferenceAri,
    /// Directed modularity of the caller annotation (requires an annotation).
    AnnotationModularity,
    /// Nominal assortativity of the caller annotation (requires an annotation).
    AnnotationAssortativity,
    /// Adjusted Rand index of the inferred partition against the annotation.
    InferredVsAnnotationAri,
}

impl V888GrowthEnsembleDescriptor {
    /// Every descriptor in stable report order.
    pub const ALL: [Self; 13] = [
        Self::ReplacedEdgeFraction,
        Self::ReachableOrderedPairs,
        Self::MedianOutReach,
        Self::MeanOutHarmonic,
        Self::MeanInHarmonic,
        Self::MeanBetweenness,
        Self::MaxBetweenness,
        Self::InferredCommunityCount,
        Self::InferredModularity,
        Self::InferredVsReferenceAri,
        Self::AnnotationModularity,
        Self::AnnotationAssortativity,
        Self::InferredVsAnnotationAri,
    ];

    /// Stable machine name used in exporters and Thor summaries.
    pub const fn name(self) -> &'static str {
        match self
        {
            Self::ReplacedEdgeFraction => "replaced_edge_fraction",
            Self::ReachableOrderedPairs => "reachable_ordered_pairs",
            Self::MedianOutReach => "median_out_reach",
            Self::MeanOutHarmonic => "mean_out_harmonic",
            Self::MeanInHarmonic => "mean_in_harmonic",
            Self::MeanBetweenness => "mean_betweenness",
            Self::MaxBetweenness => "max_betweenness",
            Self::InferredCommunityCount => "inferred_community_count",
            Self::InferredModularity => "inferred_modularity",
            Self::InferredVsReferenceAri => "inferred_vs_reference_ari",
            Self::AnnotationModularity => "annotation_modularity",
            Self::AnnotationAssortativity => "annotation_assortativity",
            Self::InferredVsAnnotationAri => "inferred_vs_annotation_ari",
        }
    }

    /// Whether the descriptor is only defined when an annotation is supplied.
    pub const fn requires_annotation(self) -> bool {
        matches!(
            self,
            Self::AnnotationModularity
                | Self::AnnotationAssortativity
                | Self::InferredVsAnnotationAri
        )
    }
}

/// Descriptor values of one arm (`None` when undefined, e.g. a zero denominator).
pub type V888GrowthEnsembleValues = BTreeMap<V888GrowthEnsembleDescriptor, Option<f64>>;

/// Descriptor values of one control arm generated from one ensemble seed.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthEnsembleSample {
    /// Ensemble seed passed to the VG-3B ladder.
    pub seed: u64,
    /// Control arm.
    pub arm: V888GrowthControlArm,
    /// Accepted double-edge switches for this draw (zero for the edge-count arm).
    pub accepted_swaps: usize,
    /// Descriptor values for this draw.
    pub values: V888GrowthEnsembleValues,
}

/// Deterministic dispersion summary of one descriptor across ensemble draws.
///
/// `below_reference`, `equal_reference` and `above_reference` compare samples
/// to the reference after nano-rounding (see [`v888_growth_f64_to_nano`]) so
/// float noise below `1e-9` never flips a count. They are descriptive counts,
/// not a significance test.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthDispersion {
    /// Reference value, if defined.
    pub reference: Option<f64>,
    /// Number of draws with a defined value.
    pub defined: usize,
    /// Number of draws whose value was undefined.
    pub undefined: usize,
    /// Minimum defined value.
    pub min: Option<f64>,
    /// Maximum defined value.
    pub max: Option<f64>,
    /// Arithmetic mean of defined values.
    pub mean: Option<f64>,
    /// Sample standard deviation (`n - 1` denominator); `None` below two values.
    pub std_dev: Option<f64>,
    /// Median of defined values (mean of the two middle values when even).
    pub median: Option<f64>,
    /// Defined draws strictly below the reference (zero without a reference).
    pub below_reference: usize,
    /// Defined draws equal to the reference at nano resolution.
    pub equal_reference: usize,
    /// Defined draws strictly above the reference.
    pub above_reference: usize,
}

impl V888GrowthDispersion {
    /// Nano-rounded mean, for integer-stable exporters.
    #[must_use]
    pub fn mean_nano(&self) -> Option<i64> {
        self.mean.map(v888_growth_f64_to_nano)
    }

    /// Nano-rounded sample standard deviation, for integer-stable exporters.
    #[must_use]
    pub fn std_dev_nano(&self) -> Option<i64> {
        self.std_dev.map(v888_growth_f64_to_nano)
    }
}

/// Per-arm ensemble summary.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthEnsembleArmSummary {
    /// Control arm.
    pub arm: V888GrowthControlArm,
    /// Number of seeds evaluated for this arm.
    pub draws: usize,
    /// Number of distinct edge sets produced across seeds.
    pub distinct_graphs: usize,
    /// Dispersion per descriptor, in [`V888GrowthEnsembleDescriptor::ALL`] order.
    pub dispersion: BTreeMap<V888GrowthEnsembleDescriptor, V888GrowthDispersion>,
}

/// Complete multi-seed ensemble report.
#[derive(Debug, Clone, PartialEq)]
pub struct V888GrowthEnsembleReport {
    /// Dense node count.
    pub node_count: usize,
    /// Reference unit-edge count.
    pub reference_edges: usize,
    /// Seeds in caller order.
    pub seeds: Vec<u64>,
    /// Whether an annotation was supplied.
    pub annotated: bool,
    /// Whether the block constraint was vacuous (single block label).
    pub block_constraint_vacuous: bool,
    /// Reference descriptor values.
    pub reference: V888GrowthEnsembleValues,
    /// Raw samples in seed-major, ladder-minor order.
    pub samples: Vec<V888GrowthEnsembleSample>,
    /// Per-arm summaries in ladder order.
    pub arms: Vec<V888GrowthEnsembleArmSummary>,
}

/// Fail-closed errors from ensemble generation.
#[derive(Debug)]
pub enum V888GrowthEnsembleError {
    /// Fewer than two seeds were supplied.
    TooFewSeeds {
        /// Supplied seed count.
        observed: usize,
    },
    /// More than [`V888_GROWTH_ENSEMBLE_MAX_SEEDS`] seeds were supplied.
    TooManySeeds {
        /// Supplied seed count.
        observed: usize,
    },
    /// A seed appeared more than once.
    DuplicateSeed {
        /// Repeated seed.
        seed: u64,
    },
    /// VG-3B control generation failed.
    Control(V888GrowthControlError),
    /// VG-3C metric computation failed.
    Metrics(V888GrowthMetricsError),
    /// VG-3D modularity computation failed.
    Modularity(V888GrowthModularityError),
}

impl fmt::Display for V888GrowthEnsembleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooFewSeeds { observed } => write!(
                formatter,
                "control ensemble needs at least two seeds, got {observed}"
            ),
            Self::TooManySeeds { observed } => write!(
                formatter,
                "control ensemble accepts at most {V888_GROWTH_ENSEMBLE_MAX_SEEDS} seeds, got \
                 {observed}"
            ),
            Self::DuplicateSeed { seed } =>
            {
                write!(formatter, "control ensemble seed {seed} is repeated")
            },
            Self::Control(error) => write!(formatter, "{error}"),
            Self::Metrics(error) => write!(formatter, "{error}"),
            Self::Modularity(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for V888GrowthEnsembleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Control(error) => Some(error),
            Self::Metrics(error) => Some(error),
            Self::Modularity(error) => Some(error),
            _ => None,
        }
    }
}

impl From<V888GrowthControlError> for V888GrowthEnsembleError {
    fn from(value: V888GrowthControlError) -> Self {
        Self::Control(value)
    }
}

impl From<V888GrowthMetricsError> for V888GrowthEnsembleError {
    fn from(value: V888GrowthMetricsError) -> Self {
        Self::Metrics(value)
    }
}

impl From<V888GrowthModularityError> for V888GrowthEnsembleError {
    fn from(value: V888GrowthModularityError) -> Self {
        Self::Modularity(value)
    }
}

/// Deterministic dispersion of `values` around an optional reference value.
///
/// `None` entries are counted as undefined and excluded from every statistic.
/// The mean is accumulated in input order and the variance uses a second pass
/// around that mean, so results are bit-for-bit reproducible for a fixed input.
///
/// ```
/// use scirust_graph::v888_growth::v888_growth_dispersion;
///
/// let summary = v888_growth_dispersion(Some(2.0), &[Some(1.0), Some(3.0), None, Some(2.0)]);
/// assert_eq!(summary.defined, 3);
/// assert_eq!(summary.undefined, 1);
/// assert_eq!(summary.mean, Some(2.0));
/// assert_eq!(summary.median, Some(2.0));
/// assert_eq!(summary.std_dev, Some(1.0));
/// assert_eq!(
///     (summary.below_reference, summary.equal_reference, summary.above_reference),
///     (1, 1, 1)
/// );
/// ```
#[must_use]
pub fn v888_growth_dispersion(
    reference: Option<f64>,
    values: &[Option<f64>],
) -> V888GrowthDispersion {
    let defined: Vec<f64> = values.iter().filter_map(|value| *value).collect();
    let undefined = values.len() - defined.len();
    let count = defined.len();
    let mut summary = V888GrowthDispersion {
        reference,
        defined: count,
        undefined,
        min: None,
        max: None,
        mean: None,
        std_dev: None,
        median: None,
        below_reference: 0,
        equal_reference: 0,
        above_reference: 0,
    };
    if count == 0
    {
        return summary;
    }

    let mut sorted = defined.clone();
    sorted.sort_by(f64::total_cmp);
    summary.min = sorted.first().copied();
    summary.max = sorted.last().copied();
    summary.median = Some(
        if count % 2 == 1
        {
            sorted[count / 2]
        }
        else
        {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        },
    );

    let mean = defined.iter().sum::<f64>() / count as f64;
    summary.mean = Some(mean);
    if count >= 2
    {
        let squares: f64 = defined
            .iter()
            .map(|value| {
                let centred = value - mean;
                centred * centred
            })
            .sum();
        summary.std_dev = Some((squares / (count - 1) as f64).sqrt());
    }

    if let Some(reference_value) = reference
    {
        let reference_nano = v888_growth_f64_to_nano(reference_value);
        for &value in &defined
        {
            match v888_growth_f64_to_nano(value).cmp(&reference_nano)
            {
                core::cmp::Ordering::Less => summary.below_reference += 1,
                core::cmp::Ordering::Equal => summary.equal_reference += 1,
                core::cmp::Ordering::Greater => summary.above_reference += 1,
            }
        }
    }
    summary
}

/// Deterministic ensemble seed list `base, base + 1, …, base + count - 1`.
///
/// Addition wraps at `u64::MAX`, so every seed is distinct for any accepted
/// `count`. The first seed equals `base`, which lets an exporter's single-draw
/// rows (generated from `base`) coincide with the ensemble's first sample.
///
/// `# Errors`
///
/// Returns [`V888GrowthEnsembleError::TooFewSeeds`] below two seeds and
/// [`V888GrowthEnsembleError::TooManySeeds`] above
/// [`V888_GROWTH_ENSEMBLE_MAX_SEEDS`], the same bounds as
/// [`v888_growth_control_ensemble`].
///
/// ```
/// use scirust_graph::v888_growth::v888_growth_ensemble_seeds;
///
/// assert_eq!(v888_growth_ensemble_seeds(29, 3)?, vec![29, 30, 31]);
/// assert_eq!(v888_growth_ensemble_seeds(u64::MAX, 2)?, vec![u64::MAX, 0]);
/// assert!(v888_growth_ensemble_seeds(29, 1).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn v888_growth_ensemble_seeds(
    base: u64,
    count: usize,
) -> Result<Vec<u64>, V888GrowthEnsembleError> {
    if count < 2
    {
        return Err(V888GrowthEnsembleError::TooFewSeeds { observed: count });
    }
    if count > V888_GROWTH_ENSEMBLE_MAX_SEEDS
    {
        return Err(V888GrowthEnsembleError::TooManySeeds { observed: count });
    }
    Ok((0..count as u64)
        .map(|offset| base.wrapping_add(offset))
        .collect())
}

/// Run the VG-3B control ladder for every seed and summarise per-arm dispersion.
///
/// `seeds` must hold between two and [`V888_GROWTH_ENSEMBLE_MAX_SEEDS`]
/// distinct values; each one is passed unchanged to
/// [`v888_growth_matched_controls`]. `annotation`, when supplied, is one label
/// per node and is evaluated unchanged on the reference and every arm.
///
/// `# Errors`
///
/// Returns an error for a bad seed list, wrong-length groups or annotation,
/// invalid edges, or a control arm that fails its declared invariants.
///
/// `# Examples`
///
/// ~~~rust
/// use scirust_graph::v888_growth::{
///     V888GrowthControlArm, V888GrowthEnsembleDescriptor, V888GrowthLouvainOptions,
///     v888_growth_control_ensemble, v888_growth_edge_count_control,
/// };
///
/// let reference = v888_growth_edge_count_control(16, 48, 3)?;
/// let groups = vec![0_u32, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
/// let report = v888_growth_control_ensemble(
///     &reference,
///     16,
///     &groups,
///     &[11, 12, 13, 14],
///     None,
///     V888GrowthLouvainOptions::default(),
/// )?;
/// assert_eq!(report.samples.len(), 4 * 4);
/// let degree = &report.arms[1];
/// assert_eq!(degree.arm, V888GrowthControlArm::Degree);
/// let reach = &degree.dispersion[&V888GrowthEnsembleDescriptor::ReachableOrderedPairs];
/// assert_eq!(reach.defined, 4);
/// assert!(reach.min <= reach.max);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ~~~
pub fn v888_growth_control_ensemble(
    reference: &V888GrowthUnitEdges,
    node_count: usize,
    groups: &[u32],
    seeds: &[u64],
    annotation: Option<&[usize]>,
    options: V888GrowthLouvainOptions,
) -> Result<V888GrowthEnsembleReport, V888GrowthEnsembleError> {
    if seeds.len() < 2
    {
        return Err(V888GrowthEnsembleError::TooFewSeeds {
            observed: seeds.len(),
        });
    }
    if seeds.len() > V888_GROWTH_ENSEMBLE_MAX_SEEDS
    {
        return Err(V888GrowthEnsembleError::TooManySeeds {
            observed: seeds.len(),
        });
    }
    let mut seen = BTreeSet::new();
    for &seed in seeds
    {
        if !seen.insert(seed)
        {
            return Err(V888GrowthEnsembleError::DuplicateSeed { seed });
        }
    }

    let reference_metrics = v888_growth_metric_bundle("reference", node_count, reference)?;
    let reference_partition = v888_growth_infer_communities(node_count, reference, options)?;
    let reference_values = arm_values(
        node_count,
        reference,
        0.0,
        &reference_metrics,
        annotation,
        &reference_partition.labels,
        options,
    )?;

    let mut samples = Vec::new();
    // Ladder order with the distinct edge sets observed for each arm.
    let mut ladder: Vec<(V888GrowthControlArm, BTreeSet<V888GrowthUnitEdges>)> = Vec::new();
    let mut block_constraint_vacuous = false;
    for &seed in seeds
    {
        let bundle = v888_growth_matched_controls(reference, node_count, groups, seed)?;
        block_constraint_vacuous = bundle.block_constraint_vacuous;
        for arm in &bundle.arms
        {
            let slot = match ladder.iter().position(|(known, _)| *known == arm.arm)
            {
                Some(slot) => slot,
                None =>
                {
                    ladder.push((arm.arm, BTreeSet::new()));
                    ladder.len() - 1
                },
            };
            let replaced = if reference.is_empty()
            {
                0.0
            }
            else
            {
                arm.replaced_edges as f64 / reference.len() as f64
            };
            let metrics = v888_growth_metric_bundle(arm.arm.name(), node_count, &arm.edges)?;
            let values = arm_values(
                node_count,
                &arm.edges,
                replaced,
                &metrics,
                annotation,
                &reference_partition.labels,
                options,
            )?;
            ladder[slot].1.insert(arm.edges.clone());
            samples.push(V888GrowthEnsembleSample {
                seed,
                arm: arm.arm,
                accepted_swaps: arm.accepted_swaps,
                values,
            });
        }
    }

    let arms = ladder
        .into_iter()
        .map(|(arm, distinct)| {
            let arm_samples: Vec<&V888GrowthEnsembleSample> =
                samples.iter().filter(|sample| sample.arm == arm).collect();
            let dispersion = V888GrowthEnsembleDescriptor::ALL
                .into_iter()
                .map(|descriptor| {
                    let values: Vec<Option<f64>> = arm_samples
                        .iter()
                        .map(|sample| sample.values.get(&descriptor).copied().flatten())
                        .collect();
                    let reference_value = reference_values.get(&descriptor).copied().flatten();
                    (descriptor, v888_growth_dispersion(reference_value, &values))
                })
                .collect();
            V888GrowthEnsembleArmSummary {
                arm,
                draws: arm_samples.len(),
                distinct_graphs: distinct.len(),
                dispersion,
            }
        })
        .collect();

    Ok(V888GrowthEnsembleReport {
        node_count,
        reference_edges: reference.len(),
        seeds: seeds.to_vec(),
        annotated: annotation.is_some(),
        block_constraint_vacuous,
        reference: reference_values,
        samples,
        arms,
    })
}

fn arm_values(
    node_count: usize,
    edges: &V888GrowthUnitEdges,
    replaced_edge_fraction: f64,
    metrics: &V888GrowthMetricBundle,
    annotation: Option<&[usize]>,
    reference_labels: &[usize],
    options: V888GrowthLouvainOptions,
) -> Result<V888GrowthEnsembleValues, V888GrowthEnsembleError> {
    use V888GrowthEnsembleDescriptor as D;

    let inferred = v888_growth_infer_communities(node_count, edges, options)?;
    let vs_reference = v888_growth_partition_agreement(&inferred.labels, reference_labels)?;
    let mut values = V888GrowthEnsembleValues::new();
    values.insert(D::ReplacedEdgeFraction, Some(replaced_edge_fraction));
    values.insert(
        D::ReachableOrderedPairs,
        Some(metrics.reachability.reachable_ordered_pairs as f64),
    );
    values.insert(
        D::MedianOutReach,
        Some(metrics.reachability.median_out_reach as f64),
    );
    values.insert(D::MeanOutHarmonic, mean(&metrics.harmonic.out_harmonic));
    values.insert(D::MeanInHarmonic, mean(&metrics.harmonic.in_harmonic));
    values.insert(D::MeanBetweenness, mean(&metrics.betweenness.normalized));
    values.insert(
        D::MaxBetweenness,
        metrics
            .betweenness
            .normalized
            .iter()
            .copied()
            .max_by(f64::total_cmp),
    );
    values.insert(
        D::InferredCommunityCount,
        Some(inferred.community_sizes.len() as f64),
    );
    values.insert(D::InferredModularity, inferred.modularity.q);
    values.insert(D::InferredVsReferenceAri, vs_reference.adjusted_rand);

    let (annotation_q, annotation_r, vs_annotation) = match annotation
    {
        Some(labels) => (
            v888_growth_directed_modularity(node_count, edges, labels)?.q,
            v888_growth_label_mixing(node_count, edges, labels)?.assortativity,
            v888_growth_partition_agreement(&inferred.labels, labels)?.adjusted_rand,
        ),
        None => (None, None, None),
    };
    values.insert(D::AnnotationModularity, annotation_q);
    values.insert(D::AnnotationAssortativity, annotation_r);
    values.insert(D::InferredVsAnnotationAri, vs_annotation);
    Ok(values)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty()
    {
        None
    }
    else
    {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v888_growth_controls::v888_growth_edge_count_control;

    fn two_module_reference() -> (V888GrowthUnitEdges, Vec<u32>, Vec<usize>) {
        // Two dense 6-node directed modules joined by two bridges.
        let mut edges = V888GrowthUnitEdges::new();
        for module in 0..2_usize
        {
            let base = module * 6;
            for i in 0..6
            {
                for step in 1..=2
                {
                    edges.insert((base + i, base + (i + step) % 6));
                }
            }
        }
        edges.insert((2, 8));
        edges.insert((9, 3));
        let groups = vec![0_u32, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let annotation = vec![0_usize, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        (edges, groups, annotation)
    }

    #[test]
    fn dispersion_handles_empty_single_and_even_inputs() {
        let empty = v888_growth_dispersion(Some(1.0), &[None, None]);
        assert_eq!(empty.defined, 0);
        assert_eq!(empty.undefined, 2);
        assert_eq!(empty.mean, None);
        assert_eq!(empty.median, None);
        assert_eq!(
            empty.below_reference + empty.equal_reference + empty.above_reference,
            0
        );

        let single = v888_growth_dispersion(None, &[Some(4.0)]);
        assert_eq!(single.mean, Some(4.0));
        assert_eq!(single.std_dev, None);
        assert_eq!(single.median, Some(4.0));
        assert_eq!(
            single.below_reference + single.equal_reference + single.above_reference,
            0
        );

        let even = v888_growth_dispersion(Some(0.0), &[Some(4.0), Some(1.0), Some(3.0), Some(2.0)]);
        assert_eq!(even.min, Some(1.0));
        assert_eq!(even.max, Some(4.0));
        assert_eq!(even.median, Some(2.5));
        assert_eq!(even.mean, Some(2.5));
        let expected_sd = (5.0_f64 / 3.0).sqrt();
        assert!((even.std_dev.unwrap() - expected_sd).abs() < 1e-15);
        assert_eq!(even.above_reference, 4);
        assert_eq!(even.mean_nano(), Some(2_500_000_000));
    }

    #[test]
    fn dispersion_reference_comparison_ignores_sub_nano_noise() {
        let summary = v888_growth_dispersion(Some(0.1 + 0.2), &[Some(0.3), Some(0.3 + 1e-12)]);
        assert_eq!(summary.equal_reference, 2);
        assert_eq!(summary.below_reference + summary.above_reference, 0);
    }

    #[test]
    fn ensemble_seeds_are_consecutive_distinct_and_bounded() {
        assert_eq!(v888_growth_ensemble_seeds(7, 2).unwrap(), vec![7, 8]);
        let wrapped = v888_growth_ensemble_seeds(u64::MAX - 1, 4).unwrap();
        assert_eq!(wrapped, vec![u64::MAX - 1, u64::MAX, 0, 1]);
        let full = v888_growth_ensemble_seeds(u64::MAX, V888_GROWTH_ENSEMBLE_MAX_SEEDS).unwrap();
        assert_eq!(full.len(), V888_GROWTH_ENSEMBLE_MAX_SEEDS);
        assert_eq!(full.iter().collect::<BTreeSet<_>>().len(), full.len());
        assert!(matches!(
            v888_growth_ensemble_seeds(0, 0),
            Err(V888GrowthEnsembleError::TooFewSeeds { observed: 0 })
        ));
        assert!(matches!(
            v888_growth_ensemble_seeds(0, V888_GROWTH_ENSEMBLE_MAX_SEEDS + 1),
            Err(V888GrowthEnsembleError::TooManySeeds { .. })
        ));
    }

    #[test]
    fn ensemble_rejects_bad_seed_lists() {
        let (edges, groups, _) = two_module_reference();
        let options = V888GrowthLouvainOptions::default();
        assert!(matches!(
            v888_growth_control_ensemble(&edges, 12, &groups, &[1], None, options),
            Err(V888GrowthEnsembleError::TooFewSeeds { observed: 1 })
        ));
        assert!(matches!(
            v888_growth_control_ensemble(&edges, 12, &groups, &[5, 6, 5], None, options),
            Err(V888GrowthEnsembleError::DuplicateSeed { seed: 5 })
        ));
        let many: Vec<u64> = (0..=V888_GROWTH_ENSEMBLE_MAX_SEEDS as u64).collect();
        assert!(matches!(
            v888_growth_control_ensemble(&edges, 12, &groups, &many, None, options),
            Err(V888GrowthEnsembleError::TooManySeeds { .. })
        ));
        assert!(matches!(
            v888_growth_control_ensemble(&edges, 12, &groups[..11], &[1, 2], None, options),
            Err(V888GrowthEnsembleError::Control(_))
        ));
        let short_annotation = vec![0_usize; 11];
        assert!(matches!(
            v888_growth_control_ensemble(
                &edges,
                12,
                &groups,
                &[1, 2],
                Some(&short_annotation),
                options
            ),
            Err(V888GrowthEnsembleError::Modularity(_))
        ));
    }

    #[test]
    fn ensemble_is_deterministic_and_matches_single_bundles() {
        let (edges, groups, annotation) = two_module_reference();
        let seeds = [3_u64, 17, 29, 41, 53];
        let options = V888GrowthLouvainOptions::default();
        let first =
            v888_growth_control_ensemble(&edges, 12, &groups, &seeds, Some(&annotation), options)
                .unwrap();
        let second =
            v888_growth_control_ensemble(&edges, 12, &groups, &seeds, Some(&annotation), options)
                .unwrap();
        assert_eq!(first, second);
        assert!(first.annotated);
        assert_eq!(first.seeds, seeds);
        assert_eq!(first.samples.len(), seeds.len() * 4);
        assert_eq!(first.arms.len(), 4);

        // Each sample replays the single-seed VG-3B bundle exactly.
        for (index, &seed) in seeds.iter().enumerate()
        {
            let bundle = v888_growth_matched_controls(&edges, 12, &groups, seed).unwrap();
            for (offset, arm) in bundle.arms.iter().enumerate()
            {
                let sample = &first.samples[index * 4 + offset];
                assert_eq!(sample.seed, seed);
                assert_eq!(sample.arm, arm.arm);
                assert_eq!(sample.accepted_swaps, arm.accepted_swaps);
                let expected = arm.replaced_edges as f64 / edges.len() as f64;
                assert_eq!(
                    sample.values[&V888GrowthEnsembleDescriptor::ReplacedEdgeFraction],
                    Some(expected)
                );
            }
        }

        // The reference is its own inferred partition, so ARI is exactly one.
        assert_eq!(
            first.reference[&V888GrowthEnsembleDescriptor::InferredVsReferenceAri],
            Some(1.0)
        );
        assert_eq!(
            first.reference[&V888GrowthEnsembleDescriptor::ReplacedEdgeFraction],
            Some(0.0)
        );
        for summary in &first.arms
        {
            assert_eq!(summary.draws, seeds.len());
            assert!(summary.distinct_graphs >= 1 && summary.distinct_graphs <= seeds.len());
            assert_eq!(
                summary.dispersion.len(),
                V888GrowthEnsembleDescriptor::ALL.len()
            );
            for (descriptor, dispersion) in &summary.dispersion
            {
                assert_eq!(dispersion.defined + dispersion.undefined, seeds.len());
                if let (Some(min), Some(max), Some(mean)) =
                    (dispersion.min, dispersion.max, dispersion.mean)
                {
                    assert!(min <= mean + 1e-12 && mean <= max + 1e-12, "{descriptor:?}");
                }
                if dispersion.reference.is_some()
                {
                    assert_eq!(
                        dispersion.below_reference
                            + dispersion.equal_reference
                            + dispersion.above_reference,
                        dispersion.defined
                    );
                }
            }
        }
    }

    #[test]
    fn block_arm_preserves_annotation_mixing_and_edge_count_arm_varies() {
        let (edges, groups, annotation) = two_module_reference();
        let seeds = [101_u64, 202, 303, 404];
        let report = v888_growth_control_ensemble(
            &edges,
            12,
            &groups,
            &seeds,
            Some(&annotation),
            V888GrowthLouvainOptions::default(),
        )
        .unwrap();

        // Block-matched rewiring keeps the block-pair matrix, and the annotation
        // equals the blocks here, so annotation modularity and assortativity are
        // invariant across every draw.
        let block = report
            .arms
            .iter()
            .find(|arm| arm.arm == V888GrowthControlArm::DegreeReciprocalBlock)
            .unwrap();
        for descriptor in [
            V888GrowthEnsembleDescriptor::AnnotationModularity,
            V888GrowthEnsembleDescriptor::AnnotationAssortativity,
        ]
        {
            let dispersion = &block.dispersion[&descriptor];
            assert_eq!(dispersion.equal_reference, seeds.len(), "{descriptor:?}");
            assert!(dispersion.std_dev.unwrap() < 1e-12);
        }

        // The unconstrained edge-count arm produces distinct graphs per seed.
        let edge_count = &report.arms[0];
        assert_eq!(edge_count.arm, V888GrowthControlArm::EdgeCount);
        assert_eq!(edge_count.distinct_graphs, seeds.len());
    }

    #[test]
    fn unannotated_ensemble_leaves_annotation_descriptors_undefined() {
        let reference = v888_growth_edge_count_control(10, 24, 9).unwrap();
        let groups = vec![0_u32; 10];
        let report = v888_growth_control_ensemble(
            &reference,
            10,
            &groups,
            &[1, 2, 3],
            None,
            V888GrowthLouvainOptions::default(),
        )
        .unwrap();
        assert!(!report.annotated);
        assert!(report.block_constraint_vacuous);
        for summary in &report.arms
        {
            for descriptor in V888GrowthEnsembleDescriptor::ALL
            {
                let dispersion = &summary.dispersion[&descriptor];
                if descriptor.requires_annotation()
                {
                    assert_eq!(dispersion.defined, 0);
                    assert_eq!(dispersion.undefined, 3);
                }
                else
                {
                    assert_eq!(dispersion.undefined, 0, "{descriptor:?}");
                }
            }
        }
    }
}
