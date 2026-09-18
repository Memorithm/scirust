//! Typed foundations for SciRust optimization runtimes.
//!
//! This crate deliberately contains no sampler algorithm and no external
//! dependency. It defines the stable substrate that TPE, Gaussian-process,
//! evolutionary, pruning and distributed layers can share:
//!
//! - dense [`ParamId`] identifiers compiled from an API-facing search space;
//! - typed distributions and acyclic activation conditions;
//! - a row-oriented [`Candidate`] only at the proposal boundary;
//! - column-oriented parameter and objective storage inside [`TrialStore`];
//! - monotone trial reservation and explicit state transitions;
//! - idempotent [`Study::tell`] semantics suitable for retries.
//!
//! Strings are retained for diagnostics and public API ergonomics, but the
//! hot-path storage is indexed by integer identifiers.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

/// Dense parameter identifier used by the compiled search-space IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamId(u32);

impl ParamId {
    /// Construct an identifier from its dense zero-based index.
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Return the zero-based index represented by this identifier.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Value assigned to one optimization parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamValue {
    /// Floating-point value for continuous or log-uniform parameters.
    Float(f64),
    /// Signed integer value.
    Int(i64),
    /// Zero-based categorical choice index.
    Categorical(u32),
}

/// Supported parameter distributions in the core search-space IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    /// Continuous uniform interval, inclusive for validation.
    Uniform {
        /// Lower finite bound.
        low: f64,
        /// Upper finite bound.
        high: f64,
    },
    /// Positive continuous interval sampled in logarithmic space.
    LogUniform {
        /// Strictly positive lower finite bound.
        low: f64,
        /// Upper finite bound.
        high: f64,
    },
    /// Inclusive signed integer interval.
    IntRange {
        /// Lower inclusive bound.
        low: i64,
        /// Upper inclusive bound.
        high: i64,
    },
    /// Categorical domain represented by dense choice indices.
    Categorical {
        /// Number of available choices; must be non-zero.
        cardinality: u32,
    },
}

impl Distribution {
    fn accepts(&self, value: ParamValue) -> bool {
        match (self, value)
        {
            (Self::Uniform { low, high }, ParamValue::Float(v)) =>
            {
                v.is_finite() && v >= *low && v <= *high
            },
            (Self::LogUniform { low, high }, ParamValue::Float(v)) =>
            {
                v.is_finite() && v > 0.0 && v >= *low && v <= *high
            },
            (Self::IntRange { low, high }, ParamValue::Int(v)) => v >= *low && v <= *high,
            (Self::Categorical { cardinality }, ParamValue::Categorical(v)) => v < *cardinality,
            _ => false,
        }
    }

    fn validate(&self) -> bool {
        match self
        {
            Self::Uniform { low, high } => low.is_finite() && high.is_finite() && low < high,
            Self::LogUniform { low, high } =>
            {
                low.is_finite() && high.is_finite() && *low > 0.0 && low < high
            },
            Self::IntRange { low, high } => low <= high,
            Self::Categorical { cardinality } => *cardinality != 0,
        }
    }
}

/// Activation condition for a parameter in a conditional search space.
///
/// Requiring parents to have a lower [`ParamId`] than their child makes the
/// compiled condition graph acyclic by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    /// Parameter is always active.
    Always,
    /// Parameter is active when a categorical parent equals one choice.
    CategoricalEquals {
        /// Parent parameter.
        parent: ParamId,
        /// Required categorical choice.
        choice: u32,
    },
    /// Parameter is active when an integer parent equals one value.
    IntEquals {
        /// Parent parameter.
        parent: ParamId,
        /// Required integer value.
        value: i64,
    },
}

/// One compiled search-space parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSpec {
    /// Dense identifier. Compiled spaces require `id.index() == position`.
    pub id: ParamId,
    /// Human-readable stable name used outside the hot path.
    pub name: String,
    /// Parameter distribution.
    pub distribution: Distribution,
    /// Activation condition.
    pub condition: Condition,
}

impl ParameterSpec {
    /// Construct a parameter specification.
    pub fn new(
        id: ParamId,
        name: impl Into<String>,
        distribution: Distribution,
        condition: Condition,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            distribution,
            condition,
        }
    }
}

/// Error returned while compiling a search space.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceError {
    /// Parameter identifiers were not dense and ordered.
    NonDenseId {
        /// Position in the provided specification list.
        position: usize,
        /// Identifier found at that position.
        found: ParamId,
    },
    /// Two parameters used the same public name.
    DuplicateName(String),
    /// Distribution bounds or cardinality were invalid.
    InvalidDistribution(ParamId),
    /// A condition referenced an unknown or non-earlier parent.
    InvalidConditionParent {
        /// Child parameter.
        child: ParamId,
        /// Referenced parent.
        parent: ParamId,
    },
    /// Condition kind did not match the parent's distribution.
    ConditionTypeMismatch {
        /// Child parameter.
        child: ParamId,
        /// Referenced parent.
        parent: ParamId,
    },
    /// Condition requested a categorical choice outside the parent's domain.
    ConditionChoiceOutOfRange {
        /// Child parameter.
        child: ParamId,
        /// Referenced parent.
        parent: ParamId,
    },
    /// Condition requested an integer value outside the parent's domain.
    ConditionValueOutOfRange {
        /// Child parameter.
        child: ParamId,
        /// Referenced parent.
        parent: ParamId,
    },
}

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonDenseId { position, found } =>
            {
                write!(
                    f,
                    "parameter at position {position} has non-dense id {found:?}"
                )
            },
            Self::DuplicateName(name) => write!(f, "duplicate parameter name `{name}`"),
            Self::InvalidDistribution(id) => write!(f, "invalid distribution for {id:?}"),
            Self::InvalidConditionParent { child, parent } =>
            {
                write!(f, "invalid condition parent {parent:?} for child {child:?}")
            },
            Self::ConditionTypeMismatch { child, parent } =>
            {
                write!(
                    f,
                    "condition type mismatch between {child:?} and {parent:?}"
                )
            },
            Self::ConditionChoiceOutOfRange { child, parent } =>
            {
                write!(
                    f,
                    "categorical condition for {child:?} is outside {parent:?}"
                )
            },
            Self::ConditionValueOutOfRange { child, parent } =>
            {
                write!(f, "integer condition for {child:?} is outside {parent:?}")
            },
        }
    }
}

impl std::error::Error for SpaceError {}

/// Validated, densely indexed optimization search space.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSpace {
    params: Vec<ParameterSpec>,
}

impl SearchSpace {
    /// Validate and compile a parameter list.
    pub fn compile(params: Vec<ParameterSpec>) -> Result<Self, SpaceError> {
        for (position, param) in params.iter().enumerate()
        {
            if param.id.index() != position
            {
                return Err(SpaceError::NonDenseId {
                    position,
                    found: param.id,
                });
            }
            if !param.distribution.validate()
            {
                return Err(SpaceError::InvalidDistribution(param.id));
            }
            if params[..position]
                .iter()
                .any(|previous| previous.name == param.name)
            {
                return Err(SpaceError::DuplicateName(param.name.clone()));
            }
            Self::validate_condition(&params, position, param)?;
        }
        Ok(Self { params })
    }

    fn validate_condition(
        params: &[ParameterSpec],
        position: usize,
        child: &ParameterSpec,
    ) -> Result<(), SpaceError> {
        let (parent, condition_kind) = match child.condition
        {
            Condition::Always => return Ok(()),
            Condition::CategoricalEquals { parent, choice } => (parent, Some((choice, None))),
            Condition::IntEquals { parent, value } => (parent, Some((0, Some(value)))),
        };

        if parent.index() >= position
        {
            return Err(SpaceError::InvalidConditionParent {
                child: child.id,
                parent,
            });
        }
        let Some(parent_spec) = params.get(parent.index())
        else
        {
            return Err(SpaceError::InvalidConditionParent {
                child: child.id,
                parent,
            });
        };

        match (child.condition, &parent_spec.distribution, condition_kind)
        {
            (
                Condition::CategoricalEquals { choice, .. },
                Distribution::Categorical { cardinality },
                _,
            ) if choice < *cardinality => Ok(()),
            (Condition::CategoricalEquals { .. }, Distribution::Categorical { .. }, _) =>
            {
                Err(SpaceError::ConditionChoiceOutOfRange {
                    child: child.id,
                    parent,
                })
            },
            (Condition::IntEquals { value, .. }, Distribution::IntRange { low, high }, _)
                if value >= *low && value <= *high =>
            {
                Ok(())
            },
            (Condition::IntEquals { .. }, Distribution::IntRange { .. }, _) =>
            {
                Err(SpaceError::ConditionValueOutOfRange {
                    child: child.id,
                    parent,
                })
            },
            _ => Err(SpaceError::ConditionTypeMismatch {
                child: child.id,
                parent,
            }),
        }
    }

    /// Number of parameters in the compiled space.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the space contains no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Return one specification by dense identifier.
    pub fn parameter(&self, id: ParamId) -> Option<&ParameterSpec> {
        self.params.get(id.index()).filter(|spec| spec.id == id)
    }

    /// Iterate over parameters in deterministic identifier order.
    pub fn parameters(&self) -> impl ExactSizeIterator<Item = &ParameterSpec> {
        self.params.iter()
    }

    /// Validate a completed sampler proposal against this search space.
    ///
    /// Every active parameter must be assigned, every inactive parameter must
    /// remain absent, and every assigned value must match its distribution.
    /// This final validation is intentionally separate from [`Candidate::set`]:
    /// a sampler may change a parent after assigning a conditional child, so
    /// only the finished proposal can certify global conditional consistency.
    pub fn validate_candidate(&self, candidate: &Candidate) -> Result<(), CandidateError> {
        if candidate.len() != self.len()
        {
            return Err(CandidateError::ShapeMismatch {
                expected: self.len(),
                found: candidate.len(),
            });
        }

        for spec in &self.params
        {
            let active = self.is_active(spec.id, candidate);
            let value = candidate.value(spec.id);
            match (active, value)
            {
                (true, None) => return Err(CandidateError::MissingActiveParameter(spec.id)),
                (false, Some(_)) => return Err(CandidateError::InactiveParameter(spec.id)),
                (_, Some(value)) if !spec.distribution.accepts(value) =>
                {
                    return Err(CandidateError::InvalidValue(spec.id));
                },
                _ => {},
            }
        }
        Ok(())
    }

    /// Test whether a parameter is active for a row-oriented candidate.
    pub fn is_active(&self, id: ParamId, candidate: &Candidate) -> bool {
        self.is_active_with(id, |parent| candidate.value(parent))
    }

    fn is_active_with(
        &self,
        id: ParamId,
        mut lookup: impl FnMut(ParamId) -> Option<ParamValue>,
    ) -> bool {
        let Some(spec) = self.parameter(id)
        else
        {
            return false;
        };
        match spec.condition
        {
            Condition::Always => true,
            Condition::CategoricalEquals { parent, choice } =>
            {
                lookup(parent) == Some(ParamValue::Categorical(choice))
            },
            Condition::IntEquals { parent, value } =>
            {
                lookup(parent) == Some(ParamValue::Int(value))
            },
        }
    }
}

/// Row-oriented proposal representation used at sampler/API boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    values: Vec<Option<ParamValue>>,
}

impl Candidate {
    /// Construct an empty proposal sized for one compiled search space.
    pub fn empty(space: &SearchSpace) -> Self {
        Self {
            values: vec![None; space.len()],
        }
    }

    /// Assign one active parameter after validating type and bounds.
    pub fn set(
        &mut self,
        space: &SearchSpace,
        id: ParamId,
        value: ParamValue,
    ) -> Result<(), CandidateError> {
        let Some(spec) = space.parameter(id)
        else
        {
            return Err(CandidateError::UnknownParameter(id));
        };
        if !space.is_active(id, self)
        {
            return Err(CandidateError::InactiveParameter(id));
        }
        if !spec.distribution.accepts(value)
        {
            return Err(CandidateError::InvalidValue(id));
        }
        self.values[id.index()] = Some(value);
        Ok(())
    }

    /// Return one assigned parameter value.
    pub fn value(&self, id: ParamId) -> Option<ParamValue> {
        self.values.get(id.index()).copied().flatten()
    }

    /// Return the number of parameter slots represented by this candidate.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the candidate contains no parameter slots.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over assigned values in increasing [`ParamId`] order.
    pub fn assigned_values(&self) -> impl Iterator<Item = (ParamId, ParamValue)> + '_ {
        self.values
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, value)| {
                value.map(|value| {
                    (
                        ParamId::new(u32::try_from(index).expect("candidate index fits ParamId")),
                        value,
                    )
                })
            })
    }
}

/// Candidate construction or final-validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateError {
    /// Unknown parameter identifier.
    UnknownParameter(ParamId),
    /// Candidate shape did not match the compiled search space.
    ShapeMismatch {
        /// Number of parameter slots expected by the search space.
        expected: usize,
        /// Number of parameter slots present in the candidate.
        found: usize,
    },
    /// An active parameter was not assigned by the sampler.
    MissingActiveParameter(ParamId),
    /// Parameter is inactive because its condition is not currently satisfied.
    InactiveParameter(ParamId),
    /// Value had the wrong type or was outside the parameter domain.
    InvalidValue(ParamId),
}

impl fmt::Display for CandidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::UnknownParameter(id) => write!(f, "unknown parameter {id:?}"),
            Self::ShapeMismatch { expected, found } =>
            {
                write!(f, "candidate has {found} slots, expected {expected}")
            },
            Self::MissingActiveParameter(id) => write!(f, "missing active parameter {id:?}"),
            Self::InactiveParameter(id) => write!(f, "inactive parameter {id:?}"),
            Self::InvalidValue(id) => write!(f, "invalid value for parameter {id:?}"),
        }
    }
}

impl std::error::Error for CandidateError {}

/// Monotone trial identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrialId(u64);

impl TrialId {
    /// Construct a trial identifier from its raw value.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Return the raw monotone identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Lifecycle state of one trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialState {
    /// Identifier reserved but worker execution has not started.
    Reserved,
    /// Worker execution is active.
    Running,
    /// Objective values were committed successfully.
    Complete,
    /// Trial was stopped early by a pruning policy.
    Pruned,
    /// Trial ended without a valid objective result.
    Failed,
}

/// Terminal result accepted by [`Study::tell`].
#[derive(Debug, Clone, PartialEq)]
pub enum TrialOutcome {
    /// Successful completion with one value per configured objective.
    Complete(Vec<f64>),
    /// Pruned trial.
    Pruned,
    /// Failed trial.
    Failed,
}

#[derive(Debug, Clone)]
enum ParamColumn {
    Float {
        values: Vec<f64>,
        present: Vec<bool>,
    },
    Int {
        values: Vec<i64>,
        present: Vec<bool>,
    },
    Categorical {
        values: Vec<u32>,
        present: Vec<bool>,
    },
}

impl ParamColumn {
    fn new(distribution: &Distribution) -> Self {
        match distribution
        {
            Distribution::Uniform { .. } | Distribution::LogUniform { .. } => Self::Float {
                values: Vec::new(),
                present: Vec::new(),
            },
            Distribution::IntRange { .. } => Self::Int {
                values: Vec::new(),
                present: Vec::new(),
            },
            Distribution::Categorical { .. } => Self::Categorical {
                values: Vec::new(),
                present: Vec::new(),
            },
        }
    }

    fn push_missing(&mut self) {
        match self
        {
            Self::Float { values, present } =>
            {
                values.push(0.0);
                present.push(false);
            },
            Self::Int { values, present } =>
            {
                values.push(0);
                present.push(false);
            },
            Self::Categorical { values, present } =>
            {
                values.push(0);
                present.push(false);
            },
        }
    }

    fn set(&mut self, row: usize, value: ParamValue) -> bool {
        match (self, value)
        {
            (Self::Float { values, present }, ParamValue::Float(v)) =>
            {
                values[row] = v;
                present[row] = true;
                true
            },
            (Self::Int { values, present }, ParamValue::Int(v)) =>
            {
                values[row] = v;
                present[row] = true;
                true
            },
            (Self::Categorical { values, present }, ParamValue::Categorical(v)) =>
            {
                values[row] = v;
                present[row] = true;
                true
            },
            _ => false,
        }
    }

    fn get(&self, row: usize) -> Option<ParamValue> {
        match self
        {
            Self::Float { values, present } => present
                .get(row)
                .copied()
                .filter(|present| *present)
                .map(|_| ParamValue::Float(values[row])),
            Self::Int { values, present } => present
                .get(row)
                .copied()
                .filter(|present| *present)
                .map(|_| ParamValue::Int(values[row])),
            Self::Categorical { values, present } => present
                .get(row)
                .copied()
                .filter(|present| *present)
                .map(|_| ParamValue::Categorical(values[row])),
        }
    }
}

/// Column-oriented storage for trial states, parameters and objectives.
///
/// One typed parameter column is allocated per compiled [`ParamId`]. Objective
/// values are also stored by objective column rather than as one heap object per
/// trial. This structure is intentionally independent of any persistent backend.
#[derive(Debug, Clone)]
pub struct TrialStore {
    next_id: u64,
    ids: Vec<TrialId>,
    states: Vec<TrialState>,
    parameter_columns: Vec<ParamColumn>,
    objective_columns: Vec<Vec<f64>>,
    objective_present: Vec<bool>,
}

impl TrialStore {
    fn new(space: &SearchSpace, objective_count: usize) -> Self {
        Self {
            next_id: 0,
            ids: Vec::new(),
            states: Vec::new(),
            parameter_columns: space
                .parameters()
                .map(|spec| ParamColumn::new(&spec.distribution))
                .collect(),
            objective_columns: (0..objective_count).map(|_| Vec::new()).collect(),
            objective_present: Vec::new(),
        }
    }

    fn row(&self, id: TrialId) -> Option<usize> {
        let row = usize::try_from(id.get()).ok()?;
        self.ids.get(row).copied().filter(|stored| *stored == id)?;
        Some(row)
    }

    fn reserve(&mut self) -> Result<TrialId, TrialError> {
        let id = TrialId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(TrialError::TrialIdExhausted)?;
        self.ids.push(id);
        self.states.push(TrialState::Reserved);
        for column in &mut self.parameter_columns
        {
            column.push_missing();
        }
        for column in &mut self.objective_columns
        {
            column.push(0.0);
        }
        self.objective_present.push(false);
        Ok(id)
    }

    fn set_param(&mut self, row: usize, param: ParamId, value: ParamValue) -> bool {
        self.parameter_columns
            .get_mut(param.index())
            .is_some_and(|column| column.set(row, value))
    }

    fn commit_outcome(
        &mut self,
        row: usize,
        id: TrialId,
        outcome: TrialOutcome,
    ) -> Result<(), TrialError> {
        let current = self.states[row];
        if matches!(
            current,
            TrialState::Complete | TrialState::Pruned | TrialState::Failed
        )
        {
            let existing = self
                .outcome_by_row(row)
                .expect("terminal trial has an outcome");
            return if existing == outcome
            {
                Ok(())
            }
            else
            {
                Err(TrialError::ConflictingTell(id))
            };
        }

        match outcome
        {
            TrialOutcome::Complete(values) =>
            {
                if values.len() != self.objective_columns.len()
                {
                    return Err(TrialError::ObjectiveCountMismatch {
                        expected: self.objective_columns.len(),
                        found: values.len(),
                    });
                }
                if values.iter().any(|value| !value.is_finite())
                {
                    return Err(TrialError::NonFiniteObjective);
                }
                for (column, value) in self.objective_columns.iter_mut().zip(values)
                {
                    column[row] = value;
                }
                self.objective_present[row] = true;
                self.states[row] = TrialState::Complete;
            },
            TrialOutcome::Pruned =>
            {
                self.states[row] = TrialState::Pruned;
            },
            TrialOutcome::Failed =>
            {
                self.states[row] = TrialState::Failed;
            },
        }
        Ok(())
    }

    fn outcome_by_row(&self, row: usize) -> Option<TrialOutcome> {
        match self.states.get(row).copied()?
        {
            TrialState::Complete if self.objective_present[row] => Some(TrialOutcome::Complete(
                self.objective_columns
                    .iter()
                    .map(|column| column[row])
                    .collect(),
            )),
            TrialState::Pruned => Some(TrialOutcome::Pruned),
            TrialState::Failed => Some(TrialOutcome::Failed),
            TrialState::Reserved | TrialState::Running | TrialState::Complete => None,
        }
    }

    /// Number of reserved trial rows.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no trial has been reserved.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Iterate over trial identifiers in monotone reservation order.
    pub fn trial_ids(&self) -> impl ExactSizeIterator<Item = TrialId> + '_ {
        self.ids.iter().copied()
    }

    /// Count trials currently in one lifecycle state.
    pub fn count_state(&self, state: TrialState) -> usize {
        self.states
            .iter()
            .copied()
            .filter(|current| *current == state)
            .count()
    }

    /// Return one trial's lifecycle state.
    pub fn state(&self, id: TrialId) -> Option<TrialState> {
        self.row(id).map(|row| self.states[row])
    }

    /// Return one terminal outcome, allocating only for completed objectives.
    pub fn outcome(&self, id: TrialId) -> Option<TrialOutcome> {
        self.row(id).and_then(|row| self.outcome_by_row(row))
    }

    /// Return one stored parameter value.
    pub fn param_value(&self, id: TrialId, param: ParamId) -> Option<ParamValue> {
        let row = self.row(id)?;
        self.parameter_columns.get(param.index())?.get(row)
    }

    /// Number of configured objective columns.
    pub fn objective_count(&self) -> usize {
        self.objective_columns.len()
    }
}

/// Error returned while creating a study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyError {
    /// Studies require at least one objective.
    ZeroObjectives,
}

impl fmt::Display for StudyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroObjectives => write!(f, "a study requires at least one objective"),
        }
    }
}

impl std::error::Error for StudyError {}

/// Mutation or lifecycle error for a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialError {
    /// Trial identifier was not reserved in this store.
    UnknownTrial(TrialId),
    /// Parameter identifier was not present in the compiled space.
    UnknownParameter(ParamId),
    /// Parameter's activation condition was not satisfied.
    InactiveParameter {
        /// Trial being modified.
        trial: TrialId,
        /// Inactive parameter.
        param: ParamId,
    },
    /// Parameter type or bounds were invalid.
    InvalidParameterValue(ParamId),
    /// Requested state transition was invalid.
    InvalidTransition {
        /// Trial being transitioned.
        trial: TrialId,
        /// Current state.
        from: TrialState,
        /// Requested state.
        to: TrialState,
    },
    /// Completed result had the wrong number of objectives.
    ObjectiveCountMismatch {
        /// Configured objective count.
        expected: usize,
        /// Provided objective count.
        found: usize,
    },
    /// Completed result contained NaN or infinity.
    NonFiniteObjective,
    /// Retry attempted to commit a different terminal outcome.
    ConflictingTell(TrialId),
    /// Monotone trial identifier space was exhausted.
    TrialIdExhausted,
}

impl fmt::Display for TrialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::UnknownTrial(id) => write!(f, "unknown trial {id:?}"),
            Self::UnknownParameter(id) => write!(f, "unknown parameter {id:?}"),
            Self::InactiveParameter { trial, param } =>
            {
                write!(f, "parameter {param:?} is inactive for trial {trial:?}")
            },
            Self::InvalidParameterValue(id) => write!(f, "invalid parameter value for {id:?}"),
            Self::InvalidTransition { trial, from, to } =>
            {
                write!(f, "invalid transition for {trial:?}: {from:?} -> {to:?}")
            },
            Self::ObjectiveCountMismatch { expected, found } =>
            {
                write!(f, "expected {expected} objective values, found {found}")
            },
            Self::NonFiniteObjective => write!(f, "objective values must be finite"),
            Self::ConflictingTell(id) => write!(f, "conflicting terminal result for {id:?}"),
            Self::TrialIdExhausted => write!(f, "trial identifier space exhausted"),
        }
    }
}

impl std::error::Error for TrialError {}

/// Read-only study state supplied to a sampler.
///
/// A sampler sees reservations and running trials that already exist in the
/// store. During [`Study::ask`], the identifier currently being sampled is
/// already present in state [`TrialState::Reserved`]; the sampler receives that
/// identifier separately so it can exclude its own empty reservation.
#[derive(Debug, Clone, Copy)]
pub struct StudyView<'a> {
    space: &'a SearchSpace,
    trials: &'a TrialStore,
}

impl<'a> StudyView<'a> {
    fn new(space: &'a SearchSpace, trials: &'a TrialStore) -> Self {
        Self { space, trials }
    }

    /// Return the compiled search space.
    pub const fn search_space(self) -> &'a SearchSpace {
        self.space
    }

    /// Return the read-only trial store.
    pub const fn trials(self) -> &'a TrialStore {
        self.trials
    }
}

/// Proposal algorithm used by [`Study::ask`] and [`Study::ask_batch`].
///
/// Algorithms own their mutable state. The study supplies an immutable view of
/// completed and pending history plus the freshly reserved [`TrialId`].
pub trait Sampler {
    /// Algorithm-specific error type.
    type Error;

    /// Produce a complete, conditionally valid candidate for one reserved trial.
    fn sample(
        &mut self,
        study: StudyView<'_>,
        trial: TrialId,
    ) -> Result<Candidate, Self::Error>;
}

/// Successful result of [`Study::ask`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrialProposal {
    /// Reserved trial identifier, already transitioned to Running.
    pub trial: TrialId,
    /// Complete candidate assigned to the trial.
    pub candidate: Candidate,
}

/// Failure while reserving or constructing an asked trial.
#[derive(Debug)]
pub enum AskError<E> {
    /// The trial identifier could not be reserved.
    Reservation(TrialError),
    /// The sampler failed after reservation. The reserved trial is marked Failed.
    Sampler {
        /// Reserved trial that failed.
        trial: TrialId,
        /// Algorithm-specific failure.
        source: E,
    },
    /// The sampler returned a candidate that failed final validation. The trial
    /// is marked Failed.
    InvalidCandidate {
        /// Reserved trial that failed.
        trial: TrialId,
        /// Candidate validation failure.
        source: CandidateError,
    },
    /// A validated candidate could not be committed to the trial store. The
    /// trial is marked Failed.
    Commit {
        /// Reserved trial that failed.
        trial: TrialId,
        /// Storage/lifecycle failure.
        source: TrialError,
    },
}

impl<E: fmt::Display> fmt::Display for AskError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Reservation(source) => write!(f, "trial reservation failed: {source}"),
            Self::Sampler { trial, source } =>
            {
                write!(f, "sampler failed for {trial:?}: {source}")
            },
            Self::InvalidCandidate { trial, source } =>
            {
                write!(f, "invalid candidate for {trial:?}: {source}")
            },
            Self::Commit { trial, source } =>
            {
                write!(f, "candidate commit failed for {trial:?}: {source}")
            },
        }
    }
}

impl<E> std::error::Error for AskError<E>
where
    E: std::error::Error + 'static,
{
}

/// One in-memory optimization study over a compiled search space.
///
/// Persistent/distributed backends can mirror these semantics while replacing
/// the in-memory [`TrialStore`].
#[derive(Debug, Clone)]
pub struct Study {
    space: SearchSpace,
    store: TrialStore,
}

impl Study {
    /// Create a study with a fixed objective count.
    pub fn new(space: SearchSpace, objective_count: usize) -> Result<Self, StudyError> {
        if objective_count == 0
        {
            return Err(StudyError::ZeroObjectives);
        }
        let store = TrialStore::new(&space, objective_count);
        Ok(Self { space, store })
    }

    /// Return the compiled search space.
    pub fn search_space(&self) -> &SearchSpace {
        &self.space
    }

    /// Return the read-only trial store.
    pub fn trials(&self) -> &TrialStore {
        &self.store
    }

    /// Reserve the next monotone trial identifier.
    pub fn reserve(&mut self) -> Result<TrialId, TrialError> {
        self.store.reserve()
    }

    fn fail_reserved(&mut self, trial: TrialId) {
        if let Some(row) = self.store.row(trial)
        {
            self.store.states[row] = TrialState::Failed;
        }
    }

    /// Reserve, sample, validate, assign and start one trial.
    ///
    /// Reservation happens before the sampler is called. This makes the fresh
    /// identifier visible as Reserved and makes proposals returned by earlier
    /// calls visible as Running, which is the substrate required by
    /// pending-aware parallel samplers.
    pub fn ask<S>(&mut self, sampler: &mut S) -> Result<TrialProposal, AskError<S::Error>>
    where
        S: Sampler,
    {
        let trial = self.reserve().map_err(AskError::Reservation)?;
        let candidate = {
            let view = StudyView::new(&self.space, &self.store);
            match sampler.sample(view, trial)
            {
                Ok(candidate) => candidate,
                Err(source) =>
                {
                    self.fail_reserved(trial);
                    return Err(AskError::Sampler { trial, source });
                },
            }
        };

        if let Err(source) = self.space.validate_candidate(&candidate)
        {
            self.fail_reserved(trial);
            return Err(AskError::InvalidCandidate { trial, source });
        }

        for (param, value) in candidate.assigned_values()
        {
            if let Err(source) = self.set_param(trial, param, value)
            {
                self.fail_reserved(trial);
                return Err(AskError::Commit { trial, source });
            }
        }
        if let Err(source) = self.start(trial)
        {
            self.fail_reserved(trial);
            return Err(AskError::Commit { trial, source });
        }

        Ok(TrialProposal { trial, candidate })
    }

    /// Produce a rolling batch of running proposals.
    ///
    /// Proposals are generated sequentially without a synchronization barrier.
    /// After each successful proposal is committed and marked Running, the next
    /// sampler call sees it in [`StudyView`]. This permits constant-liar,
    /// fantasy and other pending-aware strategies without a separate side
    /// channel.
    ///
    /// If one proposal fails, earlier proposals remain Running and are visible
    /// through [`Self::trials`]; the failing reservation is marked Failed.
    pub fn ask_batch<S>(
        &mut self,
        sampler: &mut S,
        batch_size: usize,
    ) -> Result<Vec<TrialProposal>, AskError<S::Error>>
    where
        S: Sampler,
    {
        let mut proposals = Vec::with_capacity(batch_size);
        for _ in 0..batch_size
        {
            proposals.push(self.ask(sampler)?);
        }
        Ok(proposals)
    }

    /// Mark a reserved trial as running.
    ///
    /// Repeating the operation on an already-running trial is idempotent.
    pub fn start(&mut self, trial: TrialId) -> Result<(), TrialError> {
        let row = self
            .store
            .row(trial)
            .ok_or(TrialError::UnknownTrial(trial))?;
        match self.store.states[row]
        {
            TrialState::Reserved =>
            {
                self.store.states[row] = TrialState::Running;
                Ok(())
            },
            TrialState::Running => Ok(()),
            from => Err(TrialError::InvalidTransition {
                trial,
                from,
                to: TrialState::Running,
            }),
        }
    }

    /// Assign one parameter to a reserved or running trial.
    ///
    /// Conditions are evaluated against parameter values already present in the
    /// same trial, which permits dynamic/conditional suggestion order.
    pub fn set_param(
        &mut self,
        trial: TrialId,
        param: ParamId,
        value: ParamValue,
    ) -> Result<(), TrialError> {
        let row = self
            .store
            .row(trial)
            .ok_or(TrialError::UnknownTrial(trial))?;
        let state = self.store.states[row];
        if !matches!(state, TrialState::Reserved | TrialState::Running)
        {
            return Err(TrialError::InvalidTransition {
                trial,
                from: state,
                to: state,
            });
        }
        let Some(spec) = self.space.parameter(param)
        else
        {
            return Err(TrialError::UnknownParameter(param));
        };
        let active = self
            .space
            .is_active_with(param, |parent| self.store.param_value(trial, parent));
        if !active
        {
            return Err(TrialError::InactiveParameter { trial, param });
        }
        if !spec.distribution.accepts(value)
        {
            return Err(TrialError::InvalidParameterValue(param));
        }
        if !self.store.set_param(row, param, value)
        {
            return Err(TrialError::InvalidParameterValue(param));
        }
        Ok(())
    }

    /// Commit a terminal result.
    ///
    /// Repeating exactly the same terminal result is accepted. Retrying with a
    /// different result returns [`TrialError::ConflictingTell`].
    pub fn tell(&mut self, trial: TrialId, outcome: TrialOutcome) -> Result<(), TrialError> {
        let row = self
            .store
            .row(trial)
            .ok_or(TrialError::UnknownTrial(trial))?;
        self.store.commit_outcome(row, trial, outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conditional_space() -> SearchSpace {
        SearchSpace::compile(vec![
            ParameterSpec::new(
                ParamId::new(0),
                "optimizer",
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
        .unwrap()
    }

    #[test]
    fn compiled_space_is_dense_and_deterministic() {
        let space = conditional_space();
        let ids: Vec<_> = space.parameters().map(|spec| spec.id).collect();
        assert_eq!(ids, vec![ParamId::new(0), ParamId::new(1), ParamId::new(2)]);
        assert_eq!(space.len(), 3);
    }

    #[test]
    fn invalid_log_domain_is_rejected() {
        let err = SearchSpace::compile(vec![ParameterSpec::new(
            ParamId::new(0),
            "lr",
            Distribution::LogUniform {
                low: 0.0,
                high: 1.0,
            },
            Condition::Always,
        )])
        .unwrap_err();
        assert_eq!(err, SpaceError::InvalidDistribution(ParamId::new(0)));
    }

    #[test]
    fn condition_parent_must_precede_child() {
        let err = SearchSpace::compile(vec![ParameterSpec::new(
            ParamId::new(0),
            "child",
            Distribution::IntRange { low: 0, high: 2 },
            Condition::IntEquals {
                parent: ParamId::new(0),
                value: 1,
            },
        )])
        .unwrap_err();
        assert_eq!(
            err,
            SpaceError::InvalidConditionParent {
                child: ParamId::new(0),
                parent: ParamId::new(0),
            }
        );
    }

    #[test]
    fn candidate_condition_is_enforced() {
        let space = conditional_space();
        let mut candidate = Candidate::empty(&space);
        assert_eq!(
            candidate.set(&space, ParamId::new(1), ParamValue::Int(4)),
            Err(CandidateError::InactiveParameter(ParamId::new(1)))
        );
        candidate
            .set(&space, ParamId::new(0), ParamValue::Categorical(1))
            .unwrap();
        candidate
            .set(&space, ParamId::new(1), ParamValue::Int(4))
            .unwrap();
        assert_eq!(candidate.value(ParamId::new(1)), Some(ParamValue::Int(4)));
    }

    #[test]
    fn reservations_are_monotone_and_o1_addressable() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let a = study.reserve().unwrap();
        let b = study.reserve().unwrap();
        assert_eq!((a.get(), b.get()), (0, 1));
        assert_eq!(study.trials().state(a), Some(TrialState::Reserved));
        assert_eq!(study.trials().state(b), Some(TrialState::Reserved));
    }

    #[test]
    fn store_keeps_typed_parameter_columns() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let trial = study.reserve().unwrap();
        study
            .set_param(trial, ParamId::new(0), ParamValue::Categorical(1))
            .unwrap();
        study
            .set_param(trial, ParamId::new(1), ParamValue::Int(7))
            .unwrap();
        study
            .set_param(trial, ParamId::new(2), ParamValue::Float(1e-3))
            .unwrap();
        assert_eq!(
            study.trials().param_value(trial, ParamId::new(0)),
            Some(ParamValue::Categorical(1))
        );
        assert_eq!(
            study.trials().param_value(trial, ParamId::new(1)),
            Some(ParamValue::Int(7))
        );
        assert_eq!(
            study.trials().param_value(trial, ParamId::new(2)),
            Some(ParamValue::Float(1e-3))
        );
    }

    #[test]
    fn tell_is_idempotent_for_same_result() {
        let mut study = Study::new(conditional_space(), 2).unwrap();
        let trial = study.reserve().unwrap();
        study.start(trial).unwrap();
        let result = TrialOutcome::Complete(vec![1.25, 3.5]);
        study.tell(trial, result.clone()).unwrap();
        study.tell(trial, result.clone()).unwrap();
        assert_eq!(study.trials().outcome(trial), Some(result));
    }

    #[test]
    fn conflicting_retry_is_rejected() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let trial = study.reserve().unwrap();
        study
            .tell(trial, TrialOutcome::Complete(vec![1.0]))
            .unwrap();
        assert_eq!(
            study.tell(trial, TrialOutcome::Complete(vec![2.0])),
            Err(TrialError::ConflictingTell(trial))
        );
    }

    #[test]
    fn objective_shape_and_finiteness_are_checked() {
        let mut study = Study::new(conditional_space(), 2).unwrap();
        let trial = study.reserve().unwrap();
        assert_eq!(
            study.tell(trial, TrialOutcome::Complete(vec![1.0])),
            Err(TrialError::ObjectiveCountMismatch {
                expected: 2,
                found: 1,
            })
        );
        assert_eq!(
            study.tell(trial, TrialOutcome::Complete(vec![1.0, f64::NAN])),
            Err(TrialError::NonFiniteObjective)
        );
    }

    #[test]
    fn terminal_trials_reject_parameter_mutation() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let trial = study.reserve().unwrap();
        study.tell(trial, TrialOutcome::Pruned).unwrap();
        let err = study
            .set_param(trial, ParamId::new(0), ParamValue::Categorical(0))
            .unwrap_err();
        assert!(matches!(
            err,
            TrialError::InvalidTransition {
                from: TrialState::Pruned,
                ..
            }
        ));
    }

    struct PendingAwareSampler {
        seen_running: Vec<usize>,
        choice: u32,
    }

    impl Sampler for PendingAwareSampler {
        type Error = &'static str;

        fn sample(
            &mut self,
            study: StudyView<'_>,
            _trial: TrialId,
        ) -> Result<Candidate, Self::Error> {
            self.seen_running
                .push(study.trials().count_state(TrialState::Running));
            let mut candidate = Candidate::empty(study.search_space());
            candidate
                .set(
                    study.search_space(),
                    ParamId::new(0),
                    ParamValue::Categorical(self.choice),
                )
                .map_err(|_| "categorical assignment failed")?;
            if self.choice == 1
            {
                candidate
                    .set(study.search_space(), ParamId::new(1), ParamValue::Int(4))
                    .map_err(|_| "conditional assignment failed")?;
            }
            candidate
                .set(
                    study.search_space(),
                    ParamId::new(2),
                    ParamValue::Float(1e-3),
                )
                .map_err(|_| "float assignment failed")?;
            Ok(candidate)
        }
    }

    #[test]
    fn ask_batch_exposes_previous_proposals_as_running() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let mut sampler = PendingAwareSampler {
            seen_running: Vec::new(),
            choice: 1,
        };

        let proposals = study.ask_batch(&mut sampler, 3).unwrap();

        assert_eq!(sampler.seen_running, vec![0, 1, 2]);
        assert_eq!(
            proposals.iter().map(|proposal| proposal.trial.get()).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(study.trials().count_state(TrialState::Running), 3);
        assert_eq!(study.trials().count_state(TrialState::Reserved), 0);
    }

    struct IncompleteSampler;

    impl Sampler for IncompleteSampler {
        type Error = &'static str;

        fn sample(
            &mut self,
            study: StudyView<'_>,
            _trial: TrialId,
        ) -> Result<Candidate, Self::Error> {
            Ok(Candidate::empty(study.search_space()))
        }
    }

    #[test]
    fn invalid_sampler_candidate_marks_reservation_failed() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let err = study.ask(&mut IncompleteSampler).unwrap_err();
        let trial = match err
        {
            AskError::InvalidCandidate {
                trial,
                source: CandidateError::MissingActiveParameter(ParamId(0)),
            } => trial,
            other => panic!("unexpected ask error: {other:?}"),
        };
        assert_eq!(study.trials().state(trial), Some(TrialState::Failed));
    }

    struct FailingSampler;

    impl Sampler for FailingSampler {
        type Error = &'static str;

        fn sample(
            &mut self,
            _study: StudyView<'_>,
            _trial: TrialId,
        ) -> Result<Candidate, Self::Error> {
            Err("synthetic sampler failure")
        }
    }

    #[test]
    fn sampler_failure_marks_reservation_failed() {
        let mut study = Study::new(conditional_space(), 1).unwrap();
        let err = study.ask(&mut FailingSampler).unwrap_err();
        let trial = match err
        {
            AskError::Sampler {
                trial,
                source: "synthetic sampler failure",
            } => trial,
            other => panic!("unexpected ask error: {other:?}"),
        };
        assert_eq!(study.trials().state(trial), Some(TrialState::Failed));
    }

    #[test]
    fn final_candidate_validation_catches_parent_rewrite() {
        let space = conditional_space();
        let mut candidate = Candidate::empty(&space);
        candidate
            .set(&space, ParamId::new(0), ParamValue::Categorical(1))
            .unwrap();
        candidate
            .set(&space, ParamId::new(1), ParamValue::Int(4))
            .unwrap();
        candidate
            .set(&space, ParamId::new(0), ParamValue::Categorical(0))
            .unwrap();
        candidate
            .set(&space, ParamId::new(2), ParamValue::Float(1e-3))
            .unwrap();

        assert_eq!(
            space.validate_candidate(&candidate),
            Err(CandidateError::InactiveParameter(ParamId::new(1)))
        );
    }
}
