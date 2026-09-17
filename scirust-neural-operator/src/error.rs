use thiserror::Error;

/// Result type used by the neural-operator crate.
pub type Result<T> = std::result::Result<T, NeuralOperatorError>;

/// Fail-closed validation errors for operator-learning inputs.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum NeuralOperatorError {
    #[error("{what} must be non-empty")]
    Empty { what: &'static str },

    #[error("{what} contains a non-finite value at index {index}")]
    NonFinite { what: &'static str, index: usize },

    #[error("shape mismatch for {what}: expected {expected} values, got {got}")]
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("invalid channel count {channels}; channel count must be positive")]
    InvalidChannels { channels: usize },

    #[error("row width {width} is not divisible by channel count {channels}")]
    ChannelMismatch { width: usize, channels: usize },

    #[error("FFT axis length {len} must be a power of two")]
    NonRadix2 { len: usize },

    #[error("physical domain length must be finite and > 0, got {length}")]
    InvalidDomainLength { length: f64 },

    #[error("derivative order must be positive")]
    ZeroDerivativeOrder,

    #[error("axis {axis} is invalid for a 2-D field; expected 0 or 1")]
    InvalidAxis { axis: usize },

    #[error("dataset must contain at least one sample")]
    EmptyDataset,

    #[error("sample {sample} has incompatible {what} length: expected {expected}, got {got}")]
    DatasetShapeMismatch {
        sample: usize,
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("normalizer has not been fitted")]
    NormalizerNotFitted,

    #[error("standard deviation for channel {channel} is not finite or below epsilon")]
    DegenerateChannel { channel: usize },

    #[error("LpLoss supports p >= 1, got {p}")]
    InvalidP { p: u32 },

    #[error("learning rate must be finite, got {lr}")]
    InvalidLearningRate { lr: f32 },

    #[error("Boolean shape mismatch for {what}: expected {expected}, got {got}")]
    BooleanShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("Boolean index {index} is out of bounds for length {len}")]
    BooleanIndexOutOfBounds { index: usize, len: usize },

    #[error("{variables} Boolean variables exceed the supported maximum {maximum}")]
    TooManyBooleanVariables { variables: usize, maximum: usize },

    #[error("truth table for {variables} variables must contain {expected} rows, got {got}")]
    TruthTableLength {
        variables: usize,
        expected: usize,
        got: usize,
    },

    #[error("ANF monomial mask {mask:#x} references a variable outside arity {variables}")]
    InvalidMonomialMask { mask: u64, variables: usize },

    #[error("spectral mode {mode} is outside configured range 0..{configured_modes}")]
    InvalidSpectralMode {
        mode: usize,
        configured_modes: usize,
    },

    #[error("spectral mode indices must be strictly increasing and unique")]
    NonCanonicalSpectralModes,

    #[error("Boolean development example {example} has width {got}; expected {expected}")]
    BooleanDatasetShapeMismatch {
        example: usize,
        expected: usize,
        got: usize,
    },

    #[error("duplicate Boolean development assignment {assignment:#x}")]
    DuplicateBooleanAssignment { assignment: u64 },

    #[error("ANF search degree {degree} exceeds variable count {variables}")]
    InvalidAnfSearchDegree { degree: usize, variables: usize },

    #[error("ANF search max_terms must be positive, got {terms}")]
    InvalidAnfSearchTerms { terms: usize },

    #[error("ANF candidate universe needs {required} evaluations but budget is {budget}")]
    CandidateBudgetExceeded { required: usize, budget: usize },

    #[error("no exact ANF candidate exists inside the declared bounded grammar")]
    NoExactAnfCandidate,

    #[error("benchmark repeats must be positive, got {repeats}")]
    InvalidBenchmarkRepeats { repeats: usize },

    #[error("verification tolerance must be finite and non-negative, got {tolerance}")]
    InvalidVerificationTolerance { tolerance: f64 },

    #[error("surrogate cost inputs must be finite and non-negative")]
    InvalidCost,
}
