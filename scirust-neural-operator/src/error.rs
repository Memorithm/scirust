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

    #[error("surrogate cost inputs must be finite and non-negative")]
    InvalidCost,
}
