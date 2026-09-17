//! # SciRust Neural Operator
//!
//! Native Rust building blocks for **operator learning**: learning mappings
//! between discretised functions rather than only fixed-dimensional vectors.
//!
//! This crate is intentionally independent from the Python `neuraloperator`
//! package. It reuses SciRust's own deterministic autodiff, FNO, DeepONet and
//! portable FFT implementations and adds the missing library-level contracts
//! around them: datasets, normalization, losses, spectral differential
//! operators, trainable FNO wrappers and surrogate qualification metrics.
//!
//! The scientific contract is conservative: an operator surrogate is an
//! approximation to a reference solver, never a replacement for validation.
//! Exact/reference simulations remain the oracle for qualification and for
//! out-of-distribution fallback.

pub mod anf_search;
pub mod boolean;
pub mod dataset;
pub mod error;
pub mod fno;
pub mod grid;
pub mod hybrid;
pub mod loss;
pub mod metrics;
pub mod normalizer;
pub mod spectral;
pub mod spectral_routing;

pub use anf_search::{
    BooleanDevelopmentExample, BooleanDevelopmentSet, SparseAnfSearchConfig, SparseAnfSearchResult,
    search_sparse_anf,
};
pub use boolean::{AnfOperator, AnfPolynomial, BooleanComplexity, F2LinearOperator, PackedBits};
pub use dataset::{OperatorDataset1d, OperatorSample1d};
pub use error::{NeuralOperatorError, Result};
pub use fno::{FitReport, Fno1dConfig, Fno1dOperator, LearnedOperator};
pub use grid::PeriodicGrid1d;
pub use hybrid::{
    BooleanRouteRule, BooleanRouter, HybridAction, HybridExecution, HybridExecutor,
    RouterComplexity,
};
pub use loss::{LpLoss, relative_l2};
pub use metrics::{OperatorMetrics, SurrogateEconomics};
pub use normalizer::ChannelNormalizer;
pub use spectral::{
    spectral_derivative_1d, spectral_derivative_2d, spectral_laplacian_1d, spectral_laplacian_2d,
};
pub use spectral_routing::{
    BooleanSpectralRouter, SpectralModePlan, SpectralRouterComplexity, SpectralWorkEstimate,
};

// Existing SciRust operator-learning primitives remain available through the
// dedicated crate so callers do not have to know their historical locations.
pub use scirust_core::nn::deeponet::DeepONet;
pub use scirust_core::nn::fno::{FnoSpectralConv1d, NdFno};
pub use scirust_core::nn::pinn::{Pinn1D, PinnSolution, solve_harmonic};
