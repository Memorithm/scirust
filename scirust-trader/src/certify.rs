//! Certified bounds — lightweight Interval Bound Propagation (IBP) for the
//! trading model.
//!
//! Given a perturbation radius `eps` around the input feature vector, we
//! propagate numerical interval estimates through the exported network.
//! This f32 implementation has no directed rounding and is not a formal
//! floating-point enclosure, a market-return guarantee, or an LLM constraint.
//!
//! This is a simplified IBP: for a linear layer `y = Wx + b`, if `x ∈ [x_lo, x_hi]`
//! then `y_j ∈ [sum(W_ji * x_lo_i if W_ji>0 else W_ji * x_hi_i) + b_j,
//!              sum(W_ji * x_hi_i if W_ji>0 else W_ji * x_lo_i) + b_j]`.
//! For ReLU, `[max(0, lo), max(0, hi)]`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::model::ModelWeights;

/// Invalid input never produces a partial or zero-width certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificationError {
    InvalidRadius,
    InvalidInput,
    InvalidLayer(usize),
    NonScalarOutput,
    NonFiniteArithmetic,
    InvalidAgentConfiguration,
}

impl std::fmt::Display for CertificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "interval estimation failed: {self:?}")
    }
}

impl std::error::Error for CertificationError {}

/// An interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f32,
    pub hi: f32,
}

impl Interval {
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }

    pub fn point(x: f32) -> Self {
        Self { lo: x, hi: x }
    }

    pub fn width(&self) -> f32 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f32 {
        (self.lo + self.hi) / 2.0
    }
}

/// Certified output of the model on an input box of radius `eps`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedBounds {
    /// Input perturbation radius (L∞ ball).
    pub eps: f32,
    /// Output interval for the predicted return.
    pub output: Interval,
    /// Midpoint prediction.
    pub midpoint: f32,
    /// Half-width of the output interval (uncertainty).
    pub uncertainty: f32,
    /// Hash of the weights used (for the proof file).
    pub weights_fingerprint: String,
}

impl CertifiedBounds {
    /// Serialize to JSON for the proof file.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Propagate interval bounds through the network's weights.
///
/// The layout matches [`PricePredictor::export_weights`](crate::model::PricePredictor::export_weights):
/// each entry in `weights.layers` is **one** `Linear` layer, stored as its bias
/// (`out` values) immediately followed by its weight matrix (`in*out`, row-major
/// `(in, out)`). ReLU is applied after every layer except the last, mirroring the
/// `Sequential` network. `input` is the feature vector; `eps` is the L∞ radius.
///
/// # Panics
/// Panics on invalid input. Use `try_certify` at untrusted/public boundaries.
pub fn certify(weights: &ModelWeights, input: &[f32], eps: f32) -> CertifiedBounds {
    try_certify(weights, input, eps).expect("invalid certification input; use try_certify")
}

/// Checked numerical IBP for the exported linear/ReLU scalar network.
///
/// Rejects malformed layers, dimensional mismatch, nonfinite inputs/weights,
/// invalid radius and nonfinite intermediate arithmetic. This does not verify
/// training provenance or floating-point enclosure soundness.
pub fn try_certify(
    weights: &ModelWeights,
    input: &[f32],
    eps: f32,
) -> Result<CertifiedBounds, CertificationError> {
    if !eps.is_finite() || eps < 0.0
    {
        return Err(CertificationError::InvalidRadius);
    }
    if input.is_empty() || input.len() != weights.input_dim || input.iter().any(|x| !x.is_finite())
    {
        return Err(CertificationError::InvalidInput);
    }
    if weights.layers.is_empty()
    {
        return Err(CertificationError::InvalidLayer(0));
    }
    let mut bounds: Vec<Interval> = input
        .iter()
        .map(|&x| Interval::new(x - eps, x + eps))
        .collect();
    if bounds.iter().any(|b| !b.lo.is_finite() || !b.hi.is_finite())
    {
        return Err(CertificationError::NonFiniteArithmetic);
    }

    let n_layers = weights.layers.len();
    for (li, entry) in weights.layers.iter().enumerate()
    {
        let in_dim = bounds.len();
        // entry = bias(out) ++ weight(in*out)  =>  len = out*(in+1)  =>  out = len/(in+1)
        if in_dim == 0 || entry.len() % (in_dim + 1) != 0 || entry.iter().any(|x| !x.is_finite())
        {
            return Err(CertificationError::InvalidLayer(li));
        }
        let out_dim = entry.len() / (in_dim + 1);
        if out_dim == 0
        {
            return Err(CertificationError::InvalidLayer(li));
        }
        let (b_flat, w_flat) = entry.split_at(out_dim);
        let mut out_bounds = Vec::with_capacity(out_dim);
        for j in 0..out_dim
        {
            let mut lo = b_flat[j];
            let mut hi = b_flat[j];
            for k in 0..in_dim
            {
                let w = w_flat[k * out_dim + j];
                let (x_lo, x_hi) = (bounds[k].lo, bounds[k].hi);
                if w >= 0.0
                {
                    lo += w * x_lo;
                    hi += w * x_hi;
                }
                else
                {
                    lo += w * x_hi;
                    hi += w * x_lo;
                }
            }
            if !lo.is_finite() || !hi.is_finite() || lo > hi
            {
                return Err(CertificationError::NonFiniteArithmetic);
            }
            out_bounds.push(Interval::new(lo, hi));
        }
        bounds = out_bounds;
        // ReLU after every layer except the last.
        if li + 1 < n_layers
        {
            for b in &mut bounds
            {
                b.lo = b.lo.max(0.0);
                b.hi = b.hi.max(0.0);
            }
        }
    }
    if bounds.len() != 1
    {
        return Err(CertificationError::NonScalarOutput);
    }
    let out = bounds[0];
    let midpoint = out.midpoint();
    let uncertainty = out.width() / 2.0;
    if !midpoint.is_finite() || !uncertainty.is_finite()
    {
        return Err(CertificationError::NonFiniteArithmetic);
    }
    Ok(CertifiedBounds {
        eps,
        output: out,
        midpoint,
        uncertainty,
        weights_fingerprint: weights.fingerprint.clone(),
    })
}

/// Feature attribution using Integrated Gradients.
pub fn feature_attribution(
    model: &mut crate::model::PricePredictor,
    features: &[f32],
    names: &[String],
) -> BTreeMap<String, f32> {
    use scirust_core::autodiff::reverse::Tensor;
    use scirust_core::nn::Module;
    use scirust_core::xai::integrated_gradients;

    let input = Tensor::from_vec(features.to_vec(), 1, features.len());
    let baseline = Tensor::zeros(1, features.len());

    let attr = integrated_gradients(&input, &baseline, 20, |x| model.net.forward(x.tape(), *x));

    let total: f32 = attr.data.iter().map(|f| f.abs()).sum::<f32>().max(1e-6);
    attr.data
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let name = names.get(i).cloned().unwrap_or_else(|| format!("f{}", i));
            (name, f.abs() / total)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PricePredictor;

    fn linear_weights(layers: Vec<Vec<f32>>, input_dim: usize) -> ModelWeights {
        ModelWeights { layers, input_dim, fingerprint: "test-only".into() }
    }

    #[test]
    fn checked_certification_rejects_malformed_inputs() {
        let weights = linear_weights(vec![vec![1.0, 2.0]], 1);
        for eps in [-1.0, f32::NAN, f32::INFINITY]
        {
            assert_eq!(try_certify(&weights, &[1.0], eps).unwrap_err(), CertificationError::InvalidRadius);
        }
        for input in [vec![], vec![1.0, 2.0], vec![f32::NAN], vec![f32::INFINITY]]
        {
            assert_eq!(try_certify(&weights, &input, 0.0).unwrap_err(), CertificationError::InvalidInput);
        }
        for layers in [vec![], vec![vec![]], vec![vec![1.0]], vec![vec![0.0, f32::NAN]],
                       vec![vec![0.0, 1.0], vec![1.0]]]
        {
            assert!(matches!(try_certify(&linear_weights(layers, 1), &[1.0], 0.0), Err(CertificationError::InvalidLayer(_))));
        }
        assert_eq!(try_certify(&linear_weights(vec![vec![0.0; 4]], 1), &[1.0], 0.0).unwrap_err(), CertificationError::NonScalarOutput);
        assert_eq!(try_certify(&weights, &[f32::MAX], f32::MAX).unwrap_err(), CertificationError::NonFiniteArithmetic);
        assert_eq!(try_certify(&linear_weights(vec![vec![0.0, f32::MAX]], 1), &[2.0], 0.0).unwrap_err(), CertificationError::NonFiniteArithmetic);
    }

    #[test]
    fn checked_linear_interval_matches_independent_corner_oracle() {
        // y = 1 - 2*x0 + 3*x1; exact integer-valued corner arithmetic.
        let weights = linear_weights(vec![vec![1.0, -2.0, 3.0]], 2);
        let bounds = try_certify(&weights, &[2.0, 3.0], 1.0).unwrap();
        let corners: Vec<f32> = [1.0, 3.0].iter().flat_map(|x| [2.0, 4.0].iter().map(move |y| 1.0 - 2.0*x + 3.0*y)).collect();
        assert_eq!(bounds.output.lo, corners.iter().copied().fold(f32::INFINITY, f32::min));
        assert_eq!(bounds.output.hi, corners.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    }

    #[test]
    fn interval_basic() {
        let i = Interval::new(1.0, 3.0);
        assert_eq!(i.width(), 2.0);
        assert_eq!(i.midpoint(), 2.0);
    }

    #[test]
    fn certify_returns_finite_bounds() {
        let mut model = PricePredictor::new(5, &[8], 42);
        let features = vec![0.1, -0.05, 0.2, 0.0, 0.15];
        let _ = model.predict(&features);
        let weights = model.export_weights();
        let bounds = certify(&weights, &features, 0.01);
        assert!(bounds.output.lo.is_finite());
        assert!(bounds.output.hi.is_finite());
        assert!(bounds.output.lo <= bounds.output.hi);
        assert_eq!(bounds.eps, 0.01);
        assert_eq!(bounds.weights_fingerprint, weights.fingerprint);
    }

    #[test]
    fn certified_interval_contains_model_output() {
        // Regression for the weight-layout mismatch: the certified interval must
        // actually BOUND the network output. The strongest check is eps=0, where
        // sound IBP collapses onto the exact forward pass.
        let mut model = PricePredictor::new(13, &[16, 8], 42);
        let features: Vec<f32> = (0..13).map(|i| 0.01 * i as f32 - 0.05).collect();
        let y = model.predict(&features);
        let weights = model.export_weights();

        let b = certify(&weights, &features, 0.01);
        assert!(
            b.output.lo <= y && y <= b.output.hi,
            "certified [{}, {}] must contain model output {y}",
            b.output.lo,
            b.output.hi
        );

        let b0 = certify(&weights, &features, 0.0);
        assert!(
            (b0.output.lo - y).abs() < 1e-4 && (b0.output.hi - y).abs() < 1e-4,
            "eps=0 must reproduce the exact output: got [{}, {}] vs {y}",
            b0.output.lo,
            b0.output.hi
        );
    }

    #[test]
    fn larger_eps_widens_output() {
        let mut model = PricePredictor::new(5, &[8], 42);
        let features = vec![0.1, -0.05, 0.2, 0.0, 0.15];
        let _ = model.predict(&features);
        let weights = model.export_weights();
        let small = certify(&weights, &features, 0.001);
        let large = certify(&weights, &features, 0.1);
        assert!(large.uncertainty >= small.uncertainty);
    }

    #[test]
    fn attribution_sums_to_one() {
        let mut model = PricePredictor::new(3, &[8], 42);
        let features = vec![0.5, -0.3, 0.2];
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let attr = feature_attribution(&mut model, &features, &names);
        let total: f32 = attr.values().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }
}
