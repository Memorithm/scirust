// scirust-core/src/nn/activation.rs
//
// Wrappers Module pour les activations sans paramètres.
//
// Pourquoi en faire des Module ? Pour pouvoir les empiler dans Sequential
// au même titre que Linear. Sans paramètres : parameter_indices() retourne
// un Vec vide, sync() est un no-op.

use crate::autodiff::reverse::{Tape, Var};
use crate::error::Result;
use crate::nn::module::Module;

// ---------- ReLU ---------- //

pub struct ReLU;

impl ReLU {
    /// Construct a parameter-free rectified-linear activation.
    ///
    /// The module applies `max(0, x)` elementwise and therefore has no trainable
    /// parameter indices to synchronize.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::reverse::{Tape, Tensor};
    /// use scirust_core::nn::{Module, ReLU};
    /// let tape = Tape::new();
    /// let x = tape.input(Tensor::from_vec(vec![-1.0, 2.0], 1, 2));
    /// let y = ReLU::new().forward(&tape, x);
    /// assert_eq!(tape.value(y.idx()).data, vec![0.0, 2.0]);
    /// ```
    pub fn new() -> Self {
        ReLU
    }
}

impl Default for ReLU {
    fn default() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        input.relu()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }
    fn sync(&mut self, _tape: &Tape) {}
}

// ---------- Sigmoid ---------- //

pub struct Sigmoid;

impl Sigmoid {
    /// Construct a parameter-free logistic sigmoid activation.
    ///
    /// The forward pass maps each finite input through `1 / (1 + exp(-x))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::reverse::{Tape, Tensor};
    /// use scirust_core::nn::{Module, Sigmoid};
    /// let tape = Tape::new();
    /// let x = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
    /// let y = Sigmoid::new().forward(&tape, x);
    /// assert!((tape.value(y.idx()).data[0] - 0.5).abs() < 1e-6);
    /// ```
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Sigmoid
    }
}

impl Module for Sigmoid {
    fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        input.sigmoid()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }
    fn sync(&mut self, _tape: &Tape) {}
}

// ---------- Softmax ---------- //
//
// ATTENTION : CrossEntropyLoss intègre déjà le softmax + log + négation.
// Si vous utilisez CrossEntropyLoss, ne passez PAS par Softmax/LogSoftmax
// en amont — vous feriez un softmax suivi d'un log(softmax), ce qui est
// redondant et moins stable numériquement.
//
// Softmax seul est utile quand vous voulez explicitement des probabilités
// (par exemple pour l'inférence ou l'affichage).

pub struct Softmax {
    pub axis: u8,
}

impl Softmax {
    /// Construct a softmax module normalized along `axis`.
    ///
    /// The axis is validated by the fallible [`Module::try_forward`] path; the
    /// compatibility [`Module::forward`] path unwraps that result and therefore
    /// panics for an unsupported axis. Do not place this module before a loss that
    /// already incorporates softmax (for example cross entropy).
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::reverse::{Tape, Tensor};
    /// use scirust_core::nn::Module;
    /// use scirust_core::nn::activation::Softmax;
    /// let tape = Tape::new();
    /// let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
    /// let y = Softmax::new(1).forward(&tape, x);
    /// let sum: f32 = tape.value(y.idx()).data.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// ```
    pub fn new(axis: u8) -> Self {
        Softmax { axis }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Softmax { axis: 1 }
    }
}

impl Module for Softmax {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        self.try_forward(tape, input).unwrap()
    }

    fn try_forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Result<Var<'t>> {
        input.try_softmax(self.axis)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }
    fn sync(&mut self, _tape: &Tape) {}
}

// ---------- LogSoftmax ---------- //
//
// LogSoftmax est conçu pour être suivi de NLLLoss (Negative Log-Likelihood).
// Cette combinaison LogSoftmax + NLLLoss est mathématiquement équivalente
// à CrossEntropyLoss, mais découpée en deux modules explicites.
//
//   let log_probs = model.forward(&tape, x).log_softmax(1);
//   let loss = NLLLoss::new().forward(&tape, log_probs, target);
//
// N'utilisez LogSoftmax devant CrossEntropyLoss — vous feriez deux softmax.

pub struct LogSoftmax {
    pub axis: u8,
}

impl LogSoftmax {
    /// Construct a log-softmax module normalized along `axis`.
    ///
    /// This is the explicit log-probability companion for NLL-style losses. The
    /// axis is validated by [`Module::try_forward`]; the compatibility
    /// [`Module::forward`] method unwraps that result.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::reverse::{Tape, Tensor};
    /// use scirust_core::nn::Module;
    /// use scirust_core::nn::activation::LogSoftmax;
    /// let tape = Tape::new();
    /// let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
    /// let y = LogSoftmax::new(1).forward(&tape, x);
    /// assert!(tape.value(y.idx()).data.iter().all(|v| *v <= 0.0));
    /// ```
    pub fn new(axis: u8) -> Self {
        LogSoftmax { axis }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        LogSoftmax { axis: 1 }
    }
}

impl Module for LogSoftmax {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        self.try_forward(tape, input).unwrap()
    }

    fn try_forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Result<Var<'t>> {
        input.try_log_softmax(self.axis)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }
    fn sync(&mut self, _tape: &Tape) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;

    #[test]
    fn relu_forward_clamps_negatives() {
        let mut act = ReLU::new();
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], 1, 4));
        let y = act.forward(&tape, x);
        assert_eq!(tape.value(y.idx()).data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn sigmoid_forward_in_zero_one() {
        let mut act = Sigmoid::new();
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![-100.0, 0.0, 100.0], 1, 3));
        let y = act.forward(&tape, x);
        let v = tape.value(y.idx());
        assert!(v.data[0] >= 0.0 && v.data[0] < 0.01);
        assert!((v.data[1] - 0.5).abs() < 1e-6);
        assert!(v.data[2] > 0.99 && v.data[2] <= 1.0);
    }

    #[test]
    fn activations_have_no_parameters() {
        let relu = ReLU::new();
        let sig = Sigmoid::new();
        assert!(relu.parameter_indices().is_empty());
        assert!(sig.parameter_indices().is_empty());
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut act = Softmax::new(1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let y = act.forward(&tape, x);
        let v = tape.value(y.idx());
        let sum: f32 = v.data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn softmax_stable_with_large_logits() {
        let mut act = Softmax::new(1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1000.0, 999.0, 998.0], 1, 3));
        let y = act.forward(&tape, x);
        let v = tape.value(y.idx());
        assert!(
            v.data.iter().all(|f| f.is_finite()),
            "softmax should be finite with large logits, got {:?}",
            v.data
        );
        let sum: f32 = v.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn log_softmax_rows_sum_less_than_zero() {
        let mut act = LogSoftmax::new(1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let y = act.forward(&tape, x);
        let v = tape.value(y.idx());
        // log(softmax) values should be <= 0
        assert!(
            v.data.iter().all(|f| *f <= 0.0),
            "log_softmax values should be <= 0, got {:?}",
            v.data
        );
    }

    #[test]
    fn log_softmax_stable_with_large_logits() {
        let mut act = LogSoftmax::new(1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1000.0, 999.0, 998.0], 1, 3));
        let y = act.forward(&tape, x);
        let v = tape.value(y.idx());
        assert!(
            v.data.iter().all(|f| f.is_finite()),
            "log_softmax should be finite with large logits, got {:?}",
            v.data
        );
    }

    #[test]
    fn softmax_gradient_flows() {
        let mut act = Softmax::new(1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let y = act.forward(&tape, x);
        // sum(softmax) == 1 (constant), so gradient would be zero.
        // Use a weighted sum to get non-zero gradients.
        let weights = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let loss = y.hadamard(weights).sum();
        loss.backward();
        let g = tape.grad(x.idx());
        assert!(
            g.data.iter().all(|v| v.abs() > 1e-6),
            "softmax gradient is zero"
        );
    }

    #[test]
    fn log_softmax_gradient_flows() {
        let mut act = LogSoftmax::new(1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let y = act.forward(&tape, x);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x.idx());
        assert!(
            g.data.iter().all(|v| v.abs() > 1e-6),
            "log_softmax gradient is zero"
        );
    }
}
