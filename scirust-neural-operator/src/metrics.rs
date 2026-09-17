use crate::{NeuralOperatorError, Result, relative_l2};

/// Accuracy report for a surrogate against an exact/reference solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatorMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub relative_l2: f64,
    pub max_abs: f64,
}

impl OperatorMetrics {
    pub fn from_fields(prediction: &[f32], target: &[f32]) -> Result<Self> {
        if prediction.len() != target.len()
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "operator metrics",
                expected: target.len(),
                got: prediction.len(),
            });
        }
        if prediction.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "operator metrics field",
            });
        }
        let mut abs_sum = 0.0;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0f64;
        for (&p, &t) in prediction.iter().zip(target)
        {
            let e = (p - t) as f64;
            abs_sum += e.abs();
            sq_sum += e * e;
            max_abs = max_abs.max(e.abs());
        }
        let n = prediction.len() as f64;
        Ok(Self {
            mae: abs_sum / n,
            rmse: (sq_sum / n).sqrt(),
            relative_l2: relative_l2(prediction, target)?,
            max_abs,
        })
    }
}

/// Amortisation model for replacing repeated exact simulations with a trained surrogate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurrogateEconomics {
    pub exact_cost_per_eval: f64,
    pub surrogate_cost_per_eval: f64,
    pub training_cost: f64,
}

impl SurrogateEconomics {
    pub fn new(
        exact_cost_per_eval: f64,
        surrogate_cost_per_eval: f64,
        training_cost: f64,
    ) -> Result<Self> {
        if [exact_cost_per_eval, surrogate_cost_per_eval, training_cost]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(NeuralOperatorError::InvalidCost);
        }
        Ok(Self {
            exact_cost_per_eval,
            surrogate_cost_per_eval,
            training_cost,
        })
    }
    pub fn online_speedup(self) -> f64 {
        if self.surrogate_cost_per_eval == 0.0
        {
            f64::INFINITY
        }
        else
        {
            self.exact_cost_per_eval / self.surrogate_cost_per_eval
        }
    }
    pub fn break_even_evaluations(self) -> Option<u64> {
        let saving = self.exact_cost_per_eval - self.surrogate_cost_per_eval;
        if saving <= 0.0
        {
            return None;
        }
        Some((self.training_cost / saving).ceil().max(0.0) as u64)
    }
    pub fn total_exact_cost(self, evaluations: u64) -> f64 {
        evaluations as f64 * self.exact_cost_per_eval
    }
    pub fn total_surrogate_cost(self, evaluations: u64) -> f64 {
        self.training_cost + evaluations as f64 * self.surrogate_cost_per_eval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn economics_computes_break_even() {
        let e = SurrogateEconomics::new(10.0, 0.1, 99.0).unwrap();
        assert_eq!(e.break_even_evaluations(), Some(10));
        assert!((e.online_speedup() - 100.0).abs() < 1e-12);
    }
}
