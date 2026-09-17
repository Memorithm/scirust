use crate::{NeuralOperatorError, Result};

/// Absolute/relative discrete Lp loss for operator fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpLoss {
    p: u32,
    relative: bool,
}

impl LpLoss {
    pub fn absolute(p: u32) -> Result<Self> {
        Self::new(p, false)
    }
    pub fn relative(p: u32) -> Result<Self> {
        Self::new(p, true)
    }

    fn new(p: u32, relative: bool) -> Result<Self> {
        if p == 0
        {
            return Err(NeuralOperatorError::InvalidP { p });
        }
        Ok(Self { p, relative })
    }

    pub fn evaluate(self, prediction: &[f32], target: &[f32]) -> Result<f64> {
        if prediction.len() != target.len()
        {
            return Err(NeuralOperatorError::ShapeMismatch {
                what: "LpLoss",
                expected: target.len(),
                got: prediction.len(),
            });
        }
        if prediction.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "LpLoss field",
            });
        }
        let p = self.p as f64;
        let err_sum = prediction
            .iter()
            .zip(target)
            .map(|(&a, &b)| ((a - b) as f64).abs().powf(p))
            .sum::<f64>();
        let err = err_sum.powf(1.0 / p);
        if !self.relative
        {
            return Ok(err);
        }
        let denom = target
            .iter()
            .map(|&x| (x as f64).abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p);
        Ok(
            if denom > 0.0
            {
                err / denom
            }
            else if err == 0.0
            {
                0.0
            }
            else
            {
                f64::INFINITY
            },
        )
    }
}

pub fn relative_l2(prediction: &[f32], target: &[f32]) -> Result<f64> {
    LpLoss::relative(2)?.evaluate(prediction, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_l2_matches_hand_calculation() {
        let got = relative_l2(&[2.0, 0.0], &[1.0, 0.0]).unwrap();
        assert!((got - 1.0).abs() < 1e-15);
    }
}
