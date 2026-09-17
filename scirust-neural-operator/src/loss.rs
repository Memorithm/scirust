use crate::{NeuralOperatorError, Result};

/// Absolute/relative discrete Lp loss for operator fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpLoss {
    p: u32,
    relative: bool,
}

impl LpLoss {
    /// Construct an absolute discrete Lp loss for `p >= 1`.
    pub fn absolute(p: u32) -> Result<Self> {
        Self::new(p, false)
    }
    /// Construct a target-relative discrete Lp loss for `p >= 1`.
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

    /// Evaluate the loss with finite-input checks and overflow-safe scaled accumulation.
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
        validate_finite("LpLoss prediction", prediction)?;
        validate_finite("LpLoss target", target)?;
        let p = self.p as f64;
        let err = stable_lp_norm(prediction.len(), p, |index| {
            ((prediction[index] as f64) - (target[index] as f64)).abs()
        });
        if !self.relative
        {
            return Ok(err);
        }
        let denom = stable_lp_norm(target.len(), p, |index| (target[index] as f64).abs());
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

fn validate_finite(what: &'static str, values: &[f32]) -> Result<()> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(NeuralOperatorError::NonFinite { what, index });
    }
    Ok(())
}

fn stable_lp_norm<F>(len: usize, p: f64, magnitude_at: F) -> f64
where
    F: Fn(usize) -> f64,
{
    let mut scale = 0.0_f64;
    for index in 0..len
    {
        scale = scale.max(magnitude_at(index));
    }
    if scale == 0.0
    {
        return 0.0;
    }

    let mut normalized_power_sum = 0.0_f64;
    for index in 0..len
    {
        normalized_power_sum += (magnitude_at(index) / scale).powf(p);
    }
    scale * normalized_power_sum.powf(1.0 / p)
}

/// Evaluate the discrete target-relative L2 loss.
pub fn relative_l2(prediction: &[f32], target: &[f32]) -> Result<f64> {
    LpLoss::relative(2)?.evaluate(prediction, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_p_norm_uses_overflow_safe_scaling() {
        let absolute = LpLoss::absolute(2048)
            .unwrap()
            .evaluate(&[2.0], &[0.0])
            .unwrap();
        assert_eq!(absolute, 2.0);

        let relative = LpLoss::relative(2048)
            .unwrap()
            .evaluate(&[4.0], &[2.0])
            .unwrap();
        assert_eq!(relative, 1.0);
    }

    #[test]
    fn lp_loss_rejects_non_finite_fields() {
        assert!(matches!(
            LpLoss::absolute(2).unwrap().evaluate(&[f32::NAN], &[0.0]),
            Err(NeuralOperatorError::NonFinite {
                what: "LpLoss prediction",
                index: 0
            })
        ));
        assert!(matches!(
            LpLoss::relative(2)
                .unwrap()
                .evaluate(&[0.0], &[f32::INFINITY]),
            Err(NeuralOperatorError::NonFinite {
                what: "LpLoss target",
                index: 0
            })
        ));
    }

    #[test]
    fn relative_l2_matches_hand_calculation() {
        let got = relative_l2(&[2.0, 0.0], &[1.0, 0.0]).unwrap();
        assert!((got - 1.0).abs() < 1e-15);
    }
}
