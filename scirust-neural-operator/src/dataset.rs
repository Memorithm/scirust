use crate::{NeuralOperatorError, Result};

/// One function-to-function training example sampled on a regular 1-D grid.
/// Data are row-major `(points, channels)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorSample1d {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

impl OperatorSample1d {
    pub fn new(input: Vec<f32>, target: Vec<f32>) -> Result<Self> {
        validate_finite("operator input", &input)?;
        validate_finite("operator target", &target)?;
        if input.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "operator input",
            });
        }
        if target.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "operator target",
            });
        }
        Ok(Self { input, target })
    }
}

fn validate_finite(what: &'static str, values: &[f32]) -> Result<()> {
    if let Some((index, _)) = values.iter().enumerate().find(|(_, v)| !v.is_finite())
    {
        return Err(NeuralOperatorError::NonFinite { what, index });
    }
    Ok(())
}

/// Shape-checked collection of 1-D operator samples.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorDataset1d {
    samples: Vec<OperatorSample1d>,
    points: usize,
    in_channels: usize,
    out_channels: usize,
}

impl OperatorDataset1d {
    pub fn new(
        samples: Vec<OperatorSample1d>,
        points: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        if samples.is_empty()
        {
            return Err(NeuralOperatorError::EmptyDataset);
        }
        if in_channels == 0
        {
            return Err(NeuralOperatorError::InvalidChannels {
                channels: in_channels,
            });
        }
        if out_channels == 0
        {
            return Err(NeuralOperatorError::InvalidChannels {
                channels: out_channels,
            });
        }
        let expected_in = points.saturating_mul(in_channels);
        let expected_out = points.saturating_mul(out_channels);
        for (sample, item) in samples.iter().enumerate()
        {
            if item.input.len() != expected_in
            {
                return Err(NeuralOperatorError::DatasetShapeMismatch {
                    sample,
                    what: "input",
                    expected: expected_in,
                    got: item.input.len(),
                });
            }
            if item.target.len() != expected_out
            {
                return Err(NeuralOperatorError::DatasetShapeMismatch {
                    sample,
                    what: "target",
                    expected: expected_out,
                    got: item.target.len(),
                });
            }
        }
        Ok(Self {
            samples,
            points,
            in_channels,
            out_channels,
        })
    }

    pub fn samples(&self) -> &[OperatorSample1d] {
        &self.samples
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    pub fn points(&self) -> usize {
        self.points
    }
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_rejects_shape_mismatch() {
        let s = OperatorSample1d::new(vec![0.0; 8], vec![0.0; 4]).unwrap();
        let e = OperatorDataset1d::new(vec![s], 4, 2, 2).unwrap_err();
        assert!(matches!(
            e,
            NeuralOperatorError::DatasetShapeMismatch { what: "target", .. }
        ));
    }
}
