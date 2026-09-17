use crate::{NeuralOperatorError, Result};

/// One function-to-function training example sampled on a regular 1-D grid.
/// Data are row-major `(points, channels)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorSample1d {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

impl OperatorSample1d {
    /// Construct a finite, non-empty operator-learning sample.
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
    /// Construct a shape-checked dataset and revalidate every sample as finite.
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
            validate_finite("operator input", &item.input)?;
            validate_finite("operator target", &item.target)?;
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

    /// Return the samples in deterministic dataset order.
    pub fn samples(&self) -> &[OperatorSample1d] {
        &self.samples
    }
    /// Return the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    /// Return whether the dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    /// Return the number of grid points per sample.
    pub fn points(&self) -> usize {
        self.points
    }
    /// Return the input-channel count per grid point.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    /// Return the output-channel count per grid point.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

/// One function-to-function training example sampled on a regular 2-D grid.
/// Data are row-major `(rows, cols, channels)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorSample2d {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

impl OperatorSample2d {
    /// Construct a finite, non-empty 2-D operator-learning sample.
    pub fn new(input: Vec<f32>, target: Vec<f32>) -> Result<Self> {
        validate_finite("2-D operator input", &input)?;
        validate_finite("2-D operator target", &target)?;
        if input.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "2-D operator input",
            });
        }
        if target.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "2-D operator target",
            });
        }
        Ok(Self { input, target })
    }
}

/// Shape-checked collection of 2-D operator samples.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorDataset2d {
    samples: Vec<OperatorSample2d>,
    rows: usize,
    cols: usize,
    in_channels: usize,
    out_channels: usize,
}

impl OperatorDataset2d {
    /// Construct a finite 2-D dataset with fixed spatial/channel shape.
    pub fn new(
        samples: Vec<OperatorSample2d>,
        rows: usize,
        cols: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        if samples.is_empty()
        {
            return Err(NeuralOperatorError::EmptyDataset);
        }
        if rows == 0 || cols == 0
        {
            return Err(NeuralOperatorError::Empty {
                what: "2-D operator grid",
            });
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
        let points = rows.saturating_mul(cols);
        let expected_in = points.saturating_mul(in_channels);
        let expected_out = points.saturating_mul(out_channels);
        for (sample, item) in samples.iter().enumerate()
        {
            validate_finite("2-D operator input", &item.input)?;
            validate_finite("2-D operator target", &item.target)?;
            if item.input.len() != expected_in
            {
                return Err(NeuralOperatorError::DatasetShapeMismatch {
                    sample,
                    what: "2-D input",
                    expected: expected_in,
                    got: item.input.len(),
                });
            }
            if item.target.len() != expected_out
            {
                return Err(NeuralOperatorError::DatasetShapeMismatch {
                    sample,
                    what: "2-D target",
                    expected: expected_out,
                    got: item.target.len(),
                });
            }
        }
        Ok(Self {
            samples,
            rows,
            cols,
            in_channels,
            out_channels,
        })
    }

    /// Return samples in deterministic declaration order.
    pub fn samples(&self) -> &[OperatorSample2d] {
        &self.samples
    }
    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    /// Whether the dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    /// Number of rows in each sampled field.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns in each sampled field.
    pub const fn cols(&self) -> usize {
        self.cols
    }
    /// Input-channel count per spatial point.
    pub const fn in_channels(&self) -> usize {
        self.in_channels
    }
    /// Output-channel count per spatial point.
    pub const fn out_channels(&self) -> usize {
        self.out_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_revalidates_mutable_sample_finiteness() {
        let mut sample = OperatorSample1d::new(vec![0.0; 4], vec![0.0; 4]).unwrap();
        sample.input[2] = f32::NAN;
        assert_eq!(
            OperatorDataset1d::new(vec![sample], 4, 1, 1).unwrap_err(),
            NeuralOperatorError::NonFinite {
                what: "operator input",
                index: 2,
            }
        );

        let mut sample = OperatorSample1d::new(vec![0.0; 4], vec![0.0; 4]).unwrap();
        sample.target[1] = f32::INFINITY;
        assert_eq!(
            OperatorDataset1d::new(vec![sample], 4, 1, 1).unwrap_err(),
            NeuralOperatorError::NonFinite {
                what: "operator target",
                index: 1,
            }
        );
    }

    #[test]
    fn dataset_rejects_shape_mismatch() {
        let s = OperatorSample1d::new(vec![0.0; 8], vec![0.0; 4]).unwrap();
        let e = OperatorDataset1d::new(vec![s], 4, 2, 2).unwrap_err();
        assert!(matches!(
            e,
            NeuralOperatorError::DatasetShapeMismatch { what: "target", .. }
        ));
    }

    #[test]
    fn dataset2d_rejects_shape_mismatch() {
        let sample = OperatorSample2d::new(vec![0.0; 12], vec![0.0; 6]).unwrap();
        let error = OperatorDataset2d::new(vec![sample], 2, 3, 2, 2).unwrap_err();
        assert!(matches!(
            error,
            NeuralOperatorError::DatasetShapeMismatch {
                what: "2-D target",
                ..
            }
        ));
    }
}
