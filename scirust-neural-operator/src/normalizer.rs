use crate::{NeuralOperatorError, Result};

/// Per-channel affine normalizer for row-major `(points, channels)` fields.
/// Statistics are accumulated in `f64` and the transform is applied in `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelNormalizer {
    channels: usize,
    epsilon: f64,
    mean: Vec<f64>,
    std: Vec<f64>,
    fitted: bool,
}

impl ChannelNormalizer {
    /// Construct an unfitted per-channel normalizer with a positive finite epsilon.
    pub fn new(channels: usize, epsilon: f64) -> Result<Self> {
        if channels == 0
        {
            return Err(NeuralOperatorError::InvalidChannels { channels });
        }
        if !epsilon.is_finite() || epsilon <= 0.0
        {
            return Err(NeuralOperatorError::DegenerateChannel { channel: 0 });
        }
        Ok(Self {
            channels,
            epsilon,
            mean: vec![0.0; channels],
            std: vec![1.0; channels],
            fitted: false,
        })
    }

    /// Fit per-channel mean and population standard deviation across all rows.
    pub fn fit<'a, I>(&mut self, fields: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        let mut count = vec![0usize; self.channels];
        let mut mean = vec![0.0f64; self.channels];
        let mut m2 = vec![0.0f64; self.channels];
        let mut any = false;
        for field in fields
        {
            any = true;
            if !field.len().is_multiple_of(self.channels)
            {
                return Err(NeuralOperatorError::ChannelMismatch {
                    width: field.len(),
                    channels: self.channels,
                });
            }
            for (i, &value) in field.iter().enumerate()
            {
                if !value.is_finite()
                {
                    return Err(NeuralOperatorError::NonFinite {
                        what: "normalizer field",
                        index: i,
                    });
                }
                let c = i % self.channels;
                count[c] += 1;
                let x = value as f64;
                let delta = x - mean[c];
                mean[c] += delta / count[c] as f64;
                let delta2 = x - mean[c];
                m2[c] += delta * delta2;
            }
        }
        if !any
        {
            return Err(NeuralOperatorError::EmptyDataset);
        }
        let mut std = vec![0.0; self.channels];
        for c in 0..self.channels
        {
            std[c] = (m2[c] / count[c] as f64).sqrt();
            if !std[c].is_finite() || std[c] < self.epsilon
            {
                return Err(NeuralOperatorError::DegenerateChannel { channel: c });
            }
        }
        self.mean = mean;
        self.std = std;
        self.fitted = true;
        Ok(())
    }

    /// Standardize a row-major field using fitted per-channel statistics.
    pub fn encode(&self, field: &[f32]) -> Result<Vec<f32>> {
        self.transform(field, false)
    }
    /// Invert standardization using the fitted per-channel statistics.
    pub fn decode(&self, field: &[f32]) -> Result<Vec<f32>> {
        self.transform(field, true)
    }

    fn transform(&self, field: &[f32], inverse: bool) -> Result<Vec<f32>> {
        if !self.fitted
        {
            return Err(NeuralOperatorError::NormalizerNotFitted);
        }
        if !field.len().is_multiple_of(self.channels)
        {
            return Err(NeuralOperatorError::ChannelMismatch {
                width: field.len(),
                channels: self.channels,
            });
        }
        if let Some((index, _)) = field
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NeuralOperatorError::NonFinite {
                what: "normalizer field",
                index,
            });
        }
        let mut out = Vec::with_capacity(field.len());
        for (i, &x) in field.iter().enumerate()
        {
            let c = i % self.channels;
            let y = if inverse
            {
                x as f64 * self.std[c] + self.mean[c]
            }
            else
            {
                (x as f64 - self.mean[c]) / self.std[c]
            };
            out.push(y as f32);
        }
        Ok(out)
    }

    /// Return fitted per-channel means, or fail if fitting has not completed.
    pub fn mean(&self) -> Result<&[f64]> {
        self.fitted
            .then_some(self.mean.as_slice())
            .ok_or(NeuralOperatorError::NormalizerNotFitted)
    }
    /// Return fitted population standard deviations, or fail before fitting.
    pub fn std(&self) -> Result<&[f64]> {
        self.fitted
            .then_some(self.std.as_slice())
            .ok_or(NeuralOperatorError::NormalizerNotFitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitted_normalizer_rejects_non_finite_transform_inputs() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut normalizer = ChannelNormalizer::new(1, 1e-12).unwrap();
        normalizer.fit([data.as_slice()]).unwrap();
        assert!(matches!(
            normalizer.encode(&[f32::NAN]),
            Err(NeuralOperatorError::NonFinite {
                what: "normalizer field",
                index: 0
            })
        ));
        assert!(matches!(
            normalizer.decode(&[f32::INFINITY]),
            Err(NeuralOperatorError::NonFinite {
                what: "normalizer field",
                index: 0
            })
        ));
    }

    #[test]
    fn channel_normalizer_round_trips() {
        let a = vec![1.0f32, 10.0, 3.0, 14.0];
        let b = vec![5.0f32, 18.0, 7.0, 22.0];
        let mut n = ChannelNormalizer::new(2, 1e-12).unwrap();
        n.fit([a.as_slice(), b.as_slice()]).unwrap();
        let encoded = n.encode(&a).unwrap();
        let decoded = n.decode(&encoded).unwrap();
        for (x, y) in a.iter().zip(decoded)
        {
            assert!((*x - y).abs() < 1e-5);
        }
    }
}
