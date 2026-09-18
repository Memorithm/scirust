//! Mini-batch DataLoader with deterministic shuffling and a prefetch configuration hint.
//!
//! Builds on the `Dataset` trait to provide ergonomic mini-batch iteration
//! with optional shuffle via Fisher-Yates and deterministic seeding.
//!
//! # Example
//!
//! ```
//! use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
//!
//! let dataset = InMemoryDataset::new(
//!     vec![1.0, 2.0, 3.0, 4.0],
//!     vec![0.0, 1.0],
//!     2,
//!     1,
//! );
//! let loader = DataLoader::builder(dataset)
//!     .batch_size(1)
//!     .shuffle(false)
//!     .build();
//! let batches: Vec<_> = loader.collect();
//! assert_eq!(batches.len(), 2);
//! assert_eq!(batches[0].0, vec![1.0, 2.0]);
//! ```

use crate::data::Dataset;
use crate::nn::rng::PcgEngine;

/// Configuration for a DataLoader.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Number of samples per mini-batch.
    pub batch_size: usize,
    /// Whether to shuffle indices at the start of each epoch.
    pub shuffle: bool,
    /// Random seed for reproducible shuffling.
    pub seed: u64,
    /// Number of batches to prefetch (0 = no prefetch).
    pub prefetch: usize,
    /// Drop the last incomplete batch if true.
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            seed: 42,
            prefetch: 0,
            drop_last: false,
        }
    }
}

/// Builder for DataLoader.
pub struct DataLoaderBuilder<D: Dataset> {
    dataset: D,
    config: DataLoaderConfig,
}

impl<D: Dataset> DataLoaderBuilder<D> {
    /// Start a builder with [`DataLoaderConfig::default`].
    ///
    /// The dataset is moved into the builder and later into the constructed loader.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoaderBuilder};
    /// let ds = InMemoryDataset::new(vec![1.0], vec![2.0], 1, 1);
    /// let loader = DataLoaderBuilder::new(ds).shuffle(false).build();
    /// assert_eq!(loader.n_batches(), 1);
    /// ```
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            config: DataLoaderConfig::default(),
        }
    }

    /// Set the number of samples emitted per mini-batch.
    ///
    /// `n` must be strictly positive. The last batch may contain fewer samples
    /// unless [`Self::drop_last`] is enabled.
    ///
    /// # Panics
    ///
    /// Panics when `n == 0`; a zero-sized batch cannot advance the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0], 1, 1);
    /// let loader = DataLoader::builder(ds).batch_size(2).shuffle(false).build();
    /// assert_eq!(loader.n_batches(), 2);
    /// ```
    pub fn batch_size(mut self, n: usize) -> Self {
        assert!(
            n > 0,
            "DataLoaderBuilder::batch_size: batch size must be > 0"
        );
        self.config.batch_size = n;
        self
    }

    /// Enable or disable Fisher-Yates shuffling at each [`DataLoader::reset`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0], vec![0.0, 1.0], 1, 1);
    /// let batches: Vec<_> = DataLoader::builder(ds).shuffle(false).batch_size(1).build().collect();
    /// assert_eq!(batches[0].0, vec![1.0]);
    /// assert_eq!(batches[1].0, vec![2.0]);
    /// ```
    pub fn shuffle(mut self, yes: bool) -> Self {
        self.config.shuffle = yes;
        self
    }

    /// Set the deterministic PCG seed used by epoch shuffling.
    ///
    /// Two loaders with identical datasets, configuration and seed generate the
    /// same shuffled order. Each reset advances the loader's RNG state.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0, 3.0], vec![0.0; 3], 1, 1);
    /// let a: Vec<_> = DataLoader::builder(ds.clone()).seed(7).batch_size(3).build().collect();
    /// let b: Vec<_> = DataLoader::builder(ds).seed(7).batch_size(3).build().collect();
    /// assert_eq!(a, b);
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Record the requested prefetch depth in the loader configuration.
    ///
    /// The current iterator is synchronous: this value is a forward-compatible
    /// scheduling hint and does not spawn a background prefetch worker yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0], vec![0.0], 1, 1);
    /// let batches: Vec<_> = DataLoader::builder(ds).prefetch(2).shuffle(false).build().collect();
    /// assert_eq!(batches.len(), 1);
    /// ```
    pub fn prefetch(mut self, n: usize) -> Self {
        self.config.prefetch = n;
        self
    }

    /// Choose whether an incomplete final mini-batch is omitted.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0, 3.0], vec![0.0; 3], 1, 1);
    /// let loader = DataLoader::builder(ds).batch_size(2).drop_last(true).shuffle(false).build();
    /// assert_eq!(loader.n_batches(), 1);
    /// assert_eq!(loader.count(), 1);
    /// ```
    pub fn drop_last(mut self, yes: bool) -> Self {
        self.config.drop_last = yes;
        self
    }

    /// Build a loader and seed its deterministic shuffle engine.
    ///
    /// Indices are initialized lazily on the first call to [`Iterator::next`] or
    /// explicitly by [`DataLoader::reset`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0], vec![0.0, 1.0], 1, 1);
    /// let mut loader = DataLoader::builder(ds).batch_size(1).shuffle(false).build();
    /// assert_eq!(loader.next().unwrap().0, vec![1.0]);
    /// ```
    pub fn build(self) -> DataLoader<D> {
        let seed = self.config.seed;
        DataLoader {
            dataset: self.dataset,
            config: self.config,
            indices: Vec::new(),
            current: 0,
            rng: PcgEngine::new(seed),
        }
    }
}

/// Iterable DataLoader yielding mini-batches of `(features, labels)`.
pub struct DataLoader<D: Dataset> {
    dataset: D,
    config: DataLoaderConfig,
    indices: Vec<usize>,
    current: usize,
    rng: PcgEngine,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a [`DataLoaderBuilder`] for `dataset`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0], vec![0.0, 1.0], 1, 1);
    /// let loader = DataLoader::builder(ds).batch_size(2).shuffle(false).build();
    /// assert_eq!(loader.n_batches(), 1);
    /// ```
    pub fn builder(dataset: D) -> DataLoaderBuilder<D> {
        DataLoaderBuilder::new(dataset)
    }

    /// Reset the loader for a new epoch (re-shuffles if configured).
    ///
    /// With shuffling disabled, a reset reproduces the same sample order. With
    /// shuffling enabled the same RNG instance advances to the next deterministic
    /// epoch permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0], vec![0.0, 1.0], 1, 1);
    /// let mut loader = DataLoader::builder(ds).batch_size(1).shuffle(false).build();
    /// assert_eq!(loader.next().unwrap().0, vec![1.0]);
    /// loader.reset();
    /// assert_eq!(loader.next().unwrap().0, vec![1.0]);
    /// ```
    pub fn reset(&mut self) {
        let n = self.dataset.n_samples();
        self.indices = (0..n).collect();

        if self.config.shuffle
        {
            // Fisher-Yates shuffle
            for i in (1..self.indices.len()).rev()
            {
                let j = (self.rng.next_u32() as usize) % (i + 1);
                self.indices.swap(i, j);
            }
        }

        self.current = 0;
    }

    /// Return the number of mini-batches in one epoch.
    ///
    /// The result is a ceiling division when incomplete batches are retained and
    /// a floor division when [`DataLoaderBuilder::drop_last`] is enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::data::{InMemoryDataset, dataloader::DataLoader};
    /// let ds = InMemoryDataset::new(vec![1.0, 2.0, 3.0], vec![0.0; 3], 1, 1);
    /// let keep = DataLoader::builder(ds.clone()).batch_size(2).shuffle(false).build();
    /// let drop = DataLoader::builder(ds).batch_size(2).drop_last(true).shuffle(false).build();
    /// assert_eq!(keep.n_batches(), 2);
    /// assert_eq!(drop.n_batches(), 1);
    /// ```
    pub fn n_batches(&self) -> usize {
        let n = self.dataset.n_samples();
        let bs = self.config.batch_size;
        if self.config.drop_last
        {
            n / bs
        }
        else
        {
            n.div_ceil(bs)
        }
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Vec<f32>, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.dataset.n_samples();
        let bs = self.config.batch_size;

        // Initialize on first call
        if self.indices.is_empty()
        {
            self.reset();
        }

        if self.current >= n
        {
            return None;
        }

        let remaining = n - self.current;
        let actual_bs = if self.config.drop_last && remaining < bs
        {
            return None;
        }
        else
        {
            remaining.min(bs)
        };

        // Determine feature/label dimensions from the first sample
        let (x0, y0) = self.dataset.sample(self.indices[self.current]);
        let in_dim = x0.len();
        let out_dim = y0.len();

        let mut batch_x = Vec::with_capacity(actual_bs * in_dim);
        let mut batch_y = Vec::with_capacity(actual_bs * out_dim);

        for k in 0..actual_bs
        {
            let (x, y) = self.dataset.sample(self.indices[self.current + k]);
            debug_assert_eq!(x.len(), in_dim, "inconsistent feature dimension");
            debug_assert_eq!(y.len(), out_dim, "inconsistent label dimension");
            batch_x.extend_from_slice(x);
            batch_y.extend_from_slice(y);
        }

        self.current += actual_bs;
        Some((batch_x, batch_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::InMemoryDataset;

    fn make_dataset(n: usize, in_dim: usize, out_dim: usize) -> InMemoryDataset {
        let xs: Vec<f32> = (0..(n * in_dim)).map(|i| i as f32).collect();
        let ys: Vec<f32> = (0..(n * out_dim)).map(|i| i as f32 * 0.1).collect();
        InMemoryDataset::new(xs, ys, in_dim, out_dim)
    }

    #[test]
    fn test_dataloader_basic() {
        let dataset = make_dataset(10, 3, 2);
        let loader = DataLoader::builder(dataset)
            .batch_size(3)
            .shuffle(false)
            .build();

        let batches: Vec<_> = loader.collect();
        // 10 samples, batch_size=3 → ceil(10/3) = 4 batches: 3+3+3+1
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].0.len(), 3 * 3);
        assert_eq!(batches[0].1.len(), 3 * 2);
        assert_eq!(batches[3].0.len(), 3);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let dataset = make_dataset(5, 3, 2);
        let loader = DataLoader::builder(dataset)
            .batch_size(2)
            .shuffle(false)
            .drop_last(true)
            .build();

        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 2); // floor(5/2)
    }

    #[test]
    fn test_dataloader_shuffle_deterministic() {
        let dataset1 = make_dataset(10, 3, 2);
        let dataset2 = make_dataset(10, 3, 2);

        let mut loader1 = DataLoader::builder(dataset1)
            .batch_size(5)
            .shuffle(true)
            .seed(123)
            .build();
        let mut loader2 = DataLoader::builder(dataset2)
            .batch_size(5)
            .shuffle(true)
            .seed(123)
            .build();

        loader1.reset();
        loader2.reset();

        let b1 = loader1.next().unwrap();
        let b2 = loader2.next().unwrap();
        assert_eq!(b1.0, b2.0);
        assert_eq!(b1.1, b2.1);
    }

    #[test]
    #[should_panic(expected = "batch size must be > 0")]
    fn builder_rejects_zero_batch_size() {
        let dataset = make_dataset(2, 1, 1);
        let _ = DataLoader::builder(dataset).batch_size(0);
    }

    #[test]
    fn test_dataloader_reset() {
        let dataset = make_dataset(8, 3, 2);
        let mut loader = DataLoader::builder(dataset)
            .batch_size(3)
            .shuffle(false)
            .build();

        // Collect first pass into a vec
        let mut first_pass = Vec::new();
        for batch in loader.by_ref()
        {
            first_pass.push(batch);
        }

        loader.reset();

        let mut second_pass = Vec::new();
        for batch in loader.by_ref()
        {
            second_pass.push(batch);
        }

        assert_eq!(first_pass.len(), second_pass.len());
    }
}
