//! Demonstration data-parallel training over [`ParallelTape`].
//!
//! This module exists primarily to exercise deterministic worker-order gradient
//! reduction. It is not the production training API. Each worker owns one
//! [`ParallelTape`]; sequential and threaded batches collect a gradient vector
//! per worker and reduce workers in ascending worker-index order.
//!
//! Deterministic reduction here means that, for deterministic per-worker work,
//! identical worker gradient shapes, and identical worker-index ordering, the
//! aggregation order is independent of OS-thread completion order. It does not
//! make nondeterministic user closures, external state, or backend kernels
//! deterministic.

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::parallel::ParallelTape;

/// Length mismatch between one worker gradient and worker 0's gradient shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientLengthMismatch {
    /// Zero-based worker whose vector length differed.
    pub worker: usize,
    /// Required gradient-vector length, defined by worker 0.
    pub expected: usize,
    /// Actual vector length returned by `worker`.
    pub got: usize,
}

impl fmt::Display for GradientLengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "worker {} gradient length mismatch: expected {}, got {}",
            self.worker, self.expected, self.got
        )
    }
}

impl std::error::Error for GradientLengthMismatch {}

/// Deterministic worker-order gradient aggregation helpers.
pub struct GradientAggregator;

impl GradientAggregator {
    fn validate_lengths(grads: &[Vec<f64>]) -> Result<(), GradientLengthMismatch> {
        let Some(first) = grads.first() else {
            return Ok(());
        };
        let expected = first.len();
        for (worker, gradient) in grads.iter().enumerate().skip(1) {
            if gradient.len() != expected {
                return Err(GradientLengthMismatch {
                    worker,
                    expected,
                    got: gradient.len(),
                });
            }
        }
        Ok(())
    }

    /// Returns the element-wise worker sum after validating all vector lengths.
    ///
    /// Workers are accumulated in slice order. Empty input returns an empty
    /// vector.
    ///
    /// # Errors
    ///
    /// Returns [`GradientLengthMismatch`] when a worker vector differs in length
    /// from worker 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// let sum = GradientAggregator::try_reduce_sum(&[
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0],
    /// ]).unwrap();
    /// assert_eq!(sum, vec![4.0, 6.0]);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// let err = GradientAggregator::try_reduce_sum(&[
    ///     vec![1.0, 2.0],
    ///     vec![3.0],
    /// ]).unwrap_err();
    /// assert_eq!((err.worker, err.expected, err.got), (1, 2, 1));
    /// ```
    pub fn try_reduce_sum(grads: &[Vec<f64>]) -> Result<Vec<f64>, GradientLengthMismatch> {
        Self::validate_lengths(grads)?;
        let Some(first) = grads.first() else {
            return Ok(Vec::new());
        };
        let mut result = vec![0.0; first.len()];
        for worker_grads in grads {
            for (dst, &value) in result.iter_mut().zip(worker_grads) {
                *dst += value;
            }
        }
        Ok(result)
    }

    /// Returns the element-wise worker sum.
    ///
    /// Empty input returns an empty vector. Prefer [`Self::try_reduce_sum`] at
    /// public/fallible boundaries.
    ///
    /// # Panics
    ///
    /// Panics in both debug and release builds when worker gradient lengths do
    /// not match.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// assert_eq!(
    ///     GradientAggregator::reduce_sum(&[vec![1.0], vec![2.0]]),
    ///     vec![3.0]
    /// );
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// assert!(GradientAggregator::reduce_sum(&[]).is_empty());
    /// ```
    pub fn reduce_sum(grads: &[Vec<f64>]) -> Vec<f64> {
        Self::try_reduce_sum(grads).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Returns the element-wise arithmetic mean after shape validation.
    ///
    /// Empty input returns an empty vector. Worker accumulation order is fixed
    /// by the input slice order.
    ///
    /// # Errors
    ///
    /// Returns [`GradientLengthMismatch`] when worker vector lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// let mean = GradientAggregator::try_reduce_mean(&[
    ///     vec![2.0, 4.0],
    ///     vec![4.0, 6.0],
    /// ]).unwrap();
    /// assert_eq!(mean, vec![3.0, 5.0]);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// assert!(GradientAggregator::try_reduce_mean(&[]).unwrap().is_empty());
    /// ```
    pub fn try_reduce_mean(grads: &[Vec<f64>]) -> Result<Vec<f64>, GradientLengthMismatch> {
        let mut result = Self::try_reduce_sum(grads)?;
        if !grads.is_empty() {
            let denominator = grads.len() as f64;
            for value in &mut result {
                *value /= denominator;
            }
        }
        Ok(result)
    }

    /// Returns the element-wise arithmetic mean.
    ///
    /// # Panics
    ///
    /// Panics in both debug and release builds when worker gradient lengths do
    /// not match. Prefer [`Self::try_reduce_mean`] when mismatch is recoverable.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// assert_eq!(
    ///     GradientAggregator::reduce_mean(&[vec![2.0], vec![4.0]]),
    ///     vec![3.0]
    /// );
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::GradientAggregator;
    /// assert!(GradientAggregator::reduce_mean(&[]).is_empty());
    /// ```
    pub fn reduce_mean(grads: &[Vec<f64>]) -> Vec<f64> {
        Self::try_reduce_mean(grads).unwrap_or_else(|error| panic!("{error}"))
    }
}

/// One [`ParallelTape`] per worker plus deterministic worker-order reduction.
///
/// The trainer is a proof/test harness, not a production optimizer. Tapes persist
/// across calls, so closures that append graph state are responsible for their
/// own reuse semantics.
pub struct DataParallelTrainer {
    n_workers: usize,
    tapes: Vec<ParallelTape>,
}

impl DataParallelTrainer {
    /// Creates `n_workers` empty worker tapes.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(4);
    /// assert_eq!(trainer.n_workers(), 4);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(0);
    /// assert_eq!(trainer.n_workers(), 0);
    /// ```
    #[must_use]
    pub fn new(n_workers: usize) -> Self {
        let tapes = (0..n_workers).map(|_| ParallelTape::new()).collect();
        Self { n_workers, tapes }
    }

    /// Runs all workers sequentially and returns their validated mean gradient.
    ///
    /// # Errors
    ///
    /// Returns [`GradientLengthMismatch`] if worker closures return vectors with
    /// different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(3);
    /// let mean = trainer.try_train_batch(|_, worker| vec![worker as f64]).unwrap();
    /// assert_eq!(mean, vec![1.0]);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(2);
    /// let err = trainer.try_train_batch(|_, worker| {
    ///     if worker == 0 { vec![1.0, 2.0] } else { vec![3.0] }
    /// }).unwrap_err();
    /// assert_eq!(err.worker, 1);
    /// ```
    pub fn try_train_batch<F>(&mut self, batch_fn: F) -> Result<Vec<f64>, GradientLengthMismatch>
    where
        F: Fn(&ParallelTape, usize) -> Vec<f64>,
    {
        let mut all_grads = Vec::with_capacity(self.n_workers);
        for worker in 0..self.n_workers {
            all_grads.push(batch_fn(&self.tapes[worker], worker));
        }
        GradientAggregator::try_reduce_mean(&all_grads)
    }

    /// Runs all workers sequentially and returns their mean gradient.
    ///
    /// # Panics
    ///
    /// Panics when workers return unequal gradient-vector lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(2);
    /// assert_eq!(trainer.train_batch(|_, w| vec![w as f64]), vec![0.5]);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(0);
    /// assert!(trainer.train_batch(|_, _| vec![1.0]).is_empty());
    /// ```
    pub fn train_batch<F>(&mut self, batch_fn: F) -> Vec<f64>
    where
        F: Fn(&ParallelTape, usize) -> Vec<f64>,
    {
        self.try_train_batch(batch_fn)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Runs workers on scoped OS threads and returns their validated mean.
    ///
    /// Worker scheduling uses an atomic index, but results are stored in
    /// worker-indexed slots and reduced in worker order. `n_threads == 0` is
    /// treated as one thread when at least one worker exists.
    ///
    /// # Errors
    ///
    /// Returns [`GradientLengthMismatch`] when worker result lengths differ.
    ///
    /// # Panics
    ///
    /// A panic in `batch_fn` propagates through `thread::scope`. Internal slot
    /// mutex poisoning also panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(4);
    /// let result = trainer.try_train_batch_threaded(2, |_, w| vec![w as f64]).unwrap();
    /// assert_eq!(result, vec![1.5]);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(2);
    /// let one = trainer.try_train_batch_threaded(1, |_, w| vec![w as f64, 1.0]).unwrap();
    /// let two = trainer.try_train_batch_threaded(2, |_, w| vec![w as f64, 1.0]).unwrap();
    /// assert_eq!(one, two);
    /// ```
    pub fn try_train_batch_threaded<F>(
        &self,
        n_threads: usize,
        batch_fn: F,
    ) -> Result<Vec<f64>, GradientLengthMismatch>
    where
        F: Fn(&ParallelTape, usize) -> Vec<f64> + Sync,
    {
        let n = self.n_workers;
        if n == 0 {
            return Ok(Vec::new());
        }
        let n_threads = n_threads.clamp(1, n);
        let slots: Vec<Mutex<Option<Vec<f64>>>> = (0..n).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _ in 0..n_threads {
                scope.spawn(|| loop {
                    let worker = next.fetch_add(1, Ordering::Relaxed);
                    if worker >= n {
                        break;
                    }
                    let gradient = batch_fn(&self.tapes[worker], worker);
                    *slots[worker]
                        .lock()
                        .expect("data-parallel slot poisoned") = Some(gradient);
                });
            }
        });

        let all_grads = slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .expect("data-parallel slot poisoned")
                    .expect("data-parallel worker did not run")
            })
            .collect::<Vec<_>>();
        GradientAggregator::try_reduce_mean(&all_grads)
    }

    /// Threaded compatibility wrapper around [`Self::try_train_batch_threaded`].
    ///
    /// # Panics
    ///
    /// Panics for unequal worker gradient lengths, a panicking worker closure,
    /// or poisoned internal slot mutexes.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(2);
    /// assert_eq!(trainer.train_batch_threaded(2, |_, w| vec![w as f64]), vec![0.5]);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(0);
    /// assert!(trainer.train_batch_threaded(8, |_, _| vec![1.0]).is_empty());
    /// ```
    pub fn train_batch_threaded<F>(&self, n_threads: usize, batch_fn: F) -> Vec<f64>
    where
        F: Fn(&ParallelTape, usize) -> Vec<f64> + Sync,
    {
        self.try_train_batch_threaded(n_threads, batch_fn)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Returns the worker tape when `worker` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(1);
    /// assert_eq!(trainer.try_tape(0).unwrap().num_nodes(), 0);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(1);
    /// assert!(trainer.try_tape(1).is_none());
    /// ```
    #[must_use]
    pub fn try_tape(&self, worker: usize) -> Option<&ParallelTape> {
        self.tapes.get(worker)
    }

    /// Returns the worker tape.
    ///
    /// # Panics
    ///
    /// Panics when `worker >= self.n_workers()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(1);
    /// assert_eq!(trainer.tape(0).num_nodes(), 0);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let trainer = DataParallelTrainer::new(2);
    /// assert!(std::ptr::eq(trainer.tape(0), trainer.try_tape(0).unwrap()));
    /// ```
    #[must_use]
    pub fn tape(&self, worker: usize) -> &ParallelTape {
        &self.tapes[worker]
    }

    /// Returns a mutable worker-tape reference when the worker exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(1);
    /// assert_eq!(trainer.try_tape_mut(0).unwrap().num_nodes(), 0);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(0);
    /// assert!(trainer.try_tape_mut(0).is_none());
    /// ```
    pub fn try_tape_mut(&mut self, worker: usize) -> Option<&mut ParallelTape> {
        self.tapes.get_mut(worker)
    }

    /// Returns a mutable reference to a worker tape.
    ///
    /// # Panics
    ///
    /// Panics when `worker >= self.n_workers()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(1);
    /// assert_eq!(trainer.tape_mut(0).num_nodes(), 0);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// let mut trainer = DataParallelTrainer::new(2);
    /// assert_eq!(trainer.tape_mut(1).num_nodes(), 0);
    /// ```
    pub fn tape_mut(&mut self, worker: usize) -> &mut ParallelTape {
        &mut self.tapes[worker]
    }

    /// Returns the configured worker count.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// assert_eq!(DataParallelTrainer::new(3).n_workers(), 3);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::data_parallel::DataParallelTrainer;
    /// assert_eq!(DataParallelTrainer::new(0).n_workers(), 0);
    /// ```
    #[must_use]
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Node, Op, SavedData, Tape, Tensor};

    #[test]
    fn reduce_sum_and_mean_cover_empty_and_multiple_workers() {
        assert!(GradientAggregator::reduce_sum(&[]).is_empty());
        assert_eq!(
            GradientAggregator::reduce_sum(&[vec![1.0, 2.0], vec![4.0, 5.0]]),
            vec![5.0, 7.0]
        );
        assert_eq!(
            GradientAggregator::reduce_mean(&[vec![2.0, 4.0], vec![4.0, 6.0]]),
            vec![3.0, 5.0]
        );
    }

    #[test]
    fn checked_reduction_rejects_shorter_and_longer_worker_vectors() {
        assert_eq!(
            GradientAggregator::try_reduce_sum(&[vec![1.0, 2.0], vec![3.0]]),
            Err(GradientLengthMismatch {
                worker: 1,
                expected: 2,
                got: 1,
            })
        );
        assert_eq!(
            GradientAggregator::try_reduce_mean(&[vec![1.0], vec![2.0, 3.0]]),
            Err(GradientLengthMismatch {
                worker: 1,
                expected: 1,
                got: 2,
            })
        );
    }

    #[test]
    fn compatibility_reduction_panics_on_shape_mismatch_in_all_profiles() {
        let result = std::panic::catch_unwind(|| {
            GradientAggregator::reduce_sum(&[vec![1.0, 2.0], vec![3.0]])
        });
        assert!(result.is_err());
    }

    #[test]
    fn train_batch_reports_worker_shape_mismatch() {
        let mut trainer = DataParallelTrainer::new(2);
        let error = trainer
            .try_train_batch(|_, worker| {
                if worker == 0 {
                    vec![1.0, 2.0]
                } else {
                    vec![3.0]
                }
            })
            .unwrap_err();
        assert_eq!(error.worker, 1);
    }

    #[test]
    fn trainer_one_worker_matches_sequential_autodiff() {
        let seq_tape = Tape::new();
        let x = seq_tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let x_idx = x.idx();
        let y = x.scale(2.0);
        y.backward();
        let seq_grad = seq_tape.grad(x_idx).sum() as f64;

        let mut trainer = DataParallelTrainer::new(1);
        let avg = trainer.train_batch(|tape, _| {
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 1),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 2.0,
                },
                shape: (1, 1),
                saved: SavedData::None,
            });
            tape.set_value(x, &[3.0]);
            tape.set_value(y, &[6.0]);
            tape.backward(y);
            vec![tape.grad(x)]
        });
        assert_eq!(avg, vec![seq_grad]);
    }

    #[test]
    fn train_batch_threaded_is_thread_count_invariant() {
        let batch = |_tape: &ParallelTape, worker: usize| {
            let sensitive = match worker % 4 {
                0 => 1e16,
                1 => 1.0,
                2 => -1e16,
                _ => 3.0,
            };
            vec![sensitive, (worker as f64 + 1.0).recip()]
        };
        let run = |threads| DataParallelTrainer::new(8).train_batch_threaded(threads, batch);
        let one = run(1);
        assert_eq!(one, run(2));
        assert_eq!(one, run(4));
        assert_eq!(one, run(8));
        let mut sequential = DataParallelTrainer::new(8);
        assert_eq!(one, sequential.train_batch(batch));
    }

    #[test]
    fn parallel_tape_training_is_deterministic_across_threads() {
        let batch = |tape: &ParallelTape, worker: usize| {
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 3),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 2.0,
                },
                shape: (1, 3),
                saved: SavedData::None,
            });
            let xv: Vec<f32> = (0..3)
                .map(|j| ((worker * 3 + j) as f32).sin())
                .collect();
            let yv: Vec<f32> = xv.iter().map(|value| value * 2.0).collect();
            tape.set_value(x, &xv);
            tape.set_value(y, &yv);
            tape.backward(y);
            vec![tape.grad(x)]
        };
        let run = |threads| DataParallelTrainer::new(4).train_batch_threaded(threads, batch);
        let one = run(1);
        assert_eq!(one, run(2));
        assert_eq!(one, run(4));
    }

    #[test]
    fn multi_step_training_is_thread_count_invariant() {
        fn train(threads: usize) -> Vec<f32> {
            let (in_dim, out_dim, n_workers, steps, lr) =
                (3usize, 2usize, 4usize, 8usize, 0.05f32);
            let mut weights: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| (i as f32 * 0.1).sin())
                .collect();
            for _ in 0..steps {
                let trainer = DataParallelTrainer::new(n_workers);
                let current = &weights;
                let grads = trainer.train_batch_threaded(threads, |_parallel_tape, worker| {
                    let input: Vec<f32> = (0..in_dim)
                        .map(|j| (((worker * in_dim + j) as f32) * 0.3).cos())
                        .collect();
                    let target: Vec<f32> = (0..out_dim)
                        .map(|j| (((worker + j) as f32) * 0.2).sin())
                        .collect();
                    let tape = Tape::new();
                    let x = tape.input(Tensor::from_vec(input, 1, in_dim));
                    let w = tape.input(Tensor::from_vec(current.clone(), in_dim, out_dim));
                    let target = tape.input(Tensor::from_vec(target, 1, out_dim));
                    let output = x.matmul(w);
                    let loss = output.sub(target).pow(2.0).sum();
                    tape.backward(loss.idx());
                    tape.grad(w.idx())
                        .data
                        .iter()
                        .map(|&value| value as f64)
                        .collect()
                });
                for (weight, &gradient) in weights.iter_mut().zip(&grads) {
                    *weight -= lr * gradient as f32;
                }
            }
            weights
        }

        let one = train(1);
        assert_eq!(one, train(2));
        assert_eq!(one, train(4));
        let initial: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();
        assert_ne!(one, initial);
    }
}
