//! Deterministic learning-rate schedules for SciRust optimizers.
//!
//! Implementations of [`LrSchedule`] are stateless: the learning rate is a
//! deterministic function of the integer training step. They can therefore be
//! reconstructed after a checkpoint without serializing scheduler state.
//!
//! [`ReduceOnPlateau`] is deliberately different: it depends on the observed
//! validation-loss history, is stateful, and therefore does not implement
//! [`LrSchedule`].
//!
//! [`LrSchedule::drive`] targets the cross-family
//! [`HasLearningRate`] interface, so the same schedule can drive tape, N-D, and
//! raw-slice optimizers.

use crate::optim::HasLearningRate;

/// Raises `base` to a non-negative `usize` exponent without narrowing the
/// exponent to `i32`.
///
/// The common path keeps the previous `powi(i32)` behavior exactly. Only steps
/// larger than `i32::MAX` use exponentiation by squaring, which avoids the
/// wraparound that an unchecked `usize as i32` conversion would introduce.
fn powi_usize(base: f32, exponent: usize) -> f32 {
    if let Ok(exponent) = i32::try_from(exponent)
    {
        return base.powi(exponent);
    }

    let mut base = base;
    let mut exponent = exponent;
    let mut result = 1.0_f32;
    while exponent != 0
    {
        if exponent & 1 == 1
        {
            result *= base;
        }
        exponent >>= 1;
        if exponent != 0
        {
            base *= base;
        }
    }
    result
}

/// Stateless deterministic learning-rate schedule.
///
/// Implementations must return the same value for the same `step` and must not
/// depend on hidden mutable state.
///
/// # Examples
///
/// ```
/// use scirust_core::autodiff::scheduler::{ConstantLr, LrSchedule};
/// let schedule = ConstantLr::new(0.01);
/// assert_eq!(schedule.lr_at(0), 0.01);
/// assert_eq!(schedule.lr_at(10_000), 0.01);
/// ```
///
/// ```
/// use scirust_core::autodiff::scheduler::{LrSchedule, StepLr};
/// let schedule = StepLr::new(0.1, 0.5, 10);
/// assert_eq!(schedule.lr_at(0), 0.1);
/// assert_eq!(schedule.lr_at(10), 0.05);
/// ```
pub trait LrSchedule: Send + Sync {
    /// Returns the learning rate for `step`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{ConstantLr, LrSchedule};
    /// assert_eq!(ConstantLr::new(0.25).lr_at(123), 0.25);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{LrSchedule, StepLr};
    /// let schedule = StepLr::new(1.0, 0.1, 5);
    /// assert!((schedule.lr_at(5) - 0.1).abs() < 1e-7);
    /// ```
    fn lr_at(&self, step: usize) -> f32;

    /// Writes the scheduled learning rate into any [`HasLearningRate`] target.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{ConstantLr, LrSchedule};
    /// use scirust_core::optim::HasLearningRate;
    /// struct Opt(f32);
    /// impl HasLearningRate for Opt {
    ///     fn lr(&self) -> f32 { self.0 }
    ///     fn set_lr(&mut self, lr: f32) { self.0 = lr; }
    /// }
    /// let mut opt = Opt(1.0);
    /// ConstantLr::new(0.02).drive(&mut opt, 50);
    /// assert_eq!(opt.lr(), 0.02);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{LrSchedule, StepLr};
    /// use scirust_core::optim::HasLearningRate;
    /// struct Opt(f32);
    /// impl HasLearningRate for Opt {
    ///     fn lr(&self) -> f32 { self.0 }
    ///     fn set_lr(&mut self, lr: f32) { self.0 = lr; }
    /// }
    /// let mut opt = Opt(0.0);
    /// StepLr::new(0.1, 0.5, 10).drive(&mut opt, 10);
    /// assert_eq!(opt.lr(), 0.05);
    /// ```
    fn drive(&self, opt: &mut dyn HasLearningRate, step: usize) {
        opt.set_lr(self.lr_at(step));
    }
}

impl<T: LrSchedule + ?Sized> LrSchedule for Box<T> {
    fn lr_at(&self, step: usize) -> f32 {
        (**self).lr_at(step)
    }
}

/// Constant learning rate, useful as a baseline or ablation control.
#[derive(Clone, Debug)]
pub struct ConstantLr {
    /// Learning rate returned for every step.
    pub lr: f32,
}

impl ConstantLr {
    /// Creates a constant schedule.
    ///
    /// The value is stored verbatim; this constructor does not impose a sign or
    /// finiteness policy on caller-defined optimizer conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{ConstantLr, LrSchedule};
    /// let schedule = ConstantLr::new(0.03);
    /// assert_eq!(schedule.lr_at(7), 0.03);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ConstantLr;
    /// let schedule = ConstantLr::new(-0.01);
    /// assert_eq!(schedule.lr, -0.01);
    /// ```
    #[must_use]
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LrSchedule for ConstantLr {
    fn lr_at(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Step decay: `initial * decay^(step / step_size)`.
#[derive(Clone, Debug)]
pub struct StepLr {
    /// Learning rate before the first decay boundary.
    pub initial: f32,
    /// Multiplicative factor applied at each boundary.
    pub decay: f32,
    /// Number of steps between decay events.
    pub step_size: usize,
}

impl StepLr {
    /// Creates a step schedule.
    ///
    /// # Panics
    ///
    /// Panics when `step_size == 0`. `initial` and `decay` are otherwise stored
    /// verbatim, permitting caller-defined growth schedules as well as decay.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{LrSchedule, StepLr};
    /// let schedule = StepLr::new(0.1, 0.1, 30);
    /// assert_eq!(schedule.lr_at(29), 0.1);
    /// assert!((schedule.lr_at(30) - 0.01).abs() < 1e-7);
    /// ```
    ///
    /// ```should_panic
    /// use scirust_core::autodiff::scheduler::StepLr;
    /// let _ = StepLr::new(0.1, 0.5, 0);
    /// ```
    #[must_use]
    pub fn new(initial: f32, decay: f32, step_size: usize) -> Self {
        assert!(step_size > 0, "StepLr: step_size > 0 requis");
        Self {
            initial,
            decay,
            step_size,
        }
    }
}

impl LrSchedule for StepLr {
    fn lr_at(&self, step: usize) -> f32 {
        let n_decays = step / self.step_size;
        self.initial * powi_usize(self.decay, n_decays)
    }
}

/// Continuous exponential decay: `initial * gamma^step`.
#[derive(Clone, Debug)]
pub struct ExponentialLr {
    /// Learning rate at step zero.
    pub initial: f32,
    /// Per-step multiplicative decay in `(0, 1]`.
    pub gamma: f32,
}

impl ExponentialLr {
    /// Creates an exponential-decay schedule.
    ///
    /// # Panics
    ///
    /// Panics unless `gamma` is in `(0, 1]`. NaN is rejected by the same
    /// comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{ExponentialLr, LrSchedule};
    /// let schedule = ExponentialLr::new(1.0, 0.9);
    /// assert!((schedule.lr_at(2) - 0.81).abs() < 1e-6);
    /// ```
    ///
    /// ```should_panic
    /// use scirust_core::autodiff::scheduler::ExponentialLr;
    /// let _ = ExponentialLr::new(1.0, 1.01);
    /// ```
    #[must_use]
    pub fn new(initial: f32, gamma: f32) -> Self {
        assert!(gamma > 0.0 && gamma <= 1.0, "ExponentialLr: gamma ∈ (0, 1]");
        Self { initial, gamma }
    }
}

impl LrSchedule for ExponentialLr {
    fn lr_at(&self, step: usize) -> f32 {
        self.initial * powi_usize(self.gamma, step)
    }
}

/// Cosine annealing from `initial` to `min_lr` over `period` steps.
///
/// Values at and after `period` are clamped to `min_lr`; this implementation
/// does not perform warm restarts.
#[derive(Clone, Debug)]
pub struct CosineAnnealing {
    /// Learning rate at step zero.
    pub initial: f32,
    /// Learning-rate floor reached at `period`.
    pub min_lr: f32,
    /// Annealing period in steps.
    pub period: usize,
}

impl CosineAnnealing {
    /// Creates a cosine-annealing schedule.
    ///
    /// # Panics
    ///
    /// Panics when `period == 0` or `min_lr > initial`. NaN inputs participating
    /// in the ordering check are rejected.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{CosineAnnealing, LrSchedule};
    /// let schedule = CosineAnnealing::new(0.1, 0.001, 100);
    /// assert!((schedule.lr_at(0) - 0.1).abs() < 1e-6);
    /// assert!((schedule.lr_at(100) - 0.001).abs() < 1e-6);
    /// ```
    ///
    /// ```should_panic
    /// use scirust_core::autodiff::scheduler::CosineAnnealing;
    /// let _ = CosineAnnealing::new(0.1, 0.2, 100);
    /// ```
    #[must_use]
    pub fn new(initial: f32, min_lr: f32, period: usize) -> Self {
        assert!(period > 0, "CosineAnnealing: period > 0 requis");
        assert!(
            min_lr <= initial,
            "CosineAnnealing: min_lr <= initial requis"
        );
        Self {
            initial,
            min_lr,
            period,
        }
    }
}

impl LrSchedule for CosineAnnealing {
    fn lr_at(&self, step: usize) -> f32 {
        if step >= self.period
        {
            return self.min_lr;
        }
        let progress = step as f32 / self.period as f32;
        let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.initial - self.min_lr) * cos_factor
    }
}

/// Linear warmup followed by cosine annealing.
///
/// For `warmup_steps > 0`, steps `[0, warmup_steps)` rise linearly from zero.
/// At `warmup_steps`, the cosine phase starts at `base`. A zero-length warmup is
/// valid and starts directly at `base`.
#[derive(Clone, Debug)]
pub struct WarmupCosine {
    /// Peak/base learning rate at the warmup-to-cosine transition.
    pub base: f32,
    /// Learning-rate floor after `total_steps`.
    pub min_lr: f32,
    /// Number of linear warmup steps; zero is permitted.
    pub warmup_steps: usize,
    /// Total schedule horizon; must exceed `warmup_steps`.
    pub total_steps: usize,
}

impl WarmupCosine {
    /// Creates a warmup-plus-cosine schedule.
    ///
    /// # Panics
    ///
    /// Panics unless `warmup_steps < total_steps` and `min_lr <= base`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{LrSchedule, WarmupCosine};
    /// let schedule = WarmupCosine::new(0.01, 0.0001, 100, 1000);
    /// assert_eq!(schedule.lr_at(0), 0.0);
    /// assert!((schedule.lr_at(100) - 0.01).abs() < 1e-6);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::{LrSchedule, WarmupCosine};
    /// let schedule = WarmupCosine::new(0.01, 0.001, 0, 10);
    /// assert!((schedule.lr_at(0) - 0.01).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn new(base: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        assert!(
            warmup_steps < total_steps,
            "WarmupCosine: warmup_steps < total_steps requis"
        );
        assert!(min_lr <= base, "WarmupCosine: min_lr <= base requis");
        Self {
            base,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LrSchedule for WarmupCosine {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps
        {
            self.base * (step as f32 / self.warmup_steps as f32)
        }
        else
        {
            let post = step - self.warmup_steps;
            let period = self.total_steps - self.warmup_steps;
            if post >= period
            {
                return self.min_lr;
            }
            let progress = post as f32 / period as f32;
            let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.base - self.min_lr) * cos_factor
        }
    }
}

/// Stateful reduce-on-plateau scheduler driven by validation loss.
///
/// A strictly smaller loss resets the bad-epoch counter. Any other value,
/// including an equal or NaN loss, counts as a non-improvement. Once
/// `patience` non-improving observations have accumulated, the current learning
/// rate is multiplied by `factor` and clamped to `min_lr`.
pub struct ReduceOnPlateau {
    initial: f32,
    /// Multiplicative reduction factor in `(0, 1)`.
    pub factor: f32,
    /// Number of consecutive non-improving observations before a reduction.
    pub patience: usize,
    /// Absolute lower bound for the learning rate.
    pub min_lr: f32,
    current: f32,
    best_loss: f32,
    n_bad_epochs: usize,
}

impl ReduceOnPlateau {
    /// Creates a stateful plateau scheduler.
    ///
    /// # Panics
    ///
    /// Panics unless `factor` is in `(0, 1)` and `min_lr <= initial`. The floor
    /// check prevents a "reduction" event from increasing the learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let schedule = ReduceOnPlateau::new(0.1, 0.5, 3, 0.001);
    /// assert_eq!(schedule.current_lr(), 0.1);
    /// ```
    ///
    /// ```should_panic
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let _ = ReduceOnPlateau::new(0.1, 0.5, 3, 0.2);
    /// ```
    #[must_use]
    pub fn new(initial: f32, factor: f32, patience: usize, min_lr: f32) -> Self {
        assert!(factor > 0.0 && factor < 1.0, "factor ∈ (0, 1)");
        assert!(
            min_lr <= initial,
            "ReduceOnPlateau: min_lr <= initial requis"
        );
        Self {
            initial,
            factor,
            patience,
            min_lr,
            current: initial,
            best_loss: f32::INFINITY,
            n_bad_epochs: 0,
        }
    }

    /// Returns the current stateful learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let schedule = ReduceOnPlateau::new(0.1, 0.5, 2, 0.0);
    /// assert_eq!(schedule.current_lr(), 0.1);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let mut schedule = ReduceOnPlateau::new(0.1, 0.5, 1, 0.0);
    /// schedule.step(1.0);
    /// schedule.step(1.0);
    /// assert_eq!(schedule.current_lr(), 0.05);
    /// ```
    #[must_use]
    pub fn current_lr(&self) -> f32 {
        self.current
    }

    /// Records one validation loss and returns the resulting learning rate.
    ///
    /// A strictly smaller loss is an improvement. Equal, greater, infinite, or
    /// NaN values take the non-improvement branch. With `patience == 0`, the
    /// first non-improving observation reduces the rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let mut schedule = ReduceOnPlateau::new(0.1, 0.5, 2, 0.0);
    /// schedule.step(1.0);
    /// assert_eq!(schedule.step(1.1), 0.1);
    /// assert_eq!(schedule.step(1.2), 0.05);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let mut schedule = ReduceOnPlateau::new(0.1, 0.5, 1, 0.025);
    /// schedule.step(1.0);
    /// assert_eq!(schedule.step(f32::NAN), 0.05);
    /// assert_eq!(schedule.step(f32::NAN), 0.025);
    /// assert_eq!(schedule.step(f32::NAN), 0.025);
    /// ```
    pub fn step(&mut self, val_loss: f32) -> f32 {
        if val_loss < self.best_loss
        {
            self.best_loss = val_loss;
            self.n_bad_epochs = 0;
        }
        else
        {
            self.n_bad_epochs += 1;
            if self.n_bad_epochs >= self.patience
            {
                self.current = (self.current * self.factor).max(self.min_lr);
                self.n_bad_epochs = 0;
            }
        }
        self.current
    }

    /// Restores the initial learning rate and clears validation history.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let mut schedule = ReduceOnPlateau::new(0.1, 0.5, 1, 0.0);
    /// schedule.step(1.0);
    /// schedule.step(2.0);
    /// assert_eq!(schedule.current_lr(), 0.05);
    /// schedule.reset();
    /// assert_eq!(schedule.current_lr(), 0.1);
    /// ```
    ///
    /// ```
    /// use scirust_core::autodiff::scheduler::ReduceOnPlateau;
    /// let mut schedule = ReduceOnPlateau::new(0.1, 0.5, 2, 0.0);
    /// schedule.step(0.5);
    /// schedule.reset();
    /// assert_eq!(schedule.step(1.0), 0.1);
    /// ```
    pub fn reset(&mut self) {
        self.current = self.initial;
        self.best_loss = f32::INFINITY;
        self.n_bad_epochs = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_lr() {
        let s = ConstantLr::new(0.05);
        for step in [0, 1, 100, 10_000]
        {
            assert_eq!(s.lr_at(step), 0.05);
        }
    }

    #[test]
    fn step_lr_decays_at_boundaries() {
        let s = StepLr::new(0.1, 0.1, 30);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-7);
        assert!((s.lr_at(29) - 0.1).abs() < 1e-7);
        assert!((s.lr_at(30) - 0.01).abs() < 1e-7);
        assert!((s.lr_at(60) - 0.001).abs() < 1e-7);
        assert!((s.lr_at(90) - 0.0001).abs() < 1e-7);
    }

    #[test]
    fn step_lr_does_not_wrap_decay_count_past_i32_max() {
        let s = StepLr::new(1.0, 0.5, 1);
        let step = i32::MAX as usize + 1;
        assert_eq!(s.lr_at(step), 0.0);
    }

    #[test]
    fn exponential_decays_smoothly() {
        let s = ExponentialLr::new(1.0, 0.9);
        assert!((s.lr_at(0) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(1) - 0.9).abs() < 1e-6);
        assert!((s.lr_at(10) - 0.9_f32.powi(10)).abs() < 1e-6);
    }

    #[test]
    fn exponential_does_not_turn_decay_into_growth_past_i32_max() {
        let s = ExponentialLr::new(1.0, 0.999_999_94);
        let step = i32::MAX as usize + 1;
        let lr = s.lr_at(step);
        assert!(lr.is_finite());
        assert_eq!(lr, 0.0);
    }

    #[test]
    fn cosine_starts_at_initial_ends_at_min() {
        let s = CosineAnnealing::new(0.1, 0.001, 100);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-5);
        assert!((s.lr_at(100) - 0.001).abs() < 1e-5);
        assert!((s.lr_at(200) - 0.001).abs() < 1e-5);
        let mid = s.lr_at(50);
        assert!(mid > 0.04 && mid < 0.06, "mid lr = {mid}");
    }

    #[test]
    fn cosine_monotonic_decreasing_in_period() {
        let s = CosineAnnealing::new(1.0, 0.0, 100);
        let mut prev = f32::INFINITY;
        for step in 0..=100
        {
            let lr = s.lr_at(step);
            assert!(
                lr <= prev + 1e-7,
                "non-monotonic at step {step}: {prev} → {lr}"
            );
            prev = lr;
        }
    }

    #[test]
    fn warmup_cosine_phases() {
        let s = WarmupCosine::new(0.01, 0.0001, 100, 1000);
        assert!(s.lr_at(0) < 1e-6);
        assert!((s.lr_at(50) - 0.005).abs() < 1e-5);
        assert!((s.lr_at(100) - 0.01).abs() < 1e-5);
        assert!(s.lr_at(999) < 0.001);
    }

    #[test]
    fn zero_warmup_starts_directly_at_base() {
        let s = WarmupCosine::new(0.01, 0.001, 0, 10);
        assert!((s.lr_at(0) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn warmup_cosine_continuous_at_transition() {
        let s = WarmupCosine::new(0.1, 0.001, 100, 1000);
        let just_before = s.lr_at(99);
        let just_after = s.lr_at(100);
        assert!(
            (just_after - just_before).abs() < 0.01,
            "discontinuity at warmup transition: {just_before} vs {just_after}"
        );
    }

    #[test]
    fn reduce_on_plateau_reduces_after_patience() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 3, 0.0);
        s.step(1.0);
        assert_eq!(s.current_lr(), 0.1);
        s.step(1.1);
        s.step(1.05);
        s.step(1.05);
        assert!((s.current_lr() - 0.05).abs() < 1e-7);
    }

    #[test]
    fn reduce_on_plateau_resets_on_improvement() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 2, 0.0);
        s.step(1.0);
        s.step(1.5);
        s.step(0.5);
        s.step(0.6);
        assert_eq!(s.current_lr(), 0.1);
    }

    #[test]
    #[should_panic(expected = "min_lr <= initial")]
    fn reduce_on_plateau_rejects_floor_above_initial() {
        let _ = ReduceOnPlateau::new(0.1, 0.5, 2, 0.2);
    }

    #[test]
    fn boxed_scheduler_works() {
        let s: Box<dyn LrSchedule> = Box::new(StepLr::new(0.1, 0.5, 10));
        assert_eq!(s.lr_at(0), 0.1);
        assert_eq!(s.lr_at(10), 0.05);
    }

    #[test]
    fn drive_schedules_tape_and_nd_optimizers() {
        use crate::autodiff::optim::Adam;
        use crate::nn::nd_optim::NdAdam;

        let sched = CosineAnnealing::new(0.1, 0.001, 100);
        let mut tape_opt = Adam::new(0.7);
        let mut nd_opt = NdAdam::with_lr(0.7);

        let mut prev = f32::INFINITY;
        for step in [0_usize, 25, 50, 75, 100]
        {
            sched.drive(&mut tape_opt, step);
            sched.drive(&mut nd_opt, step);

            let expected = sched.lr_at(step);
            assert_eq!(HasLearningRate::lr(&tape_opt), expected, "step {step}");
            assert_eq!(HasLearningRate::lr(&nd_opt), expected, "step {step}");
            assert!(
                expected < prev,
                "learning rate must decrease: step {step}, {prev} → {expected}"
            );
            prev = expected;
        }
        assert!((HasLearningRate::lr(&nd_opt) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn drive_works_through_boxed_scheduler() {
        use crate::nn::nd_optim::NdAdam;

        let s: Box<dyn LrSchedule> = Box::new(StepLr::new(0.1, 0.5, 10));
        let mut opt = NdAdam::with_lr(0.9);
        s.drive(&mut opt, 10);
        assert_eq!(HasLearningRate::lr(&opt), 0.05);
    }
}
