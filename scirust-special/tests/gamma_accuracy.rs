//! Regression tests for the accuracy of `ln_gamma` and `gamma`.
//!
//! Reference values are 50-digit `mpmath` evaluations (`mp.loggamma`,
//! `mp.gamma`) at the exact `f64` arguments. Tolerances are relative and leave
//! room for a few rounding units of libm (and Miri's float perturbation), but
//! sit well below the errors of the previous implementation, which are quoted
//! next to each case.

use scirust_special::{gamma, ln_gamma};

fn rel_err(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

/// `ln Γ` near its zeros at 1 and 2 must keep *relative* accuracy.
/// Previously the Lanczos sum had an absolute error of ~1e-16 there, so
/// `ln_gamma(1 + 1e-10)` had a relative error of 4.4e-6 and `ln_gamma(1.0)`
/// returned −8.9e-16.
#[test]
fn ln_gamma_is_relatively_accurate_near_its_zeros() {
    let cases = [
        (1.000_000_000_1, -5.772_157_125_783_244e-11), // old rel. error 4.4e-6
        (0.999_999_9, 5.772_157_468_444_193e-8),       // old 6.8e-9
        (1.999_999_9, -4.227_843_030_986_13e-8),       // old 3.7e-8
        (2.000_000_1, 4.227_843_666_532_498e-8),       // old 2.3e-8
        (1.5, -0.120_782_237_635_245_22),
        (0.5, 0.572_364_942_924_700_1),
    ];
    for (x, want) in cases
    {
        let got = ln_gamma(x);
        assert!(
            rel_err(got, want) < 1e-14,
            "ln_gamma({x:e}) = {got:e}, want {want:e}"
        );
    }
    assert!(
        ln_gamma(1.0).abs() < 1e-30,
        "ln_gamma(1) = {:e}",
        ln_gamma(1.0)
    );
    assert!(
        ln_gamma(2.0).abs() < 1e-30,
        "ln_gamma(2) = {:e}",
        ln_gamma(2.0)
    );
}

/// Subnormal arguments: `ln Γ(x) ≈ −ln|x|` is finite. The reflection formula
/// used before computed `π / sin(πx)`, which overflowed to `+∞`.
#[test]
fn ln_gamma_is_finite_at_subnormal_arguments() {
    for x in [1e-310, -1e-310]
    {
        let got = ln_gamma(x);
        assert!(
            rel_err(got, 713.801_378_828_154_2) < 1e-14,
            "ln_gamma({x:e}) = {got:e}"
        );
    }
}

/// Large positive arguments: the result used to be `exp(ln Γ(x))`, whose
/// relative error is about `ε·ln Γ(x)` on top of the Lanczos sum's own bias
/// (old errors 1.1e-13 at 100.5, 1.2e-13 at 150, 2.2e-13 at 170.5).
#[test]
fn gamma_is_accurate_for_large_arguments() {
    let cases = [
        (50.25, 1.614_476_471_241_244_1e63),
        (100.5, 9.320_963_104_082_717e156),
        (150.0, 3.808_922_637_630_57e260),
        (170.5, 5.562_092_414_56e305),
        (171.6, 1.585_896_909_667_256_5e308),
    ];
    for (x, want) in cases
    {
        let got = gamma(x);
        assert!(
            rel_err(got, want) < 2e-14,
            "gamma({x}) = {got:e}, want {want:e}"
        );
    }
    assert_eq!(gamma(171.7), f64::INFINITY);
    assert_eq!(gamma(1e10), f64::INFINITY);
}

/// Integer arguments: `Γ(n + 1) = n!`. Old relative errors reached 4.8e-15
/// (`gamma(10)` returned 362880.00000000175).
#[test]
fn gamma_matches_factorials() {
    let mut fact = 1.0_f64; // (n − 1)!, exact in f64 up to 22!
    for n in 1..=23_u32
    {
        let got = gamma(f64::from(n));
        assert!(
            rel_err(got, fact) < 2e-15,
            "gamma({n}) = {got:e}, want {fact:e}"
        );
        fact *= f64::from(n);
    }
}

/// Negative arguments, including results in the subnormal range, which the
/// old reflection `π / (sin(πx)·Γ(1−x))` flushed to zero once `Γ(1−x)`
/// overflowed.
#[test]
fn gamma_negative_arguments_and_subnormal_results() {
    let cases = [
        (-15.710_507_361_877_104, 4.272_859_325_005_75e-13),
        (-100.3, -1.043_143_249_578_308_9e-158),
        (-170.3, -1.144_927_998_387_812_2e-307),
    ];
    for (x, want) in cases
    {
        let got = gamma(x);
        assert!(
            rel_err(got, want) < 2e-14,
            "gamma({x}) = {got:e}, want {want:e}"
        );
    }
    // Γ(−175.5) = 2.107473…e-319 (subnormal; old result: 0.0).
    let got = gamma(-175.5);
    assert!(rel_err(got, 2.107_5e-319) < 1e-4, "gamma(-175.5) = {got:e}");
    // Far below the subnormal range the result is a signed zero.
    assert_eq!(gamma(-200.5), 0.0);
}

/// Functional equation `Γ(x + 1) = x·Γ(x)` across the internal branch
/// boundaries (½, 1.5, 2.5, 30) and on the negative axis.
#[test]
fn gamma_recurrence_holds_across_branches() {
    let xs = [
        -2.6, -2.4, -1.5, -0.6, -0.4, 0.3, 0.49, 0.51, 1.49, 1.51, 2.49, 2.51, 7.3, 28.9, 29.5,
        30.5, 64.2,
    ];
    for x in xs
    {
        let lhs = gamma(x + 1.0);
        let rhs = x * gamma(x);
        assert!(
            rel_err(lhs, rhs) < 4e-15,
            "Γ({x}+1) = {lhs:e} vs x·Γ(x) = {rhs:e}"
        );
        let dl = ln_gamma(x + 1.0) - ln_gamma(x) - x.abs().ln();
        let scale = ln_gamma(x + 1.0).abs().max(1.0);
        assert!(
            dl.abs() < 2e-15 * scale,
            "ln-form recurrence at {x}: {dl:e}"
        );
    }
}
