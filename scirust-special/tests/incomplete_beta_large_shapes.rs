//! Regression tests: regularized incomplete beta `I_x(a, b)` when both shape
//! parameters are large.
//!
//! The prefactor `x^a (1−x)^b / B(a, b)` used to be formed as
//! `exp(a·ln x + b·ln(1−x) − ln B(a, b))`. Near the mode those terms are of
//! size `≈ (a+b)·ln 2` while their sum is only `≈ ½·ln(a+b)`, so the relative
//! error grew like `ε·(a+b)`: `I_½(1e12, 1e12)` was `0.499975` and
//! `I_½(1e14, 1e14)` was `0.5044` instead of `0.5`, and
//! `I_{1/3}(1e10, 2e10)` was `0.50000247` instead of `0.50000054`.
//!
//! References are mpmath 1.4 at 60 significant digits on the exact `f64`
//! inputs: adaptive quadrature of the density on a window of ±80 standard
//! deviations around the mode (cross-checked against `mpmath.betainc` on the
//! moderate cases, where both agree to 20 digits), and the exact value `½`
//! for `I_½(a, a)`. Both the lower value and its complement are checked, so
//! the two sides of the continued-fraction swap are covered. Tolerances are
//! relative; nothing is compared bit for bit. The remaining error near the
//! mode grows roughly like `ε·√(a+b)` (continued-fraction rounding), which
//! sets the tolerance tiers below.

use scirust_special::regularized_incomplete_beta;

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let rel = ((got - want) / want).abs();
    assert!(
        got.is_finite() && rel <= tol,
        "{what}: got {got:e}, want {want:e} (rel err {rel:e})"
    );
}

/// `(a, b, x, I_x(a, b), 1 − I_x(a, b))`.
type Case = (f64, f64, f64, f64, f64);

fn check(cases: &[Case], tol: f64) {
    for &(a, b, x, lower, upper) in cases
    {
        let got = regularized_incomplete_beta(a, b, x);
        assert_rel(got, lower, tol, &format!("I_{x}({a:e}, {b:e})"));
        // `1 − I_x(a, b) = I_{1−x}(b, a)`.
        assert_rel(
            regularized_incomplete_beta(b, a, 1.0 - x),
            upper,
            tol,
            &format!("I_{{1-{x}}}({b:e}, {a:e})"),
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // ~10³–2·10⁵ continued-fraction steps: too slow under Miri.
fn symmetric_shapes_at_one_half_are_exactly_one_half() {
    // Before: 0.499975 at 1e12 and 0.5044 at 1e14.
    for &(a, tol) in &[
        (1e6, 1e-12),
        (1e8, 1e-10),
        (1e10, 1e-10),
        (1e12, 1e-9),
        (1e14, 1e-8),
    ]
    {
        assert_rel(
            regularized_incomplete_beta(a, a, 0.5),
            0.5,
            tol,
            &format!("I_0.5({a:e}, {a:e})"),
        );
    }
}

#[test]
fn near_the_mode_up_to_a_million() {
    check(
        &[
            (
                1e6,
                1e6,
                0.5005,
                0.921_350_422_419_075_3,
                0.078_649_577_580_924_67,
            ),
            (
                5e5,
                5e5,
                0.4999,
                0.420_740_309_330_952_66,
                0.579_259_690_669_047_3,
            ),
        ],
        1e-12,
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~10⁴ continued-fraction steps: too slow under Miri.
fn near_the_mode_up_to_tens_of_billions() {
    check(
        &[
            (
                1e8,
                3e8,
                0.25002,
                0.822_195_204_495_161_2,
                0.177_804_795_504_838_84,
            ),
            (
                1e10,
                2e10,
                1.0 / 3.0,
                0.500_000_542_888_967_6,
                0.499_999_457_111_032_4,
            ),
            (
                1e10,
                2e10,
                1.0 / 3.0 + 1e-6,
                0.643_348_852_002_839_7,
                0.356_651_147_997_160_3,
            ),
            (
                1e10,
                2e10,
                1.0 / 3.0 - 1e-6,
                0.356_652_025_890_801_1,
                0.643_347_974_109_198_9,
            ),
        ],
        1e-10,
    );
}

#[test]
#[cfg_attr(miri, ignore)] // ~4·10⁴ continued-fraction steps: too slow under Miri.
fn near_the_mode_at_a_trillion() {
    check(
        &[(
            1e12,
            1e12,
            0.5 + 5e-7,
            0.921_350_396_457_808_6,
            0.078_649_603_542_191_37,
        )],
        1e-9,
    );
}

#[test]
fn moderate_and_mixed_shapes() {
    // Both sides of the saddle-point threshold (both shapes ≥ 10). The
    // `(3e4, 7e4)` case was off by 2.9e-12 before; the other three were
    // already accurate and guard against a regression from the switch.
    check(
        &[
            (
                3e4,
                7e4,
                0.3,
                0.500_367_062_336_610_9,
                0.499_632_937_663_389_1,
            ),
            (
                2e3,
                3e3,
                0.41,
                0.925_262_126_208_304_2,
                0.074_737_873_791_695_82,
            ),
            (
                25.0,
                40.0,
                0.38,
                0.477_414_674_253_387_94,
                0.522_585_325_746_612_1,
            ),
            (
                1e7,
                10.0,
                0.999_999,
                0.457_928_525_890_501_17,
                0.542_071_474_109_498_8,
            ),
        ],
        1e-12,
    );
}
