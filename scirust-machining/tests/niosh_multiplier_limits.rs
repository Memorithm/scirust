//! Regression tests: the NIOSH multipliers follow the limits of the
//! Applications Manual for the Revised NIOSH Lifting Equation (Waters,
//! Putz-Anderson & Garg, DHHS (NIOSH) Publication No. 94-110, §1.3,
//! Tables 1–4).
//!
//! * `D < 25 cm` is set to `D = 25 cm`, so `DM = 1` (it is never above 1);
//! * `H > 63 cm`, `V > 175 cm`, `D > 175 cm` and `A > 135°` give a
//!   multiplier of 0 (the task is outside the equation's limits and the
//!   recommended weight limit is 0).

use approx::assert_relative_eq;
use scirust_machining::{
    lifting_recommended_weight_limit, niosh_asymmetry_multiplier, niosh_distance_multiplier,
    niosh_horizontal_multiplier, niosh_vertical_multiplier,
};

#[test]
fn distance_multiplier_is_one_for_travel_below_25_cm() {
    // Manual §1.3.3.3: "For D less than 10 inches (25 cm) D is assumed to be
    // 10 inches (25 cm), and DM is 1.0" — DM ranges from 1.0 to 0.85 as D
    // varies from 0 to 175 cm.
    for d in [0.0, 1.0, 10.0, 20.0, 24.9, 25.0]
    {
        let dm = niosh_distance_multiplier(d);
        assert_relative_eq!(dm, 1.0, epsilon = 1e-12);
    }
    // Inside the range the continuous formula is unchanged (Table 3: D = 40
    // cm gives 0.93, D = 175 cm gives 0.85).
    assert_relative_eq!(
        niosh_distance_multiplier(40.0),
        0.82 + 4.5 / 40.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        niosh_distance_multiplier(175.0),
        0.82 + 4.5 / 175.0,
        epsilon = 1e-12
    );
}

#[test]
fn short_travel_lift_gives_the_load_constant_under_ideal_conditions() {
    // H = 25, V = 75, D = 10 cm, A = 0, FM = CM = 1: every multiplier is 1,
    // so RWL = LC = 23 kg. Before the fix DM(10) = 1.27 and this call
    // panicked on the "multiplier in [0, 1]" check.
    let rwl = lifting_recommended_weight_limit(
        niosh_horizontal_multiplier(25.0),
        niosh_vertical_multiplier(75.0),
        niosh_distance_multiplier(10.0),
        niosh_asymmetry_multiplier(0.0),
        1.0,
        1.0,
    );
    assert_relative_eq!(rwl, 23.0, epsilon = 1e-12);
}

#[test]
fn multipliers_are_zero_beyond_the_equation_limits() {
    // Table 1 (HM): H > 63 cm -> 0. Table 2 (VM): V > 175 cm -> 0.
    // Table 3 (DM): D > 175 cm -> 0. Table 4 (AM): A > 135° -> 0.
    assert_eq!(niosh_horizontal_multiplier(63.5), 0.0);
    assert_eq!(niosh_horizontal_multiplier(80.0), 0.0);
    assert_eq!(niosh_vertical_multiplier(176.0), 0.0);
    assert_eq!(niosh_vertical_multiplier(500.0), 0.0);
    assert_eq!(niosh_distance_multiplier(180.0), 0.0);
    assert_eq!(niosh_asymmetry_multiplier(136.0), 0.0);
    assert_eq!(niosh_asymmetry_multiplier(180.0), 0.0);

    // The tabulated end points themselves still use the formulas.
    assert_relative_eq!(
        niosh_horizontal_multiplier(63.0),
        25.0 / 63.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(niosh_vertical_multiplier(175.0), 0.70, epsilon = 1e-12);
    assert_relative_eq!(niosh_vertical_multiplier(0.0), 0.775, epsilon = 1e-12);
    assert_relative_eq!(niosh_asymmetry_multiplier(135.0), 0.568, epsilon = 1e-12);
}

#[test]
fn far_overhead_grip_no_longer_panics_the_weight_limit() {
    // V = 500 cm: the old 1 − 0.003·|V − 75| = −0.275 made
    // `lifting_recommended_weight_limit` panic; the manual gives VM = 0.
    let rwl = lifting_recommended_weight_limit(
        niosh_horizontal_multiplier(30.0),
        niosh_vertical_multiplier(500.0),
        niosh_distance_multiplier(30.0),
        niosh_asymmetry_multiplier(0.0),
        1.0,
        1.0,
    );
    assert_eq!(rwl, 0.0);
}
