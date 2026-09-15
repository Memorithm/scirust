//! End-to-end annualisation tests against the production MCP registry.

use scirust_mcp::default_registry;
use serde_json::{Value, json};

/// Small explicit chronological fixture; interval metadata is tested separately.
fn candles() -> Value {
    json!([
        [1_000, 100.0, 102.0, 99.0, 101.0, 10.0],
        [2_000, 101.0, 103.0, 100.0, 102.0, 11.0],
        [3_000, 102.0, 103.0, 99.0, 100.0, 12.0]
    ])
}

#[test]
fn every_direct_annualisation_tool_rejects_invalid_intervals() {
    let registry = default_registry();
    let listed = registry.list_json();
    let tools = [
        "trader_backtest",
        "trader_metrics",
        "trader_walkforward",
        "trader_monte_carlo",
        "trader_portfolio_construct",
        "trader_regime",
        "trader_optimize",
    ];
    for name in tools
    {
        let schema = &listed
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == name)
            .unwrap()["inputSchema"];
        assert_eq!(schema["properties"]["interval"]["type"], "string");

        for interval in [
            "",
            "unknown",
            "15",
            "h",
            "0h",
            "-1h",
            "1fortnight",
            "0.000000000000000000000000000000000000000001s",
            "999999999999999999999999999999999999999999999999999999999999h",
        ]
        {
            let error = registry
                .call(
                    name,
                    json!({
                        "interval": interval,
                        "ohlcv": candles(),
                        "strategy": "momentum",
                        "equity": [100.0, 110.0, 99.0],
                        "assets": [
                            {"symbol": "A", "returns": [0.01, -0.02]},
                            {"symbol": "B", "returns": [-0.01, 0.03]}
                        ]
                    }),
                )
                .unwrap_err();
            assert!(
                error.starts_with("invalid_arguments:") && error.contains("$.interval"),
                "{name} accepted or misclassified interval {interval:?}: {error}",
            );
        }
    }
}

#[test]
fn nested_market_intervals_cannot_fall_back_to_hourly() {
    let registry = default_registry();
    for name in ["trader_scan_opportunities", "trader_dashboard"]
    {
        for interval in [json!("unknown"), json!("0m"), json!(null), json!(15), json!(false)]
        {
            let error = registry
                .call(
                    name,
                    json!({
                        "series": [{"symbol": "X", "interval": interval, "ohlcv": candles()}]
                    }),
                )
                .unwrap_err();
            assert!(error.starts_with("invalid_arguments:"), "{name}: {error}");
            assert!(error.contains("interval"), "{name}: {error}");
        }
    }
}

#[test]
fn dashboard_guard_identifies_the_actual_bad_series() {
    let registry = default_registry();
    let error = registry
        .call(
            "trader_dashboard",
            json!({
                "series": [
                    {"symbol": "A", "interval": "1h", "ohlcv": candles()},
                    {"symbol": "B", "interval": "invalid", "ohlcv": candles()}
                ]
            }),
        )
        .unwrap_err();
    assert!(error.contains("$.series[1].interval"), "{error}");
}

#[test]
fn metrics_preserve_subannual_frequency_and_match_an_independent_oracle() {
    let registry = default_registry();
    let result = registry
        .call(
            "trader_metrics",
            json!({"interval": "730d", "equity": [100.0, 110.0, 99.0]}),
        )
        .unwrap();
    assert_eq!(result["periods_per_year"], json!(0.5));
    // Returns +0.1 and -0.1 have sample variance 0.02. Annualised variance
    // at 0.5 periods/year is 0.01, hence volatility 0.1 (not sqrt(0.02)).
    let volatility = result["volatility"].as_f64().unwrap();
    assert!((volatility - 0.1).abs() < 1e-5);
    let total_return = result["total_return"].as_f64().unwrap();
    assert!((total_return + 0.01).abs() < 1e-5);
}

#[test]
fn omitted_metric_interval_retains_daily_default_and_aliases_work() {
    let registry = default_registry();
    let omitted = registry
        .call("trader_metrics", json!({"equity": [100.0, 110.0, 99.0]}))
        .unwrap();
    let daily = registry
        .call(
            "trader_metrics",
            json!({"interval": "1d", "equity": [100.0, 110.0, 99.0]}),
        )
        .unwrap();
    assert_eq!(omitted, daily);
    assert_eq!(daily["periods_per_year"], json!(365.0));

    let four_hour = registry
        .call(
            "trader_metrics",
            json!({"interval": " 4HR ", "equity": [100.0, 110.0, 99.0]}),
        )
        .unwrap();
    assert_eq!(four_hour["periods_per_year"], json!(2190.0));
}

#[test]
fn backtest_handler_reports_the_same_checked_frequency() {
    let registry = default_registry();
    for (interval, expected) in [("1h", 8760.0), ("730d", 0.5)]
    {
        let report = registry
            .call(
                "trader_backtest",
                json!({"interval": interval, "ohlcv": candles(), "strategy": "momentum"}),
            )
            .unwrap();
        assert_eq!(report["performance"]["periods_per_year"], json!(expected));
        assert_eq!(report["equity_curve"].as_array().unwrap().len(), 3);
    }
}
