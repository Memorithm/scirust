use scirust_mcp::default_registry;
use serde_json::{Value, json};

fn valid_ohlcv(count: usize) -> Value {
    let rows = (0..count)
        .map(|i| {
            let open = 100.0 + ((i * 7) % 13) as f64;
            let close = open
                + match i % 4
                {
                    0 => 2.0,
                    1 => -1.0,
                    2 => 3.0,
                    _ => -2.0,
                };
            let high = open.max(close) + 1.0;
            let low = open.min(close) - 1.0;
            json!([
                1_700_000_000_000_i64 + i as i64 * 60_000,
                open,
                high,
                low,
                close,
                100.0 + i as f64
            ])
        })
        .collect();
    Value::Array(rows)
}

fn tool_schema<'a>(list: &'a Value, name: &str) -> &'a Value {
    list.as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == name)
        .map(|tool| &tool["inputSchema"])
        .unwrap()
}

#[test]
fn indicators_request_is_built_from_discovered_schema_and_invokes_real_handler() {
    let registry = default_registry();
    let advertised = registry.list_json();
    let schema = tool_schema(&advertised, "trader_indicators");

    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["params"]["type"], "object");
    assert!(
        schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "ohlcv")
    );

    let short_period = registry
        .call(
            "trader_indicators",
            json!({"ohlcv": valid_ohlcv(40), "params": {"rsi": 3}}),
        )
        .unwrap();
    let default_period = registry
        .call("trader_indicators", json!({"ohlcv": valid_ohlcv(40)}))
        .unwrap();

    let short_rsi = short_period["indicators"]["rsi"].as_f64().unwrap();
    let default_rsi = default_period["indicators"]["rsi"].as_f64().unwrap();
    assert_ne!(short_rsi.to_bits(), default_rsi.to_bits());
}

#[test]
fn malformed_or_timestamp_free_market_rows_are_rejected_before_handler_defaults() {
    let registry = default_registry();

    let malformed = registry
        .call(
            "trader_indicators",
            json!({
                "ohlcv": [[1_000, 100.0, 101.0, 99.0, "not-a-number", 10.0]]
            }),
        )
        .unwrap_err();
    assert!(malformed.starts_with("invalid_arguments:"));

    let missing_timestamp = registry
        .call(
            "trader_indicators",
            json!({"ohlcv": [[100.0, 101.0, 99.0, 100.0, 10.0]]}),
        )
        .unwrap_err();
    assert!(missing_timestamp.contains("timestamp/volume are not inferred"));
}

#[test]
fn incoherent_ohlc_and_duplicate_timestamps_are_rejected() {
    let registry = default_registry();

    let incoherent = registry
        .call(
            "trader_backtest",
            json!({
                "ohlcv": [[1_000, 100.0, 99.0, 98.0, 101.0, 10.0]],
                "strategy": "momentum"
            }),
        )
        .unwrap_err();
    assert!(incoherent.contains("incoherent OHLC"));

    let duplicate = registry
        .call(
            "trader_indicators",
            json!({
                "ohlcv": [
                    [1_000, 100.0, 101.0, 99.0, 100.0, 10.0],
                    [1_000, 100.0, 101.0, 99.0, 100.0, 11.0]
                ]
            }),
        )
        .unwrap_err();
    assert!(duplicate.contains("strictly increasing"));
}

#[test]
fn market_data_never_silently_falls_back_or_relabels_mock_data() {
    let registry = default_registry();

    let unsupported_source = registry
        .call("trader_market_data", json!({"source": "binance"}))
        .unwrap_err();
    assert!(unsupported_source.starts_with("invalid_arguments:"));

    let relabel = registry
        .call(
            "trader_market_data",
            json!({"source": "mock", "symbol": "ETH/USDT"}),
        )
        .unwrap_err();
    assert!(relabel.starts_with("unsupported_market_identity:"));

    let wrong_interval = registry
        .call(
            "trader_market_data",
            json!({"source": "mock", "interval": "1h"}),
        )
        .unwrap_err();
    assert!(wrong_interval.starts_with("unsupported_market_interval:"));
}

#[test]
fn accepted_mock_metadata_matches_its_actual_identity_and_cadence() {
    let registry = default_registry();
    let first = registry
        .call(
            "trader_market_data",
            json!({"source": "mock", "symbol": "BTC/USDT", "interval": "1m", "limit": 3, "seed": 17}),
        )
        .unwrap();
    let replay = registry
        .call(
            "trader_market_data",
            json!({"source": "mock", "symbol": "BTC/USDT", "interval": "1m", "limit": 3, "seed": 17}),
        )
        .unwrap();
    let other_seed = registry
        .call(
            "trader_market_data",
            json!({"source": "mock", "symbol": "BTC/USDT", "interval": "1m", "limit": 3, "seed": 18}),
        )
        .unwrap();

    assert_eq!(first["symbol"], "BTC/USDT");
    assert_eq!(first["interval"], "1m");
    let rows = first["ohlcv"].as_array().unwrap();
    assert_eq!(
        rows[1][0].as_i64().unwrap() - rows[0][0].as_i64().unwrap(),
        60_000
    );
    assert_eq!(
        rows[2][0].as_i64().unwrap() - rows[1][0].as_i64().unwrap(),
        60_000
    );
    assert_eq!(first["fingerprint"], replay["fingerprint"]);
    assert_ne!(first["fingerprint"], other_seed["fingerprint"]);
}

#[test]
fn mock_limit_is_a_rejection_boundary_not_a_silent_clamp() {
    let registry = default_registry();
    assert!(
        registry
            .call("trader_market_data", json!({"limit": 5_001}))
            .unwrap_err()
            .contains("not silently clamped")
    );
    assert!(
        registry
            .call("trader_market_data", json!({"limit": 0}))
            .is_err()
    );
}
