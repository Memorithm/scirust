//! Validation at the MCP boundary for trading tools.
//!
//! The historical trading handlers predate strict untrusted-input handling and
//! some of them accept shorthand OHLCV rows by inventing timestamps or filling
//! malformed numeric cells with zero.  The MCP server must not expose those
//! permissive fallbacks to an agent.  This module therefore validates market
//! data before dispatch and only forwards inputs for which the legacy parsing
//! path is lossless.
//!
//! This is intentionally a boundary guard, not a second trading engine. Domain
//! calculations remain in `scirust-trader`.

use serde_json::Value;

const MOCK_SYMBOL: &str = "BTC/USDT";
const MOCK_INTERVAL: &str = "1m";
const MAX_MOCK_CANDLES: u64 = 5_000;

pub(crate) fn prepare_arguments(name: &str, mut arguments: Value) -> Result<Value, String> {
    if !name.starts_with("trader_") {
        return Ok(arguments);
    }

    validate_nested_ohlcv(&arguments, "$")?;

    match name {
        "trader_market_data" => validate_mock_market_data_args(&arguments)?,
        "trader_indicators" => bridge_indicator_params(&mut arguments)?,
        _ => {},
    }

    Ok(arguments)
}

fn validate_nested_ohlcv(value: &Value, path: &str) -> Result<(), String> {
    match value {
        Value::Object(map) => {
            if let Some(ohlcv) = map.get("ohlcv") {
                validate_ohlcv(ohlcv, &format!("{path}.ohlcv"))?;
            }
            for (key, child) in map {
                if key != "ohlcv" {
                    validate_nested_ohlcv(child, &format!("{path}.{key}"))?;
                }
            }
        },
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                validate_nested_ohlcv(child, &format!("{path}[{index}]"))?;
            }
        },
        _ => {},
    }
    Ok(())
}

fn validate_ohlcv(value: &Value, path: &str) -> Result<(), String> {
    let rows = value
        .as_array()
        .ok_or_else(|| invalid(path, "must be an array of OHLCV rows"))?;
    if rows.is_empty() {
        return Err(invalid(path, "must contain at least one candle"));
    }

    let mut previous_ts = None;
    for (index, row) in rows.iter().enumerate() {
        let row_path = format!("{path}[{index}]");
        let (ts_ms, open, high, low, close, volume) = parse_strict_row(row, &row_path)?;

        if let Some(previous) = previous_ts {
            if ts_ms <= previous {
                return Err(invalid(
                    &format!("{row_path}.ts"),
                    "timestamps must be strictly increasing; duplicates and reordering are rejected",
                ));
            }
        }
        previous_ts = Some(ts_ms);

        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return Err(invalid(
                &row_path,
                "open, high, low and close must all be strictly positive",
            ));
        }
        if volume < 0.0 {
            return Err(invalid(&format!("{row_path}.volume"), "must be non-negative"));
        }
        if high < low || high < open || high < close || low > open || low > close {
            return Err(invalid(
                &row_path,
                "incoherent OHLC values: high must bound open/close and low, and low must bound open/close",
            ));
        }
    }

    Ok(())
}

fn parse_strict_row(row: &Value, path: &str) -> Result<(i64, f32, f32, f32, f32, f32), String> {
    if let Some(columns) = row.as_array() {
        if columns.len() != 6 {
            return Err(invalid(
                path,
                "array rows must be exactly [ts_ms, open, high, low, close, volume]; timestamp/volume are not inferred",
            ));
        }
        let ts_ms = columns[0]
            .as_i64()
            .ok_or_else(|| invalid(&format!("{path}[0]"), "timestamp must be an i64 millisecond value"))?;
        return Ok((
            ts_ms,
            finite_f32(&columns[1], &format!("{path}[1]"))?,
            finite_f32(&columns[2], &format!("{path}[2]"))?,
            finite_f32(&columns[3], &format!("{path}[3]"))?,
            finite_f32(&columns[4], &format!("{path}[4]"))?,
            finite_f32(&columns[5], &format!("{path}[5]"))?,
        ));
    }

    let object = row
        .as_object()
        .ok_or_else(|| invalid(path, "row must be a six-column array or an OHLCV object"))?;
    let ts_ms = object
        .get("ts")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid(&format!("{path}.ts"), "required i64 millisecond timestamp is missing or invalid"))?;
    Ok((
        ts_ms,
        finite_required(object.get("open"), &format!("{path}.open"))?,
        finite_required(object.get("high"), &format!("{path}.high"))?,
        finite_required(object.get("low"), &format!("{path}.low"))?,
        finite_required(object.get("close"), &format!("{path}.close"))?,
        finite_required(object.get("volume"), &format!("{path}.volume"))?,
    ))
}

fn finite_required(value: Option<&Value>, path: &str) -> Result<f32, String> {
    finite_f32(
        value.ok_or_else(|| invalid(path, "required numeric field is missing"))?,
        path,
    )
}

fn finite_f32(value: &Value, path: &str) -> Result<f32, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| invalid(path, "must be a JSON number"))?;
    if !number.is_finite() {
        return Err(invalid(path, "must be finite"));
    }
    let narrowed = number as f32;
    if !narrowed.is_finite() {
        return Err(invalid(path, "is outside the finite f32 range used by scirust-trader"));
    }
    Ok(narrowed)
}

fn validate_mock_market_data_args(arguments: &Value) -> Result<(), String> {
    let object = arguments
        .as_object()
        .ok_or_else(|| invalid("$", "trader_market_data arguments must be an object"))?;

    if let Some(source) = object.get("source") {
        if source.as_str() != Some("mock") {
            return Err("unsupported_source: trader_market_data currently supports only source='mock'"
                .to_string());
        }
    }

    if let Some(symbol) = object.get("symbol") {
        if symbol.as_str() != Some(MOCK_SYMBOL) {
            return Err(format!(
                "unsupported_market_identity: mock data is currently generated as {MOCK_SYMBOL}; refusing to relabel it"
            ));
        }
    }
    if let Some(interval) = object.get("interval") {
        if interval.as_str() != Some(MOCK_INTERVAL) {
            return Err(format!(
                "unsupported_market_interval: mock cadence is currently {MOCK_INTERVAL}; refusing to relabel it"
            ));
        }
    }

    if let Some(limit) = object.get("limit") {
        let limit = limit
            .as_u64()
            .ok_or_else(|| invalid("$.limit", "must be a positive integer"))?;
        if limit == 0 || limit > MAX_MOCK_CANDLES {
            return Err(invalid(
                "$.limit",
                &format!("must be in 1..={MAX_MOCK_CANDLES}; values are not silently clamped"),
            ));
        }
    }

    if let Some(start_price) = object.get("start_price") {
        let start_price = finite_f32(start_price, "$.start_price")?;
        if start_price <= 0.0 {
            return Err(invalid("$.start_price", "must be strictly positive"));
        }
    }

    Ok(())
}

fn bridge_indicator_params(arguments: &mut Value) -> Result<(), String> {
    let Some(object) = arguments.as_object_mut() else {
        return Err(invalid("$", "trader_indicators arguments must be an object"));
    };
    let Some(params) = object.get("params").cloned() else {
        return Ok(());
    };
    let params = params
        .as_object()
        .ok_or_else(|| invalid("$.params", "must be an object"))?;

    for (key, value) in params {
        let _ = finite_f32(value, &format!("$.params.{key}"))?;
        if let Some(existing) = object.get(key) {
            if existing != value {
                return Err(invalid(
                    &format!("$.params.{key}"),
                    "conflicts with the same legacy root parameter",
                ));
            }
        } else {
            object.insert(key.clone(), value.clone());
        }
    }
    Ok(())
}

fn invalid(path: &str, message: &str) -> String {
    format!("invalid_arguments: {path}: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_ohlcv_accepts_complete_ordered_rows() {
        let value = json!([
            [1_000, 100.0, 102.0, 99.0, 101.0, 10.0],
            [2_000, 101.0, 103.0, 100.0, 102.0, 11.0]
        ]);
        assert!(validate_ohlcv(&value, "$.ohlcv").is_ok());
    }

    #[test]
    fn shorthand_rows_are_rejected_instead_of_getting_invented_timestamps() {
        let value = json!([[100.0, 102.0, 99.0, 101.0, 10.0]]);
        let error = validate_ohlcv(&value, "$.ohlcv").unwrap_err();
        assert!(error.contains("timestamp/volume are not inferred"));
    }

    #[test]
    fn malformed_numeric_cells_and_ohlc_are_rejected() {
        let text = json!([[1_000, 100.0, 102.0, 99.0, "bad", 10.0]]);
        assert!(validate_ohlcv(&text, "$.ohlcv").is_err());

        let incoherent = json!([[1_000, 100.0, 99.0, 98.0, 101.0, 10.0]]);
        assert!(validate_ohlcv(&incoherent, "$.ohlcv").is_err());
    }

    #[test]
    fn timestamps_must_be_strictly_increasing() {
        let duplicate = json!([
            [1_000, 100.0, 102.0, 99.0, 101.0, 10.0],
            [1_000, 101.0, 103.0, 100.0, 102.0, 11.0]
        ]);
        assert!(validate_ohlcv(&duplicate, "$.ohlcv").is_err());
    }

    #[test]
    fn mock_identity_is_not_silently_relabelled() {
        let error = prepare_arguments(
            "trader_market_data",
            json!({"source":"mock", "symbol":"ETH/USDT"}),
        )
        .unwrap_err();
        assert!(error.starts_with("unsupported_market_identity:"));

        let error = prepare_arguments(
            "trader_market_data",
            json!({"source":"mock", "interval":"1h"}),
        )
        .unwrap_err();
        assert!(error.starts_with("unsupported_market_interval:"));
    }

    #[test]
    fn market_limit_is_rejected_not_clamped() {
        assert!(prepare_arguments("trader_market_data", json!({"limit": 5_001})).is_err());
        assert!(prepare_arguments("trader_market_data", json!({"limit": 0})).is_err());
    }

    #[test]
    fn nested_indicator_params_are_forwarded_to_the_legacy_handler_contract() {
        let prepared = prepare_arguments(
            "trader_indicators",
            json!({
                "ohlcv": [[1_000, 100.0, 101.0, 99.0, 100.0, 10.0]],
                "params": {"rsi": 7, "bb_k": 2.5}
            }),
        )
        .unwrap();
        assert_eq!(prepared["rsi"], json!(7));
        assert_eq!(prepared["bb_k"], json!(2.5));
    }
}
