//! Structural adapter for the frozen SML PVP6 fixture contract.
//!
//! This executable intentionally validates the cross-repository interchange
//! boundary only. It does not replace SML's independent semantic oracle and
//! it does not make a performance or architecture claim.

use std::collections::HashSet;
use std::env;
use std::fs;

const DENSITIES: [u32; 4] = [4, 32, 256, 1024];
const SCHEDULES: [&str; 3] = ["AFFINE_UNIFORM", "LOW_POPCOUNT", "HIGH_POPCOUNT"];
const BANK_WORDS: usize = 2048;
const QUERY_ROWS: usize = 6144;
const OUTPUT_ROWS: usize = 24_576;
const GATES: u32 = 16;

fn usage() -> ! {
    eprintln!("usage: pvp6_fixture_check --fixture PATH");
    std::process::exit(2);
}

fn fixture_path() -> String {
    let args: Vec<String> = env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == "--fixture")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| usage())
}

fn parse_word(raw: &str, field: &str) -> u64 {
    assert_eq!(raw.len(), 16, "{field} must contain exactly 16 hex digits");
    assert!(
        raw.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{field} must use lowercase hexadecimal"
    );
    u64::from_str_radix(raw, 16).unwrap_or_else(|_| panic!("{field} is not hexadecimal"))
}

fn main() {
    let path = fixture_path();
    let text = fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    assert!(text.is_ascii(), "fixture must be ASCII-compatible UTF-8");
    assert!(!text.starts_with('\u{feff}'), "fixture must not contain a BOM");
    assert!(text.ends_with('\n'), "fixture must end with exactly one LF");
    assert!(!text.ends_with("\n\n"), "fixture must contain one trailing LF");

    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines.first().copied(),
        Some("PVP6_FIXTURE,1,K,2048,G,16,word_bits,64,bit_order,lsb0"),
        "invalid fixture header"
    );

    let mut cursor = 1usize;
    let mut banks = HashSet::new();
    for density in DENSITIES {
        for gate in 0..GATES {
            for word in 0..32usize {
                let fields: Vec<&str> = lines[cursor].split(',').collect();
                assert_eq!(fields.len(), 6, "BANK record must have six fields");
                assert_eq!(fields[0], "BANK");
                assert_eq!(fields[1].parse::<u32>().unwrap(), density);
                assert_eq!(fields[2].parse::<u32>().unwrap(), gate);
                assert_eq!(fields[3].parse::<usize>().unwrap(), word);
                let coefficient = parse_word(fields[4], "coefficient_hex");
                let truth = parse_word(fields[5], "truth_hex");
                banks.insert((density, gate, word));
                let _ = (coefficient, truth);
                cursor += 1;
            }
        }
    }
    assert_eq!(banks.len(), 2048, "BANK records must be unique");

    let mut queries = HashSet::new();
    for schedule in SCHEDULES {
        for position in 0..BANK_WORDS {
            let fields: Vec<&str> = lines[cursor].split(',').collect();
            assert_eq!(fields.len(), 4, "QUERY record must have four fields");
            assert_eq!(fields[0], "QUERY");
            assert_eq!(fields[1], schedule);
            assert_eq!(fields[2].parse::<usize>().unwrap(), position);
            let address = fields[3].parse::<usize>().unwrap();
            assert!(address < 2048, "QUERY address out of range");
            assert!(queries.insert((schedule, position)), "duplicate QUERY record");
            cursor += 1;
        }
    }
    assert_eq!(queries.len(), QUERY_ROWS, "unexpected QUERY count");

    let mut outputs = HashSet::new();
    for density in DENSITIES {
        for schedule in SCHEDULES {
            for position in 0..BANK_WORDS {
                let fields: Vec<&str> = lines[cursor].split(',').collect();
                assert_eq!(fields.len(), 5, "OUTPUT record must have five fields");
                assert_eq!(fields[0], "OUTPUT");
                assert_eq!(fields[1].parse::<u32>().unwrap(), density);
                assert_eq!(fields[2], schedule);
                assert_eq!(fields[3].parse::<usize>().unwrap(), position);
                let word = parse_word(fields[4], "word_hex");
                assert_eq!(word >> GATES, 0, "OUTPUT padding bits must be zero");
                assert!(outputs.insert((density, schedule, position)), "duplicate OUTPUT record");
                cursor += 1;
            }
        }
    }
    assert_eq!(outputs.len(), OUTPUT_ROWS, "unexpected OUTPUT count");

    let end: Vec<&str> = lines[cursor].split(',').collect();
    assert_eq!(end, ["END", "banks", "4", "bank_words", "2048", "query_rows", "6144", "output_rows", "24576"]);
    assert_eq!(cursor + 1, lines.len(), "unexpected records after END");

    println!(
        "{{\"contract\":\"sml-pvp6-anf-fixture/v1\",\"banks\":2048,\"query_rows\":6144,\"output_rows\":24576,\"status\":\"structural-ok\",\"semantic_oracle\":\"sml-owned\"}}"
    );
}
