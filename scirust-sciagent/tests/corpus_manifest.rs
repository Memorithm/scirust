use scirust_sciagent::artifact_provenance::artifact_sha256;
use scirust_sciagent::train::corpus_manifest::{Domain, Manifest, Record, Split, validate};

fn row(id: &str, split: Split, domain: Domain) -> Record {
    Record {
        id: id.into(),
        source_uri: "synthetic://test-only".into(),
        revision: "fixture-v1".into(),
        license: "test-only".into(),
        rights_review: "synthetic fixture, not a corpus".into(),
        group: id.into(),
        text: id.into(),
        sha256: artifact_sha256(id.as_bytes()).as_str().into(),
        split,
        domain,
    }
}
fn finance(id: &str, decision_at: u64, label_end: u64, split: Split) -> Record {
    row(
        id,
        split,
        Domain::Finance {
            available_at: decision_at,
            decision_at,
            label_end,
        },
    )
}
fn manifest(records: Vec<Record>) -> Manifest {
    Manifest {
        schema_version: 1,
        train_end: 100,
        validation_end: 200,
        embargo_ms: 10,
        records,
    }
}
#[test]
fn chronology_purges_horizons_and_embargo_without_reassigning() {
    let m = manifest(vec![
        finance("a", 50, 99, Split::Train),
        finance("b", 90, 100, Split::Train),
        finance("c", 100, 105, Split::Validation),
        finance("d", 110, 199, Split::Validation),
        finance("e", 190, 200, Split::Validation),
        finance("f", 200, 205, Split::Test),
        finance("g", 210, 220, Split::Test),
    ]);
    let got = validate(&m).unwrap();
    assert_eq!(
        got.iter().map(|r| r.split).collect::<Vec<_>>(),
        vec![
            Some(Split::Train),
            None,
            None,
            Some(Split::Validation),
            None,
            None,
            Some(Split::Test)
        ]
    );
}
#[test]
fn future_information_and_wrong_split_are_rejected() {
    let mut r = finance("a", 50, 60, Split::Test);
    assert!(validate(&manifest(vec![r.clone()])).is_err());
    r.split = Split::Train;
    r.domain = Domain::Finance {
        available_at: 51,
        decision_at: 50,
        label_end: 60,
    };
    assert!(validate(&manifest(vec![r])).is_err());
}
#[test]
fn upstream_project_and_event_cannot_cross_splits() {
    for financial in [false, true]
    {
        let mut a = if financial
        {
            finance("a", 50, 60, Split::Train)
        }
        else
        {
            row("a", Split::Train, Domain::Rust)
        };
        let mut b = if financial
        {
            finance("b", 150, 160, Split::Validation)
        }
        else
        {
            row("b", Split::Validation, Domain::Rust)
        };
        a.group = "canonical-origin".into();
        b.group = a.group.clone();
        assert!(validate(&manifest(vec![a, b])).is_err());
    }
}
#[test]
fn content_tampering_duplicates_and_missing_review_are_rejected() {
    let a = row("a", Split::Train, Domain::Rust);
    let mut b = a.clone();
    b.id = "different-id".into();
    assert!(validate(&manifest(vec![a.clone(), b])).is_err());
    let mut b = a.clone();
    b.text = "tampered".into();
    assert!(validate(&manifest(vec![b])).is_err());
    let mut b = a;
    b.rights_review = " ".into();
    assert!(validate(&manifest(vec![b])).is_err());
}
#[test]
fn strict_schema_empty_manifest_and_overflow_fail_closed() {
    assert!(serde_json::from_str::<Manifest>(r#"{"schema_version":1,"train_end":100,"validation_end":200,"embargo_ms":0,"records":[],"typo":true}"#).is_err());
    assert!(validate(&manifest(vec![])).is_err());
    let mut m = manifest(vec![row("a", Split::Train, Domain::Rust)]);
    m.embargo_ms = u64::MAX;
    assert!(validate(&m).is_err());
}
#[test]
fn multiple_files_of_one_project_may_share_a_split() {
    let a = row("a", Split::Train, Domain::Rust);
    let mut b = row("b", Split::Train, Domain::Rust);
    b.group = a.group.clone();
    assert_eq!(validate(&manifest(vec![a, b])).unwrap().len(), 2);
}

#[test]
fn actual_cli_accepts_valid_manifest_and_rejects_tampering() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "sciagent-corpus-check-{}-{stamp}.json",
        std::process::id()
    ));
    let mut m = manifest(vec![row("synthetic", Split::Train, Domain::Rust)]);
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    let good = std::process::Command::new(env!("CARGO_BIN_EXE_sciagent-corpus-check"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(good.status.success());
    let report: serde_json::Value = serde_json::from_slice(&good.stdout).unwrap();
    assert_eq!(report["training_performed"], false);
    assert_eq!(report["records"][0]["split"], "train");
    m.records[0].text = "changed after hash".into();
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_sciagent-corpus-check"))
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(!bad.status.success());
    assert!(bad.stdout.is_empty());
}
