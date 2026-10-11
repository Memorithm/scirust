//! End-to-end: transpile a real NumPy snippet with `scirust-transpiler`, sign the
//! emitted Rust with a demo signer, and prove the provenance flow on a genuine
//! artifact — verification passes, survives reformatting, and catches tampering,
//! while the signed artifact stays digest-identical to the unsigned one.

use scirust_license::hashsig::{self, MerkleSigner};
use scirust_provenance::{
    BANNER_PREFIX, EMIT_ROOT_HEX, Verdict, canonicalize, emit_root, sign_artifact, strip_banner,
    verify_artifact,
};

fn demo_signer() -> MerkleSigner {
    MerkleSigner::from_seed(&scirust_license::demo_seed(), scirust_license::DEMO_HEIGHT)
}

fn emit_real_artifact() -> String {
    // A small NumPy-subset function the transpiler lowers to deterministic Rust.
    let py = "def f(x, y):\n    return x * y + x\n";
    scirust_transpiler::transpile(py).expect("transpile should succeed")
}

#[test]
fn real_transpiler_artifact_signs_and_verifies() {
    let art = emit_real_artifact();
    assert!(
        art.contains("pub fn f("),
        "unexpected emitter output:\n{art}"
    );

    let signed = sign_artifact(&art, &demo_signer(), 0);
    assert!(signed.contains(BANNER_PREFIX));

    match verify_artifact(&signed, &emit_root())
    {
        Verdict::Verified { leaf } => assert_eq!(leaf, 0),
        other => panic!("expected Verified, got {other:?}"),
    }
}

#[test]
fn signing_is_digest_neutral_on_a_real_artifact() {
    let art = emit_real_artifact();
    let signed = sign_artifact(&art, &demo_signer(), 1);

    // The banner is a comment: the canonical (hashable) form is unchanged, and the
    // only textual difference is the appended provenance line.
    assert_eq!(canonicalize(&signed), canonicalize(&art));
    assert_eq!(strip_banner(&signed).trim(), art.trim());

    // Everything the signer added is a single comment line — Rust-neutral by
    // construction (the lexer discards it before codegen).
    let added: Vec<&str> = signed
        .lines()
        .filter(|l| !art.lines().any(|a| a == *l))
        .collect();
    assert!(
        added.iter().all(|l| l.trim_start().starts_with("//")),
        "signing added non-comment text: {added:?}"
    );
}

#[test]
fn reformatting_the_real_artifact_still_verifies() {
    let art = emit_real_artifact();
    let signed = sign_artifact(&art, &demo_signer(), 2);

    // Whitespace churn + an extra comment must not break verification.
    let reformatted = format!("// vendored copy\n{}", signed.replace('\n', "\n  "));
    assert_eq!(
        verify_artifact(&reformatted, &emit_root()),
        Verdict::Verified { leaf: 2 }
    );
}

#[test]
fn tampering_with_the_real_artifact_is_caught() {
    let art = emit_real_artifact();
    let signed = sign_artifact(&art, &demo_signer(), 3);

    // Flip a `*` to `+` somewhere in the emitted body — a genuine token edit.
    assert!(
        signed.contains('*'),
        "artifact should contain a '*' to flip"
    );
    let tampered = signed.replacen('*', "+", 1);
    assert_eq!(verify_artifact(&tampered, &emit_root()), Verdict::Forged);
}

#[test]
fn cli_requires_an_explicit_production_root_and_demo_is_opt_in() {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    let signed = sign_artifact(&emit_real_artifact(), &demo_signer(), 4);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "scirust-provenance-{}-{nonce}.rs",
        std::process::id()
    ));
    std::fs::write(&path, signed).expect("write temporary signed artifact");
    let path_arg = path.to_string_lossy().into_owned();

    let without_root = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args(["verify", &path_arg])
        .status()
        .expect("run verifier without root");
    assert_eq!(without_root.code(), Some(2));

    let demo_disguised_as_production = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args(["verify", "--root", EMIT_ROOT_HEX, &path_arg])
        .status()
        .expect("run verifier with demo root in production mode");
    assert_eq!(demo_disguised_as_production.code(), Some(2));

    let explicit_demo = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args(["verify", "--demo", &path_arg])
        .status()
        .expect("run verifier in explicit demo mode");
    assert!(explicit_demo.success());

    std::fs::remove_file(path).expect("remove temporary signed artifact");
}

#[test]
fn production_cli_requires_a_published_record_and_refuses_ots_leaf_reuse() {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_nanos();
    let prefix = format!("scirust-provenance-prod-{}-{nonce}", std::process::id());
    let artifact = std::env::temp_dir().join(format!("{prefix}.rs"));
    let journal = std::env::temp_dir().join(format!("{prefix}.journal"));
    let record = std::env::temp_dir().join(format!("{prefix}.record"));
    let conflicting_record = std::env::temp_dir().join(format!("{prefix}-conflict.record"));

    std::fs::write(&artifact, "pub fn value() -> u32 { 1 }\n")
        .expect("write unsigned production artifact");

    let seed = [0x11u8; 32];
    let seed_hex = hashsig::hex_encode(&seed);
    let signer = MerkleSigner::from_seed(&seed, 4);
    let root_hex = hashsig::hex_encode(&signer.root());
    let artifact_arg = artifact.to_string_lossy().into_owned();
    let journal_arg = journal.to_string_lossy().into_owned();
    let record_arg = record.to_string_lossy().into_owned();
    let conflicting_record_arg = conflicting_record.to_string_lossy().into_owned();

    let sign = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args([
            "sign",
            "--seed",
            &seed_hex,
            "--height",
            "4",
            "--leaf",
            "0",
            "--journal",
            &journal_arg,
            "--record-out",
            &record_arg,
            "--in-place",
            &artifact_arg,
        ])
        .status()
        .expect("run production signer");
    assert!(sign.success());

    let missing_record = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args(["verify", "--root", &root_hex, &artifact_arg])
        .status()
        .expect("run production verifier without published record");
    assert_eq!(missing_record.code(), Some(2));

    let verify = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args([
            "verify",
            "--root",
            &root_hex,
            "--published-record",
            &record_arg,
            &artifact_arg,
        ])
        .status()
        .expect("run production verifier with published record");
    assert!(verify.success());

    std::fs::write(&artifact, "pub fn value() -> u32 { 2 }\n").expect("write conflicting artifact");
    let reuse = Command::new(env!("CARGO_BIN_EXE_prov"))
        .args([
            "sign",
            "--seed",
            &seed_hex,
            "--height",
            "4",
            "--leaf",
            "0",
            "--journal",
            &journal_arg,
            "--record-out",
            &conflicting_record_arg,
            "--in-place",
            &artifact_arg,
        ])
        .status()
        .expect("run signer with reused OTS leaf");
    assert_eq!(reuse.code(), Some(2));
    assert!(!conflicting_record.exists());

    for path in [&artifact, &journal, &record]
    {
        std::fs::remove_file(path).expect("remove temporary provenance file");
    }
}
