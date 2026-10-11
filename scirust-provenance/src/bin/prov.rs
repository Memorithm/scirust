//! `prov` — offline signer and public verifier for SciRust-emitted artifacts.
//!
//! ```text
//! prov sign   --seed <hex64> --journal <path> --record-out <path> [--leaf <n>] [--height <h>] [--in-place] <file>
//! prov sign   --seed-file <path> --journal <path> --record-out <path> [--leaf <n>] [--height <h>] [--in-place] <file>
//! prov verify --root <hex64> --published-record <path> <file>
//! prov verify --demo <file>
//! ```
//!
//! `sign` is the **offline vendor step** — it needs the secret master seed and
//! must run on a trusted host (an HSM export, never a shipped binary). By default
//! it writes the signed artifact to stdout; `--in-place` rewrites `<file>`.
//!
//! `verify` ships to anyone: production verification requires an explicit public
//! root and refuses the public demo root. `--demo` is the only way to opt into the
//! pinned demonstration root. Exit code `0` means verified, `1` means
//! the banner is missing/malformed/forged, `2` means a usage error.

use scirust_license::hashsig::{self, Hash, MerkleSigner};
use scirust_provenance::{Verdict, digest_of_source, sign_artifact, verify_artifact};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
prov — SciRust codegen provenance tool

USAGE:
  prov sign   --seed <hex64> | --seed-file <path> --journal <path> --record-out <path> [--leaf <n>] [--height <h>] [--in-place] <file>
  prov verify --root <hex64> --published-record <path> <file>
  prov verify --demo <file>

SIGN (offline, needs the secret seed):
  --seed <hex64>     32-byte master seed as 64 hex chars
  --seed-file <path> read the seed (hex, whitespace-trimmed) from a file
  --leaf <n>         one-time leaf index (default 0); never reuse for two artifacts
  --height <h>       Merkle height, 1..=20 (default 10)
  --journal <path>   persistent OTS leaf ledger; reuse for a different digest is refused
  --record-out <path> write the immutable record that must be published with the artifact
  --in-place         rewrite <file> instead of printing to stdout

VERIFY (public, needs only the root and published record):
  --root <hex64>     explicit trusted 32-byte production Merkle root
  --published-record <path> exact root/leaf/digest record published before distribution
  --demo             explicitly use the public demo root (never attribution evidence)

Exit: 0 verified · 1 not verified · 2 usage error";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str)
    {
        Some("sign") => cmd_sign(&args[1..]),
        Some("verify") => cmd_verify(&args[1..]),
        Some("-h") | Some("--help") | None =>
        {
            println!("{USAGE}");
            ExitCode::SUCCESS
        },
        Some(other) =>
        {
            eprintln!("prov: unknown subcommand `{other}`\n\n{USAGE}");
            ExitCode::from(2)
        },
    }
}

/// A single `--flag value` / positional parser — no external dependency, keeping
/// the tool auditable in the SciRust spirit.
struct Args {
    seed: Option<String>,
    seed_file: Option<String>,
    root: Option<String>,
    journal: Option<String>,
    record_out: Option<String>,
    published_record: Option<String>,
    leaf: u32,
    height: u32,
    in_place: bool,
    demo: bool,
    file: Option<String>,
}

fn parse(rest: &[String]) -> Result<Args, String> {
    let mut a = Args {
        seed: None,
        seed_file: None,
        root: None,
        journal: None,
        record_out: None,
        published_record: None,
        leaf: 0,
        height: 10,
        in_place: false,
        demo: false,
        file: None,
    };
    let mut it = rest.iter();
    while let Some(tok) = it.next()
    {
        match tok.as_str()
        {
            "--seed" => a.seed = Some(next(&mut it, "--seed")?),
            "--seed-file" => a.seed_file = Some(next(&mut it, "--seed-file")?),
            "--root" => a.root = Some(next(&mut it, "--root")?),
            "--journal" => a.journal = Some(next(&mut it, "--journal")?),
            "--record-out" => a.record_out = Some(next(&mut it, "--record-out")?),
            "--published-record" => a.published_record = Some(next(&mut it, "--published-record")?),
            "--leaf" =>
            {
                a.leaf = next(&mut it, "--leaf")?
                    .parse()
                    .map_err(|_| "--leaf must be a u32".to_string())?
            },
            "--height" =>
            {
                a.height = next(&mut it, "--height")?
                    .parse()
                    .map_err(|_| "--height must be a u32".to_string())?
            },
            "--in-place" => a.in_place = true,
            "--demo" => a.demo = true,
            other if other.starts_with("--") =>
            {
                return Err(format!("unknown flag `{other}`"));
            },
            positional =>
            {
                if a.file.is_some()
                {
                    return Err("expected exactly one <file>".to_string());
                }
                a.file = Some(positional.to_string());
            },
        }
    }
    Ok(a)
}

fn next(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    it.next()
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn seed_bytes(a: &Args) -> Result<[u8; 32], String> {
    let hex = if let Some(h) = &a.seed
    {
        h.trim().to_string()
    }
    else if let Some(path) = &a.seed_file
    {
        std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read seed file {path}: {e}"))?
            .trim()
            .to_string()
    }
    else
    {
        return Err("sign needs --seed <hex64> or --seed-file <path>".to_string());
    };
    let bytes = hashsig::hex_decode(&hex).ok_or("seed is not valid hex")?;
    if bytes.len() != 32
    {
        return Err(format!("seed must be 32 bytes (got {})", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

const RECORD_VERSION: &str = "scirust-prov-record-v1";

struct JournalLock {
    path: PathBuf,
}

impl JournalLock {
    fn acquire(journal: &Path) -> Result<Self, String> {
        let mut lock = journal.as_os_str().to_os_string();
        lock.push(".lock");
        let path = PathBuf::from(lock);
        std::fs::create_dir(&path).map_err(|e| {
            format!(
                "cannot acquire OTS journal lock {}: {e}; if it is stale, verify no signer is running before removing it",
                path.display()
            )
        })?;
        Ok(Self { path })
    }
}

impl Drop for JournalLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.path);
    }
}

fn record_line(root: &Hash, leaf: u32, digest: &Hash) -> String {
    format!(
        "{RECORD_VERSION}\t{}\t{leaf}\t{}\n",
        hashsig::hex_encode(root),
        hashsig::hex_encode(digest)
    )
}

fn decode_hash(value: &str, field: &str) -> Result<Hash, String> {
    let bytes = hashsig::hex_decode(value).ok_or_else(|| format!("{field} is not valid hex"))?;
    if bytes.len() != 32
    {
        return Err(format!("{field} must be 32 bytes"));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn parse_record(line: &str) -> Result<(Hash, u32, Hash), String> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 4 || fields[0] != RECORD_VERSION
    {
        return Err("record must contain version, root, leaf and digest".to_string());
    }
    let root = decode_hash(fields[1], "record root")?;
    let leaf = fields[2]
        .parse::<u32>()
        .map_err(|_| "record leaf must be a u32".to_string())?;
    let digest = decode_hash(fields[3], "record digest")?;
    Ok((root, leaf, digest))
}

fn reserve_ots_leaf(journal: &Path, root: &Hash, leaf: u32, digest: &Hash) -> Result<(), String> {
    let _lock = JournalLock::acquire(journal)?;
    let existing = match std::fs::read_to_string(journal)
    {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) =>
        {
            return Err(format!(
                "cannot read OTS journal {}: {e}",
                journal.display()
            ));
        },
    };
    for (index, line) in existing.lines().enumerate()
    {
        if line.trim().is_empty()
        {
            continue;
        }
        let (seen_root, seen_leaf, seen_digest) = parse_record(line)
            .map_err(|e| format!("malformed OTS journal line {}: {e}", index + 1))?;
        if seen_root == *root && seen_leaf == leaf
        {
            if seen_digest == *digest
            {
                return Ok(());
            }
            return Err(format!(
                "OTS leaf {leaf} is already reserved for a different digest under this root"
            ));
        }
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(journal)
        .map_err(|e| format!("cannot open OTS journal {}: {e}", journal.display()))?;
    file.write_all(record_line(root, leaf, digest).as_bytes())
        .map_err(|e| format!("cannot append OTS journal {}: {e}", journal.display()))?;
    file.sync_all()
        .map_err(|e| format!("cannot sync OTS journal {}: {e}", journal.display()))
}

fn write_published_record(
    path: &Path,
    root: &Hash,
    leaf: u32,
    digest: &Hash,
) -> Result<(), String> {
    let expected = record_line(root, leaf, digest);
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut file) =>
        {
            file.write_all(expected.as_bytes())
                .map_err(|e| format!("cannot write published record {}: {e}", path.display()))?;
            file.sync_all()
                .map_err(|e| format!("cannot sync published record {}: {e}", path.display()))
        },
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists =>
        {
            let current = std::fs::read_to_string(path).map_err(|read| {
                format!("cannot read published record {}: {read}", path.display())
            })?;
            if current == expected
            {
                Ok(())
            }
            else
            {
                Err(format!(
                    "published record {} already exists with different content",
                    path.display()
                ))
            }
        },
        Err(e) => Err(format!(
            "cannot create published record {}: {e}",
            path.display()
        )),
    }
}

fn published_record_matches(
    path: &Path,
    root: &Hash,
    leaf: u32,
    digest: &Hash,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read published record {}: {e}", path.display()))?;
    let mut records = text.lines().filter(|line| !line.trim().is_empty());
    let line = records
        .next()
        .ok_or_else(|| "published record is empty".to_string())?;
    if records.next().is_some()
    {
        return Err("published record must contain exactly one record".to_string());
    }
    let (published_root, published_leaf, published_digest) = parse_record(line)?;
    if published_root != *root || published_leaf != leaf || published_digest != *digest
    {
        return Err("published record does not match the verified artifact".to_string());
    }
    Ok(())
}

fn cmd_sign(rest: &[String]) -> ExitCode {
    let a = match parse(rest)
    {
        Ok(a) => a,
        Err(e) => return usage_err(&e),
    };
    let Some(file) = a.file.clone()
    else
    {
        return usage_err("sign needs a <file>");
    };
    let seed = match seed_bytes(&a)
    {
        Ok(s) => s,
        Err(e) => return usage_err(&e),
    };
    let src = match std::fs::read_to_string(&file)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("prov: cannot read {file}: {e}");
            return ExitCode::from(2);
        },
    };
    let signer = MerkleSigner::from_seed(&seed, a.height);
    if a.leaf as usize >= signer.capacity()
    {
        return usage_err(&format!(
            "--leaf {} out of range for height {} (capacity {})",
            a.leaf,
            a.height,
            signer.capacity()
        ));
    }
    let Some(journal) = a.journal.as_deref()
    else
    {
        return usage_err("production signing requires --journal <path>");
    };
    let Some(record_out) = a.record_out.as_deref()
    else
    {
        return usage_err("production signing requires --record-out <path>");
    };
    let root = signer.root();
    let digest = digest_of_source(&src);
    if let Err(e) = reserve_ots_leaf(Path::new(journal), &root, a.leaf, &digest)
    {
        return usage_err(&e);
    }
    if let Err(e) = write_published_record(Path::new(record_out), &root, a.leaf, &digest)
    {
        return usage_err(&e);
    }
    let signed = sign_artifact(&src, &signer, a.leaf);

    if a.in_place
    {
        if let Err(e) = std::fs::write(&file, &signed)
        {
            eprintln!("prov: cannot write {file}: {e}");
            return ExitCode::from(2);
        }
        eprintln!(
            "prov: signed {file} in place (root {}, leaf {})",
            hashsig::hex_encode(&signer.root()[..4]),
            a.leaf
        );
    }
    else
    {
        print!("{signed}");
    }
    ExitCode::SUCCESS
}

fn verification_root(a: &Args) -> Result<[u8; 32], String> {
    match (&a.root, a.demo)
    {
        (Some(_), true) =>
        {
            Err("--root and --demo are mutually exclusive".to_string())
        },
        (None, true) => Ok(scirust_license::demo_root()),
        (None, false) =>
        {
            Err("production verification requires --root <hex64>; use --demo only for non-attribution demonstrations".to_string())
        },
        (Some(hex), false) =>
        {
            let bytes = hashsig::hex_decode(hex.trim())
                .ok_or_else(|| "--root must be 64 hex chars (32 bytes)".to_string())?;
            if bytes.len() != 32
            {
                return Err("--root must be 64 hex chars (32 bytes)".to_string());
            }
            let mut root = [0u8; 32];
            root.copy_from_slice(&bytes);
            if root == scirust_license::demo_root()
            {
                return Err("production verification refuses the public demo root; use --demo only for non-attribution demonstrations".to_string());
            }
            Ok(root)
        },
    }
}

fn cmd_verify(rest: &[String]) -> ExitCode {
    let a = match parse(rest)
    {
        Ok(a) => a,
        Err(e) => return usage_err(&e),
    };
    let Some(file) = a.file.clone()
    else
    {
        return usage_err("verify needs a <file>");
    };
    let root = match verification_root(&a)
    {
        Ok(root) => root,
        Err(e) => return usage_err(&e),
    };
    let published_record = if a.demo
    {
        if a.published_record.is_some()
        {
            return usage_err("--published-record is a production-only verification input");
        }
        None
    }
    else
    {
        let Some(path) = a.published_record.as_deref()
        else
        {
            return usage_err("production verification requires --published-record <path>");
        };
        Some(path)
    };
    let src = match std::fs::read_to_string(&file)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("prov: cannot read {file}: {e}");
            return ExitCode::from(2);
        },
    };
    match verify_artifact(&src, &root)
    {
        Verdict::Verified { leaf } =>
        {
            if let Some(path) = published_record
            {
                let digest = digest_of_source(&src);
                if let Err(e) = published_record_matches(Path::new(path), &root, leaf, &digest)
                {
                    println!("UNPUBLISHED {file}  ({e})");
                    return ExitCode::from(1);
                }
            }
            println!(
                "VERIFIED  {file}  (root {}, leaf {leaf})",
                hashsig::hex_encode(&root[..4])
            );
            ExitCode::SUCCESS
        },
        Verdict::NoBanner =>
        {
            println!("NO-BANNER {file}  (no srl-emit provenance mark found)");
            ExitCode::from(1)
        },
        Verdict::MalformedBanner =>
        {
            println!("MALFORMED {file}  (banner present but signature unreadable)");
            ExitCode::from(1)
        },
        Verdict::Forged =>
        {
            println!("FORGED    {file}  (banner does not verify against this root)");
            ExitCode::from(1)
        },
    }
}

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("prov: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}
