//! Opt-in SciAgent to FLAT RAE-9 research bridge for genuine on-tape GQA.
//! Inference only: no training-gradient promise, WGPU backend change or speedup.
//! The operator supplies an exact-SHA-pinned RAE-9 executable; no auto-download.
use scirust_core::autodiff::reverse::Tensor;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static CAPTURE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct Rae9ResearchBridge {
    binary: PathBuf,
    pub page_size: usize,
    pub query_head: usize,
    observations: Mutex<Vec<Rae9HeadEvidence>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rae9HeadEvidence {
    pub route: String,
    pub keys: usize,
    pub executed_keys: usize,
    pub scored_keys: usize,
    pub maximum_output_error: f64,
    pub native_component_delta: f64,
    pub injected: bool,
}

impl Rae9ResearchBridge {
    pub fn new(binary: PathBuf, page_size: usize, query_head: usize) -> Result<Self, String> {
        if !binary.is_file() || page_size == 0 || page_size > 4096
        {
            return Err("RAE9_INVALID_CONFIGURATION".into());
        }
        Ok(Self {
            binary,
            page_size,
            query_head,
            observations: Mutex::new(Vec::new()),
        })
    }

    /// Read non-secret numerical execution evidence for the latest experiment.
    pub fn observations(&self) -> Result<Vec<Rae9HeadEvidence>, String> {
        self.observations
            .lock()
            .map(|guard| guard.clone())
            .map_err(|_| "RAE9_EVIDENCE_LOCK_POISONED".into())
    }

    /// Input Q/K are already RoPE-rotated by SciAgent's differentiable tape.
    /// V is projected; no causal padding/additive mask is admitted.
    /// The original SciAgent dense head is the independent native oracle.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        native: &Tensor,
        q_heads: usize,
        kv_heads: usize,
        head: usize,
        scale: f32,
    ) -> Result<(Option<Vec<f32>>, Rae9HeadEvidence), String> {
        let n = q.rows;
        let d = q.cols;
        if head != self.query_head
            || !(2..=4096).contains(&n)
            || !(1..=256).contains(&d)
            || k.rows != n
            || k.cols != d
            || v.rows != n
            || !(1..=256).contains(&v.cols)
            || native.rows != n
            || native.cols != v.cols
            || q_heads == 0
            || kv_heads == 0
            || !q_heads.is_multiple_of(kv_heads)
            || head >= q_heads
            || self.page_size > n
            || n.div_ceil(self.page_size) > 512
            || !scale.is_finite()
            || scale < 0.0
        {
            return Err("RAE9_INVALID_CAUSAL_GQA_SHAPE".into());
        }
        if [&q.data, &k.data, &v.data, &native.data]
            .iter()
            .any(|x| !x.iter().all(|y| y.is_finite()))
        {
            return Err("RAE9_NONFINITE_TENSOR".into());
        }

        let capture = Capture::create()?;
        {
            let mut f = OpenOptions::new()
                .write(true)
                .open(&capture.path)
                .map_err(|e| format!("RAE9_OPEN_CAPTURE: {e}"))?;
            let tick = CAPTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let kv_head = head / (q_heads / kv_heads);
            writeln!(f,
                "RAE9QKV1 sciagent {n} {d} {} {} {:.17e} {tick} {q_heads} {kv_heads} {head} {kv_head}",
                v.cols, self.page_size, f64::from(scale)
            ).map_err(|e| format!("RAE9_WRITE_HEADER: {e}"))?;
            for row in [
                &q.data[(n - 2) * d..(n - 1) * d],
                &q.data[(n - 1) * d..n * d],
                &k.data,
                &v.data,
            ]
            {
                for value in row
                {
                    write!(f, "{:.17e} ", f64::from(*value))
                        .map_err(|e| format!("RAE9_WRITE_DATA: {e}"))?;
                }
                writeln!(f).map_err(|e| format!("RAE9_WRITE_EOL: {e}"))?;
            }
            f.flush().map_err(|e| format!("RAE9_FLUSH: {e}"))?;
        }
        let text = run_child(&self.binary, &capture.path)?;
        let (mut evidence, output) = parse_result(&text, v.cols)?;
        if evidence.keys != n || evidence.executed_keys > n || evidence.scored_keys > n
        {
            return Err("RAE9_INVALID_RESULT_COUNTS".into());
        }
        let reference = &native.data[(n - 1) * v.cols..n * v.cols];
        let difference = output
            .iter()
            .zip(reference)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0_f64, f64::max);
        if !difference.is_finite() || difference > evidence.maximum_output_error + 0.001
        {
            return Err(format!(
                "RAE9_NATIVE_HEAD_PARITY_FAILED difference={difference:.8e} bound={:.8e}",
                evidence.maximum_output_error
            ));
        }
        let admitted = evidence.route == "bounded_sparse";
        if !admitted && !evidence.route.starts_with("dense_")
        {
            return Err("RAE9_UNKNOWN_RESULT_ROUTE".into());
        }
        evidence.native_component_delta = difference;
        evidence.injected = admitted;
        self.observations
            .lock()
            .map_err(|_| "RAE9_EVIDENCE_LOCK_POISONED")?
            .push(evidence.clone());
        Ok((admitted.then_some(output), evidence))
    }
}

struct Capture {
    path: PathBuf,
}
impl Capture {
    fn create() -> Result<Self, String> {
        for _ in 0..8
        {
            let seq = CAPTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("sciagent-rae9-{}-{seq}.txt", std::process::id()));
            let mut options = OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path)
            {
                Ok(_) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(e) => return Err(format!("RAE9_CAPTURE_CREATE: {e}")),
            }
        }
        Err("RAE9_CAPTURE_NAME_COLLISION".into())
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn run_child(binary: &Path, snapshot: &Path) -> Result<String, String> {
    let mut child = Command::new(binary)
        .arg(snapshot)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("RAE9_SPAWN: {e}"))?;
    let begin = Instant::now();
    loop
    {
        match child.try_wait()
        {
            Ok(Some(status)) =>
            {
                let output = child
                    .wait_with_output()
                    .map_err(|e| format!("RAE9_WAIT_OUTPUT: {e}"))?;
                if !status.success()
                {
                    return Err(format!("RAE9_REJECTED: {status}"));
                }
                if output.stdout.len() > 16_384
                {
                    return Err("RAE9_OVERSIZE_OUTPUT".into());
                }
                return String::from_utf8(output.stdout).map_err(|_| "RAE9_NONUTF8_RESULT".into());
            },
            Ok(None) if begin.elapsed() < Duration::from_secs(15) =>
            {
                std::thread::sleep(Duration::from_millis(25))
            },
            Ok(None) =>
            {
                let _ = child.kill();
                let _ = child.wait();
                return Err("RAE9_TIMEOUT".into());
            },
            Err(e) =>
            {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("RAE9_POLL: {e}"));
            },
        }
    }
}

fn parse_result(text: &str, width: usize) -> Result<(Rae9HeadEvidence, Vec<f32>), String> {
    let mut record = None;
    let mut output = None;
    for line in text.lines()
    {
        if let Some(rest) = line.strip_prefix("RAE9_RESULT ")
        {
            if record.is_some()
            {
                return Err("RAE9_DUPLICATE_RESULT".into());
            }
            let kv: std::collections::HashMap<&str, &str> = rest
                .split_whitespace()
                .filter_map(|x| x.split_once('='))
                .collect();
            let value = |k: &str| -> Result<&str, String> {
                kv.get(k)
                    .copied()
                    .ok_or_else(|| format!("RAE9_MISSING_{k}"))
            };
            let size = |k: &str| -> Result<usize, String> {
                value(k)?.parse().map_err(|_| format!("RAE9_BAD_{k}"))
            };
            let bound: f64 = value("max_error_bound")?
                .parse()
                .map_err(|_| "RAE9_BAD_ERROR_BOUND".to_string())?;
            if !bound.is_finite() || bound < 0.0
            {
                return Err("RAE9_INVALID_ERROR_BOUND".into());
            }
            record = Some(Rae9HeadEvidence {
                route: value("route")?.to_owned(),
                keys: size("keys")?,
                executed_keys: size("executed_keys")?,
                scored_keys: size("scored_keys")?,
                maximum_output_error: bound,
                native_component_delta: 0.0,
                injected: false,
            });
        }
        if let Some(rest) = line.strip_prefix("RAE9_OUTPUT ")
        {
            if output.is_some()
            {
                return Err("RAE9_DUPLICATE_VALUES".into());
            }
            let values: Vec<f32> = rest
                .split_whitespace()
                .map(|s| s.parse().map_err(|_| "RAE9_BAD_NUMBER".to_string()))
                .collect::<Result<_, _>>()?;
            if values.len() != width || !values.iter().all(|v| v.is_finite())
            {
                return Err("RAE9_INVALID_VECTOR".into());
            }
            output = Some(values);
        }
    }
    match (record, output)
    {
        (Some(r), Some(v)) => Ok((r, v)),
        _ => Err("RAE9_INCOMPLETE_RESULT".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_rejects_malformed_and_accepts_bounded_result() {
        let (r, vec) = parse_result(
            "RAE9_RESULT route=bounded_sparse keys=3 executed_keys=2 scored_keys=2 max_error_bound=0.01\nRAE9_OUTPUT 1.0 -2.0\n", 2
        ).unwrap();
        assert_eq!(r.executed_keys, 2);
        assert_eq!(vec, [1.0, -2.0]);
        assert!(parse_result("RAE9_RESULT keys=3\nRAE9_OUTPUT 0\n", 1).is_err());
        assert!(parse_result(
            "RAE9_RESULT route=bounded_sparse keys=3 executed_keys=2 scored_keys=2 max_error_bound=NaN\nRAE9_OUTPUT 0\n", 1
        ).is_err());
    }
    #[test]
    fn private_capture_is_removed_after_use() {
        let path = {
            let c = Capture::create().unwrap();
            assert!(c.path.is_file());
            c.path.clone()
        };
        assert!(!path.exists());
    }
}
