//! Unified, process-global compute-capability registry.
//!
//! The registry reports compute paths known to the current process. It is a
//! diagnostic/provenance surface, not a unified execution dispatcher: CPU SIMD,
//! portable GPU and CUDA retain their own execution abstractions.
//!
//! The first access seeds one CPU SIMD entry from
//! `scirust_simd::dispatch::detect_backend()`. GPU/CUDA integrations register
//! their own entries. Availability is deliberately tri-state:
//!
//! - `Some(true)`: the path was probed and reported usable;
//! - `Some(false)`: the path was probed and reported unusable;
//! - `None`: no usability result has been registered yet.
//!
//! `compiled` is independent of `available`; callers must not treat compilation
//! as evidence that a GPU/CUDA path executed on physical hardware.

use scirust_simd::dispatch::detect_backend;
use std::sync::{Mutex, OnceLock};

/// Compute domain represented by a [`Capability`].
///
/// # Examples
///
/// ```
/// use scirust_core::compute_capability::ComputeDomain;
/// assert_eq!(ComputeDomain::CpuSimd.label(), "cpu-simd");
/// ```
///
/// ```
/// use scirust_core::compute_capability::ComputeDomain;
/// assert!(ComputeDomain::CpuSimd < ComputeDomain::GpuPortable);
/// assert!(ComputeDomain::GpuPortable < ComputeDomain::Cuda);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComputeDomain {
    /// CPU SIMD tier selected by runtime CPU-feature detection.
    CpuSimd,
    /// Portable GPU compute path registered by the GPU integration.
    GpuPortable,
    /// CUDA/Tensor-core path registered by the CUDA/GPU integration.
    Cuda,
}

impl ComputeDomain {
    /// Returns the stable short label used by [`capability_summary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scirust_core::compute_capability::ComputeDomain;
    /// assert_eq!(ComputeDomain::GpuPortable.label(), "gpu-portable");
    /// ```
    ///
    /// ```
    /// use scirust_core::compute_capability::ComputeDomain;
    /// assert_eq!(ComputeDomain::Cuda.label(), "cuda");
    /// ```
    #[must_use]
    pub fn label(self) -> &'static str {
        match self
        {
            ComputeDomain::CpuSimd => "cpu-simd",
            ComputeDomain::GpuPortable => "gpu-portable",
            ComputeDomain::Cuda => "cuda",
        }
    }
}

/// One compute path known to the current process.
///
/// `compiled` states whether the registering integration says the code path is
/// present. [`available`](Capability::available) records probe status separately;
/// these fields intentionally do not imply one another.
///
/// # Examples
///
/// ```
/// use scirust_core::compute_capability::{Capability, ComputeDomain};
/// let cap = Capability {
///     domain: ComputeDomain::GpuPortable,
///     label: "wgpu".into(),
///     compiled: true,
///     available: None,
///     detail: "not probed yet".into(),
/// };
/// assert!(cap.compiled);
/// assert_eq!(cap.available, None);
/// ```
///
/// ```
/// use scirust_core::compute_capability::{Capability, ComputeDomain};
/// let cap = Capability {
///     domain: ComputeDomain::Cuda,
///     label: "cuda".into(),
///     compiled: true,
///     available: Some(false),
///     detail: "probe failed".into(),
/// };
/// assert_eq!(cap.available, Some(false));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Capability {
    /// Domain to which this compute path belongs.
    pub domain: ComputeDomain,
    /// Identifier within the domain, for example `"wgpu"`.
    pub label: String,
    /// Whether the registering integration reports this path compiled in.
    pub compiled: bool,
    /// Probe result: usable, unusable, or not yet probed/registered.
    pub available: Option<bool>,
    /// Free-form diagnostic/provenance detail such as an adapter name.
    pub detail: String,
}

fn registry() -> &'static Mutex<Vec<Capability>> {
    static REGISTRY: OnceLock<Mutex<Vec<Capability>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(vec![Capability {
            domain: ComputeDomain::CpuSimd,
            label: detect_backend().label().to_string(),
            compiled: true,
            available: Some(true),
            detail: "runtime CPU-feature detection (scirust_simd::dispatch::detect_backend)"
                .to_string(),
        }])
    })
}

/// Registers or replaces a process-global compute capability.
///
/// Entries are keyed by `(domain, label)`. Re-registering the same key replaces
/// the complete previous entry, allowing a later probe to refine an earlier
/// `available: None` announcement. Other keys are preserved.
///
/// The registry is shared by all callers in the process; this function is not a
/// per-object configuration method.
///
/// # Panics
///
/// Panics if the process-global capability-registry mutex has previously been
/// poisoned by a panic while held.
///
/// # Examples
///
/// ```
/// use scirust_core::compute_capability::{register_capability, compute_capabilities, Capability, ComputeDomain};
/// let label = "doc-example-unprobed";
/// register_capability(Capability {
///     domain: ComputeDomain::GpuPortable,
///     label: label.into(), compiled: true, available: None, detail: "announced".into(),
/// });
/// assert!(compute_capabilities().iter().any(|c| c.label == label && c.available.is_none()));
/// ```
///
/// Re-registering the same key is an upsert, not an append:
///
/// ```
/// use scirust_core::compute_capability::{register_capability, compute_capabilities, Capability, ComputeDomain};
/// let label = "doc-example-upsert";
/// register_capability(Capability {
///     domain: ComputeDomain::Cuda, label: label.into(), compiled: true,
///     available: None, detail: "announced".into(),
/// });
/// register_capability(Capability {
///     domain: ComputeDomain::Cuda, label: label.into(), compiled: true,
///     available: Some(false), detail: "probed".into(),
/// });
/// let matches: Vec<_> = compute_capabilities().into_iter().filter(|c| c.label == label).collect();
/// assert_eq!(matches.len(), 1);
/// assert_eq!(matches[0].available, Some(false));
/// ```
pub fn register_capability(cap: Capability) {
    let mut reg = registry().lock().expect("capability registry poisoned");
    match reg
        .iter_mut()
        .find(|c| c.domain == cap.domain && c.label == cap.label)
    {
        Some(existing) => *existing = cap,
        None => reg.push(cap),
    }
}

/// Returns a cloned snapshot sorted by `(domain, label)`.
///
/// Mutating the returned vector or its entries does not modify the global
/// registry. The CPU SIMD capability is seeded on first registry access and is
/// therefore present in every successful snapshot.
///
/// # Panics
///
/// Panics if the process-global capability-registry mutex is poisoned.
///
/// # Examples
///
/// ```
/// use scirust_core::compute_capability::{compute_capabilities, ComputeDomain};
/// let caps = compute_capabilities();
/// assert!(caps.iter().any(|c| c.domain == ComputeDomain::CpuSimd && c.available == Some(true)));
/// ```
///
/// ```
/// use scirust_core::compute_capability::compute_capabilities;
/// let caps = compute_capabilities();
/// assert!(caps.windows(2).all(|pair| (pair[0].domain, &pair[0].label) <= (pair[1].domain, &pair[1].label)));
/// ```
#[must_use]
pub fn compute_capabilities() -> Vec<Capability> {
    let reg = registry().lock().expect("capability registry poisoned");
    let mut out = reg.clone();
    out.sort_by(|a, b| (a.domain, &a.label).cmp(&(b.domain, &b.label)));
    out
}

/// Formats all known capabilities as a deterministic one-line summary.
///
/// Each entry has `domain:label=status`, where status is `yes`, `no`, or
/// `unprobed` according to [`Capability::available`]. Entries follow the same
/// `(domain, label)` order as [`compute_capabilities`]. The summary reports
/// registered state only; it does not itself probe GPU/CUDA hardware.
///
/// # Panics
///
/// Panics if [`compute_capabilities`] cannot acquire the poisoned registry mutex.
///
/// # Examples
///
/// ```
/// use scirust_core::compute_capability::capability_summary;
/// let summary = capability_summary();
/// assert!(summary.contains("cpu-simd:"));
/// ```
///
/// ```
/// use scirust_core::compute_capability::{capability_summary, register_capability, Capability, ComputeDomain};
/// register_capability(Capability {
///     domain: ComputeDomain::GpuPortable, label: "doc-summary".into(), compiled: true,
///     available: None, detail: String::new(),
/// });
/// assert!(capability_summary().contains("gpu-portable:doc-summary=unprobed"));
/// ```
#[must_use]
pub fn capability_summary() -> String {
    compute_capabilities()
        .iter()
        .map(|c| {
            let avail = match c.available
            {
                Some(true) => "yes",
                Some(false) => "no",
                None => "unprobed",
            };
            format!("{}:{}={avail}", c.domain.label(), c.label)
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_simd_tier_is_always_seeded_from_the_real_detector() {
        let caps = compute_capabilities();
        let cpu: Vec<_> = caps
            .iter()
            .filter(|c| c.domain == ComputeDomain::CpuSimd)
            .collect();
        assert!(!cpu.is_empty(), "CPU tier must always be present");
        assert!(
            cpu.iter()
                .any(|c| c.label == detect_backend().label() && c.available == Some(true)),
            "seeded CPU entry must match the live detector: {cpu:?}"
        );
    }

    #[test]
    fn register_is_an_upsert_keyed_on_domain_and_label() {
        let label = "test-upsert-gpu";
        register_capability(Capability {
            domain: ComputeDomain::GpuPortable,
            label: label.to_string(),
            compiled: true,
            available: None,
            detail: "announced".into(),
        });
        register_capability(Capability {
            domain: ComputeDomain::GpuPortable,
            label: label.to_string(),
            compiled: true,
            available: Some(true),
            detail: "probed".into(),
        });
        let matches: Vec<_> = compute_capabilities()
            .into_iter()
            .filter(|c| c.label == label)
            .collect();
        assert_eq!(matches.len(), 1, "upsert must not duplicate: {matches:?}");
        assert_eq!(matches[0].available, Some(true));
        assert_eq!(matches[0].detail, "probed");
    }

    #[test]
    fn snapshot_is_deterministically_sorted_and_summary_reports_every_entry() {
        register_capability(Capability {
            domain: ComputeDomain::Cuda,
            label: "test-summary-cuda".into(),
            compiled: true,
            available: Some(false),
            detail: String::new(),
        });
        let caps = compute_capabilities();
        let mut sorted = caps.clone();
        sorted.sort_by(|a, b| (a.domain, &a.label).cmp(&(b.domain, &b.label)));
        assert_eq!(caps, sorted, "snapshot must come out sorted");
        let summary = capability_summary();
        assert!(summary.contains("cuda:test-summary-cuda=no"), "{summary}");
        assert!(summary.contains("cpu-simd:"), "{summary}");
    }
}
