//! Where the ONNX Runtime library comes from.
//!
//! bobine loads ONNX Runtime dynamically (`ort` with `load-dynamic`), so
//! before the first session it has to answer one question: *which* library,
//! and where did that answer come from. The search itself is embroider's
//! locator — the same one every embroider model open and okfgraph run — so
//! bobine cannot drift from the rest of the stack. This module adds only
//! what is bobine's own:
//!
//! - the process-level bootstrap (once, before the first session),
//! - the CUDA lane's DLL directories ([`embroider::diag::prepare_cuda_dirs`]),
//!   which is what keeps a GPU build from failing on `cudnn64_9.dll`,
//! - the report shape the Python layer hands back (`runtime_info()`),
//!   including *who* chose the library: `env` (an explicit
//!   `ORT_DYLIB_PATH`), `bobine` (a located wheel), `none`.
//!
//! The locator never imports `onnxruntime`; it reads dist metadata and
//! looks for the library inside the package folder. That is the whole
//! point of it — a host process keeps one runtime, not two. When nothing is
//! found, `dylib = None` is the answer and nothing else is touched: no
//! probe, no loader call, no CUDA prep.

use std::path::PathBuf;
use std::sync::OnceLock;

/// The answer bobine gives once per process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeReport {
    /// The library `ort` will load, or `None` when nothing was found.
    pub dylib: Option<String>,
    /// `env`, `pip:<distribution>`, or `os` (not found).
    pub source: String,
    /// The distribution's version (`pip:` sources only).
    pub version: Option<String>,
    /// Who chose it: `env`, `bobine`, or `none`.
    pub set_by: &'static str,
    /// CUDA folders this process put on `PATH` before the first session.
    pub cuda_dirs: Vec<String>,
}

impl RuntimeReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "dylib": self.dylib,
            "source": self.source,
            "version": self.version,
            "set_by": self.set_by,
            "cuda_dirs": self.cuda_dirs,
        })
    }
}

/// `set_by` from the locator's answer — pure, so the table is testable.
pub fn set_by(source: &str, dylib: Option<&str>) -> &'static str {
    if source == "env" {
        "env"
    } else if dylib.is_some() {
        "bobine"
    } else {
        "none"
    }
}

/// The `prepended` list from [`embroider::diag::prepare_cuda_dirs`].
pub fn cuda_dirs_from(report: &serde_json::Value) -> Vec<String> {
    report
        .get("prepended")
        .and_then(|v| v.as_array())
        .map(|dirs| {
            dirs.iter()
                .filter_map(|d| d.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

static REPORT: OnceLock<RuntimeReport> = OnceLock::new();

/// Locate the runtime once per process and remember the answer.
///
/// *hints* are folders to search before the interpreter prefix (the Python
/// layer passes `sys.path`, so user site, `PYTHONPATH` and editable
/// layouts count). Idempotent: the first call wins, later ones return the
/// same report — a library choice made after the first session could not
/// take effect anyway.
pub fn bootstrap(hints: &[String]) -> &'static RuntimeReport {
    REPORT.get_or_init(|| {
        let hints: Vec<PathBuf> = hints
            .iter()
            .filter(|h| !h.is_empty())
            .map(PathBuf::from)
            .collect();
        // `set_env = true`: ort reads ORT_DYLIB_PATH when the first
        // session builds, and this runs before that.
        let located = embroider::diag::locate_runtime(&hints, true, true);
        // No library, no CUDA lane to prepare: `prepare_cuda_dirs` ends in
        // embroider's runtime report, which probes CUDA, and on a machine
        // whose only `onnxruntime` is a stale copy on the OS loader path
        // that probe makes ort panic (caught inside `cuda_available`, but
        // it prints a panic banner). Skipping it keeps "no runtime" quiet
        // and costs nothing — there are no DLLs to prepend.
        let cuda = if located.dylib.is_some() {
            embroider::diag::prepare_cuda_dirs(None)
        } else {
            serde_json::json!({
                "prepended": [],
                "candidates": 0,
                "dylib": serde_json::Value::Null,
            })
        };
        let dylib = located
            .dylib
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());
        RuntimeReport {
            set_by: set_by(&located.source, dylib.as_deref()),
            dylib,
            source: located.source,
            version: located.version,
            cuda_dirs: cuda_dirs_from(&cuda),
        }
    })
}

/// The report from [`bootstrap`], bootstrapping with no hints if the Python
/// layer never asked (a Rust-only caller, a test).
pub fn report() -> &'static RuntimeReport {
    bootstrap(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_by_names_the_chooser() {
        assert_eq!(set_by("env", Some("/lib/onnxruntime.so")), "env");
        assert_eq!(set_by("pip:onnxruntime-gpu", Some("/lib/onnxruntime.so")), "bobine");
        assert_eq!(set_by("os", None), "none");
        // A locator that reports `os` but somehow has a path is still
        // "bobine": the library is there, bobine is what found it.
        assert_eq!(set_by("os", Some("/lib/onnxruntime.so")), "bobine");
    }

    #[test]
    fn cuda_dirs_reads_the_prepended_list() {
        let report = serde_json::json!({
            "prepended": ["/a", "/b"],
            "candidates": 2,
            "dylib": serde_json::Value::Null,
        });
        assert_eq!(cuda_dirs_from(&report), vec!["/a".to_string(), "/b".to_string()]);
        assert!(cuda_dirs_from(&serde_json::json!({})).is_empty());
        assert!(cuda_dirs_from(&serde_json::json!({"prepended": [1, 2]})).is_empty());
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let first = bootstrap(&[]);
        let second = bootstrap(&["/nonexistent-hint".to_string()]);
        assert!(std::ptr::eq(first, second), "the first call must win");
        assert_eq!(first.set_by, second.set_by);
    }
}
