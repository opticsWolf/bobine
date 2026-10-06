//! Hub-cache acquisition (cache alignment across the stack).
//!
//! Every converter-model fetch goes through the standard HuggingFace hub
//! cache in cache-mode — no `local_dir` side-caches, no destination
//! probes. `cache_dir` parameters across this crate mean the **hub-cache
//! root** (override); where optional, `None` resolves the env default.
//! (Before 0.6.0 they meant a flat directory of model files, usually
//! `<output>/.cache` — a layout the hub cache cannot reuse, so the first
//! fetch after upgrading downloads once per machine.)
//!
//! [`hub_fetch`] is offline-first: it probes through embroider's shared
//! `cache_info_files` surface and returns the snapshot path with zero
//! network when the file is already cached ("using cached"); otherwise
//! the file downloads once per machine into the shared cache. Repeated
//! opens never re-download — hf-hub consults the blob store after a
//! cheap HEAD — so okfgraph's per-ingest temp-dir flow is fixed by
//! construction: models land in the shared cache, not a deleted dir.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::error::{BobineError, Result};

/// Build an HF client rooted at `cache_dir` (hub-cache root override).
pub(crate) fn hub_client(cache_dir: &Path) -> Result<hf_hub::HFClientSync> {
    Ok(hf_hub::HFClientBuilder::new()
        .cache_dir(cache_dir.to_path_buf())
        .build()
        .map(hf_hub::HFClientSync::from_inner)
        .map_err(|e| BobineError::Ort(format!("hf-hub init: {e}")))?
        .map_err(|e| BobineError::Ort(format!("hf-hub init: {e}")))?)
}

/// Fetch one file into the hub cache, returning its snapshot path to load.
///
/// Offline-first: an already-cached file returns its snapshot location
/// with zero network. A miss downloads once per machine (cache-mode, so
/// every later caller — and every okfgraph ingest — reuses the blob).
pub(crate) fn hub_fetch(
    cache_dir: &Path,
    owner: &str,
    name: &str,
    filename: &str,
    what: &str,
) -> Result<PathBuf> {
    let repo_id = format!("{owner}/{name}");
    // Offline probe through the shared surface: a hit never touches the
    // network. A probe error (impossible repo ids — all call sites use
    // constants) falls through to the download, which fails loudly.
    match embroider::cache_info_files(
        &repo_id,
        &[(filename, true)],
        None,
        Some(cache_dir.to_path_buf()),
    ) {
        Ok(rep) if rep.cached => {
            // The map is filename → Option<path>; flatten the double Option.
            let path = rep
                .files
                .get(filename)
                .cloned()
                .flatten()
                .ok_or_else(|| {
                    BobineError::Ort(format!("cache probe hit but {filename} has no path"))
                })?;
            info!("{what}: using cached {}", path.display());
            return Ok(path);
        }
        _ => {}
    }
    info!("Downloading {what} from {repo_id}...");
    let path = hub_client(cache_dir)?
        .model(owner, name)
        .download_file()
        .filename(filename.to_string())
        .send()
        .map_err(|e| BobineError::Ort(format!("download {what}: {e}")))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point HF at an unroutable endpoint: any accidental network access
    /// fails fast and loudly instead of hanging the suite.
    fn endpoint_guard() -> Option<String> {
        let saved = std::env::var("HF_ENDPOINT").ok();
        // SAFETY: test-only; no other thread reads HF_ENDPOINT here.
        unsafe { std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1"); }
        saved
    }

    fn endpoint_restore(saved: Option<String>) {
        // SAFETY: test-only; restores the pre-test value.
        unsafe {
            match saved {
                Some(v) => std::env::set_var("HF_ENDPOINT", v),
                None => std::env::remove_var("HF_ENDPOINT"),
            }
        }
    }

    /// Seed a fake hub-cache repo: refs/main + snapshots/<sha>/<file>.
    fn seed_repo(root: &Path, folder: &str, filename: &str, data: &[u8]) -> PathBuf {
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        let repo_dir = root.join(folder);
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), format!("{sha}\n")).unwrap();
        let snap = repo_dir.join("snapshots").join(sha);
        let p = snap.join(filename);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, data).unwrap();
        snap
    }

    #[test]
    fn hub_fetch_serves_seeded_cache_with_zero_network() {
        let saved = endpoint_guard();
        let root =
            std::env::temp_dir().join(format!("bobine-hub-hit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let snap = seed_repo(&root, "models--org--probe", "w.onnx", &[3u8; 48]);

        // The endpoint is unroutable: a hit must return without any request.
        let path = hub_fetch(&root, "org", "probe", "w.onnx", "probe-file").unwrap();
        assert_eq!(path, snap.join("w.onnx"));

        endpoint_restore(saved);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hub_fetch_miss_surfaces_the_download_error() {
        let saved = endpoint_guard();
        let root =
            std::env::temp_dir().join(format!("bobine-hub-miss-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // Empty cache + unroutable endpoint: the download fails fast and
        // the error names what was being fetched.
        let err = hub_fetch(&root, "org", "probe", "w.onnx", "probe-file")
            .unwrap_err()
            .to_string();
        assert!(err.contains("download probe-file"), "{err}");

        endpoint_restore(saved);
        let _ = std::fs::remove_dir_all(&root);
    }
}
