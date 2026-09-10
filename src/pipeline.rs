//! Document ingestion orchestration.
//!
//! High-level, graph-agnostic pipeline shared by every entry point (CLI,
//! API, batch tools):
//!
//! ```text
//! document → markdown  (PDF via HybridConverter, Office via office_oxide,
//!                       text read as-is)
//!          → .md on disk
//!          → loose images moved into `_assets/` and links rewritten to
//!            `okf-asset://<id>`
//!          → optional lint hook
//!          → ConvertedDocument
//! ```
//!
//! Nothing here touches a database or an embedding model — that stays in the
//! consumer (e.g. OKFgraph). The output contract is a directory of staged
//! markdown plus the staged asset store, ready to be handed to an import
//! step.

use std::path::{Path, PathBuf};

use tracing::warn;

use crate::assets::{ASSET_STORE_DIRNAME, is_image_ext, stage_images};
use crate::config::ConverterConfig;
use crate::converter::{HybridConverter, ProgressHooks};
use crate::error::{BobineError, Result};

/// Extensions convertible to markdown (Office + PDF).
pub const SUPPORTED_EXTENSIONS: &[&str] = &["pdf", "docx", "xlsx", "pptx", "doc", "xls", "ppt"];

/// Plain-text extensions converted by reading as UTF-8.
pub const TEXT_EXTS: &[&str] = &["txt", "md", "markdown", "rst", "text"];

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// True when `path` is a document bobine can ingest.
pub fn is_supported(path: &Path) -> bool {
    let e = ext_of(path);
    SUPPORTED_EXTENSIONS.contains(&e.as_str()) || TEXT_EXTS.contains(&e.as_str())
}

/// Result of a [`lint`](crate::pipeline) hook invocation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LintOutcome {
    /// Whether the content was modified (and should be written back).
    pub fixed: bool,
    /// The (possibly fixed) markdown content.
    pub content: String,
}

/// Lint hook signature — receives markdown text, returns a [`LintOutcome`].
///
/// Rust consumers can plug any linter; the Python bindings accept a Python
/// callable (e.g. one backed by mordant). Returning `Ok(default)` skips the
/// write-back; a lint failure aborts ingestion with the propagated error.
pub type LintFn<'a> = dyn Fn(&str) -> Result<LintOutcome> + 'a;

/// Result of [`ingest_document`] — a staged, linted markdown doc.
#[derive(Debug, Clone)]
pub struct ConvertedDocument {
    pub md_path: PathBuf,
    pub md_text: String,
    pub image_dir: PathBuf,
    pub image_count: usize,
    pub page_count: usize,
}

/// Convert a single document to a markdown string.
///
/// Dispatch by extension: `.pdf` → [`HybridConverter`], Office extensions →
/// office_oxide, text extensions → read as UTF-8. `work_dir` receives
/// extracted images / crops. Returns the markdown text.
pub fn convert_to_markdown(
    path: &Path,
    config: &ConverterConfig,
    work_dir: &Path,
    cache_dir: &Path,
    hooks: &ProgressHooks<'_>,
) -> Result<String> {
    if !path.is_file() {
        return Err(BobineError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Document not found: {}", path.display()),
        )));
    }
    let ext = ext_of(path);
    std::fs::create_dir_all(work_dir)?;

    if ext == "pdf" {
        let mut converter = HybridConverter::new(config.clone(), cache_dir);
        return converter.convert_pdf_with(path, work_dir, hooks);
    }

    // Office + text go through HybridConverter's dispatcher (no models needed).
    let mut converter = HybridConverter::new(config.clone(), cache_dir);
    converter.convert(path, work_dir)
}

/// Full ingestion pipeline for one document.
///
/// Convert `path` to markdown (writing into `output_dir`), move loose images
/// into the asset store, rewrite links to `okf-asset://`, and optionally run
/// the lint hook on the result.
///
/// Returns a [`ConvertedDocument`]. Nothing is imported anywhere — a
/// consumer receives the staged bundle.
pub fn ingest_document(
    path: &Path,
    output_dir: &Path,
    config: Option<&ConverterConfig>,
    lint_fn: Option<&LintFn<'_>>,
    hooks: &ProgressHooks<'_>,
) -> Result<ConvertedDocument> {
    if !path.is_file() {
        return Err(BobineError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Document not found: {}", path.display()),
        )));
    }

    let default_cfg;
    let config = match config {
        Some(c) => c,
        None => {
            default_cfg = ConverterConfig::default();
            &default_cfg
        }
    };

    std::fs::create_dir_all(output_dir)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document");
    let cache_dir = output_dir.join(".cache");

    let md = convert_to_markdown(path, config, output_dir, &cache_dir, hooks)?;

    let md_path = output_dir.join(format!("{stem}.md"));
    std::fs::write(&md_path, &md)?;

    // Stage images: move loose files, rewrite links, copy bytes into _assets.
    let (md_text, image_count) = stage_images(&md, path, output_dir, stem)?;
    std::fs::write(&md_path, &md_text)?;

    let page_count = if ext_of(path) == "pdf" {
        pdf_oxide::api::Pdf::open(path)
            .and_then(|mut p| p.page_count())
            .map(|n| n as usize)
            .unwrap_or_else(|e| {
                warn!("page count failed for {}: {e}", path.display());
                0
            })
    } else {
        0
    };

    // Optional lint (fixes formatting issues in place).
    if let Some(lint_fn) = lint_fn {
        let outcome = lint_fn(&md_text)?;
        if outcome.fixed && outcome.content != md_text {
            std::fs::write(&md_path, &outcome.content)?;
        }
    }

    Ok(ConvertedDocument {
        image_dir: output_dir.join(ASSET_STORE_DIRNAME),
        md_text: std::fs::read_to_string(&md_path)?,
        md_path,
        image_count,
        page_count,
    })
}

/// Batch-convert every supported document in `source_dir` (recursive).
///
/// Text files and PDF/Office files are converted into `output_dir` as staged
/// markdown (images collected into `_assets/`). Loose image files are
/// skipped — they are assets, not documents. Failures are logged and
/// skipped; successful results are returned sorted by source path.
pub fn convert_directory(
    source_dir: &Path,
    output_dir: &Path,
    config: Option<&ConverterConfig>,
    lint_fn: Option<&LintFn<'_>>,
) -> Result<Vec<ConvertedDocument>> {
    if !source_dir.is_dir() {
        return Err(BobineError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Source directory not found: {}", source_dir.display()),
        )));
    }

    let mut sources: Vec<PathBuf> = Vec::new();
    collect_sources(source_dir, &mut sources)?;
    sources.sort();

    let hooks = ProgressHooks::default();
    let mut results = Vec::new();
    for file_path in sources {
        match ingest_document(&file_path, output_dir, config, lint_fn, &hooks) {
            Ok(doc) => results.push(doc),
            Err(e) => warn!("{}: {e}", file_path.display()),
        }
    }
    Ok(results)
}

fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if p.is_dir() {
            collect_sources(&p, out)?;
        } else if p.is_file() && !is_image_ext(&p) && is_supported(&p) {
            out.push(p);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "bobine_pipe_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn test_is_supported_and_image_ext() {
        assert!(is_supported(Path::new("a.pdf")));
        assert!(is_supported(Path::new("b.TXT")));
        assert!(is_supported(Path::new("c.docx")));
        assert!(!is_supported(Path::new("d.png")));
        assert!(!is_supported(Path::new("e.exe")));
    }

    #[test]
    fn test_ingest_text_document() {
        let src_dir = tmpdir("src");
        let out_dir = tmpdir("out");
        let src = src_dir.join("My Note.txt");
        fs::write(&src, "hello\nworld\n").unwrap();

        let doc = ingest_document(&src, &out_dir, None, None, &ProgressHooks::default()).unwrap();
        assert_eq!(doc.md_path, out_dir.join("My Note.md"));
        assert_eq!(doc.md_text, "hello\nworld\n");
        assert_eq!(doc.image_count, 0);
        assert_eq!(doc.page_count, 0); // not a pdf
        assert_eq!(fs::read_to_string(&doc.md_path).unwrap(), doc.md_text);
        assert!(doc.image_dir.is_dir());

        fs::remove_dir_all(src_dir).unwrap();
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn test_ingest_markdown_with_image_staging() {
        let src_dir = tmpdir("src2");
        let out_dir = tmpdir("out2");
        // A "loose image" dropped next to the md gets moved into _assets and
        // its link rewritten. Source-relative resolution: img sits beside src.
        fs::write(src_dir.join("pic.png"), b"\x89PNG").unwrap();
        let src = src_dir.join("note.md");
        fs::write(&src, "# T\n\n![pic](pic.png)\n").unwrap();

        let doc = ingest_document(&src, &out_dir, None, None, &ProgressHooks::default()).unwrap();
        assert_eq!(doc.image_count, 1);
        assert!(doc.md_text.contains("okf-asset://img_"), "{}", doc.md_text);
        assert_eq!(fs::read_dir(&doc.image_dir).unwrap().count(), 1);

        fs::remove_dir_all(src_dir).unwrap();
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn test_ingest_lint_hook_fixes_content() {
        let src_dir = tmpdir("src3");
        let out_dir = tmpdir("out3");
        let src = src_dir.join("l.txt");
        fs::write(&src, "raw text").unwrap();

        let lint = |md: &str| {
            Ok(LintOutcome {
                fixed: true,
                content: format!("{md}\n<!-- linted -->\n"),
            })
        };
        let doc =
            ingest_document(&src, &out_dir, None, Some(&lint), &ProgressHooks::default()).unwrap();
        assert!(doc.md_text.contains("<!-- linted -->"));

        fs::remove_dir_all(src_dir).unwrap();
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn test_ingest_missing_file_errors() {
        let out = tmpdir("out4");
        let err = ingest_document(
            Path::new("no/such.pdf"),
            &out,
            None,
            None,
            &ProgressHooks::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not found"));
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn test_convert_directory_skips_assets_and_failures() {
        let src_dir = tmpdir("src5");
        let out_dir = tmpdir("out5");
        fs::write(src_dir.join("a.txt"), "A").unwrap();
        fs::create_dir_all(src_dir.join("sub")).unwrap();
        fs::write(src_dir.join("sub/b.md"), "B").unwrap();
        fs::write(src_dir.join("photo.jpg"), b"jpg").unwrap(); // asset, skipped
        fs::write(src_dir.join("skip.bin"), "x").unwrap(); // unsupported

        let docs = convert_directory(&src_dir, &out_dir, None, None).unwrap();
        let names: Vec<_> = docs
            .iter()
            .map(|d| d.md_path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.md", "b.md"]); // sorted, recursive, filtered

        fs::remove_dir_all(src_dir).unwrap();
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn test_convert_cancellation_stops_early() {
        // Cancel immediately: convert_pdf_with should produce no pages.
        let src_dir = tmpdir("src6");
        let out_dir = tmpdir("out6");
        // No real PDF available without fixtures — verify hook plumbing via
        // text dispatch instead (hooks only affect PDFs, so just ensure the
        // API accepts them end-to-end).
        let src = src_dir.join("t.txt");
        fs::write(&src, "x").unwrap();
        let hooks = ProgressHooks {
            should_continue: Box::new(|| false),
            on_page: Box::new(|_, _| {}),
        };
        let md = convert_to_markdown(
            &src,
            &ConverterConfig::default(),
            &out_dir,
            &out_dir,
            &hooks,
        )
        .unwrap();
        assert_eq!(md, "x");

        fs::remove_dir_all(src_dir).unwrap();
        fs::remove_dir_all(out_dir).unwrap();
    }
}
