// Integration tests for bobine_rs.
//
// NOTE: PDF tests are skipped due to a stack buffer overrun in pdf_oxide
// 0.3.77 on these specific arXiv PDFs. The Python pdf_oxide bindings work
// fine — the bug is in the Rust crate's rendering pipeline. Re-enable
// after upgrading pdf_oxide.

use std::path::PathBuf;

use bobine::{ConverterConfig, HybridConverter, RoutingMode};

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join("bobine_test")
}

// ======================================================================
// Config defaults
// ======================================================================

#[test]
fn default_config_is_auto_mode() {
    let c = ConverterConfig::default();
    assert_eq!(c.routing_mode, RoutingMode::Auto);
}

#[test]
fn config_fields_accessible() {
    let mut c = ConverterConfig::default();
    assert!(c.extract_images);
    assert!(c.use_onnx);
    c.render_dpi = 150;
    assert_eq!(c.render_dpi, 150);
}

// ======================================================================
// Text file conversion
// ======================================================================

#[test]
fn text_file_conversion() {
    let config = ConverterConfig::default();
    let mut conv = HybridConverter::new(config, &temp_dir().join("cache"));
    let txt = temp_dir().join("test_hello.txt");
    std::fs::write(&txt, "# Hello\n\nWorld").unwrap();

    let md = conv.convert(&txt, &temp_dir().join("text_out")).unwrap();
    assert_eq!(md, "# Hello\n\nWorld");
}

#[test]
fn text_file_with_unicode() {
    let config = ConverterConfig::default();
    let mut conv = HybridConverter::new(config, &temp_dir().join("cache"));
    let txt = temp_dir().join("unicode.txt");
    std::fs::write(&txt, "αβγ δ\n\n数学").unwrap();

    let md = conv.convert(&txt, &temp_dir().join("unicode_out")).unwrap();
    assert_eq!(md, "αβγ δ\n\n数学");
}

// ======================================================================
// Error paths
// ======================================================================

#[test]
fn missing_file_errors() {
    let config = ConverterConfig::default();
    let mut conv = HybridConverter::new(config, &temp_dir().join("cache"));
    let result = conv.convert(
        &PathBuf::from("/nonexistent/file.pdf"),
        &temp_dir().join("out"),
    );
    assert!(result.is_err());
}

// ======================================================================
// PDF tests (skipped — pdf_oxide 0.3.77 stack overflow on these files)
// ======================================================================
// PDF conversion (fast path through pdf_oxide)
//
// NOTE: requires a modern onnxruntime.dll (>=1.19). If cargo picks up a
// stale DLL from PATH, set ORT_DYLIB_PATH, e.g.:
//   ORT_DYLIB_PATH=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll
// ======================================================================

/// Pure fast-path: ONNX disabled entirely — no DLL or models needed.
#[test]
fn pdf_fast_path_no_onnx() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut config = ConverterConfig::default();
    config.use_onnx = false;
    let mut conv = HybridConverter::new(config, &temp_dir().join("cache"));
    let work = temp_dir().join("pdf_no_onnx");
    let _ = std::fs::create_dir_all(&work);

    let md = conv
        .convert_pdf(&fixtures.join("2608.05540.pdf"), &work)
        .expect("convert_pdf should succeed on full arXiv PDF");
    assert!(!md.is_empty());
    assert!(md.contains("Abstract") || md.contains("abstract") || md.contains("Introduction"));
}

/// Default config (AUTO mode): may attempt layout/OCR model loading and
/// gracefully falls back to the fast path when models are unavailable.
#[test]
fn pdf_fast_path_full_paper() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let config = ConverterConfig::default();
    let mut conv = HybridConverter::new(config, &temp_dir().join("cache"));
    let work = temp_dir().join("pdf_full");
    let _ = std::fs::create_dir_all(&work);

    let md = conv
        .convert_pdf(&fixtures.join("2608.05540.pdf"), &work)
        .expect("convert_pdf should succeed on full arXiv PDF");
    assert!(!md.is_empty());
}
