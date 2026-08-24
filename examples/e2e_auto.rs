//! End-to-end smoke test for auto-downloaded heavy models.
//!
//! Usage: e2e_auto <pdf> [work_dir]
//!
//! Runs RoutingMode::Auto with NO explicit model paths, so RapidLayout,
//! RapidOCR and RapidTable must all come from HuggingFace auto-download.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let pdf = std::env::args().nth(1).expect("usage: e2e_auto <pdf> [work_dir]");
    let work = std::env::args().nth(2).unwrap_or_else(|| {
        std::env::temp_dir().join("bobine_test").join("e2e_out").display().to_string()
    });
    let cache = std::env::temp_dir().join("bobine_test").join("cache");

    // No set_layout_model / set_ocr_models / set_table_model calls:
    // everything must resolve via auto-download or fail loudly here.
    let mode = match std::env::var("BOB_MODE").as_deref() {
        Ok("never") => bobine::RoutingMode::Never,
        _ => bobine::RoutingMode::Always,
    };
    let mut conv = bobine::HybridConverter::new(
        bobine::ConverterConfig { routing_mode: mode, ..Default::default() },
        &cache,
    );
    println!("mode = {mode:?}");
    let t0 = Instant::now();
    let md = conv.convert_pdf(std::path::Path::new(&pdf), std::path::Path::new(&work))?;
    println!("=== converted in {:.1?}, {} bytes markdown ===", t0.elapsed(), md.len());
    let dump = std::env::temp_dir().join(format!(
        "bobine_md_{}.md",
        std::env::var("BOB_MODE").unwrap_or_else(|_| "always".into())
    ));
    std::fs::write(&dump, &md)?;
    println!("dumped to {}", dump.display());
    println!("{}", &md[..md.len().min(1500)]);
    Ok(())
}
