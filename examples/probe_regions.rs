//! Per-region timing probe for the ALWAYS pipeline.
//!
//! Usage: probe_regions <pdf> <page_index>

use std::time::Instant;

use bobine::pdf_source::PdfSource;
use pdf_oxide::api::Pdf;
use pdf_oxide::geometry::Rect;
use bobine::ConverterConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let pdf_path = std::env::args().nth(1).expect("pdf");
    let page: usize = std::env::args().nth(2).expect("page").parse()?;
    let cache = std::env::temp_dir().join("bobine_test").join("cache");

    let mut engine = bobine::engine::OnnxEngine::new(&ConverterConfig::default(), &cache);
    engine.ensure_models()?;

    let bytes = std::fs::read(&pdf_path)?;
    let mut pdf = Pdf::from_bytes(bytes).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let t0 = Instant::now();
    let img = {
        let png = pdf.render_png(page, 300)?;
        image::load_from_memory(&png)?
    };
    println!("render 300dpi: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let regions = engine.layout_regions(&img)?;
    println!("layout: {:?} ({} regions)", t0.elapsed(), regions.len());

    let scale = 300.0_f32 / 72.0;
    for r in &regions {
        let bbox = Rect::new(r.x0 / scale, r.y0 / scale, (r.x1 - r.x0) / scale, (r.y1 - r.y0) / scale);
        let lab = r.label.to_lowercase();
        println!(
            "region {lab:?} conf={:.2} {}x{}px",
            r.confidence,
            (r.x1 - r.x0) as u32,
            (r.y1 - r.y0) as u32,
        );
        // crop in render-pixel space like the converter does
        let px = |v: f32| (v * scale).max(0.0) as u32;
        let crop = image::imageops::crop_imm(
            &img,
            px(r.x0),
            px(r.y0),
            ((r.x1 - r.x0) * scale) as u32,
            ((r.y1 - r.y0) * scale) as u32,
        )
        .to_image();
        let crop_rgb = image::DynamicImage::ImageRgba8(crop.clone());
        if ["equation", "display_formula", "inline_formula", "isolate_formula", "formula"].iter().any(|k| lab.contains(k)) {
            let tmp = std::env::temp_dir().join("_probe_f.png");
            image::DynamicImage::ImageRgba8(crop).save(&tmp)?;
            if std::env::var("BOB_SKIP_FORMULA").is_ok() {
                println!("  FORMULA skipped (BOB_SKIP_FORMULA)");
                continue;
            }
            let t0 = Instant::now();
            let latex = engine.recognize_formula(&tmp)?;
            println!(
                "  FORMULA -> {:?}, latex {} chars",
                t0.elapsed(),
                latex.as_ref().map(|s| s.len()).unwrap_or(0)
            );
        } else if lab.contains("table") || lab.contains("figure") || lab.contains("image") {
            let t0 = Instant::now();
            let t = pdf.text_in_rect(page, bbox);
            println!("  {} text-layer chars={:?} ({:?})", lab, t.map(|s| s.len()), t0.elapsed());
        } else {
            let t0 = Instant::now();
            let t = pdf.text_in_rect(page, bbox);
            let n = t.as_ref().map(|s| s.len()).unwrap_or(0);
            println!("  TEXT chars={} ({:?})", n, t0.elapsed());
            if n == 0 {
                let dimg = image::DynamicImage::ImageRgba8(crop);
                let t0 = Instant::now();
                let lines = engine.ocr_lines(&dimg)?;
                println!("  OCR fallback: {} lines ({:?})", lines.len(), t0.elapsed());
            }
        }
    }
    Ok(())
}
