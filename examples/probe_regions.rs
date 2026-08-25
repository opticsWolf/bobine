//! Per-region timing probe for the ALWAYS pipeline.
//!
//! Usage: probe_regions <pdf> <page_index>

use std::time::Instant;

use bobine::ConverterConfig;
use bobine::pdf_source::PdfSource;
use pdf_oxide::api::Pdf;
use pdf_oxide::geometry::Rect;

fn is_math_font_name(f: &str) -> bool {
    let u = f.to_uppercase();
    ["CMMI", "CMSY", "CMEX", "MSAM", "MSBM", "EUFM", "EUSM"]
        .iter()
        .any(|p| u.starts_with(p))
}

fn is_math_cp(c: char) -> bool {
    let u = c as u32;
    (0x0370..=0x03FF).contains(&u) // Greek
        || (0x2200..=0x22FF).contains(&u) // math operators
        || (0x1D400..=0x1D7FF).contains(&u) // math alphanumerics
        || (0x2A00..=0x2AFF).contains(&u)
}

/// Print math-character saturation stats for the page: if "plain text"
/// regions are full of math glyphs, layout class confusion is expected.
fn math_density(pdf: &mut Pdf, page: usize) {
    let chars = match pdf.chars(page) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut fonts: std::collections::BTreeMap<String, usize> = Default::default();
    for c in &chars {
        *fonts.entry(c.font_name.clone()).or_default() += 1;
    }
    println!("  FONTS: {:?}", fonts);
    let total = chars.len();
    let math = chars
        .iter()
        .filter(|c| is_math_font_name(&c.font_name) || is_math_cp(c.char))
        .count();
    println!(
        "  MATH-DENSITY: {}/{} chars = {:.1}% carry math signal",
        math,
        total,
        100.0 * math as f32 / total.max(1) as f32
    );
    // where do the display-size math boxes sit? (coarse: union of math chars
    // grouped into rows like SURGICAL does, printed as pt rects)
    let items: Vec<_> = chars
        .iter()
        .filter(|c| is_math_font_name(&c.font_name) || is_math_cp(c.char))
        .collect();
    if items.is_empty() {
        return;
    }
    let mut hs: Vec<f32> = items.iter().map(|c| c.bbox.height).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = hs[hs.len() / 2].max(10.0);
    let line_tol = 0.8 * med_h;
    let vgap_max = 1.5 * med_h;
    let hgap = 1.5 * med_h;
    let mut sorted = items.clone();
    sorted.sort_by(|a, b| {
        (a.bbox.y + a.bbox.height / 2.0)
            .partial_cmp(&(b.bbox.y + b.bbox.height / 2.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut lines: Vec<Vec<(f32, f32, f32, f32)>> = vec![];
    for c in sorted {
        let r = (c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height);
        let cy = r.1 + r.3 / 2.0;
        if let Some(l) = lines.last() {
            let lcy = l[0].1 + l[0].3 / 2.0;
            if (cy - lcy).abs() <= line_tol {
                lines.last_mut().unwrap().push(r);
                continue;
            }
        }
        lines.push(vec![r]);
    }
    let mut runs: Vec<(f32, f32, f32, f32)> = vec![];
    for ln in &lines {
        let x0 = ln.iter().map(|r| r.0).fold(f32::INFINITY, f32::min);
        let y0 = ln.iter().map(|r| r.1).fold(f32::INFINITY, f32::min);
        let x1 = ln
            .iter()
            .map(|r| r.0 + r.2)
            .fold(f32::NEG_INFINITY, f32::max);
        let y1 = ln
            .iter()
            .map(|r| r.1 + r.3)
            .fold(f32::NEG_INFINITY, f32::max);
        runs.push((x0, y0, x1 - x0, y1 - y0));
    }
    // vertical merge (fixpoint)
    let mut changed = true;
    while changed {
        changed = false;
        let mut out: Vec<(f32, f32, f32, f32)> = vec![];
        for b in &runs {
            let mut m = false;
            for u in &mut out {
                let hx = !(b.0 > u.0 + u.2 + hgap || b.0 + b.2 < u.0 - hgap);
                if !hx {
                    continue;
                }
                let gap = (b.1 - (u.1 + u.3)).max(u.1 - (b.1 + b.3)).max(0.0);
                if gap > vgap_max {
                    continue;
                }
                let nx = u.0.min(b.0);
                let ny = u.1.min(b.1);
                let nx1 = (u.0 + u.2).max(b.0 + b.2);
                let ny1 = (u.1 + u.3).max(b.1 + b.3);
                *u = (nx, ny, nx1 - nx, ny1 - ny);
                m = true;
                changed = true;
                break;
            }
            if !m {
                out.push(*b);
            }
        }
        runs = out;
    }
    println!("  MATH-DENSITY: {} merged math block(s):", runs.len());
    for r in &runs {
        println!(
            "    math box pt: x={:.0} y={:.0} w={:.0} h={:.0}",
            r.0, r.1, r.2, r.3
        );
    }
}

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
        if std::env::var("BOB_MATH_DENSITY").is_ok() {
            math_density(&mut pdf, page);
        }
        let png = pdf.render_png(page, 300)?;
        image::load_from_memory(&png)?
    };
    println!("render 300dpi: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let regions = engine.layout_regions(&img)?;
    println!("layout: {:?} ({} regions)", t0.elapsed(), regions.len());

    let scale = 300.0_f32 / 72.0;
    for (ridx, r) in regions.iter().enumerate() {
        let bbox = Rect::new(
            r.x0 / scale,
            r.y0 / scale,
            (r.x1 - r.x0) / scale,
            (r.y1 - r.y0) / scale,
        );
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
        if [
            "equation",
            "display_formula",
            "inline_formula",
            "isolate_formula",
            "formula",
        ]
        .iter()
        .any(|k| lab.contains(k))
        {
            let tmp = std::env::temp_dir().join("_probe_f.png");
            if std::env::var("BOB_SKIP_FORMULA").is_ok() {
                let out =
                    std::env::temp_dir().join(format!("bobine_formula_p{}_r{}.png", page, ridx));
                image::DynamicImage::ImageRgba8(crop).save(&out)?;
                println!("  FORMULA skipped, crop saved: {}", out.display());
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
            println!(
                "  {} text-layer chars={:?} ({:?})",
                lab,
                t.map(|s| s.len()),
                t0.elapsed()
            );
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
