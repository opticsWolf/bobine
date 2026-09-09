//! Dump layout regions and text-layer overlap statistics for one or all
//! pages of a PDF. Diagnostic for region-overlap dedup (ALWAYS mode).
//!
//! Usage: dump_regions <pdf> <page|all>

use bobine::ConverterConfig;
use bobine::pdf_source::PdfSource;
use pdf_oxide::api::Pdf;

#[derive(Clone)]
struct RegionPt {
    label: String,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pdf_path = std::env::args().nth(1).expect("pdf");
    let which = std::env::args().nth(2).unwrap_or_else(|| "all".into());
    let cache = std::env::temp_dir().join("bobine_test").join("cache");

    let mut engine = bobine::engine::OnnxEngine::new(&ConverterConfig::default(), &cache);
    engine.ensure_models()?;

    let bytes = std::fs::read(&pdf_path)?;
    let mut pdf = Pdf::from_bytes(bytes)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let pages: Vec<usize> = if which == "all" {
        (0..pdf.page_count()?).collect()
    } else {
        vec![which.parse()?]
    };

    let mut tot_emitted = 0usize;
    let mut tot_covered = 0usize;
    for page in pages {
        let img = image::load_from_memory(&pdf.render_png(page, 300)?)?;
        let regions = engine.layout_regions(&img)?;
        let scale = 300.0_f32 / 72.0;
        println!("== page {page}: {} regions", regions.len());
        let pts: Vec<RegionPt> = regions
            .iter()
            .map(|r| RegionPt {
                label: r.label.clone(),
                x: r.x0 / scale,
                y: r.y0 / scale,
                w: (r.x1 - r.x0) / scale,
                h: (r.y1 - r.y0) / scale,
            })
            .collect();
        for (i, r) in regions.iter().enumerate() {
            let p = &pts[i];
            let la = p.label.as_str();
            let conf = r.confidence;
            let (x, y, w, h) = (p.x, p.y, p.w, p.h);
            let area = p.w * p.h;
            println!(
                "  [{i:2}] {la:<16} conf={conf:.2} pt=({x:4.0},{y:4.0}) {w:4.0}x{h:4.0} area={area:6.0}"
            );
        }
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let (a, b) = (&pts[i], &pts[j]);
                let ix0 = a.x.max(b.x);
                let iy0 = a.y.max(b.y);
                let ix1 = (a.x + a.w).min(b.x + b.w);
                let iy1 = (a.y + a.h).min(b.y + b.h);
                if ix1 <= ix0 || iy1 <= iy0 {
                    continue;
                }
                let inter = (ix1 - ix0) * (iy1 - iy0);
                let smaller = (a.w * a.h).min(b.w * b.h);
                let ratio = inter / smaller.max(1e-6);
                if ratio >= 0.25 {
                    let la = a.label.as_str();
                    let lb = b.label.as_str();
                    println!(
                        "  OVERLAP [{i:2} x {j:2}] {la:?} x {lb:?} ratio={ratio:.2} (inter={inter:.0}pt2)"
                    );
                }
            }
        }
        // Char-level multiplicity under the same Intersects semantics as
        // text_in_rect: how many regions would each covered char be emitted by.
        let chars = pdf.chars(page)?;
        let mut emitted = 0usize;
        let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
        for c in &chars {
            let cb = c.bbox;
            let n = pts
                .iter()
                .filter(|p| {
                    cb.x < p.x + p.w
                        && cb.x + cb.width > p.x
                        && cb.y < p.y + p.h
                        && cb.y + cb.height > p.y
                })
                .count();
            if n > 0 {
                emitted += n;
            }
            *hist.entry(n).or_default() += 1;
        }
        let zero = hist.get(&0).copied().unwrap_or(0);
        let covered: usize = hist.values().sum::<usize>() - zero;
        tot_emitted += emitted;
        tot_covered += covered;
        println!(
            "  CHARS total={} covered={} emitted_if_all_emit={} ratio={:.2} hist={:?}",
            chars.len(),
            covered,
            emitted,
            if covered == 0 {
                0.0
            } else {
                emitted as f32 / covered as f32
            },
            hist
        );
    }
    println!(
        "TOTAL: emitted={} covered={} overall ratio={:.2}",
        tot_emitted,
        tot_covered,
        if tot_covered == 0 {
            0.0
        } else {
            tot_emitted as f32 / tot_covered as f32
        }
    );
    Ok(())
}
